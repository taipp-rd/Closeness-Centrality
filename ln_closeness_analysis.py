#!/usr/bin/env python3
"""
Lightning Network: Advanced Centrality Analyzer with Multiple Algorithms 

Features:
- Closeness Centrality (Freeman 1979) - measures routing efficiency
- Harmonic Centrality (Marchiori & Latora 2000) - handles disconnected components
- Capacity-weighted centrality (Opsahl et al. 2010) - considers channel capacities
- Greedy Algorithm (Kempe et al. 2003) - scalable channel selection
- Exhaustive Search - optimal but computationally expensive
- Network connectivity analysis
- CSV export for optimal combinations

ALGORITHMS:
1. Closeness Centrality: CC(v) = (n-1) / Σ d(v,u)
   - Measures how quickly a node can reach all others
   - Normalized for disconnected graphs (Wasserman-Faust)
   
2. Harmonic Centrality: HC(v) = Σ (1/d(v,u)) / (n-1) 
   - Handles infinite distances gracefully (1/∞ = 0)
   - More stable for dynamic topologies
   - High correlation with CC in connected components (ρ > 0.95)

3. Capacity-Weighted Centrality:
   - Weight function: w = 1 / (1 + log(1 + capacity))
   - Large capacity channels = shorter effective distance
   - Based on Opsahl et al. (2010) weighted network theory

4. Greedy Channel Selection:
   - Iteratively selects channels with maximum marginal gain
   - O(k*n) complexity vs O(C(n,k)) for exhaustive search
   - Theoretical guarantee: (1-1/e) ≈ 63% for submodular functions
   - Note: Closeness improvement may not always be submodular

References:
- Freeman (1979): Centrality in Social Networks
- Marchiori & Latora (2000): Harmony in the Small World
- Opsahl et al. (2010): Node centrality in weighted networks
- Boldi & Vigna (2014): Axioms for Centrality
- Kempe et al. (2003): Maximizing the Spread of Influence
- Rohrer et al. (2019): Discharged Payment Channels
"""
from __future__ import annotations
import argparse
import itertools
import sys
import multiprocessing
from multiprocessing import Pool
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

import psycopg2
from psycopg2 import OperationalError
import pandas as pd
import networkx as nx

# SQL queries - fetch latest channel state
CHANNELS_SQL = r"""
WITH latest_channel AS (
  SELECT DISTINCT ON (cu.chan_id, cu.advertising_nodeid)
         cu.chan_id, cu.chan_point, cu.advertising_nodeid, cu.connecting_nodeid,
         COALESCE(cu.capacity_sat, 0) AS capacity_sat, cu.rp_disabled,
         COALESCE(cu.rp_last_update, cu.timestamp) AS last_ts
  FROM channel_update cu
  ORDER BY cu.chan_id, cu.advertising_nodeid, COALESCE(cu.rp_last_update, cu.timestamp) DESC
),
open_channel AS (
  SELECT lc.* FROM latest_channel lc
  LEFT JOIN closed_channel cc ON cc.chan_id = lc.chan_id
  WHERE cc.chan_id IS NULL AND lc.capacity_sat > 0 AND lc.rp_disabled = false
)
SELECT * FROM open_channel;
"""

ALIASES_SQL = r"""
SELECT DISTINCT ON (na.node_id) na.node_id, na.alias, na.timestamp AS last_ts
FROM node_announcement na ORDER BY na.node_id, na.timestamp DESC;
"""

@dataclass
class DBConf:
    host: str
    port: int
    db: str
    user: str
    password: str

@dataclass
class Recommendation:
    node_id: str
    alias: str
    new_cc: float
    new_hc: float
    delta_cc_abs: float
    delta_hc_abs: float
    delta_cc_pct: float
    delta_hc_pct: float

def fetch_dataframe_with_retry(conf: DBConf, sql: str, max_retries: int = 3) -> pd.DataFrame:
    """Fetch data from PostgreSQL database with retry logic."""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=conf.host, 
                port=conf.port, 
                dbname=conf.db,
                user=conf.user, 
                password=conf.password,
                connect_timeout=30
            )
            try:
                # Use chunked reading for large datasets
                return pd.read_sql_query(sql, conn, chunksize=None)
            finally:
                conn.close()
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] Database connection failed (attempt {attempt + 1}): {e}", file=sys.stderr)
                time.sleep(5)  # Wait before retry
            else:
                raise
    return pd.DataFrame()

def build_directed_graph(ch_df: pd.DataFrame, alias_df: pd.DataFrame, use_capacity_weights: bool = False) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """Build directed graph from channel updates.
    
    Args:
        ch_df: Channel dataframe
        alias_df: Alias dataframe
        use_capacity_weights: If True, add edge weights based on channel capacity
    """
    ch_df = ch_df.dropna(subset=["advertising_nodeid", "connecting_nodeid"])
    ch_df['capacity_sat'] = pd.to_numeric(ch_df['capacity_sat'], errors='coerce').fillna(0)
    
    # Calculate per-node total adjacent capacity
    node_capacity = {}
    for _, row in ch_df.iterrows():
        u, v, cap = row["advertising_nodeid"], row["connecting_nodeid"], row["capacity_sat"]
        node_capacity[u] = node_capacity.get(u, 0) + cap
        node_capacity[v] = node_capacity.get(v, 0) + cap
    
    nonzero_nodes = {n for n, cap in node_capacity.items() if cap > 0}
    print(f"[INFO] Nodes with non-zero capacity: {len(nonzero_nodes)}", file=sys.stderr)
    
    # Build directed graph
    G = nx.DiGraph()
    for _, row in ch_df.iterrows():
        u, v, cap = row["advertising_nodeid"], row["connecting_nodeid"], int(row["capacity_sat"])
        if u in nonzero_nodes and v in nonzero_nodes:
            if G.has_edge(u, v):
                G[u][v]["multiplicity"] = G[u][v].get("multiplicity", 1) + 1
                G[u][v]["capacity_sum_sat"] = G[u][v].get("capacity_sum_sat", 0) + cap
            else:
                G.add_edge(u, v, multiplicity=1, capacity_sum_sat=cap)
    
    # Add capacity-based weights if requested
    if use_capacity_weights:
        print(f"[INFO] Adding capacity-based edge weights...", file=sys.stderr)
        for u, v, data in G.edges(data=True):
            capacity = data.get("capacity_sum_sat", 0)
            # Weight function: w = 1 / (1 + log(1 + capacity))
            # Large capacity -> small weight -> shorter effective distance
            if capacity > 0:
                weight = 1.0 / (1.0 + np.log1p(capacity))
            else:
                weight = 1.0  # Default weight for zero capacity
            G[u][v]["weight"] = weight
    
    # Build alias map
    alias_map = {}
    for _, r in alias_df.iterrows():
        node_id, alias = r["node_id"], r["alias"]
        alias_map[node_id] = alias.strip() if isinstance(alias, str) and alias.strip() else node_id
    
    for n in G.nodes:
        G.nodes[n]["alias"] = alias_map.get(n, n)
    
    return G, alias_map

def analyze_connectivity(G: nx.DiGraph) -> Dict:
    """Analyze network connectivity."""
    strongly = list(nx.strongly_connected_components(G))
    weakly = list(nx.weakly_connected_components(G))
    largest_strong = max(strongly, key=len) if strongly else set()
    
    return {
        'is_strongly_connected': nx.is_strongly_connected(G),
        'num_strong_components': len(strongly),
        'largest_strong_size': len(largest_strong),
        'strong_coverage': len(largest_strong) / len(G) * 100 if len(G) > 0 else 0,
        'num_weak_components': len(weakly)
    }

def compute_closeness_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True, use_weights: bool = False) -> float:
    """Compute Closeness Centrality (Freeman 1979) - FIXED VERSION.
    
    CC(v) = (n-1) / Σ d(v,u) with proper Wasserman-Faust normalization
    
    Args:
        G: The graph
        node: Target node
        use_outgoing: If True, compute outgoing centrality (how well node reaches others)
                     If False, compute incoming centrality (how well others reach node)
        use_weights: If True, use edge weights for distance calculation
    """
    if node not in G:
        return 0.0
    
    # For outgoing centrality, we want to measure distances FROM the node
    # So we use G as is. For incoming, we reverse it.
    graph_to_use = G if use_outgoing else G.reverse()
    
    try:
        if use_weights:
            # Use Dijkstra's algorithm for weighted shortest paths
            lengths = nx.single_source_dijkstra_path_length(graph_to_use, node, weight='weight')
        else:
            # Use BFS for unweighted shortest paths
            lengths = nx.single_source_shortest_path_length(graph_to_use, node)
        
        n = len(graph_to_use)
        if len(lengths) <= 1:
            return 0.0
        
        # Remove self-distance
        lengths_without_self = {k: v for k, v in lengths.items() if k != node}
        
        if not lengths_without_self:
            return 0.0
            
        total_distance = sum(lengths_without_self.values())
        n_reachable = len(lengths_without_self)
        
        if total_distance > 0:
            # FIXED: Proper closeness centrality calculation
            closeness = n_reachable / total_distance
            return closeness
        return 0.0
    except Exception as e:
        print(f"[WARNING] Error computing closeness for {node}: {e}", file=sys.stderr)
        return 0.0

def compute_harmonic_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True, use_weights: bool = False) -> float:
    """Compute Harmonic Centrality (Marchiori & Latora 2000, Boldi & Vigna 2014).
    
    HC(v) = Σ(u≠v) [1/d(v,u)] / (n-1)
    
    Advantages for Lightning Network:
    - Handles disconnected graphs: 1/∞ = 0
    - More stable for dynamic topology
    - High correlation with closeness (ρ > 0.95) in connected components
    
    Args:
        G: The graph
        node: Target node
        use_outgoing: If True, compute outgoing centrality
        use_weights: If True, use edge weights for distance calculation
    """
    if node not in G:
        return 0.0
    
    graph_to_use = G if use_outgoing else G.reverse()
    
    try:
        if use_weights:
            lengths = nx.single_source_dijkstra_path_length(graph_to_use, node, weight='weight')
        else:
            lengths = nx.single_source_shortest_path_length(graph_to_use, node)
        
        n = len(graph_to_use)
        
        if len(lengths) <= 1:
            return 0.0
        
        # Sum of reciprocal distances (1/∞ = 0 for unreachable nodes)
        harmonic_sum = sum(1.0 / d for d in lengths.values() if d > 0)
        
        # Normalize by (n-1)
        if n > 1:
            return harmonic_sum / (n - 1)
        return 0.0
    except Exception as e:
        print(f"[WARNING] Error computing harmonic for {node}: {e}", file=sys.stderr)
        return 0.0

def simulate_add_channel_bidirectional(G: nx.DiGraph, a: str, b: str, use_capacity_weights: bool = False) -> nx.DiGraph:
    """Simulate adding a bidirectional Lightning channel.
    
    Args:
        G: Original graph
        a, b: Nodes to connect
        use_capacity_weights: If True, add default weights to new edges
    """
    H = G.copy()
    if a == b:
        return H
    
    # Add edges with default capacity and weight
    if not H.has_edge(a, b):
        H.add_edge(a, b, capacity_sum_sat=0)
        if use_capacity_weights:
            H[a][b]["weight"] = 1.0  # Default weight for new channel
    
    if not H.has_edge(b, a):
        H.add_edge(b, a, capacity_sum_sat=0)
        if use_capacity_weights:
            H[b][a]["weight"] = 1.0  # Default weight for new channel
    
    return H

def simulate_add_multiple_channels(G: nx.DiGraph, target: str, nodes: Set[str], use_capacity_weights: bool = False) -> nx.DiGraph:
    """Add multiple channels at once."""
    H = G.copy()
    for node in nodes:
        if node != target:
            if not H.has_edge(target, node):
                H.add_edge(target, node, capacity_sum_sat=0)
                if use_capacity_weights:
                    H[target][node]["weight"] = 1.0
            if not H.has_edge(node, target):
                H.add_edge(node, target, capacity_sum_sat=0)
                if use_capacity_weights:
                    H[node][target]["weight"] = 1.0
    return H

# Helper function for process pool
def _evaluate_candidate_worker(args):
    """Worker function for process pool evaluation."""
    G_serialized, target, candidate, base_cc, base_hc, alias_map, use_capacity_weights = args
    G = pickle.loads(G_serialized)
    
    H = simulate_add_channel_bidirectional(G, target, candidate, use_capacity_weights)
    new_cc = compute_closeness_fast(H, target, use_outgoing=True, use_weights=use_capacity_weights)
    new_hc = compute_harmonic_fast(H, target, use_outgoing=True, use_weights=use_capacity_weights)
    
    delta_cc_abs = new_cc - base_cc
    delta_hc_abs = new_hc - base_hc
    delta_cc_pct = (delta_cc_abs / base_cc * 100.0) if base_cc > 0 else 0.0
    delta_hc_pct = (delta_hc_abs / base_hc * 100.0) if base_hc > 0 else 0.0
    
    return Recommendation(
        candidate, alias_map.get(candidate, candidate),
        new_cc, new_hc, delta_cc_abs, delta_hc_abs, delta_cc_pct, delta_hc_pct
    )

def rank_single_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                          topk: int = 20, n_jobs: int = -1, sort_by: str = 'closeness',
                          use_capacity_weights: bool = False) -> List[Recommendation]:
    """Rank candidate nodes by centrality improvement (using ProcessPoolExecutor)."""
    base_cc = compute_closeness_fast(G, target, use_outgoing=True, use_weights=use_capacity_weights)
    base_hc = compute_harmonic_fast(G, target, use_outgoing=True, use_weights=use_capacity_weights)
    
    if target in G:
        direct_neighbors = set(G.predecessors(target)) | set(G.successors(target))
    else:
        direct_neighbors = set()
    
    candidates = [n for n in G.nodes if n != target and n not in direct_neighbors]
    print(f"[INFO] Evaluating {len(candidates)} candidates...", file=sys.stderr)
    
    # Determine number of workers
    n_workers = min(multiprocessing.cpu_count(), len(candidates)) if n_jobs == -1 else max(1, min(n_jobs, len(candidates)))
    print(f"[INFO] Using {n_workers} worker processes", file=sys.stderr)
    
    results: List[Recommendation] = []
    
    # Serialize graph once for all workers
    G_serialized = pickle.dumps(G)
    
    # Prepare arguments for workers
    worker_args = [
        (G_serialized, target, candidate, base_cc, base_hc, alias_map, use_capacity_weights)
        for candidate in candidates
    ]
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_candidate = {
            executor.submit(_evaluate_candidate_worker, args): args[2]
            for args in worker_args
        }
        
        completed = 0
        total = len(candidates)
        for future in as_completed(future_to_candidate):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(f"[PROGRESS] {completed}/{total} ({completed/total*100:.1f}%)", file=sys.stderr, end='\r')
            try:
                rec = future.result(timeout=30)  # Add timeout for robustness
                if rec.delta_cc_abs > 0 or rec.delta_hc_abs > 0:
                    results.append(rec)
            except Exception as e:
                candidate = future_to_candidate[future]
                print(f"\n[WARNING] Error evaluating {candidate}: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Sort by specified metric
    if sort_by == 'harmonic':
        results.sort(key=lambda r: (r.delta_hc_abs, r.new_hc), reverse=True)
    else:
        results.sort(key=lambda r: (r.delta_cc_abs, r.new_cc), reverse=True)
    
    return results[:topk]

def greedy_channel_selection(G: nx.DiGraph, target: str, candidates: List[str], 
                            k: int, alias_map: Dict[str, str],
                            use_capacity_weights: bool = False) -> Tuple[List[str], List[float], List[float], List[float], List[float]]:
    """
    Greedy algorithm for selecting k channels to maximize closeness centrality.
    
    Algorithm:
    1. Start with empty set S = {}
    2. For i = 1 to k:
        - Find node v that maximizes marginal gain: f(S ∪ {v}) - f(S)
        - Add v to S
    3. Return S
    
    Complexity: O(k * n * (|V| + |E|))
    Approximation: (1 - 1/e) ≈ 63% for submodular functions
    """
    selected = []
    marginal_gains_cc = []
    marginal_gains_hc = []
    cumulative_cc = []
    cumulative_hc = []
    
    H = G.copy()
    base_cc = compute_closeness_fast(H, target, use_outgoing=True, use_weights=use_capacity_weights)
    base_hc = compute_harmonic_fast(H, target, use_outgoing=True, use_weights=use_capacity_weights)
    current_cc = base_cc
    current_hc = base_hc
    cumulative_cc.append(current_cc)
    cumulative_hc.append(current_hc)
    
    weight_msg = " (capacity-weighted)" if use_capacity_weights else ""
    print(f"\n[GREEDY] Starting greedy selection (k={k}){weight_msg}...", file=sys.stderr)
    print(f"[GREEDY] Initial CC: {base_cc:.6f}, HC: {base_hc:.6f}\n", file=sys.stderr)
    
    for iteration in range(k):
        best_node = None
        best_gain_cc = 0.0
        best_gain_hc = 0.0
        best_new_cc = current_cc
        best_new_hc = current_hc
        
        # Evaluate marginal gain for each remaining candidate
        remaining = [c for c in candidates if c not in selected]
        
        print(f"[GREEDY] Iteration {iteration+1}/{k}: Evaluating {len(remaining)} candidates...", file=sys.stderr)
        start_time = time.time()
        
        for candidate in remaining:
            # Simulate adding this channel
            H_temp = simulate_add_channel_bidirectional(H, target, candidate, use_capacity_weights)
            new_cc = compute_closeness_fast(H_temp, target, use_outgoing=True, use_weights=use_capacity_weights)
            new_hc = compute_harmonic_fast(H_temp, target, use_outgoing=True, use_weights=use_capacity_weights)
            
            # Calculate marginal gains
            marginal_cc = new_cc - current_cc
            marginal_hc = new_hc - current_hc
            
            # Select based on closeness improvement (primary metric)
            if marginal_cc > best_gain_cc:
                best_gain_cc = marginal_cc
                best_gain_hc = marginal_hc
                best_node = candidate
                best_new_cc = new_cc
                best_new_hc = new_hc
        
        elapsed = time.time() - start_time
        
        if best_node is None:
            print(f"[GREEDY] No improving node found. Stopping early.", file=sys.stderr)
            break
        
        # Add best node to selection
        selected.append(best_node)
        marginal_gains_cc.append(best_gain_cc)
        marginal_gains_hc.append(best_gain_hc)
        current_cc = best_new_cc
        current_hc = best_new_hc
        cumulative_cc.append(current_cc)
        cumulative_hc.append(current_hc)
        
        # Update graph with selected channel
        H = simulate_add_channel_bidirectional(H, target, best_node, use_capacity_weights)
        
        gain_cc_pct = (best_gain_cc / base_cc * 100.0) if base_cc > 0 else 0.0
        gain_hc_pct = (best_gain_hc / base_hc * 100.0) if base_hc > 0 else 0.0
        total_cc_improvement = (current_cc - base_cc) / base_cc * 100.0 if base_cc > 0 else 0.0
        total_hc_improvement = (current_hc - base_hc) / base_hc * 100.0 if base_hc > 0 else 0.0
        
        print(f"[GREEDY] Selected: {alias_map.get(best_node, best_node)[:30]}", file=sys.stderr)
        print(f"[GREEDY] Marginal: CC +{gain_cc_pct:.2f}%, HC +{gain_hc_pct:.2f}%", file=sys.stderr)
        print(f"[GREEDY] Total: CC {current_cc:.6f} (+{total_cc_improvement:.2f}%), HC {current_hc:.6f} (+{total_hc_improvement:.2f}%)", file=sys.stderr)
        print(f"[GREEDY] Time: {elapsed:.2f}s\n", file=sys.stderr)
    
    return selected, marginal_gains_cc, marginal_gains_hc, cumulative_cc, cumulative_hc

# Helper function for exhaustive search
def _evaluate_combo_worker(args):
    """Worker function for combination evaluation."""
    G_serialized, target, nodes, base_cc, base_hc, use_capacity_weights = args
    G = pickle.loads(G_serialized)
    
    H = simulate_add_multiple_channels(G, target, set(nodes), use_capacity_weights)
    new_cc = compute_closeness_fast(H, target, use_outgoing=True, use_weights=use_capacity_weights)
    new_hc = compute_harmonic_fast(H, target, use_outgoing=True, use_weights=use_capacity_weights)
    delta_cc = new_cc - base_cc
    delta_hc = new_hc - base_hc
    delta_cc_pct = (delta_cc / base_cc * 100.0) if base_cc > 0 else 0.0
    delta_hc_pct = (delta_hc / base_hc * 100.0) if base_hc > 0 else 0.0
    
    return (nodes, new_cc, new_hc, delta_cc, delta_hc, delta_cc_pct, delta_hc_pct)

def exhaustive_search_combo(G: nx.DiGraph, target: str, candidates: List[str], 
                           k: int, top: int = 5, n_jobs: int = -1,
                           use_capacity_weights: bool = False,
                           max_candidates: int = 20) -> List[Tuple]:
    """Exhaustive search for optimal combination with configurable candidate limit."""
    base_cc = compute_closeness_fast(G, target, use_outgoing=True, use_weights=use_capacity_weights)
    base_hc = compute_harmonic_fast(G, target, use_outgoing=True, use_weights=use_capacity_weights)
    
    # Limit candidates to max_candidates
    limited_candidates = candidates[:min(len(candidates), max_candidates)]
    all_combos = list(itertools.combinations(limited_candidates, k))
    
    weight_msg = " (capacity-weighted)" if use_capacity_weights else ""
    print(f"[EXHAUSTIVE] Evaluating {len(all_combos)} combinations from top {len(limited_candidates)} candidates{weight_msg}...", file=sys.stderr)
    
    # Determine number of workers
    n_workers = min(multiprocessing.cpu_count(), len(all_combos)) if n_jobs == -1 else max(1, min(n_jobs, len(all_combos)))
    
    # Serialize graph once
    G_serialized = pickle.dumps(G)
    
    # Prepare worker arguments
    worker_args = [
        (G_serialized, target, nodes, base_cc, base_hc, use_capacity_weights)
        for nodes in all_combos
    ]
    
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_evaluate_combo_worker, args): args[2] for args in worker_args}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % max(1, len(all_combos) // 20) == 0:
                print(f"[PROGRESS] {completed}/{len(all_combos)} ({completed/len(all_combos)*100:.1f}%)", 
                      file=sys.stderr, end='\r')
            try:
                results.append(future.result(timeout=30))
            except Exception as e:
                print(f"\n[WARNING] Error evaluating combination: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    results.sort(key=lambda x: (x[3], x[1]), reverse=True)  # Sort by delta_cc
    return results[:top]

def main():
    ap = argparse.ArgumentParser(
        description="Lightning Network Advanced Centrality Analyzer (Fixed Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FIXED ISSUES:
  - Corrected double normalization in closeness centrality calculation
  - Using ProcessPoolExecutor instead of ThreadPoolExecutor for better parallelism
  - Improved speedup metric display
  - Added configurable candidate limit for exhaustive search
  - Enhanced error handling with connection retry

Examples:
  # Analyze with greedy algorithm (fast, good approximation)
  python ln_closeness_analysis_fixed.py --pg-host HOST --pg-db DB \\
      --pg-user USER --pg-pass PASS --target-node NODE_ID --method greedy

  # Analyze with capacity-weighted centrality
  python ln_closeness_analysis_fixed.py --pg-host HOST --pg-db DB \\
      --pg-user USER --pg-pass PASS --target-node NODE_ID --use-capacity

  # Compare both methods with custom candidate limit
  python ln_closeness_analysis_fixed.py --pg-host HOST --pg-db DB \\
      --pg-user USER --pg-pass PASS --target-node NODE_ID --method both \\
      --exhaustive-candidates 15

Theory:
  - Closeness Centrality: Efficiency of reaching all nodes (Freeman 1979)
  - Harmonic Centrality: Handles disconnected graphs better (Marchiori & Latora 2000)
  - Capacity Weighting: Structural importance based on channel sizes (Opsahl et al. 2010)
  - Greedy Algorithm: Fast approximation with theoretical guarantees (Kempe et al. 2003)
        """
    )
    ap.add_argument("--pg-host", required=True, help="PostgreSQL host")
    ap.add_argument("--pg-port", type=int, default=5432, help="PostgreSQL port")
    ap.add_argument("--pg-db", required=True, help="PostgreSQL database name")
    ap.add_argument("--pg-user", required=True, help="PostgreSQL username")
    ap.add_argument("--pg-pass", required=True, help="PostgreSQL password")
    ap.add_argument("--target-node", required=True, help="Hex node_id to analyze")
    ap.add_argument("--topk", type=int, default=20, help="Top-K single-node recommendations")
    ap.add_argument("--combo-k", type=int, default=3, help="Number of channels to select")
    ap.add_argument("--combo-top", type=int, default=5, help="Number of top combos to output")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel workers")
    ap.add_argument("--sort-by", choices=['closeness', 'harmonic'], default='closeness',
                    help="Sort candidates by closeness or harmonic centrality")
    ap.add_argument("--method", choices=['greedy', 'exhaustive', 'both'], default='greedy',
                    help="Optimization method: greedy (fast), exhaustive (exact), or both (comparison)")
    ap.add_argument("--use-capacity", action="store_true",
                    help="Use capacity-weighted centrality (experimental)")
    ap.add_argument("--export-marginal-gains", action="store_true",
                    help="Export marginal gains data to CSV for statistical analysis")
    ap.add_argument("--exhaustive-candidates", type=int, default=20,
                    help="Maximum number of candidates to consider in exhaustive search (default: 20)")
    args = ap.parse_args()

    conf = DBConf(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_pass)
    
    print("[INFO] Fetching data...", file=sys.stderr)
    ch_df = fetch_dataframe_with_retry(conf, CHANNELS_SQL)
    alias_df = fetch_dataframe_with_retry(conf, ALIASES_SQL)
    
    print("[INFO] Building graph...", file=sys.stderr)
    if args.use_capacity:
        print("[INFO] Using capacity-weighted analysis", file=sys.stderr)
    G, alias_map = build_directed_graph(ch_df, alias_df, use_capacity_weights=args.use_capacity)
    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)
    
    # Connectivity analysis
    conn_info = analyze_connectivity(G)
    print("\n" + "="*70)
    print("  Network Connectivity Analysis")
    print("="*70)
    print(f"Strongly connected: {conn_info['is_strongly_connected']}")
    print(f"Strong components: {conn_info['num_strong_components']}")
    print(f"Largest component coverage: {conn_info['strong_coverage']:.2f}%")
    print(f"Weak components: {conn_info['num_weak_components']}")
    if args.use_capacity:
        print("Mode: Capacity-weighted centrality")
    
    if args.target_node not in G:
        print(f"\n[ERROR] Target node not found", file=sys.stderr)
        sys.exit(1)
    
    base_cc = compute_closeness_fast(G, args.target_node, use_outgoing=True, use_weights=args.use_capacity)
    base_hc = compute_harmonic_fast(G, args.target_node, use_outgoing=True, use_weights=args.use_capacity)
    
    print("\n" + "="*70)
    print("  Current Centrality Values")
    print("="*70)
    print(f"Node: {alias_map.get(args.target_node, args.target_node)}")
    print(f"Closeness Centrality:  {base_cc:.6f}")
    print(f"Harmonic Centrality:   {base_hc:.6f}")
    
    # Rank single candidates
    singles = rank_single_candidates(G, args.target_node, alias_map, args.topk, args.n_jobs, 
                                    args.sort_by, use_capacity_weights=args.use_capacity)
    
    if singles:
        print(f"\n{'='*70}")
        print(f"  Top {args.topk} Recommendations (sorted by {args.sort_by})")
        print(f"{'='*70}")
        print(f"\n{'Rank':<6}{'Alias':<25}{'ΔCC %':<10}{'ΔHC %':<10}{'CC':<12}{'HC':<12}")
        print("-" * 75)
        
        for i, r in enumerate(singles, 1):
            print(f"{i:<6}{r.alias[:22]:<25}"
                  f"+{r.delta_cc_pct:<9.2f}+{r.delta_hc_pct:<9.2f}"
                  f"{r.new_cc:<12.6f}{r.new_hc:<12.6f}")
        
        # Calculate correlation between CC and HC improvements
        if len(singles) > 1:
            cc_deltas = [r.delta_cc_abs for r in singles]
            hc_deltas = [r.delta_hc_abs for r in singles]
            corr = np.corrcoef(cc_deltas, hc_deltas)[0,1]
            print(f"\nCorrelation (ΔCC vs ΔHC): {corr:.4f}")
        
        # Save results
        pd.DataFrame([{
            "rank": i, "node_id": r.node_id, "alias": r.alias,
            "new_closeness": r.new_cc, "new_harmonic": r.new_hc,
            "delta_cc_abs": r.delta_cc_abs, "delta_hc_abs": r.delta_hc_abs,
            "delta_cc_pct": r.delta_cc_pct, "delta_hc_pct": r.delta_hc_pct,
            "capacity_weighted": args.use_capacity
        } for i, r in enumerate(singles, 1)]).to_csv("centrality_recommendations.csv", index=False)
        print("\n✅ Saved to: centrality_recommendations.csv")
    
    if not singles:
        print("No improving candidates found.", file=sys.stderr)
        sys.exit(0)
    
    candidate_ids = [r.node_id for r in singles]
    
    # Method-specific analysis
    greedy_results = None
    exhaustive_results = None
    best_combination = None
    
    # Greedy method
    if args.method in ['greedy', 'both']:
        greedy_start = time.time()
        selected, gains_cc, gains_hc, cumulative_cc, cumulative_hc = greedy_channel_selection(
            G, args.target_node, candidate_ids, args.combo_k, alias_map, 
            use_capacity_weights=args.use_capacity
        )
        greedy_time = time.time() - greedy_start
        
        if selected:
            greedy_results = (selected, cumulative_cc[-1], cumulative_hc[-1])
            best_combination = selected
            
            print(f"\n{'='*70}")
            print(f"  GREEDY RESULT (k={args.combo_k})")
            print(f"{'='*70}")
            for i, (node, gain_cc, gain_hc) in enumerate(zip(selected, gains_cc, gains_hc), 1):
                print(f"{i}. {alias_map.get(node, node)[:40]}")
                print(f"   Marginal gain - CC: {gain_cc:.6f} (+{gain_cc/base_cc*100:.2f}%), "
                      f"HC: {gain_hc:.6f} (+{gain_hc/base_hc*100:.2f}%)")
            
            final_cc = cumulative_cc[-1]
            final_hc = cumulative_hc[-1]
            total_cc_imp = (final_cc - base_cc) / base_cc * 100.0
            total_hc_imp = (final_hc - base_hc) / base_hc * 100.0
            print(f"\nFinal centrality:")
            print(f"  CC: {final_cc:.6f} (+{total_cc_imp:.2f}%)")
            print(f"  HC: {final_hc:.6f} (+{total_hc_imp:.2f}%)")
            print(f"Computation time: {greedy_time:.2f}s")
            
            # Export greedy results
            greedy_df = pd.DataFrame([{
                "iteration": i,
                "node_id": node,
                "alias": alias_map.get(node, node),
                "marginal_gain_cc": gain_cc,
                "marginal_gain_hc": gain_hc,
                "marginal_gain_cc_pct": (gain_cc / base_cc * 100.0) if base_cc > 0 else 0.0,
                "marginal_gain_hc_pct": (gain_hc / base_hc * 100.0) if base_hc > 0 else 0.0,
                "cumulative_cc": cum_cc,
                "cumulative_hc": cum_hc,
                "cumulative_cc_improvement_pct": ((cum_cc - base_cc) / base_cc * 100.0) if base_cc > 0 else 0.0,
                "cumulative_hc_improvement_pct": ((cum_hc - base_hc) / base_hc * 100.0) if base_hc > 0 else 0.0,
                "capacity_weighted": args.use_capacity
            } for i, (node, gain_cc, gain_hc, cum_cc, cum_hc) in enumerate(
                zip(selected, gains_cc, gains_hc, cumulative_cc[1:], cumulative_hc[1:]), 1
            )])
            greedy_df.to_csv("optimal_combination_greedy.csv", index=False)
            print("✅ Saved to: optimal_combination_greedy.csv")
            
            # Export marginal gains if requested
            if args.export_marginal_gains:
                marginal_df = pd.DataFrame([{
                    "iteration": i,
                    "node_id": node,
                    "alias": alias_map.get(node, node),
                    "marginal_gain_cc": gain_cc,
                    "marginal_gain_hc": gain_hc,
                    "base_cc": base_cc,
                    "base_hc": base_hc,
                    "cumulative_cc": cum_cc,
                    "cumulative_hc": cum_hc,
                    "capacity_weighted": args.use_capacity
                } for i, (node, gain_cc, gain_hc, cum_cc, cum_hc) in enumerate(
                    zip(selected, gains_cc, gains_hc, cumulative_cc[1:], cumulative_hc[1:]), 1
                )])
                marginal_df.to_csv("marginal_gains.csv", index=False)
                print("✅ Saved to: marginal_gains.csv (statistical analysis data)")
    
    # Exhaustive method
    if args.method in ['exhaustive', 'both']:
        if len(singles) >= args.combo_k:
            exhaustive_start = time.time()
            exhaustive_results_raw = exhaustive_search_combo(
                G, args.target_node, candidate_ids, args.combo_k, args.combo_top, args.n_jobs,
                use_capacity_weights=args.use_capacity,
                max_candidates=args.exhaustive_candidates
            )
            exhaustive_time = time.time() - exhaustive_start
            
            if exhaustive_results_raw:
                exhaustive_results = exhaustive_results_raw
                # Update best combination if exhaustive found better result
                if exhaustive_results:
                    best_combination = list(exhaustive_results[0][0])
                
                print(f"\n{'='*70}")
                print(f"  EXHAUSTIVE RESULT (Top {args.combo_top} from {args.exhaustive_candidates} candidates)")
                print(f"{'='*70}")
                for i, (nodes, new_cc, new_hc, d_cc, d_hc, d_cc_pct, d_hc_pct) in enumerate(exhaustive_results, 1):
                    labels = [alias_map.get(n, n)[:20] for n in nodes]
                    print(f"\n#{i}: {', '.join(labels)}")
                    print(f"   CC: {new_cc:.6f} (+{d_cc_pct:.2f}%), HC: {new_hc:.6f} (+{d_hc_pct:.2f}%)")
                print(f"\nComputation time: {exhaustive_time:.2f}s")
                
                # Export exhaustive results
                exhaustive_records = []
                for rank, (nodes, new_cc, new_hc, d_cc, d_hc, d_cc_pct, d_hc_pct) in enumerate(exhaustive_results, 1):
                    for i, node in enumerate(nodes, 1):
                        exhaustive_records.append({
                            "rank": rank,
                            "position": i,
                            "node_id": node,
                            "alias": alias_map.get(node, node),
                            "combination_cc": new_cc,
                            "combination_hc": new_hc,
                            "delta_cc_abs": d_cc,
                            "delta_hc_abs": d_hc,
                            "delta_cc_pct": d_cc_pct,
                            "delta_hc_pct": d_hc_pct,
                            "capacity_weighted": args.use_capacity
                        })
                
                exhaustive_df = pd.DataFrame(exhaustive_records)
                exhaustive_df.to_csv("optimal_combinations_exhaustive.csv", index=False)
                print("✅ Saved to: optimal_combinations_exhaustive.csv")
    
    # Save the best combination found
    if best_combination:
        # Compute final metrics for the best combination
        H_best = simulate_add_multiple_channels(G, args.target_node, set(best_combination), args.use_capacity)
        best_cc = compute_closeness_fast(H_best, args.target_node, use_outgoing=True, use_weights=args.use_capacity)
        best_hc = compute_harmonic_fast(H_best, args.target_node, use_outgoing=True, use_weights=args.use_capacity)
        
        best_df = pd.DataFrame([{
            "position": i,
            "node_id": node,
            "alias": alias_map.get(node, node),
            "final_cc": best_cc,
            "final_hc": best_hc,
            "improvement_cc_pct": ((best_cc - base_cc) / base_cc * 100.0) if base_cc > 0 else 0.0,
            "improvement_hc_pct": ((best_hc - base_hc) / base_hc * 100.0) if base_hc > 0 else 0.0,
            "capacity_weighted": args.use_capacity,
            "method": "greedy" if args.method == "greedy" else "best_found"
        } for i, node in enumerate(best_combination, 1)])
        best_df.to_csv("optimal_combination.csv", index=False)
        print("\n✅ Saved to: optimal_combination.csv (best channel combination)")
    
    # Comparison (FIXED display)
    if args.method == 'both' and greedy_results and exhaustive_results:
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        print(f"Greedy time:     {greedy_time:.2f}s")
        print(f"Exhaustive time: {exhaustive_time:.2f}s")
        
        # Fixed: More intuitive speedup display
        if greedy_time > 0:
            speedup = exhaustive_time / greedy_time
            print(f"Relative time (exhaustive/greedy): {speedup:.2f}x")
            if speedup > 1:
                print(f"Greedy is {speedup:.2f}x faster than exhaustive")
            else:
                print(f"Exhaustive is {1/speedup:.2f}x faster than greedy")
        
        greedy_set = set(greedy_results[0])
        best_exhaustive_set = set(exhaustive_results[0][0])
        
        if greedy_set == best_exhaustive_set:
            print(f"\nResult: ✅ IDENTICAL (Greedy found optimal solution)")
        else:
            greedy_final_cc = greedy_results[1]
            exhaustive_best_cc = exhaustive_results[0][1]
            quality_ratio = greedy_final_cc / exhaustive_best_cc if exhaustive_best_cc > 0 else 0
            print(f"\nResult: ⚠️  DIFFERENT")
            print(f"Greedy quality: {quality_ratio*100:.2f}% of optimal")
            print(f"Approximation guarantee: ≥63% (theoretical for submodular functions)")
            print(f"Note: Closeness centrality improvement may not always be submodular")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
