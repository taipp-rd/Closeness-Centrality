#!/usr/bin/env python3
"""
Lightning Network: Closeness Centrality Analyzer with Greedy Optimization

NEW FEATURE: Greedy Algorithm for Channel Selection
- Scalable alternative to exhaustive search: O(k*n) vs O(C(n,k))
- Theoretical guarantee: (1-1/e)≈63% approximation for submodular functions
- Based on Kempe et al. (2003) influence maximization
"""
from __future__ import annotations
import argparse
import itertools
import sys
import multiprocessing
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import pandas as pd
import networkx as nx

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

def fetch_dataframe(conf: DBConf, sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(host=conf.host, port=conf.port, dbname=conf.db,
                            user=conf.user, password=conf.password)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df

def build_directed_graph(ch_df: pd.DataFrame, alias_df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[str, str]]:
    ch_df = ch_df.dropna(subset=["advertising_nodeid", "connecting_nodeid"])
    ch_df['capacity_sat'] = pd.to_numeric(ch_df['capacity_sat'], errors='coerce').fillna(0)
    
    node_capacity = {}
    for _, row in ch_df.iterrows():
        u, v, cap = row["advertising_nodeid"], row["connecting_nodeid"], row["capacity_sat"]
        node_capacity[u] = node_capacity.get(u, 0) + cap
        node_capacity[v] = node_capacity.get(v, 0) + cap
    
    nonzero_nodes = {n for n, cap in node_capacity.items() if cap > 0}
    print(f"[INFO] Nodes with non-zero capacity: {len(nonzero_nodes)}", file=sys.stderr)
    
    G = nx.DiGraph()
    for _, row in ch_df.iterrows():
        u, v, cap = row["advertising_nodeid"], row["connecting_nodeid"], int(row["capacity_sat"])
        if u in nonzero_nodes and v in nonzero_nodes:
            if G.has_edge(u, v):
                G[u][v]["multiplicity"] = G[u][v].get("multiplicity", 1) + 1
                G[u][v]["capacity_sum_sat"] = G[u][v].get("capacity_sum_sat", 0) + cap
            else:
                G.add_edge(u, v, multiplicity=1, capacity_sum_sat=cap)
    
    alias_map = {}
    for _, r in alias_df.iterrows():
        node_id, alias = r["node_id"], r["alias"]
        alias_map[node_id] = alias.strip() if isinstance(alias, str) and alias.strip() else node_id
    
    for n in G.nodes:
        G.nodes[n]["alias"] = alias_map.get(n, n)
    
    return G, alias_map

def compute_closeness_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True) -> float:
    if node not in G:
        return 0.0
    graph_to_use = G.reverse() if use_outgoing else G
    try:
        lengths = nx.single_source_shortest_path_length(graph_to_use, node)
        n = len(graph_to_use)
        if len(lengths) <= 1:
            return 0.0
        total_distance = sum(lengths.values())
        n_reachable = len(lengths) - 1
        if total_distance > 0:
            closeness = n_reachable / total_distance
            if n > 1:
                s = n_reachable / (n - 1)
                closeness *= s
            return closeness
        return 0.0
    except:
        return 0.0

@dataclass
class Recommendation:
    node_id: str
    alias: str
    new_cc: float
    delta_abs: float
    delta_pct: float

def simulate_add_channel_bidirectional(G: nx.DiGraph, a: str, b: str) -> nx.DiGraph:
    H = G.copy()
    if a == b:
        return H
    if not H.has_edge(a, b):
        H.add_edge(a, b, capacity_sum_sat=0)
    if not H.has_edge(b, a):
        H.add_edge(b, a, capacity_sum_sat=0)
    return H

def simulate_add_multiple_channels(G: nx.DiGraph, target: str, nodes: Set[str]) -> nx.DiGraph:
    """Add multiple channels at once."""
    H = G.copy()
    for node in nodes:
        if node != target:
            if not H.has_edge(target, node):
                H.add_edge(target, node, capacity_sum_sat=0)
            if not H.has_edge(node, target):
                H.add_edge(node, target, capacity_sum_sat=0)
    return H

def evaluate_single_candidate(G: nx.DiGraph, target: str, candidate: str, base_cc: float, 
                              alias_map: Dict[str, str]) -> Recommendation:
    H = simulate_add_channel_bidirectional(G, target, candidate)
    new_cc = compute_closeness_fast(H, target, use_outgoing=True)
    delta_abs = new_cc - base_cc
    delta_pct = (delta_abs / base_cc * 100.0) if base_cc > 0 else 0.0
    return Recommendation(candidate, alias_map.get(candidate, candidate), new_cc, delta_abs, delta_pct)

def rank_single_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                          topk: int = 20, n_jobs: int = -1) -> List[Recommendation]:
    base_cc = compute_closeness_fast(G, target, use_outgoing=True)
    
    if target in G:
        direct_neighbors = set(G.predecessors(target)) | set(G.successors(target))
    else:
        direct_neighbors = set()
    
    candidates = [n for n in G.nodes if n != target and n not in direct_neighbors]
    print(f"[INFO] Evaluating {len(candidates)} candidates...", file=sys.stderr)
    
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    print(f"[INFO] Using {n_workers} workers", file=sys.stderr)
    
    results: List[Recommendation] = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_candidate = {
            executor.submit(evaluate_single_candidate, G, target, candidate, base_cc, alias_map): candidate
            for candidate in candidates
        }
        
        completed = 0
        total = len(candidates)
        for future in as_completed(future_to_candidate):
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                print(f"[PROGRESS] {completed}/{total} ({completed/total*100:.1f}%)", file=sys.stderr, end='\r')
            try:
                rec = future.result()
                if rec.delta_abs > 0:
                    results.append(rec)
            except Exception as e:
                print(f"\n[WARNING] Error: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    results.sort(key=lambda r: (r.delta_abs, r.new_cc), reverse=True)
    return results[:topk]

def greedy_channel_selection(G: nx.DiGraph, target: str, candidates: List[str], 
                            k: int, alias_map: Dict[str, str]) -> Tuple[List[str], List[float], List[float]]:
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
    
    Args:
        G: Network graph
        target: Target node
        candidates: Candidate nodes to select from
        k: Number of channels to select
        alias_map: Node aliases
    
    Returns:
        Tuple of (selected_nodes, marginal_gains, cumulative_centrality)
    """
    selected = []
    marginal_gains = []
    cumulative_cc = []
    
    H = G.copy()
    base_cc = compute_closeness_fast(H, target, use_outgoing=True)
    current_cc = base_cc
    cumulative_cc.append(current_cc)
    
    print(f"\n[GREEDY] Starting greedy selection (k={k})...", file=sys.stderr)
    print(f"[GREEDY] Initial closeness: {base_cc:.6f}\n", file=sys.stderr)
    
    for iteration in range(k):
        best_node = None
        best_gain = 0.0
        best_new_cc = current_cc
        
        # Evaluate marginal gain for each remaining candidate
        remaining = [c for c in candidates if c not in selected]
        
        print(f"[GREEDY] Iteration {iteration+1}/{k}: Evaluating {len(remaining)} candidates...", file=sys.stderr)
        start_time = time.time()
        
        for candidate in remaining:
            # Simulate adding this channel
            H_temp = simulate_add_channel_bidirectional(H, target, candidate)
            new_cc = compute_closeness_fast(H_temp, target, use_outgoing=True)
            
            # Calculate marginal gain
            marginal_gain = new_cc - current_cc
            
            if marginal_gain > best_gain:
                best_gain = marginal_gain
                best_node = candidate
                best_new_cc = new_cc
        
        elapsed = time.time() - start_time
        
        if best_node is None:
            print(f"[GREEDY] No improving node found. Stopping early.", file=sys.stderr)
            break
        
        # Add best node to selection
        selected.append(best_node)
        marginal_gains.append(best_gain)
        current_cc = best_new_cc
        cumulative_cc.append(current_cc)
        
        # Update graph with selected channel
        H = simulate_add_channel_bidirectional(H, target, best_node)
        
        gain_pct = (best_gain / base_cc * 100.0) if base_cc > 0 else 0.0
        total_improvement = (current_cc - base_cc) / base_cc * 100.0 if base_cc > 0 else 0.0
        
        print(f"[GREEDY] Selected: {alias_map.get(best_node, best_node)[:30]}", file=sys.stderr)
        print(f"[GREEDY] Marginal gain: {best_gain:.6f} (+{gain_pct:.2f}%)", file=sys.stderr)
        print(f"[GREEDY] New closeness: {current_cc:.6f} (total: +{total_improvement:.2f}%)", file=sys.stderr)
        print(f"[GREEDY] Time: {elapsed:.2f}s\n", file=sys.stderr)
    
    return selected, marginal_gains, cumulative_cc

def exhaustive_search_combo(G: nx.DiGraph, target: str, candidates: List[str], 
                           k: int, top: int = 5, n_jobs: int = -1) -> List[Tuple]:
    """Exhaustive search for comparison (original method)."""
    base_cc = compute_closeness_fast(G, target, use_outgoing=True)
    all_combos = list(itertools.combinations(candidates[:min(len(candidates), 20)], k))
    
    print(f"[EXHAUSTIVE] Evaluating {len(all_combos)} combinations...", file=sys.stderr)
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    
    def evaluate_combo(nodes):
        H = simulate_add_multiple_channels(G, target, set(nodes))
        new_cc = compute_closeness_fast(H, target, use_outgoing=True)
        delta = new_cc - base_cc
        delta_pct = (delta / base_cc * 100.0) if base_cc > 0 else 0.0
        return (nodes, new_cc, delta, delta_pct)
    
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(evaluate_combo, nodes): nodes for nodes in all_combos}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % max(1, len(all_combos) // 20) == 0:
                print(f"[PROGRESS] {completed}/{len(all_combos)} ({completed/len(all_combos)*100:.1f}%)", 
                      file=sys.stderr, end='\r')
            try:
                results.append(future.result())
            except Exception as e:
                print(f"\n[WARNING] Error: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return results[:top]

def main():
    ap = argparse.ArgumentParser(description="Lightning Network Centrality Analyzer with Greedy Optimization")
    ap.add_argument("--pg-host", required=True)
    ap.add_argument("--pg-port", type=int, default=5432)
    ap.add_argument("--pg-db", required=True)
    ap.add_argument("--pg-user", required=True)
    ap.add_argument("--pg-pass", required=True)
    ap.add_argument("--target-node", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--combo-k", type=int, default=3)
    ap.add_argument("--combo-top", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--method", choices=['greedy', 'exhaustive', 'both'], default='greedy',
                    help="Optimization method: greedy (fast), exhaustive (exact), or both (comparison)")
    args = ap.parse_args()

    conf = DBConf(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_pass)
    
    print("[INFO] Fetching data...", file=sys.stderr)
    ch_df = fetch_dataframe(conf, CHANNELS_SQL)
    alias_df = fetch_dataframe(conf, ALIASES_SQL)
    
    print("[INFO] Building graph...", file=sys.stderr)
    G, alias_map = build_directed_graph(ch_df, alias_df)
    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", file=sys.stderr)
    
    if args.target_node not in G:
        print(f"[ERROR] Target node not found", file=sys.stderr)
        sys.exit(1)
    
    base_cc = compute_closeness_fast(G, args.target_node, use_outgoing=True)
    print(f"\n{'='*70}")
    print(f"  Current Closeness Centrality")
    print(f"{'='*70}")
    print(f"Node: {alias_map.get(args.target_node, args.target_node)}")
    print(f"Closeness: {base_cc:.6f}\n")
    
    singles = rank_single_candidates(G, args.target_node, alias_map, args.topk, args.n_jobs)
    
    if not singles:
        print("No improving candidates found.", file=sys.stderr)
        sys.exit(0)
    
    candidate_ids = [r.node_id for r in singles]
    
    # Greedy method
    if args.method in ['greedy', 'both']:
        greedy_start = time.time()
        selected, gains, cumulative = greedy_channel_selection(
            G, args.target_node, candidate_ids, args.combo_k, alias_map
        )
        greedy_time = time.time() - greedy_start
        
        print(f"\n{'='*70}")
        print(f"  GREEDY RESULT (k={args.combo_k})")
        print(f"{'='*70}")
        for i, (node, gain) in enumerate(zip(selected, gains), 1):
            print(f"{i}. {alias_map.get(node, node)[:40]}")
            print(f"   Marginal gain: {gain:.6f} (+{gain/base_cc*100:.2f}%)")
        
        final_cc = cumulative[-1]
        total_improvement = (final_cc - base_cc) / base_cc * 100.0
        print(f"\nFinal closeness: {final_cc:.6f} (+{total_improvement:.2f}%)")
        print(f"Computation time: {greedy_time:.2f}s")
    
    # Exhaustive method
    if args.method in ['exhaustive', 'both']:
        if len(singles) >= args.combo_k:
            exhaustive_start = time.time()
            exhaustive_results = exhaustive_search_combo(
                G, args.target_node, candidate_ids, args.combo_k, args.combo_top, args.n_jobs
            )
            exhaustive_time = time.time() - exhaustive_start
            
            print(f"\n{'='*70}")
            print(f"  EXHAUSTIVE RESULT (Top {args.combo_top})")
            print(f"{'='*70}")
            for i, (nodes, new_cc, delta, delta_pct) in enumerate(exhaustive_results, 1):
                labels = [alias_map.get(n, n)[:20] for n in nodes]
                print(f"\n#{i}: {', '.join(labels)}")
                print(f"   Closeness: {new_cc:.6f} (+{delta_pct:.2f}%)")
            print(f"\nComputation time: {exhaustive_time:.2f}s")
    
    # Comparison
    if args.method == 'both':
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        print(f"Greedy time:     {greedy_time:.2f}s")
        print(f"Exhaustive time: {exhaustive_time:.2f}s")
        print(f"Speedup:         {exhaustive_time/greedy_time:.2f}x")
        
        greedy_set = set(selected)
        best_exhaustive_set = set(exhaustive_results[0][0])
        
        if greedy_set == best_exhaustive_set:
            print(f"\nResult: ✅ IDENTICAL (Greedy found optimal solution)")
        else:
            greedy_final = cumulative[-1]
            exhaustive_best = exhaustive_results[0][1]
            quality_ratio = greedy_final / exhaustive_best
            print(f"\nResult: ⚠️  DIFFERENT")
            print(f"Greedy quality: {quality_ratio*100:.2f}% of optimal")
            print(f"Approximation guarantee: ≥63% (theoretical)")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
