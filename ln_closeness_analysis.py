#!/usr/bin/env python3
"""
Lightning Network: Closeness Centrality Analyzer (Optimized + Weighted)

Features:
- Connects to PostgreSQL and analyzes Lightning Network topology
- Computes outgoing closeness centrality for routing capability
- Optional capacity-weighted centrality for structural analysis
- Simulates channel openings and ranks recommendations
- Parallel processing for optimal performance

NEW FEATURE: Capacity-weighted centrality option
- Optional --use-capacity flag to consider channel capacity as weights
- Large capacity channels = shorter "effective distance"  
- Based on Opsahl et al. (2010) weighted network centrality theory

THEORETICAL NOTES:
- Capacity represents theoretical upper bound, not actual balance
- Actual balance distribution is private and bimodal (empirically proven)
- This analysis provides structural importance, not routing probability
- For production routing, use probability-based pathfinding algorithms

References:
- Opsahl et al. (2010): Node centrality in weighted networks
- Rohrer et al. (2019): Discharged Payment Channels
"""
from __future__ import annotations
import argparse
import itertools
import sys
import multiprocessing
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import pandas as pd
import networkx as nx

# SQL queries - fetch latest channel state
CHANNELS_SQL = r"""
WITH latest_channel AS (
  SELECT DISTINCT ON (cu.chan_id, cu.advertising_nodeid)
         cu.chan_id,
         cu.chan_point,
         cu.advertising_nodeid,
         cu.connecting_nodeid,
         COALESCE(cu.capacity_sat, 0) AS capacity_sat,
         cu.rp_disabled,
         COALESCE(cu.rp_last_update, cu.timestamp) AS last_ts
  FROM channel_update cu
  ORDER BY cu.chan_id, cu.advertising_nodeid, COALESCE(cu.rp_last_update, cu.timestamp) DESC
),
open_channel AS (
  SELECT lc.*
  FROM latest_channel lc
  LEFT JOIN closed_channel cc
    ON cc.chan_id = lc.chan_id
  WHERE cc.chan_id IS NULL
    AND lc.capacity_sat > 0
    AND lc.rp_disabled = false
)
SELECT * FROM open_channel;
"""

ALIASES_SQL = r"""
SELECT DISTINCT ON (na.node_id)
       na.node_id,
       na.alias,
       na.timestamp AS last_ts
FROM node_announcement na
ORDER BY na.node_id, na.timestamp DESC;
"""

@dataclass
class DBConf:
    host: str
    port: int
    db: str
    user: str
    password: str

def fetch_dataframe(conf: DBConf, sql: str) -> pd.DataFrame:
    """Fetch data from PostgreSQL database."""
    conn = psycopg2.connect(host=conf.host, port=conf.port, dbname=conf.db,
                            user=conf.user, password=conf.password)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df

def build_directed_graph(ch_df: pd.DataFrame, alias_df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[str, str]]:
    """Build directed graph from channel updates.
    
    - Filters nodes with zero total capacity
    - Adds directed edges for each channel direction
    - Aggregates multiple channels between same nodes
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
        u, v = row["advertising_nodeid"], row["connecting_nodeid"]
        cap = int(row["capacity_sat"])
        
        if u in nonzero_nodes and v in nonzero_nodes:
            if G.has_edge(u, v):
                # Multiple channels in same direction - aggregate
                G[u][v]["multiplicity"] = G[u][v].get("multiplicity", 1) + 1
                G[u][v]["capacity_sum_sat"] = G[u][v].get("capacity_sum_sat", 0) + cap
            else:
                G.add_edge(u, v, multiplicity=1, capacity_sum_sat=cap)
    
    # Build alias map
    alias_map = {}
    for _, r in alias_df.iterrows():
        node_id, alias = r["node_id"], r["alias"]
        alias_map[node_id] = alias.strip() if isinstance(alias, str) and alias.strip() else node_id
    
    for n in G.nodes:
        G.nodes[n]["alias"] = alias_map.get(n, n)
    
    return G, alias_map

def compute_closeness_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True, 
                          use_capacity_weight: bool = False) -> float:
    """Compute directed closeness centrality with optional capacity weighting.
    
    CRITICAL: Outgoing closeness measures routing capability (node sending payments).
    - For OUTGOING closeness: use original graph G (paths FROM node TO others)
    - For INCOMING closeness: use G.reverse() (paths FROM others TO node)
    
    NetworkX single_source_shortest_path_length(G, node) computes paths FROM node,
    which is exactly what we need for outgoing closeness.
    
    Args:
        G: Directed graph
        node: Target node ID
        use_outgoing: If True (default), compute outgoing closeness for routing.
                     If False, compute incoming closeness.
        use_capacity_weight: If True, use capacity as edge weights.
                           Large capacity = low weight = short effective distance.
                           
    Returns:
        Closeness centrality value (0.0 if node not in graph)
        
    References:
        - Freeman (1979): Centrality in networks
        - Opsahl et al. (2010): Node centrality in weighted networks
    """
    if node not in G:
        return 0.0
    
    # For outgoing closeness, use original graph G (not G.reverse())
    graph_to_use = G if use_outgoing else G.reverse()
    
    try:
        if use_capacity_weight:
            # Apply capacity weighting: weight = 1 / (1 + log(1 + capacity))
            # Large capacity -> small weight -> shorter effective distance
            for u, v in graph_to_use.edges():
                capacity = graph_to_use[u][v].get('capacity_sum_sat', 1)
                graph_to_use[u][v]['weight'] = 1.0 / (1.0 + np.log1p(capacity))
            
            # Use Dijkstra for weighted shortest paths
            lengths = nx.single_source_dijkstra_path_length(graph_to_use, node, weight='weight')
        else:
            # Topological: hop count only (BFS)
            lengths = nx.single_source_shortest_path_length(graph_to_use, node)
        
        n = len(graph_to_use)
        if len(lengths) <= 1:  # Only node itself
            return 0.0
        
        # Calculate closeness with Wasserman-Faust normalization
        total_distance = sum(lengths.values())
        n_reachable = len(lengths) - 1  # Exclude node itself
        
        if total_distance > 0:
            closeness = n_reachable / total_distance
            # Wasserman-Faust normalization for disconnected graphs
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
    """Return a copy of G with bidirectional channel between a and b.
    
    Simulates opening a Lightning Network channel (which is bidirectional).
    """
    H = G.copy()
    if a == b:  # No self-loops
        return H
    
    # Add both directions (Lightning channels are bidirectional)
    if not H.has_edge(a, b):
        H.add_edge(a, b, capacity_sum_sat=0)
    if not H.has_edge(b, a):
        H.add_edge(b, a, capacity_sum_sat=0)
    return H

def evaluate_single_candidate(G: nx.DiGraph, target: str, candidate: str, base_cc: float, 
                              alias_map: Dict[str, str], use_capacity: bool) -> Recommendation:
    """Evaluate a single candidate node (helper for parallel processing)."""
    H = simulate_add_channel_bidirectional(G, target, candidate)
    new_cc = compute_closeness_fast(H, target, use_outgoing=True, use_capacity_weight=use_capacity)
    delta_abs = new_cc - base_cc
    delta_pct = (delta_abs / base_cc * 100.0) if base_cc > 0 else (100.0 if new_cc > 0 else 0.0)
    return Recommendation(candidate, alias_map.get(candidate, candidate), new_cc, delta_abs, delta_pct)

def rank_single_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                          topk: int = 20, n_jobs: int = -1, use_capacity: bool = False) -> List[Recommendation]:
    """Rank candidate nodes by closeness centrality improvement (PARALLEL).
    
    Simulates opening a channel from target to each candidate node
    and measures the improvement in outgoing closeness centrality.
    """
    base_cc = compute_closeness_fast(G, target, use_outgoing=True, use_capacity_weight=use_capacity)
    
    # Candidates: nodes not directly connected to target
    if target in G:
        direct_neighbors = set(G.predecessors(target)) | set(G.successors(target))
    else:
        direct_neighbors = set()
    
    candidates = [n for n in G.nodes if n != target and n not in direct_neighbors]
    print(f"[INFO] Evaluating {len(candidates)} candidate nodes...", file=sys.stderr)
    
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    print(f"[INFO] Using {n_workers} workers, capacity_weight={use_capacity}", file=sys.stderr)
    
    results: List[Recommendation] = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_candidate = {
            executor.submit(evaluate_single_candidate, G, target, candidate, base_cc, alias_map, use_capacity): candidate
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

def evaluate_combo(G: nx.DiGraph, target: str, nodes: Tuple[str, ...], 
                  base_cc: float, use_capacity: bool) -> Tuple[Tuple[str, ...], float, float, float]:
    """Evaluate a combination of nodes (helper for parallel processing)."""
    H = G.copy()
    for n in nodes:
        H = simulate_add_channel_bidirectional(H, target, n)
    
    new_cc = compute_closeness_fast(H, target, use_outgoing=True, use_capacity_weight=use_capacity)
    delta_abs = new_cc - base_cc
    delta_pct = (delta_abs / base_cc * 100.0) if base_cc > 0 else (100.0 if new_cc > 0 else 0.0)
    return (nodes, new_cc, delta_abs, delta_pct)

def rank_combo_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                         top_singles: List[Recommendation], combo_k: int = 3, 
                         combo_top: int = 5, n_jobs: int = -1, use_capacity: bool = False) -> List:
    """Evaluate combinations of channel openings (PARALLEL)."""
    base_cc = compute_closeness_fast(G, target, use_outgoing=True, use_capacity_weight=use_capacity)
    ids = [r.node_id for r in top_singles]
    all_combos = list(itertools.combinations(ids, combo_k))
    
    print(f"[INFO] Evaluating {len(all_combos)} combinations...", file=sys.stderr)
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_combo = {
            executor.submit(evaluate_combo, G, target, nodes, base_cc, use_capacity): nodes
            for nodes in all_combos
        }
        
        completed = 0
        for future in as_completed(future_to_combo):
            completed += 1
            if completed % max(1, len(all_combos) // 20) == 0 or completed == len(all_combos):
                print(f"[PROGRESS] {completed}/{len(all_combos)} ({completed/len(all_combos)*100:.1f}%)", 
                      file=sys.stderr, end='\r')
            try:
                results.append(future.result())
            except Exception as e:
                print(f"\n[WARNING] Error: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return results[:combo_top]

def main():
    ap = argparse.ArgumentParser(
        description="Lightning Network Closeness Centrality Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Topological analysis (default)
  python ln_closeness_analysis.py --pg-host HOST --pg-port 5432 --pg-db DB \\
      --pg-user USER --pg-pass PASS --target-node NODE_ID

  # Capacity-weighted analysis (experimental)
  python ln_closeness_analysis.py --pg-host HOST --pg-port 5432 --pg-db DB \\
      --pg-user USER --pg-pass PASS --target-node NODE_ID --use-capacity

Note: Capacity-weighted centrality provides structural analysis but does not
reflect actual routing probability (balance distribution is private).
        """
    )
    ap.add_argument("--pg-host", required=True, help="PostgreSQL host")
    ap.add_argument("--pg-port", type=int, default=5432, help="PostgreSQL port")
    ap.add_argument("--pg-db", required=True, help="PostgreSQL database name")
    ap.add_argument("--pg-user", required=True, help="PostgreSQL username")
    ap.add_argument("--pg-pass", required=True, help="PostgreSQL password")
    ap.add_argument("--target-node", required=True, help="Hex node_id to analyze")
    ap.add_argument("--topk", type=int, default=20, help="Top-K single-node recommendations")
    ap.add_argument("--combo-k", type=int, default=3, help="Size of channel combinations")
    ap.add_argument("--combo-top", type=int, default=5, help="Number of top combos to output")
    ap.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel workers (-1 for all CPUs)")
    ap.add_argument("--use-capacity", action='store_true', 
                    help="Use capacity-weighted centrality (experimental, provides structural analysis)")
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
    
    base_cc = compute_closeness_fast(G, args.target_node, use_outgoing=True, use_capacity_weight=args.use_capacity)
    
    print("\n" + "="*70)
    print(f"  Current Outgoing Closeness Centrality")
    if args.use_capacity:
        print(f"  (Capacity-weighted - structural analysis)")
    else:
        print(f"  (Topological - hop count only)")
    print("="*70)
    print(f"Node:     {alias_map.get(args.target_node, args.target_node)}")
    print(f"Node ID:  {args.target_node}")
    print(f"Closeness: {base_cc:.6f}\n")
    
    singles = rank_single_candidates(G, args.target_node, alias_map, args.topk, args.n_jobs, args.use_capacity)
    
    if singles:
        print(f"\n{'Rank':<6}{'Alias':<30}{'New CC':<12}{'Δ Abs':<12}{'Δ %':<10}")
        print("-" * 70)
        for i, r in enumerate(singles, 1):
            display_alias = r.alias[:27] + "..." if len(r.alias) > 30 else r.alias
            print(f"{i:<6}{display_alias:<30}{r.new_cc:<12.6f}{r.delta_abs:<12.6f}+{r.delta_pct:<9.2f}%")
        
        pd.DataFrame([{
            "rank": i, "node_id": r.node_id, "alias": r.alias,
            "new_closeness": r.new_cc, "delta_abs": r.delta_abs, "delta_pct": r.delta_pct
        } for i, r in enumerate(singles, 1)]).to_csv("top_single_recommendations.csv", index=False)
        print("\n✅ Saved to: top_single_recommendations.csv")
    
    if singles and len(singles) >= args.combo_k:
        combos = rank_combo_candidates(G, args.target_node, alias_map, singles, 
                                      args.combo_k, args.combo_top, args.n_jobs, args.use_capacity)
        if combos:
            print(f"\n{'='*70}")
            print(f"  Top {args.combo_top} Combinations of {args.combo_k} Channels")
            print(f"{'='*70}")
            for i, (nodes, new_cc, d_abs, d_pct) in enumerate(combos, 1):
                labels = [alias_map.get(n, n) for n in nodes]
                print(f"\n#{i}")
                print(f"  Nodes:  {', '.join(labels)}")
                print(f"  New CC: {new_cc:.6f}  |  Δ: {d_abs:.6f}  |  +{d_pct:.2f}%")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
