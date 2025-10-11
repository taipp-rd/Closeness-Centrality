#!/usr/bin/env python3
"""
Lightning Network: Closeness Centrality Analyzer (Optimized + Weighted)

NEW FEATURE: Capacity-weighted centrality option
- Optional --use-capacity flag to consider channel capacity as weights
- Large capacity channels = shorter "effective distance"
- Based on Opsahl et al. (2010) weighted network centrality
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

# SQL queries remain unchanged
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
        u, v = row["advertising_nodeid"], row["connecting_nodeid"]
        cap = int(row["capacity_sat"])
        
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

def compute_closeness_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True, use_capacity_weight: bool = False) -> float:
    """
    Compute closeness centrality with optional capacity weighting.
    
    Args:
        use_capacity_weight: If True, use capacity as edge weights
                           (large capacity = low weight = short effective distance)
    """
    if node not in G:
        return 0.0
    
    graph_to_use = G.reverse() if use_outgoing else G
    
    try:
        if use_capacity_weight:
            # Prepare weighted graph: weight = 1 / log(1 + capacity)
            for u, v in graph_to_use.edges():
                capacity = graph_to_use[u][v].get('capacity_sum_sat', 1)
                # Large capacity -> small weight
                graph_to_use[u][v]['weight'] = 1.0 / (1.0 + np.log1p(capacity))
            
            lengths = nx.single_source_dijkstra_path_length(graph_to_use, node, weight='weight')
        else:
            # Topological: hop count only
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

def evaluate_single_candidate(G: nx.DiGraph, target: str, candidate: str, base_cc: float, 
                              alias_map: Dict[str, str], use_capacity: bool) -> Recommendation:
    H = simulate_add_channel_bidirectional(G, target, candidate)
    new_cc = compute_closeness_fast(H, target, use_outgoing=True, use_capacity_weight=use_capacity)
    delta_abs = new_cc - base_cc
    delta_pct = (delta_abs / base_cc * 100.0) if base_cc > 0 else (100.0 if new_cc > 0 else 0.0)
    return Recommendation(candidate, alias_map.get(candidate, candidate), new_cc, delta_abs, delta_pct)

def rank_single_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                          topk: int = 20, n_jobs: int = -1, use_capacity: bool = False) -> List[Recommendation]:
    base_cc = compute_closeness_fast(G, target, use_outgoing=True, use_capacity_weight=use_capacity)
    
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
    ap = argparse.ArgumentParser(description="Lightning Network Closeness Centrality Analyzer")
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
    ap.add_argument("--use-capacity", action='store_true', 
                    help="Use capacity-weighted centrality (experimental)")
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
    print(f"  Current Closeness (capacity_weight={args.use_capacity})")
    print("="*70)
    print(f"Node: {alias_map.get(args.target_node, args.target_node)}")
    print(f"Closeness: {base_cc:.6f}\n")
    
    singles = rank_single_candidates(G, args.target_node, alias_map, args.topk, args.n_jobs, args.use_capacity)
    
    if singles:
        print(f"\n{'Rank':<6}{'Alias':<30}{'New CC':<12}{'Δ Abs':<12}{'Δ %':<10}")
        print("-" * 70)
        for i, r in enumerate(singles, 1):
            print(f"{i:<6}{r.alias[:27]:<30}{r.new_cc:<12.6f}{r.delta_abs:<12.6f}+{r.delta_pct:<9.2f}%")
        
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
            print(f"  Top {args.combo_top} Combinations")
            print(f"{'='*70}")
            for i, (nodes, new_cc, d_abs, d_pct) in enumerate(combos, 1):
                labels = [alias_map.get(n, n) for n in nodes]
                print(f"\n#{i}: {', '.join(labels)}")
                print(f"  New CC: {new_cc:.6f} | Δ: {d_abs:.6f} | +{d_pct:.2f}%")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
