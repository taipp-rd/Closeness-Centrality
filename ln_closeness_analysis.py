#!/usr/bin/env python3
"""
Lightning Network: Closeness & Harmonic Centrality Analyzer

NEW FEATURE: Harmonic Centrality
- Alternative to closeness centrality for disconnected graphs
- HC(v) = Σ(1/d(v,u)) / (n-1)
- Handles infinite distances (unreachable nodes) as 0
- More stable for Lightning Network's dynamic topology
- References: Marchiori & Latora (2000), Boldi & Vigna (2014)
"""
from __future__ import annotations
import argparse
import itertools
import sys
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
import pandas as pd
import networkx as nx
import numpy as np

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

def analyze_connectivity(G: nx.DiGraph) -> Dict:
    """Analyze network connectivity."""
    strongly = list(nx.strongly_connected_components(G))
    weakly = list(nx.weakly_connected_components(G))
    largest_strong = max(strongly, key=len)
    
    return {
        'is_strongly_connected': nx.is_strongly_connected(G),
        'num_strong_components': len(strongly),
        'largest_strong_size': len(largest_strong),
        'strong_coverage': len(largest_strong) / len(G) * 100,
        'num_weak_components': len(weakly)
    }

def compute_closeness_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True) -> float:
    """Closeness centrality (Freeman 1979)."""
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

def compute_harmonic_fast(G: nx.DiGraph, node: str, use_outgoing: bool = True) -> float:
    """
    Harmonic centrality (Marchiori & Latora 2000, Boldi & Vigna 2014).
    
    HC(v) = Σ(u≠v) [1/d(v,u)] / (n-1)
    
    Key advantages for Lightning Network:
    - Handles disconnected graphs: 1/∞ = 0
    - More stable for dynamic topology
    - High correlation with closeness (ρ > 0.95) in connected components
    """
    if node not in G:
        return 0.0
    
    graph_to_use = G.reverse() if use_outgoing else G
    
    try:
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
    except:
        return 0.0

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

def simulate_add_channel_bidirectional(G: nx.DiGraph, a: str, b: str) -> nx.DiGraph:
    H = G.copy()
    if a == b:
        return H
    if not H.has_edge(a, b):
        H.add_edge(a, b, capacity_sum_sat=0)
    if not H.has_edge(b, a):
        H.add_edge(b, a, capacity_sum_sat=0)
    return H

def evaluate_single_candidate(G: nx.DiGraph, target: str, candidate: str, 
                              base_cc: float, base_hc: float, alias_map: Dict[str, str]) -> Recommendation:
    H = simulate_add_channel_bidirectional(G, target, candidate)
    new_cc = compute_closeness_fast(H, target, use_outgoing=True)
    new_hc = compute_harmonic_fast(H, target, use_outgoing=True)
    
    delta_cc_abs = new_cc - base_cc
    delta_hc_abs = new_hc - base_hc
    delta_cc_pct = (delta_cc_abs / base_cc * 100.0) if base_cc > 0 else 0.0
    delta_hc_pct = (delta_hc_abs / base_hc * 100.0) if base_hc > 0 else 0.0
    
    return Recommendation(
        candidate, alias_map.get(candidate, candidate),
        new_cc, new_hc, delta_cc_abs, delta_hc_abs, delta_cc_pct, delta_hc_pct
    )

def rank_single_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                          topk: int = 20, n_jobs: int = -1, sort_by: str = 'closeness') -> List[Recommendation]:
    base_cc = compute_closeness_fast(G, target, use_outgoing=True)
    base_hc = compute_harmonic_fast(G, target, use_outgoing=True)
    
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
            executor.submit(evaluate_single_candidate, G, target, candidate, base_cc, base_hc, alias_map): candidate
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
                if rec.delta_cc_abs > 0 or rec.delta_hc_abs > 0:
                    results.append(rec)
            except Exception as e:
                print(f"\n[WARNING] Error: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    # Sort by specified metric
    if sort_by == 'harmonic':
        results.sort(key=lambda r: (r.delta_hc_abs, r.new_hc), reverse=True)
    else:
        results.sort(key=lambda r: (r.delta_cc_abs, r.new_cc), reverse=True)
    
    return results[:topk]

def evaluate_combo(G: nx.DiGraph, target: str, nodes: Tuple[str, ...], 
                  base_cc: float, base_hc: float) -> Tuple:
    H = G.copy()
    for n in nodes:
        H = simulate_add_channel_bidirectional(H, target, n)
    
    new_cc = compute_closeness_fast(H, target, use_outgoing=True)
    new_hc = compute_harmonic_fast(H, target, use_outgoing=True)
    delta_cc = new_cc - base_cc
    delta_hc = new_hc - base_hc
    delta_cc_pct = (delta_cc / base_cc * 100.0) if base_cc > 0 else 0.0
    delta_hc_pct = (delta_hc / base_hc * 100.0) if base_hc > 0 else 0.0
    
    return (nodes, new_cc, new_hc, delta_cc, delta_hc, delta_cc_pct, delta_hc_pct)

def rank_combo_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], 
                         top_singles: List[Recommendation], combo_k: int = 3, 
                         combo_top: int = 5, n_jobs: int = -1) -> List:
    base_cc = compute_closeness_fast(G, target, use_outgoing=True)
    base_hc = compute_harmonic_fast(G, target, use_outgoing=True)
    ids = [r.node_id for r in top_singles]
    all_combos = list(itertools.combinations(ids, combo_k))
    
    print(f"[INFO] Evaluating {len(all_combos)} combinations...", file=sys.stderr)
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_combo = {
            executor.submit(evaluate_combo, G, target, nodes, base_cc, base_hc): nodes
            for nodes in all_combos
        }
        
        completed = 0
        for future in as_completed(future_to_combo):
            completed += 1
            if completed % max(1, len(all_combos) // 20) == 0:
                print(f"[PROGRESS] {completed}/{len(all_combos)} ({completed/len(all_combos)*100:.1f}%)", 
                      file=sys.stderr, end='\r')
            try:
                results.append(future.result())
            except Exception as e:
                print(f"\n[WARNING] Error: {e}", file=sys.stderr)
        print(file=sys.stderr)
    
    results.sort(key=lambda x: (x[3], x[1]), reverse=True)  # Sort by delta_cc
    return results[:combo_top]

def main():
    ap = argparse.ArgumentParser(description="Lightning Network Centrality Analyzer")
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
    ap.add_argument("--sort-by", choices=['closeness', 'harmonic'], default='closeness',
                    help="Sort candidates by closeness or harmonic centrality")
    args = ap.parse_args()

    conf = DBConf(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_pass)
    
    print("[INFO] Fetching data...", file=sys.stderr)
    ch_df = fetch_dataframe(conf, CHANNELS_SQL)
    alias_df = fetch_dataframe(conf, ALIASES_SQL)
    
    print("[INFO] Building graph...", file=sys.stderr)
    G, alias_map = build_directed_graph(ch_df, alias_df)
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
    
    if args.target_node not in G:
        print(f"\n[ERROR] Target node not found", file=sys.stderr)
        sys.exit(1)
    
    base_cc = compute_closeness_fast(G, args.target_node, use_outgoing=True)
    base_hc = compute_harmonic_fast(G, args.target_node, use_outgoing=True)
    
    print("\n" + "="*70)
    print("  Current Centrality Values")
    print("="*70)
    print(f"Node: {alias_map.get(args.target_node, args.target_node)}")
    print(f"Closeness Centrality:  {base_cc:.6f}")
    print(f"Harmonic Centrality:   {base_hc:.6f}")
    if base_cc > 0 and base_hc > 0:
        correlation = np.corrcoef([base_cc], [base_hc])[0,1] if base_cc != base_hc else 1.0
        print(f"Correlation: {correlation:.4f}")
    
    singles = rank_single_candidates(G, args.target_node, alias_map, args.topk, args.n_jobs, args.sort_by)
    
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
        cc_deltas = [r.delta_cc_abs for r in singles]
        hc_deltas = [r.delta_hc_abs for r in singles]
        if len(cc_deltas) > 1:
            corr = np.corrcoef(cc_deltas, hc_deltas)[0,1]
            print(f"\nCorrelation (ΔCC vs ΔHC): {corr:.4f}")
        
        pd.DataFrame([{
            "rank": i, "node_id": r.node_id, "alias": r.alias,
            "new_closeness": r.new_cc, "new_harmonic": r.new_hc,
            "delta_cc_abs": r.delta_cc_abs, "delta_hc_abs": r.delta_hc_abs,
            "delta_cc_pct": r.delta_cc_pct, "delta_hc_pct": r.delta_hc_pct
        } for i, r in enumerate(singles, 1)]).to_csv("centrality_comparison.csv", index=False)
        print("\n✅ Saved to: centrality_comparison.csv")
    
    if singles and len(singles) >= args.combo_k:
        combos = rank_combo_candidates(G, args.target_node, alias_map, singles, 
                                      args.combo_k, args.combo_top, args.n_jobs)
        if combos:
            print(f"\n{'='*70}")
            print(f"  Top {args.combo_top} Combinations")
            print(f"{'='*70}")
            for i, (nodes, new_cc, new_hc, d_cc, d_hc, d_cc_pct, d_hc_pct) in enumerate(combos, 1):
                labels = [alias_map.get(n, n)[:20] for n in nodes]
                print(f"\n#{i}: {', '.join(labels)}")
                print(f"  CC: {new_cc:.6f} (+{d_cc_pct:.2f}%) | HC: {new_hc:.6f} (+{d_hc_pct:.2f}%)")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
