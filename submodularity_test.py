#!/usr/bin/env python3
"""
Submodularity Verification for Lightning Network Closeness Centrality

Theoretical Background:
Submodular function: f(S∪{x}) - f(S) ≥ f(T∪{x}) - f(T) for all S⊆T
- Diminishing returns property
- If submodular: greedy achieves (1-1/e)≈63% approximation
- If not submodular: greedy has no theoretical guarantee

References:
- Nemhauser et al. (1978). "Approximations for submodular set functions"
- Krause & Golovin (2014). "Submodular function maximization"
"""
import argparse
import random
import sys
from typing import Set, List, Tuple, Dict
import psycopg2
import pandas as pd
import networkx as nx
import numpy as np
from dataclasses import dataclass

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

def add_channels_to_graph(G: nx.DiGraph, target: str, nodes: Set[str]) -> nx.DiGraph:
    """Add multiple bidirectional channels to graph."""
    H = G.copy()
    for node in nodes:
        if node != target:
            if not H.has_edge(target, node):
                H.add_edge(target, node, capacity_sum_sat=0)
            if not H.has_edge(node, target):
                H.add_edge(node, target, capacity_sum_sat=0)
    return H

def closeness_gain(G: nx.DiGraph, target: str, S: Set[str], x: str) -> float:
    """Calculate f(S∪{x}) - f(S) where f is closeness centrality."""
    # f(S)
    H_S = add_channels_to_graph(G, target, S)
    cc_S = compute_closeness_fast(H_S, target, use_outgoing=True)
    
    # f(S∪{x})
    H_S_x = add_channels_to_graph(G, target, S | {x})
    cc_S_x = compute_closeness_fast(H_S_x, target, use_outgoing=True)
    
    return cc_S_x - cc_S

def test_submodularity(G: nx.DiGraph, target: str, candidates: List[str], 
                       num_tests: int = 100, max_set_size: int = 5) -> Dict:
    """
    Test submodularity property empirically.
    
    For submodular f: f(S∪{x}) - f(S) ≥ f(T∪{x}) - f(T) for all S⊆T, x∉T
    
    Algorithm:
    1. Randomly sample S ⊆ T ⊆ V \ {target}
    2. Randomly sample x ∉ T
    3. Compute marginal gains: Δ_S = f(S∪{x}) - f(S), Δ_T = f(T∪{x}) - f(T)
    4. Check if Δ_S ≥ Δ_T (submodular condition)
    5. Record violation if Δ_S < Δ_T - ε (numerical tolerance)
    
    Returns:
        Dict with statistics: violations, tests, ratio, marginal_gains_data
    """
    violations = 0
    total_tests = 0
    violation_data = []
    all_marginal_gains = []
    
    epsilon = 1e-10  # Numerical tolerance
    
    print(f"\n[TEST] Starting submodularity verification...")
    print(f"[TEST] Target: {target}")
    print(f"[TEST] Candidates: {len(candidates)}")
    print(f"[TEST] Tests: {num_tests}\n")
    
    for test_num in range(num_tests):
        # Sample S ⊆ T with |S| < |T|
        size_S = random.randint(0, min(max_set_size - 1, len(candidates) - 2))
        size_T = random.randint(size_S + 1, min(max_set_size, len(candidates) - 1))
        
        # Create S
        S = set(random.sample(candidates, size_S)) if size_S > 0 else set()
        
        # Create T ⊇ S
        remaining_for_T = [c for c in candidates if c not in S]
        if len(remaining_for_T) < size_T - size_S:
            continue
        additional_for_T = set(random.sample(remaining_for_T, size_T - size_S))
        T = S | additional_for_T
        
        # Sample x ∉ T
        available_x = [c for c in candidates if c not in T]
        if not available_x:
            continue
        x = random.choice(available_x)
        
        # Compute marginal gains
        marginal_S = closeness_gain(G, target, S, x)
        marginal_T = closeness_gain(G, target, T, x)
        
        all_marginal_gains.append((marginal_S, marginal_T))
        
        # Check submodular condition: Δ_S ≥ Δ_T
        if marginal_S < marginal_T - epsilon:
            violations += 1
            violation_data.append({
                'test': test_num + 1,
                'S_size': len(S),
                'T_size': len(T),
                'marginal_S': marginal_S,
                'marginal_T': marginal_T,
                'difference': marginal_T - marginal_S
            })
        
        total_tests += 1
        
        if (test_num + 1) % 10 == 0:
            print(f"[PROGRESS] {test_num + 1}/{num_tests} tests completed", end='\r')
    
    print(f"\n\n[RESULTS] Submodularity Test Complete")
    print(f"{'='*70}")
    
    submodularity_ratio = 1.0 - (violations / total_tests) if total_tests > 0 else 0.0
    
    return {
        'violations': violations,
        'total_tests': total_tests,
        'submodularity_ratio': submodularity_ratio,
        'violation_data': violation_data,
        'marginal_gains': all_marginal_gains
    }

def analyze_results(results: Dict, alias_map: Dict[str, str]):
    """Analyze and display submodularity test results."""
    violations = results['violations']
    total = results['total_tests']
    ratio = results['submodularity_ratio']
    
    print(f"Total tests:          {total}")
    print(f"Violations:           {violations}")
    print(f"Submodularity ratio:  {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Interpretation
    if ratio >= 0.95:
        print("✅ STRONGLY SUBMODULAR")
        print("   Greedy algorithm has (1-1/e)≈63% approximation guarantee")
        print("   Lightning Network closeness improvement exhibits diminishing returns")
    elif ratio >= 0.85:
        print("⚠️  APPROXIMATELY SUBMODULAR")
        print("   Greedy algorithm likely performs well in practice")
        print("   Some violations may be due to numerical precision")
    elif ratio >= 0.70:
        print("⚠️  WEAKLY SUBMODULAR")
        print("   Greedy algorithm may still be effective")
        print("   Consider case-by-case analysis")
    else:
        print("❌ NOT SUBMODULAR")
        print("   Greedy algorithm has no theoretical guarantee")
        print("   Exhaustive search may be necessary for optimal solution")
    
    # Statistical analysis
    if results['marginal_gains']:
        marginal_S_values = [mg[0] for mg in results['marginal_gains']]
        marginal_T_values = [mg[1] for mg in results['marginal_gains']]
        
        print(f"\n{'='*70}")
        print("Statistical Analysis of Marginal Gains")
        print(f"{'='*70}")
        print(f"Marginal gain Δ_S (smaller set):")
        print(f"  Mean:   {np.mean(marginal_S_values):.8f}")
        print(f"  Median: {np.median(marginal_S_values):.8f}")
        print(f"  Std:    {np.std(marginal_S_values):.8f}")
        print(f"\nMarginal gain Δ_T (larger set):")
        print(f"  Mean:   {np.mean(marginal_T_values):.8f}")
        print(f"  Median: {np.median(marginal_T_values):.8f}")
        print(f"  Std:    {np.std(marginal_T_values):.8f}")
        
        # Correlation test
        if len(marginal_S_values) > 1:
            correlation = np.corrcoef(marginal_S_values, marginal_T_values)[0, 1]
            print(f"\nCorrelation (Δ_S vs Δ_T): {correlation:.4f}")
    
    # Show worst violations
    if results['violation_data']:
        print(f"\n{'='*70}")
        print("Top 5 Violations (Δ_T > Δ_S)")
        print(f"{'='*70}")
        sorted_violations = sorted(results['violation_data'], 
                                  key=lambda x: x['difference'], reverse=True)[:5]
        
        for i, v in enumerate(sorted_violations, 1):
            print(f"\n#{i}:")
            print(f"  |S|={v['S_size']}, |T|={v['T_size']}")
            print(f"  Δ_S = {v['marginal_S']:.8f}")
            print(f"  Δ_T = {v['marginal_T']:.8f}")
            print(f"  Violation: Δ_T - Δ_S = {v['difference']:.8f}")

def main():
    ap = argparse.ArgumentParser(
        description="Submodularity Verification for Lightning Network Closeness Centrality",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pg-host", required=True)
    ap.add_argument("--pg-port", type=int, default=5432)
    ap.add_argument("--pg-db", required=True)
    ap.add_argument("--pg-user", required=True)
    ap.add_argument("--pg-pass", required=True)
    ap.add_argument("--target-node", required=True)
    ap.add_argument("--num-tests", type=int, default=100,
                    help="Number of random tests to perform (default: 100)")
    ap.add_argument("--max-set-size", type=int, default=5,
                    help="Maximum set size for testing (default: 5)")
    ap.add_argument("--num-candidates", type=int, default=30,
                    help="Number of candidate nodes to test (default: 30)")
    args = ap.parse_args()

    conf = DBConf(args.pg_host, args.pg_port, args.pg_db, args.pg_user, args.pg_pass)
    
    print("[INFO] Fetching data...", file=sys.stderr)
    ch_df = fetch_dataframe(conf, CHANNELS_SQL)
    alias_df = fetch_dataframe(conf, ALIASES_SQL)
    
    print("[INFO] Building graph...", file=sys.stderr)
    G, alias_map = build_directed_graph(ch_df, alias_df)
    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    
    if args.target_node not in G:
        print(f"[ERROR] Target node not found", file=sys.stderr)
        sys.exit(1)
    
    # Get candidate nodes (not directly connected)
    if args.target_node in G:
        direct_neighbors = set(G.predecessors(args.target_node)) | set(G.successors(args.target_node))
    else:
        direct_neighbors = set()
    
    all_candidates = [n for n in G.nodes if n != args.target_node and n not in direct_neighbors]
    candidates = random.sample(all_candidates, min(args.num_candidates, len(all_candidates)))
    
    # Run submodularity test
    results = test_submodularity(
        G, args.target_node, candidates, 
        num_tests=args.num_tests,
        max_set_size=args.max_set_size
    )
    
    # Analyze and display results
    analyze_results(results, alias_map)
    
    # Save detailed results
    if results['violation_data']:
        df = pd.DataFrame(results['violation_data'])
        df.to_csv('submodularity_violations.csv', index=False)
        print(f"\n💾 Detailed violations saved to: submodularity_violations.csv")
    
    # Save marginal gains for further analysis
    if results['marginal_gains']:
        mg_df = pd.DataFrame(results['marginal_gains'], columns=['marginal_S', 'marginal_T'])
        mg_df.to_csv('marginal_gains.csv', index=False)
        print(f"💾 Marginal gains saved to: marginal_gains.csv")
    
    print(f"\n{'='*70}")
    print("Verification complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
