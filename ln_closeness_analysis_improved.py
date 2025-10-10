#!/usr/bin/env python3
"""
Lightning Network: Closeness Centrality Analyzer (Improved)

Features
- Connects to PostgreSQL (read-only) and pulls the *latest* graph state
  - channel_update: DISTINCT ON (chan_id, advertising_nodeid) latest rows per direction
  - node_announcement: DISTINCT ON (node_id) latest alias per node
  - Excludes closed channels (closed_channel)
- Builds a **directed** NetworkX graph with bidirectional edges
- Drops nodes whose total adjacent capacity equals 0
- Computes **outgoing closeness centrality** (G.reverse()) for routing capability
- Simulates opening channels and ranks Top 20 single-node additions
- Evaluates all 3-channel combinations from Top 20 and outputs Top 5 combos
- Prints both absolute and % improvement with node aliases

Key Improvements from Original
1. Correctly handles bidirectional Lightning Network channels
2. Uses outgoing closeness centrality (proper for payment routing)
3. Better NULL handling for capacity values
4. Improved SQL query to capture both channel directions
5. Enhanced documentation and error handling

Usage
    python ln_closeness_analysis_improved.py \
        --pg-host HOST --pg-port 5432 --pg-db DBNAME \
        --pg-user USER --pg-pass PASS \
        --target-node <hex_node_id> \
        [--topk 20] [--combo-k 3] [--combo-top 5]

Example
    python ln_closeness_analysis_improved.py \
        --pg-host localhost --pg-port 5432 --pg-db ln \
        --pg-user readonly --pg-pass '***' \
        --target-node 02abc...def

Theory & References
- Lightning Network topology analysis: Rohrer et al. (2019) 
  "Discharged Payment Channels" - https://arxiv.org/abs/1904.10253
- Closeness Centrality: Freeman (1979) - measures how efficiently 
  a node can reach all other nodes in the network
- NetworkX closeness_centrality uses INCOMING distances by default;
  for routing analysis we use OUTGOING distances via G.reverse()
"""
from __future__ import annotations
import argparse
import itertools
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List

import psycopg2
import pandas as pd
import networkx as nx

# --------------------------------------
# SQL (PostgreSQL) – latest snapshot with bidirectional channels
# --------------------------------------
CHANNELS_SQL = r"""
WITH latest_channel AS (
  -- Get latest update for each channel direction
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
    AND lc.capacity_sat > 0  -- Filter out zero capacity channels
    AND lc.rp_disabled = false  -- Only include enabled channels
)
SELECT * FROM open_channel;
"""

ALIASES_SQL = r"""
SELECT DISTINCT ON (na.node_id)
       na.node_id,
       na.alias,
       COALESCE(na.timestamp, 'epoch'::timestamp) AS last_ts
FROM node_announcement na
ORDER BY na.node_id, COALESCE(na.timestamp, 'epoch'::timestamp) DESC;
"""

# --------------------------------------
# Data classes
# --------------------------------------
@dataclass
class DBConf:
    host: str
    port: int
    db: str
    user: str
    password: str

# --------------------------------------
# Helpers
# --------------------------------------

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
    """Build a directed graph from channel updates with proper bidirectional edges.

    Key improvements:
    - Handles both directions of Lightning Network channels
    - Properly filters nodes with zero total capacity
    - Adds directed edges for each channel direction
    - Handles NULL capacity values correctly
    
    Args:
        ch_df: DataFrame with channel_update records
        alias_df: DataFrame with node_announcement records
    
    Returns:
        Tuple of (graph, alias_map)
    """
    # Safety check: remove rows with null node IDs
    ch_df = ch_df.dropna(subset=["advertising_nodeid", "connecting_nodeid"])
    
    # Ensure capacity is numeric and non-null
    ch_df['capacity_sat'] = pd.to_numeric(ch_df['capacity_sat'], errors='coerce').fillna(0)
    
    # Calculate per-node total adjacent capacity
    # Each channel contributes capacity from both perspectives
    node_capacity = {}
    for _, row in ch_df.iterrows():
        u, v, cap = row["advertising_nodeid"], row["connecting_nodeid"], row["capacity_sat"]
        node_capacity[u] = node_capacity.get(u, 0) + cap
        node_capacity[v] = node_capacity.get(v, 0) + cap
    
    # Keep only nodes with non-zero total capacity
    nonzero_nodes = {n for n, cap in node_capacity.items() if cap > 0}
    
    print(f"[INFO] Nodes with non-zero capacity: {len(nonzero_nodes)}", file=sys.stderr)
    print(f"[INFO] Nodes filtered out (zero capacity): {len(node_capacity) - len(nonzero_nodes)}", file=sys.stderr)
    
    # Build directed graph
    G = nx.DiGraph()
    
    # Add edges with metadata
    # Each row represents one direction of a channel
    for _, row in ch_df.iterrows():
        u, v = row["advertising_nodeid"], row["connecting_nodeid"]
        cap = int(row["capacity_sat"])
        
        # Only add if both nodes have non-zero capacity
        if u in nonzero_nodes and v in nonzero_nodes:
            if G.has_edge(u, v):
                # Multiple channels in same direction - aggregate
                G[u][v]["multiplicity"] = G[u][v].get("multiplicity", 1) + 1
                G[u][v]["capacity_sum_sat"] = G[u][v].get("capacity_sum_sat", 0) + cap
            else:
                G.add_edge(u, v, multiplicity=1, capacity_sum_sat=cap)
    
    # Build alias map (fallback to node_id if alias is empty/null)
    alias_map = {}
    for _, r in alias_df.iterrows():
        node_id = r["node_id"]
        alias = r["alias"]
        # Use alias only if it's a non-empty string
        if isinstance(alias, str) and alias.strip():
            alias_map[node_id] = alias.strip()
        else:
            alias_map[node_id] = node_id
    
    # Attach aliases to nodes
    for n in G.nodes:
        G.nodes[n]["alias"] = alias_map.get(n, n)
    
    return G, alias_map


def compute_closeness(G: nx.DiGraph, node: str, use_outgoing: bool = True) -> float:
    """Compute directed closeness centrality for a node.
    
    IMPORTANT: For Lightning Network routing analysis, we measure OUTGOING 
    closeness (how easily the node can send payments to others).
    
    NetworkX closeness_centrality computes INCOMING distances by default.
    To get OUTGOING distances, we reverse the graph.
    
    Args:
        G: Directed graph
        node: Target node ID
        use_outgoing: If True (default), compute outgoing closeness for routing.
                     If False, compute incoming closeness.
    
    Returns:
        Closeness centrality value (0.0 if node not in graph)
    
    Reference:
        - NetworkX docs: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html
        - Freeman (1979): "Centrality in networks: I. Conceptual clarification"
    """
    if node not in G:
        return 0.0
    
    # For outgoing closeness (routing capability), use reversed graph
    graph_to_use = G.reverse() if use_outgoing else G
    
    # wf_improved=True uses Wasserman-Faust normalization for disconnected graphs
    return nx.closeness_centrality(graph_to_use, u=node, wf_improved=True)


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
    Adds edges in both directions if they don't exist.
    
    Args:
        G: Original graph
        a: First node
        b: Second node
    
    Returns:
        New graph with channel added
    """
    H = G.copy()
    if a == b:  # No self-loops
        return H
    
    # Add both directions (Lightning channels are bidirectional)
    if not H.has_edge(a, b):
        H.add_edge(a, b, capacity_sum_sat=0)  # Capacity not relevant for topology analysis
    if not H.has_edge(b, a):
        H.add_edge(b, a, capacity_sum_sat=0)
    
    return H


def rank_single_candidates(G: nx.DiGraph, target: str, alias_map: Dict[str, str], topk: int = 20) -> List[Recommendation]:
    """Rank candidate nodes by closeness centrality improvement.
    
    Simulates opening a channel from target to each candidate node
    and measures the improvement in outgoing closeness centrality.
    
    Args:
        G: Current network graph
        target: Target node ID
        alias_map: Mapping from node ID to alias
        topk: Number of top candidates to return
    
    Returns:
        List of top recommendations sorted by improvement
    """
    base_cc = compute_closeness(G, target, use_outgoing=True)
    
    # Candidates: nodes not directly connected to target
    if target in G:
        direct_neighbors = set(G.predecessors(target)) | set(G.successors(target))
    else:
        direct_neighbors = set()
    
    candidates = [n for n in G.nodes if n != target and n not in direct_neighbors]
    
    print(f"[INFO] Evaluating {len(candidates)} candidate nodes...", file=sys.stderr)
    
    results: List[Recommendation] = []
    for n in candidates:
        H = simulate_add_channel_bidirectional(G, target, n)
        new_cc = compute_closeness(H, target, use_outgoing=True)
        delta_abs = new_cc - base_cc
        delta_pct = (delta_abs / base_cc * 100.0) if base_cc > 0 else (100.0 if new_cc > 0 else 0.0)
        
        if delta_abs > 0:  # Only include improving candidates
            results.append(Recommendation(n, alias_map.get(n, n), new_cc, delta_abs, delta_pct))
    
    # Sort by absolute improvement (descending), then by new_cc
    results.sort(key=lambda r: (r.delta_abs, r.new_cc), reverse=True)
    return results[:topk]


def rank_combo_candidates(
    G: nx.DiGraph, 
    target: str, 
    alias_map: Dict[str, str], 
    top_singles: List[Recommendation], 
    combo_k: int = 3, 
    combo_top: int = 5
) -> List[Tuple[Tuple[str, ...], float, float, float]]:
    """Evaluate combinations of channel openings.
    
    Tests all combinations of combo_k nodes from top_singles and
    measures the combined improvement in closeness centrality.
    
    Args:
        G: Current network graph
        target: Target node ID
        alias_map: Mapping from node ID to alias
        top_singles: List of top single-node recommendations
        combo_k: Size of combinations to test
        combo_top: Number of top combinations to return
    
    Returns:
        List of tuples: (node_ids_tuple, new_cc, delta_abs, delta_pct)
    """
    base_cc = compute_closeness(G, target, use_outgoing=True)
    combos = []
    
    ids = [r.node_id for r in top_singles]
    total_combos = len(list(itertools.combinations(ids, combo_k)))
    print(f"[INFO] Evaluating {total_combos} combinations of {combo_k} channels...", file=sys.stderr)
    
    for nodes in itertools.combinations(ids, combo_k):
        H = G.copy()
        # Add all channels in the combination
        for n in nodes:
            H = simulate_add_channel_bidirectional(H, target, n)
        
        new_cc = compute_closeness(H, target, use_outgoing=True)
        delta_abs = new_cc - base_cc
        delta_pct = (delta_abs / base_cc * 100.0) if base_cc > 0 else (100.0 if new_cc > 0 else 0.0)
        combos.append((nodes, new_cc, delta_abs, delta_pct))
    
    # Sort by absolute improvement (descending), then by new_cc
    combos.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return combos[:combo_top]


# --------------------------------------
# CLI
# --------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LN Closeness Centrality Analyzer (Improved - Outgoing Distance)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ln_closeness_analysis_improved.py --pg-host localhost --pg-port 5432 \\
      --pg-db ln --pg-user readonly --pg-pass 'secret' \\
      --target-node 02abc...def

Theory:
  This tool analyzes Lightning Network routing capability using outgoing 
  closeness centrality, which measures how efficiently a node can send 
  payments to all other nodes in the network.
  
  Reference: Rohrer et al. (2019) "Discharged Payment Channels"
             https://arxiv.org/abs/1904.10253
        """
    )
    ap.add_argument("--pg-host", required=True, help="PostgreSQL host")
    ap.add_argument("--pg-port", type=int, default=5432, help="PostgreSQL port")
    ap.add_argument("--pg-db", required=True, help="PostgreSQL database name")
    ap.add_argument("--pg-user", required=True, help="PostgreSQL username")
    ap.add_argument("--pg-pass", required=True, help="PostgreSQL password")
    ap.add_argument("--target-node", required=True, help="Hex node_id to analyze")
    ap.add_argument("--topk", type=int, default=20, help="Top-K single-node recommendations (default: 20)")
    ap.add_argument("--combo-k", type=int, default=3, help="Size of channel combinations (default: 3)")
    ap.add_argument("--combo-top", type=int, default=5, help="Number of top combos to output (default: 5)")

    args = ap.parse_args()

    conf = DBConf(
        host=args.pg_host, 
        port=args.pg_port, 
        db=args.pg_db, 
        user=args.pg_user, 
        password=args.pg_pass
    )

    # Fetch latest snapshots
    print("[INFO] Fetching latest open channels from database...", file=sys.stderr)
    ch_df = fetch_dataframe(conf, CHANNELS_SQL)
    print(f"[INFO] Open channel records fetched: {len(ch_df)}", file=sys.stderr)

    print("[INFO] Fetching latest node aliases...", file=sys.stderr)
    alias_df = fetch_dataframe(conf, ALIASES_SQL)
    print(f"[INFO] Node aliases fetched: {len(alias_df)}", file=sys.stderr)

    # Build graph
    print("[INFO] Building directed graph with bidirectional channels...", file=sys.stderr)
    G, alias_map = build_directed_graph(ch_df, alias_df)
    print(f"[INFO] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges", file=sys.stderr)

    # Check if target node exists
    if args.target_node not in G:
        print(f"[ERROR] Target node {args.target_node} not found in graph.", file=sys.stderr)
        print(f"[ERROR] The node may have zero capacity or no active channels.", file=sys.stderr)
        sys.exit(1)

    # Compute current closeness centrality (outgoing)
    base_cc = compute_closeness(G, args.target_node, use_outgoing=True)
    
    print("\n" + "="*70)
    print("  Current Outgoing Closeness Centrality")
    print("  (Measures routing capability: how easily node can send payments)")
    print("="*70)
    print(f"Node:     {alias_map.get(args.target_node, args.target_node)}")
    print(f"Node ID:  {args.target_node}")
    print(f"Closeness: {base_cc:.6f}")
    print()

    # Rank single-node channel openings
    print("="*70)
    print(f"  Top {args.topk} Single-Channel Openings")
    print("  (Opening one channel to these nodes improves closeness most)")
    print("="*70)
    
    singles = rank_single_candidates(G, args.target_node, alias_map, topk=args.topk)
    
    if not singles:
        print("❌ No improving single-node channel openings found.")
        print("   This may occur if the node is already optimally connected.")
    else:
        print(f"\n{'Rank':<6}{'Alias':<30}{'Node ID':<20}{'New CC':<12}{'Δ Absolute':<12}{'Δ %':<10}")
        print("-" * 90)
        
        single_rows = []
        for i, r in enumerate(singles, 1):
            # Truncate long aliases for display
            display_alias = r.alias[:27] + "..." if len(r.alias) > 30 else r.alias
            display_node = r.node_id[:17] + "..." if len(r.node_id) > 20 else r.node_id
            
            print(f"{i:<6}{display_alias:<30}{display_node:<20}{r.new_cc:<12.6f}{r.delta_abs:<12.6f}+{r.delta_pct:<9.2f}%")
            
            single_rows.append({
                "rank": i,
                "node_id": r.node_id,
                "alias": r.alias,
                "new_closeness": r.new_cc,
                "delta_abs": r.delta_abs,
                "delta_pct": r.delta_pct,
            })
        
        # Save to CSV
        pd.DataFrame(single_rows).to_csv("top_single_recommendations.csv", index=False)
        print(f"\n✅ Saved to: top_single_recommendations.csv")

    # Rank combinations
    if singles and len(singles) >= args.combo_k:
        print("\n" + "="*70)
        print(f"  Best {args.combo_k}-Channel Combinations (Top {args.combo_top})")
        print(f"  (Opening these {args.combo_k} channels together gives best improvement)")
        print("="*70)
        
        combos = rank_combo_candidates(
            G, args.target_node, alias_map, singles, 
            combo_k=args.combo_k, combo_top=args.combo_top
        )
        
        if not combos:
            print("❌ No improving combinations found.")
        else:
            combo_rows = []
            for i, (nodes, new_cc, d_abs, d_pct) in enumerate(combos, 1):
                labels = [alias_map.get(n, n) for n in nodes]
                
                print(f"\n#{i}")
                print(f"  Nodes:  {', '.join(labels)}")
                print(f"  New CC: {new_cc:.6f}  |  Δ: {d_abs:.6f}  |  +{d_pct:.2f}%")
                
                combo_rows.append({
                    "rank": i,
                    "node_ids": ";".join(nodes),
                    "aliases": ";".join(labels),
                    "new_closeness": new_cc,
                    "delta_abs": d_abs,
                    "delta_pct": d_pct,
                })
            
            pd.DataFrame(combo_rows).to_csv("top_combo_recommendations.csv", index=False)
            print(f"\n✅ Saved to: top_combo_recommendations.csv")
    elif singles:
        print(f"\n⚠️  Need at least {args.combo_k} candidates to evaluate combinations.")
        print(f"    Only {len(singles)} candidates found.")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
