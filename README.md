# Lightning Network Closeness Centrality Analyzer

Lightning Networkのノードの**近接中心性(Closeness Centrality)**を分析し、最適なチャネル開設戦略を提案するツールです。

### 最適化の特徴

・ **マルチコア並列処理** - ThreadPoolExecutorによる全候補ノードの同時評価  
・ **BFS実装** - NetworkXのcloseness_centrality関数の代わりに直接BFSで計算  
・ **進捗表示** - リアルタイムで処理状況を確認可能  


### 並列処理の制御

```bash
# 全CPUコアを使用（デフォルト）
python ln_closeness_analysis.py ... --n-jobs -1

# 4コアのみ使用（推奨：CPU負荷を抑える場合）
python ln_closeness_analysis.py ... --n-jobs 4

# シングルスレッド（デバッグ用）
python ln_closeness_analysis.py ... --n-jobs 1
```

##  機能

1. **現在の近接中心性を測定** - 指定したノードの支払い送信効率を評価
2. **最適な単一チャネルを提案** - 近接中心性を最も改善するノードをトップ20でランキング
3. **最適な組み合わせを提案** - 3つのチャネルの最適な組み合わせトップ5を提案

##  理論的背景

### 近接中心性とは

近接中心性は、ノードから他の全てのノードへの最短経路距離の逆数として定義されます：

```
CC(v) = (n-1) / Σ d(v,u)
```

Lightning Networkでは:
- **高い近接中心性** = 少ないホップ数で支払いを送信可能
- **低い近接中心性** = 支払いに多くの中継が必要

### 有向グラフと方向性

このツールは:
- Lightning Networkを**有向グラフ**として扱います
- **Outgoing Closeness Centrality**（外向き近接中心性）を測定
  - 自ノードから他のノードへ支払いを送信する能力を評価
  - **重要**: 外向き近接中心性を計算するため、元のグラフGをそのまま使用します
  - NetworkXの`single_source_shortest_path_length(G, node)`は、指定ノードからの最短経路を計算するため、外向き距離を正しく測定できます

### ⚠️ 重要な修正点

**以前の実装の誤り:**
- 外向き近接中心性の計算で`G.reverse()`を使用していました
- これは逆効果で、実際には**内向き近接中心性**を計算していました

**修正後の正しい実装:**
- 外向き近接中心性: `G`（元のグラフ）をそのまま使用
- 内向き近接中心性: `G.reverse()`（反転グラフ）を使用

**理論的根拠:**
- `single_source_shortest_path_length(G, v)`は、ノードvから他のノードへの距離を計算
- したがって、外向き距離を測定するには元のグラフGを使用すべき
- `G.reverse()`を使用すると、エッジの向きが逆転し、内向き距離を測定することになる

### 学術的根拠

- **Rohrer et al. (2019)**: ["Discharged Payment Channels"](https://arxiv.org/abs/1904.10253)
  - Lightning Networkのトポロジー分析
  - 中心性指標とネットワーク効率の関係を実証

- **Freeman (1979)**: "Centrality in networks: I. Conceptual clarification"
  - 近接中心性の数学的定義

- **Brandes & Pich (2007)**: "Centrality Estimation in Large Networks"
  - サンプリングベースの高速化手法

##  使い方

### 必要な環境

```bash
pip install psycopg2-binary pandas networkx
```

### PowerShell での実行方法

**重要**: PowerShellでは、すべての引数を**ダブルクォーテーション**で囲む必要があります。

```powershell
python ln_closeness_analysis.py --pg-host "localhost" --pg-port 5432 --pg-db "lightning_network" --pg-user "readonly" --pg-pass "your_password" --target-node "02abc123...def456" --topk 20 --combo-k 3 --combo-top 5 --n-jobs -1
```

**注意事項**:
- `<` と `>` は使用しないでください（PowerShellの予約文字）
- `[` と `]` は使用しないでください（配列構文として解釈される）
- パスワードに特殊文字が含まれる場合は必ずダブルクォーテーションで囲む

### Unix/Linux/macOS での実行方法

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'your_password' \
    --target-node 02abc123...def456 \
    --n-jobs -1
```

### オプション

```bash
--topk      # 単一チャネル推奨のトップN (デフォルト: 20)
--combo-k   # 組み合わせのサイズ (デフォルト: 3)
--combo-top # 表示する組み合わせの数 (デフォルト: 5)
--n-jobs    # 並列処理のワーカー数 (-1で全CPU使用、デフォルト: -1)
```

### 実際の実行例

```powershell
# PowerShell の例（全CPU使用）
python ln_closeness_analysis.py --pg-host "lightning-graph-db.example.com" --pg-port 19688 --pg-db "graph" --pg-user "readonly" --pg-pass "your_password" --target-node "03f5dc9f57c6c047938494ced134a485b1be5a134a6361bc5e33c2221bd9313d14" --topk 30 --combo-k 4 --combo-top 5 --n-jobs -1

# CPU負荷を抑える場合（4コアのみ使用）
python ln_closeness_analysis.py --pg-host "localhost" --pg-port 5432 --pg-db "ln" --pg-user "readonly" --pg-pass "pass" --target-node "02abc..." --n-jobs 4
```

##  出力

### コンソール出力

```
[INFO] Fetching latest open channels from database...
[INFO] Open channel records fetched: 75373
[INFO] Fetching latest node aliases...
[INFO] Node aliases fetched: 8307
[INFO] Building directed graph with bidirectional channels...
[INFO] Nodes with non-zero capacity: 8500
[INFO] Graph: 8500 nodes, 68620 directed edges

======================================================================
  Current Outgoing Closeness Centrality
  (Measures routing capability: how easily node can send payments)
======================================================================
Node:     
Node ID:  03f5dc9f・・・
Closeness: 0.353823

======================================================================
  Top 20 Single-Channel Openings
======================================================================
[INFO] Evaluating 8479 candidate nodes in parallel...
[INFO] Using 8 parallel workers
[PROGRESS] 8479/8479 candidates evaluated (100.0%)

Rank  Alias                Node ID             New CC      Δ Absolute  Δ %       
----------------------------------------------------------------------------------
1     ACINQ                03864e...f8ab       0.361245    0.007422    +2.10%
2     LNBig.com            02fd3a...9bc2       0.360891    0.007068    +2.00%
3     Bitfinex             02d96e...a8f3       0.359982    0.006159    +1.74%
...

✅ Saved to: top_single_recommendations.csv

======================================================================
  Best 3-Channel Combinations (Top 5)
======================================================================
[INFO] Evaluating 1140 combinations of 3 channels in parallel...
[INFO] Using 8 parallel workers
[PROGRESS] 1140/1140 combinations evaluated (100.0%)

#1
  Nodes:  ACINQ, LNBig.com, Bitfinex
  New CC: 0.372456  |  Δ: 0.018633  |  +5.27%

#2
  Nodes:  ACINQ, LNBig.com, Kraken
  New CC: 0.371823  |  Δ: 0.018000  |  +5.09%
...

✅ Saved to: top_combo_recommendations.csv

======================================================================
Analysis complete!
======================================================================
```

### CSVファイル

- `top_single_recommendations.csv` - 単一チャネル推奨
- `top_combo_recommendations.csv` - 組み合わせ推奨

##  技術的詳細

### アルゴリズムの最適化

#### 1. 並列BFS処理

**従来版（シングルスレッド）:**
```python
for candidate in candidates:
    new_cc = compute_closeness(G, target)  # 順次処理
```

**最適化版（マルチスレッド）:**
```python
with ThreadPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(evaluate_candidate, c): c for c in candidates}
    for future in as_completed(futures):
        result = future.result()  # 並列処理
```

#### 2. 直接BFS計算と正しいグラフの使用

NetworkXの`closeness_centrality()`関数の代わりに、`single_source_shortest_path_length()`を直接使用：

**利点:**
- 関数呼び出しのオーバーヘッド削減
- 不要な計算の省略
- メモリ効率の向上

```python
def compute_closeness_fast(G, node, use_outgoing=True):
    # 重要: 外向き近接中心性には元のグラフGを使用
    graph_to_use = G if use_outgoing else G.reverse()
    
    # ノードnodeからの最短経路を計算
    lengths = nx.single_source_shortest_path_length(graph_to_use, node)
    total_distance = sum(lengths.values())
    n_reachable = len(lengths) - 1
    
    # Wasserman-Faust正規化
    closeness = n_reachable / total_distance
    s = n_reachable / (len(G) - 1)
    return closeness * s
```

#### 3. プログレス表示

長時間実行時のユーザビリティ向上：
```python
if completed % (total // 20) == 0:
    print(f"[PROGRESS] {completed}/{total} ({progress:.1f}%)")
```

### データベーススキーマ

使用するテーブル:

1. **channel_update** - チャネルの更新情報
   - `chan_id`, `advertising_nodeid`, `connecting_nodeid`
   - `capacity_sat`, `rp_disabled`
   - `timestamp` (integer: Unix timestamp)
   - `rp_last_update` (integer: Unix timestamp)

2. **closed_channel** - 閉じられたチャネル
   - `chan_id`

3. **node_announcement** - ノード情報
   - `node_id`, `alias`
   - `timestamp` (integer: Unix timestamp)

**重要**: `timestamp`フィールドは**integer型**（Unix timestamp）です。

### グラフ構築と近接中心性計算

1. **データ取得**
   ```sql
   -- 各方向の最新レコードを取得
   SELECT DISTINCT ON (chan_id, advertising_nodeid) ...
   
   -- timestampはinteger型なのでCOALESCEの型変換は不要
   SELECT DISTINCT ON (node_id) ... ORDER BY timestamp DESC
   ```

2. **グラフ構築**
   - 有向グラフとして構築
   - 双方向チャネルを適切に表現
   - 容量ゼロのノードを除外

3. **近接中心性計算（修正済み）**
   ```python
   # 外向き近接中心性（正しい実装）
   # 元のグラフGを使用して、targetノードから他のノードへの距離を測定
   closeness = compute_closeness_fast(G, target, use_outgoing=True)
   ```

4. **シミュレーション**
   - 各候補ノードとのチャネル開設をシミュレート
   - 並列処理で全候補を同時評価
   - 改善度でランキング

##  実用例

### ルーティングノードの最適化

```
現状: 近接中心性 = 0.353823

単一チャネル推奨:
  ACINQ とのチャネル → +2.10% 改善

組み合わせ推奨:
  ACINQ + LNBig + Bitfinex → +5.27% 改善
```

**解釈:**
- 改善度が高いノードは、ネットワークの中心に位置
- 複数チャネルの組み合わせは相乗効果を生む
- 実際の運用では容量やコストも考慮が必要

### 大規模ネットワークでの使用

**10,000ノード以上のネットワークの場合:**

```bash
# 推奨設定
python ln_closeness_analysis.py \
    ... \
    --topk 30 \          # より多くの候補を評価
    --combo-k 3 \        # 組み合わせサイズは3が最適
    --combo-top 10 \     # 上位10組み合わせを表示
    --n-jobs -1          # 全CPUを使用
```

**メモリ制約がある場合:**
```bash
# メモリ節約モード
--topk 15 \
--combo-k 2 \
--n-jobs 4
```

## ⚠️ 注意事項

1. **トポロジー分析のみ** - このツールは容量や流動性を考慮しません
2. **スナップショット時点** - Lightning Networkは常に変化するため、分析結果は実行時点のもの
3. **総合的判断が必要** - 手数料、評判、安定性なども重要な考慮事項
4. **CPU負荷** - 並列処理により一時的に高いCPU使用率となります

##  トラブルシューティング

### エラー: `COALESCE types integer and timestamp cannot be matched`

**原因**: `node_announcement.timestamp`フィールドがinteger型（Unix timestamp）なのに、timestamp型として扱おうとしている

**解決済み**: 最新版では修正されています。`timestamp`をそのまま使用します。

### エラー: PowerShellでコマンドが認識されない

**原因**: PowerShellの特殊文字解釈

**解決方法**: すべての引数をダブルクォーテーションで囲む
```powershell
# ❌ 間違い
python ln_closeness_analysis.py --target-node <03f5dc...>

# ✅ 正しい
python ln_closeness_analysis.py --target-node "03f5dc..."
```

### 問題: 処理が遅い

**原因**: シングルスレッドで実行されている可能性

**解決方法**: 並列処理を有効化
```bash
# 全CPUコアを使用
python ln_closeness_analysis.py ... --n-jobs -1

# CPUコア数を確認
import multiprocessing
print(multiprocessing.cpu_count())  # 例: 8
```

### 問題: メモリ不足

**原因**: 大規模グラフ + 高い並列度

**解決方法**: 並列度を下げる
```bash
# 2コアのみ使用
python ln_closeness_analysis.py ... --n-jobs 2
```

## 🔬 さらなる最適化の可能性

### 中規模グラフ（現在の実装で最適）
- ✅ マルチスレッド並列処理
- ✅ 直接BFS計算
- ✅ プログレス表示
- ✅ 正しい外向き近接中心性の計算

### 超大規模グラフ向け（将来の拡張）
- 🔄 **Harmonic Centrality** - 非連結グラフでより安定
- 🔄 **ピボットサンプリング** - 全ノードではなくサンプルで近似
- 🔄 **Top-k最適化** - 上位ノードのみに特化した計算

これらは必要に応じて実装可能です。

##  参考文献

1. Rohrer, E., Malliaris, J., & Tschorsch, F. (2019). Discharged Payment Channels: Quantifying the Lightning Network's Resilience to Topology-Based Attacks. arXiv:1904.10253.

2. Freeman, L. C. (1979). Centrality in networks: I. Conceptual clarification. Social Networks, 1(3), 215-239.

3. Brandes, U., & Pich, C. (2007). Centrality Estimation in Large Networks. International Journal of Bifurcation and Chaos, 17(07), 2303-2318.

4. Cohen, E., et al. (2014). Computing Classic Closeness Centrality, at Scale. COSN '14.

5. NetworkX Documentation: [Closeness Centrality](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html)
   

---

**最終更新**: 2025年10月13日  
**バージョン**: 2.1（外向き近接中心性計算の修正版）
