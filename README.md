# Lightning Network Closeness Centrality Analyzer

Lightning Networkのノードの**近接中心性(Closeness Centrality)**を分析し、最適なチャネル開設戦略を提案するツールです。

## 📊 機能

1. **現在の近接中心性を測定** - 指定したノードの支払い送信効率を評価
2. **最適な単一チャネルを提案** - 近接中心性を最も改善するノードをトップ20でランキング
3. **最適な組み合わせを提案** - 3つのチャネルの最適な組み合わせトップ5を提案

## 🔬 理論的背景

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
  - NetworkXでは`G.reverse()`を使用して正しく計算

### 学術的根拠

- **Rohrer et al. (2019)**: ["Discharged Payment Channels"](https://arxiv.org/abs/1904.10253)
  - Lightning Networkのトポロジー分析
  - 中心性指標とネットワーク効率の関係を実証

- **Freeman (1979)**: "Centrality in networks: I. Conceptual clarification"
  - 近接中心性の数学的定義

## 🚀 使い方

### 必要な環境

```bash
pip install psycopg2-binary pandas networkx
```

### PowerShell での実行方法

**重要**: PowerShellでは、すべての引数を**ダブルクォーテーション**で囲む必要があります。

```powershell
python ln_closeness_analysis.py --pg-host "localhost" --pg-port 5432 --pg-db "lightning_network" --pg-user "readonly" --pg-pass "your_password" --target-node "02abc123...def456" --topk 20 --combo-k 3 --combo-top 5
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
    --target-node 02abc123...def456
```

### オプション

```bash
--topk      # 単一チャネル推奨のトップN (デフォルト: 20)
--combo-k   # 組み合わせのサイズ (デフォルト: 3)
--combo-top # 表示する組み合わせの数 (デフォルト: 5)
```

### 実際の実行例

```powershell
# PowerShell の例
python ln_closeness_analysis.py --pg-host "lightning-graph-db.c7kw0quwamx3.ap-northeast-1.rds.amazonaws.com" --pg-port 19688 --pg-db "graph" --pg-user "mikura" --pg-pass "#w!zLVhwNzzz4!r" --target-node "03f5dc9f57c6c047938494ced134a485b1be5a134a6361bc5e33c2221bd9313d14" --topk 30 --combo-k 4 --combo-top 5
```

## 📤 出力

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
Node:     bakamoto
Node ID:  03f5dc9f57c6c047938494ced134a485b1be5a134a6361bc5e33c2221bd9313d14
Closeness: 0.353823

======================================================================
  Top 20 Single-Channel Openings
======================================================================

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

#1
  Nodes:  ACINQ, LNBig.com, Bitfinex
  New CC: 0.372456  |  Δ: 0.018633  |  +5.27%

#2
  Nodes:  ACINQ, LNBig.com, Kraken
  New CC: 0.371823  |  Δ: 0.018000  |  +5.09%
...

✅ Saved to: top_combo_recommendations.csv
```

### CSVファイル

- `top_single_recommendations.csv` - 単一チャネル推奨
- `top_combo_recommendations.csv` - 組み合わせ推奨

## 🔍 技術的詳細

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

### アルゴリズム

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

3. **近接中心性計算**
   ```python
   # Outgoing Closeness Centrality
   closeness = nx.closeness_centrality(G.reverse(), u=node)
   ```

4. **シミュレーション**
   - 各候補ノードとのチャネル開設をシミュレート
   - 改善度でランキング

## 📊 実用例

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

## ⚠️ 注意事項

1. **トポロジー分析のみ** - このツールは容量や流動性を考慮しません
2. **スナップショット時点** - Lightning Networkは常に変化するため、分析結果は実行時点のもの
3. **総合的判断が必要** - 手数料、評判、安定性なども重要な考慮事項

## 🐛 トラブルシューティング

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

## 📚 参考文献

1. Rohrer, E., Malliaris, J., & Tschorsch, F. (2019). Discharged Payment Channels: Quantifying the Lightning Network's Resilience to Topology-Based Attacks. arXiv:1904.10253.

2. Freeman, L. C. (1979). Centrality in networks: I. Conceptual clarification. Social Networks, 1(3), 215-239.

3. NetworkX Documentation: [Closeness Centrality](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html)

## 📄 ライセンス

MIT License

## 👨‍💻 作者

taipp-rd

---

**最終更新**: 2025年10月10日
