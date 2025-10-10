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

### 基本的な使い方

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

### 実行例

```bash
# トップ30の単一チャネルと、4チャネルの組み合わせ5つを分析
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db ln \
    --pg-user readonly \
    --pg-pass 'secret' \
    --target-node 02abc123...def456 \
    --topk 30 \
    --combo-k 4 \
    --combo-top 5
```

## 📤 出力

### コンソール出力

```
======================================================================
  Current Outgoing Closeness Centrality
  (Measures routing capability: how easily node can send payments)
======================================================================
Node:     MyLightningNode
Node ID:  02abc123...def456
Closeness: 0.458234

======================================================================
  Top 20 Single-Channel Openings
======================================================================

Rank  Alias                Node ID             New CC      Δ Absolute  Δ %       
----------------------------------------------------------------------------------
1     ACINQ                03864e...f8ab       0.465123    0.006889    +1.50%
2     LNBig.com            02fd3a...9bc2       0.464821    0.006587    +1.44%
3     Bitfinex             02d96e...a8f3       0.463912    0.005678    +1.24%
...

✅ Saved to: top_single_recommendations.csv

======================================================================
  Best 3-Channel Combinations (Top 5)
======================================================================

#1
  Nodes:  ACINQ, LNBig.com, Bitfinex
  New CC: 0.472456  |  Δ: 0.014222  |  +3.10%

#2
  Nodes:  ACINQ, LNBig.com, Kraken
  New CC: 0.471823  |  Δ: 0.013589  |  +2.97%
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
   - `capacity_sat`, `rp_disabled`, `rp_last_update`

2. **closed_channel** - 閉じられたチャネル
   - `chan_id`

3. **node_announcement** - ノード情報
   - `node_id`, `alias`

### アルゴリズム

1. **データ取得**
   - `DISTINCT ON (chan_id, advertising_nodeid)` で各方向の最新レコード
   - 閉じられたチャネルと容量ゼロのチャネルを除外

2. **グラフ構築**
   - 有向グラフとして構築
   - 双方向チャネルを適切に表現

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
現状: 近接中心性 = 0.425

単一チャネル推奨:
  ACINQ とのチャネル → +1.8% 改善

組み合わせ推奨:
  ACINQ + LNBig + Bitfinex → +4.2% 改善
```

**解釈:**
- 改善度が高いノードは、ネットワークの中心に位置
- 複数チャネルの組み合わせは相乗効果を生む
- 実際の運用では容量やコストも考慮が必要

## ⚠️ 注意事項

1. **トポロジー分析のみ** - このツールは容量や流動性を考慮しません
2. **スナップショット時点** - Lightning Networkは常に変化するため、分析結果は実行時点のもの
3. **総合的判断が必要** - 手数料、評判、安定性なども重要な考慮事項

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
