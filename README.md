# Lightning Network Closeness Centrality Analyzer

このリポジトリには、Lightning Networkのノードの**近接中心性(Closeness Centrality)**を分析し、最適なチャネル開設戦略を提案するツールが含まれています。

## 📊 概要

このツールは以下を実現します:

1. **現在の近接中心性を測定**: 指定したノードが他のノードにどれだけ効率的に支払いを送信できるかを評価
2. **最適な単一チャネルを提案**: どのノードとチャネルを開くと最も改善するか、トップ20をランキング
3. **最適な組み合わせを提案**: トップ20の中から、3つのチャネルの最適な組み合わせトップ5を提案

## 🔬 理論的背景

### 近接中心性(Closeness Centrality)とは

近接中心性は、ノードから他の全てのノードへの最短経路距離の逆数として定義されます。Lightning Networkでは、これは**支払いの効率性**を表します:

- **高い近接中心性** = 他のノードへ少ないホップ数で支払いを送信可能
- **低い近接中心性** = 支払いに多くの中継が必要

### 有向グラフと方向性

Lightning Networkのチャネルは双方向ですが、各方向に異なる容量とポリシーを持ちます。このツールは:

- **有向グラフ**として Lightning Network を扱います
- **Outgoing Closeness Centrality** (外向き近接中心性)を測定
  - これは「自ノードから他のノードへ支払いを送信する能力」を表します
  - NetworkXでは`G.reverse()`を使用して正しく計算します

### 学術的根拠

このツールの理論は以下の研究に基づいています:

- **Rohrer et al. (2019)**: ["Discharged Payment Channels: Quantifying the Lightning Network's Resilience to Topology-Based Attacks"](https://arxiv.org/abs/1904.10253)
  - Lightning Networkはスケールフリー・スモールワールドネットワークの性質を持つ
  - 中心性に基づく攻撃に脆弱であることを示した
  - 近接中心性がネットワーク効率の重要な指標であることを実証

- **Freeman (1979)**: "Centrality in networks: I. Conceptual clarification"
  - 近接中心性の数学的定義と解釈

## 📁 ファイル説明

### `ln_closeness_analysis_improved.py` (推奨)

**改善版**のアナライザー。以下の点で元のコードより優れています:

#### 主な改善点

1. **✅ 正しい方向性**
   - Outgoing Closeness Centrality を計算 (ルーティング能力の正確な評価)
   - NetworkXの`G.reverse()`を使用して正しく実装

2. **✅ 双方向チャネルの正しい扱い**
   - Lightning Networkのチャネルは双方向であることを考慮
   - SQLクエリで両方向のチャネル更新を取得
   - グラフ構築時に両方向のエッジを適切に追加

3. **✅ データ品質の向上**
   - NULL値の適切な処理
   - ゼロ容量チャネルのフィルタリング
   - 無効化されたチャネルの除外

4. **✅ エラーハンドリング**
   - ターゲットノードが見つからない場合の明確なエラーメッセージ
   - データ検証の強化

5. **✅ 出力の改善**
   - 見やすいテーブル形式の出力
   - 詳細な進捗情報
   - CSVファイルへの自動保存

### `ln_closeness_analysis.py` (元のコード)

元の実装。以下の問題点があります:

❌ **問題点:**
- Incoming Closeness を計算 (ルーティング分析には不適切)
- 双方向チャネルの片方向しか考慮していない
- NULL値の処理が不完全

## 🚀 使い方

### 必要な環境

```bash
pip install psycopg2-binary pandas networkx
```

### 基本的な使い方

```bash
python ln_closeness_analysis_improved.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'your_password' \
    --target-node 02abc123...def456
```

### オプション

- `--topk`: 単一チャネル推奨のトップN (デフォルト: 20)
- `--combo-k`: 組み合わせのサイズ (デフォルト: 3)
- `--combo-top`: 表示する組み合わせの数 (デフォルト: 5)

### 例

```bash
# トップ30の単一チャネルと、5つの組み合わせ（4チャネルずつ）を分析
python ln_closeness_analysis_improved.py \
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
  (Opening one channel to these nodes improves closeness most)
======================================================================

Rank  Alias                         Node ID             New CC      Δ Absolute  Δ %       
------------------------------------------------------------------------------------------
1     ACINQ                         03864e...f8ab       0.465123    0.006889    +1.50%
2     LNBig.com                     02fd3a...9bc2       0.464821    0.006587    +1.44%
...
```

### CSVファイル

- `top_single_recommendations.csv`: 単一チャネル推奨のランキング
- `top_combo_recommendations.csv`: 組み合わせ推奨のランキング

## 🔍 技術的詳細

### データベーススキーマ

このツールは以下のテーブルを使用します:

1. **channel_update**: チャネルの更新情報
   - `chan_id`: チャネルID
   - `advertising_nodeid`: 広告ノードID
   - `connecting_nodeid`: 接続先ノードID
   - `capacity_sat`: 容量(satoshi)
   - `rp_disabled`: 無効フラグ
   - `rp_last_update`: 最終更新時刻

2. **closed_channel**: 閉じられたチャネル
   - `chan_id`: チャネルID

3. **node_announcement**: ノードの情報
   - `node_id`: ノードID
   - `alias`: ノードのエイリアス

### アルゴリズム

1. **グラフ構築**
   ```python
   # 最新のチャネル情報を取得
   - DISTINCT ON (chan_id, advertising_nodeid) で各方向の最新レコード
   - 閉じられたチャネルを除外
   - 容量ゼロのチャネルを除外
   ```

2. **近接中心性計算**
   ```python
   # Outgoing Closeness Centrality
   CC(v) = (n-1) / Σ d(v,u)
   
   where:
   - n = ネットワーク内のノード数
   - d(v,u) = ノードvからuへの最短経路長
   ```

3. **シミュレーション**
   - 各候補ノードとのチャネル開設をシミュレート
   - 双方向エッジを追加 (Lightning Networkチャネルの性質)
   - 新しい近接中心性を計算
   - 改善度でランキング

## 📊 実用例

### ケーススタディ: ルーティングノードの最適化

あるルーティングノードが近接中心性を改善したい場合:

1. **現状分析**: 現在の近接中心性 = 0.425
2. **単一チャネル推奨**: ACINQ とのチャネルで +1.8% 改善
3. **組み合わせ推奨**: ACINQ + LNBig + Bitfinex で +4.2% 改善

**解釈:**
- 改善度が高いノードは、ネットワークの中心に位置する
- 複数チャネルの組み合わせは相乗効果を生む
- ただし、容量やコストも考慮する必要がある

## ⚠️ 注意事項

1. **容量は考慮されない**: このツールはトポロジー分析のみ。実際のチャネル容量や流動性は別途考慮が必要
2. **動的な変化**: Lightning Networkは常に変化するため、分析結果はスナップショット時点のもの
3. **他の要因**: 手数料、評判、安定性なども重要な考慮事項

## 🔬 理論の検証

### NetworkX Closeness Centrality の方向性

```python
# NetworkX公式ドキュメントより:
# "The closeness distance function computes the incoming distance 
#  to u for directed graphs. To use outward distance, act on G.reverse()."

# 間違った実装 (元のコード)
closeness = nx.closeness_centrality(G, u=target)  # Incoming distance

# 正しい実装 (改善版)
closeness = nx.closeness_centrality(G.reverse(), u=target)  # Outgoing distance
```

### Lightning Network の双方向性

Lightning Networkのチャネルは**双方向**ですが、データベースには各方向が別レコードとして保存されます:

```sql
-- 悪い例: 片方向のみ取得
SELECT DISTINCT ON (chan_id) ...  -- ❌ 一方向のみ

-- 良い例: 両方向を取得
SELECT DISTINCT ON (chan_id, advertising_nodeid) ...  -- ✅ 両方向
```

## 📚 参考文献

1. Rohrer, E., Malliaris, J., & Tschorsch, F. (2019). Discharged Payment Channels: Quantifying the Lightning Network's Resilience to Topology-Based Attacks. arXiv:1904.10253.

2. Freeman, L. C. (1979). Centrality in networks: I. Conceptual clarification. Social Networks, 1(3), 215-239.

3. NetworkX Closeness Centrality Documentation: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.closeness_centrality.html

## 🤝 貢献

改善提案やバグ報告は Issues または Pull Requests でお願いします。

## 📄 ライセンス

MIT License

## 👨‍💻 作者

taipp-rd

---

**最終更新**: 2025年10月10日
