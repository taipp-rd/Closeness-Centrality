# Lightning Network Closeness Centrality Analyzer

Lightning Networkのノードの**近接中心性(Closeness Centrality)**と**調和中心性(Harmonic Centrality)**を分析し、最適なチャネル開設戦略を提案する高度な分析ツールです。

##  主要機能

### 1. 複数の中心性指標
- **近接中心性 (Closeness Centrality)**: ルーティング効率を測定
- **調和中心性 (Harmonic Centrality)**: 非連結グラフでより安定した指標
- **容量重み付き中心性**: チャネル容量を考慮した構造分析

### 2. 最適化アルゴリズム
- **貪欲法 (Greedy Algorithm)**: 高速で良好な近似解を提供
- **網羅的探索 (Exhaustive Search)**: 最適解を保証（小規模な場合）
- **比較モード**: 両手法の結果を比較

### 3. パフォーマンス最適化
- **マルチコア並列処理**: ThreadPoolExecutorによる高速化
- **進捗表示**: リアルタイムで処理状況を確認
- **メモリ効率**: 大規模グラフにも対応

##  理論的背景

### 近接中心性 (Freeman 1979)
```
CC(v) = (n-1) / Σ d(v,u)
```
- ノードから他の全ノードへの最短経路距離の逆数
- Lightning Networkでは支払い送信効率を表す

### 調和中心性 (Marchiori & Latora 2000)
```
HC(v) = Σ(u≠v) [1/d(v,u)] / (n-1)
```
- 非連結グラフでも安定（1/∞ = 0）
- 動的トポロジーに適している
- 連結成分では近接中心性と高相関（ρ > 0.95）

### 容量重み付き中心性（NEW - v3.0）
```
weight = 1 / (1 + log(1 + capacity))
```
- Opsahl et al. (2010)の重み付きネットワーク理論に基づく
- 大容量チャネル = 短い「効果的距離」
- **注意**: 実際の残高分布ではなく構造的重要性を示す

##  使い方

### 必要な環境

```bash
pip install psycopg2-binary pandas networkx numpy
```

### 基本的な実行

```bash
# 貪欲法によるトポロジー分析（推奨）
python ln_closeness_analysis.py \
    --pg-host localhost --pg-port 5432 \
    --pg-db lightning_network --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --method greedy

# 容量重み付き分析（実験的）
python ln_closeness_analysis.py \
    --pg-host localhost --pg-port 5432 \
    --pg-db lightning_network --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --use-capacity \
    --method greedy

# 両手法の比較
python ln_closeness_analysis.py \
    ... \
    --method both
```

### オプション詳細

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--topk` | 20 | 単一チャネル推奨のトップN |
| `--combo-k` | 3 | 組み合わせチャネル数 |
| `--combo-top` | 5 | 表示する組み合わせ数 |
| `--n-jobs` | -1 | 並列ワーカー数（-1で全CPU使用） |
| `--method` | greedy | 最適化手法（greedy/exhaustive/both） |
| `--use-capacity` | False | 容量重み付き中心性を使用 |
| `--sort-by` | closeness | ソート基準（closeness/harmonic） |

##  出力例

### コンソール出力

```
======================================================================
  Network Connectivity Analysis
======================================================================
Strongly connected: False
Strong components: 1523
Largest component coverage: 89.45%
Weak components: 12

======================================================================
  Current Centrality Values
======================================================================
Node: ACINQ
Closeness Centrality:  0.353823
Harmonic Centrality:   0.412567

======================================================================
  GREEDY RESULT (k=3)
======================================================================
[GREEDY] Iteration 1/3: Selected: LNBig.com
[GREEDY] Marginal: CC +2.10%, HC +2.34%
[GREEDY] Total: CC 0.361245 (+2.10%), HC 0.421983 (+2.34%)

[GREEDY] Iteration 2/3: Selected: Bitfinex
[GREEDY] Marginal: CC +1.68%, HC +1.85%
[GREEDY] Total: CC 0.367183 (+3.78%), HC 0.429617 (+4.19%)

[GREEDY] Iteration 3/3: Selected: Kraken
[GREEDY] Marginal: CC +1.23%, HC +1.41%
[GREEDY] Total: CC 0.371698 (+5.01%), HC 0.435433 (+5.60%)

Final centrality:
  CC: 0.371698 (+5.01%)
  HC: 0.435433 (+5.60%)
Computation time: 12.34s

======================================================================
  COMPARISON
======================================================================
Greedy time:     12.34s
Exhaustive time: 156.78s
Speedup:         12.7x

Result: ✅ IDENTICAL (Greedy found optimal solution)
```

### CSVファイル出力

- `centrality_recommendations.csv` - 単一チャネル推奨結果
- `marginal_gains.csv` - 限界効用データ（統計分析用）

##  アルゴリズムの詳細

### 1. 貪欲アルゴリズム

```python
for i = 1 to k:
    best = argmax_{v ∉ S} f(S ∪ {v}) - f(S)
    S = S ∪ {best}
```

- **複雑度**: O(k × n × (|V| + |E|))
- **近似保証**: 劣モジュラ関数の場合 (1-1/e) ≈ 63%
- **注意**: 近接中心性改善は必ずしも劣モジュラではない

### 2. 網羅的探索

- **複雑度**: O(C(n,k) × (|V| + |E|))
- **保証**: 最適解
- **実用性**: k ≤ 3, n ≤ 20 程度まで

### 3. 容量重み付き分析の理論

**重み関数の設計**:
```python
weight = 1.0 / (1.0 + np.log1p(capacity))
```

- **対数スケーリング**: 容量の影響を適切に減衰
- **`log1p`の使用**: `log(1 + x)`で数値的安定性を確保
- **逆数変換**: 大容量 → 小重み → 短い効果的距離

**理論的根拠**:
- Opsahl et al. (2010): 重み付きネットワークにおけるノード中心性
- 容量は支払い可能性の上限を示す構造的指標
- 実際のルーティングでは残高分布（非公開）が重要

## 注意事項

### 1. 容量重み付き分析について
- **容量 ≠ 実際の残高**: 容量は理論的上限であり、実際のルーティング能力とは異なります
- **構造的重要性**: ネットワークトポロジーにおける位置の重要性を示します

### 2. グラフの方向性（重要な修正）
- **v2.1での修正**: 外向き近接中心性の計算が修正されました
- **正しい実装**: `G`（元のグラフ）を使用して外向き距離を測定
- **以前の誤り**: `G.reverse()`を使用していた（内向き距離を測定）

### 3. 分析の限界
- **スナップショット分析**: Lightning Networkは常に変化しています
- **トポロジーのみ**: 手数料、評判、安定性は考慮されていません
- **総合的判断**: 実際のチャネル開設には多角的な検討が必要です

##  トラブルシューティング

### PowerShellでの実行

PowerShellではすべての引数をダブルクォーテーションで囲む必要があります：

```powershell
# 正しい例
python ln_closeness_analysis.py --target-node "02abc..."

# 間違った例（エラーになる）
python ln_closeness_analysis.py --target-node <02abc...>
```

### メモリ不足

大規模グラフで問題が発生する場合：

```bash
# ワーカー数を制限
python ln_closeness_analysis.py ... --n-jobs 2

# 評価対象を減らす
python ln_closeness_analysis.py ... --topk 10 --combo-k 2
```

### 処理速度の最適化

```bash
# CPUコア数の確認
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# 最適なワーカー数の設定（通常はCPUコア数-1）
python ln_closeness_analysis.py ... --n-jobs 7  # 8コアCPUの場合
```

##  データベーススキーマ

必要なテーブル構造：

### channel_update
- `chan_id` (bigint): チャネルID
- `advertising_nodeid` (text): 広告ノードID
- `connecting_nodeid` (text): 接続先ノードID
- `capacity_sat` (bigint): チャネル容量（satoshi）
- `rp_disabled` (boolean): 無効化フラグ
- `timestamp` (integer): Unix timestamp
- `rp_last_update` (integer): 最終更新時刻

### closed_channel
- `chan_id` (bigint): 閉じたチャネルID

### node_announcement
- `node_id` (text): ノードID
- `alias` (text): ノードのエイリアス
- `timestamp` (integer): Unix timestamp


##  参考文献

1. **Freeman, L. C. (1979)**. Centrality in networks: I. Conceptual clarification. *Social Networks*, 1(3), 215-239.

2. **Marchiori, M., & Latora, V. (2000)**. Harmony in the small world. *Physica A*, 285(3-4), 539-546.

3. **Opsahl, T., Agneessens, F., & Skvoretz, J. (2010)**. Node centrality in weighted networks: Generalizing degree and shortest paths. *Social Networks*, 32(3), 245-251.

4. **Boldi, P., & Vigna, S. (2014)**. Axioms for centrality. *Internet Mathematics*, 10(3-4), 222-262.

5. **Kempe, D., Kleinberg, J., & Tardos, É. (2003)**. Maximizing the spread of influence through a social network. *Proceedings of KDD*, 137-146.

6. **Rohrer, E., Malliaris, J., & Tschorsch, F. (2019)**. Discharged payment channels: Quantifying the Lightning Network's resilience to topology-based attacks. *arXiv:1904.10253*.

7. **Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978)**. An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14(1), 265-294.

## 更新履歴

- **v3.0** (2025-10-27): 容量重み付き中心性オプションを追加
- **v2.5** (2025-10-20): 貪欲法と網羅的探索の実装、調和中心性の追加
- **v2.1** (2025-10-13): 外向き近接中心性計算の修正
- **v2.0** (2025-10-10): マルチコア並列処理の実装
- **v1.0** (2025-10-01): 初版リリース

---

**作成者**: taipp-rd  
**ライセンス**: MIT  
**最終更新**: 2025年10月27日
