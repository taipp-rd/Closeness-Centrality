# Lightning Network Closeness Centrality Analyzer

Lightning Networkのノードの**近接中心性(Closeness Centrality)**と**調和中心性(Harmonic Centrality)**を分析し、最適なチャネル開設戦略を提案する高度な分析ツールです。

##  主要機能

### 1. 複数の中心性指標
- **近接中心性 (Closeness Centrality)**: ルーティング効率を測定
- **調和中心性 (Harmonic Centrality)**: 非連結グラフでより安定した指標
- **容量重み付き中心性**: チャネル容量を考慮した構造分析（v3.1新機能）

### 2. 最適化アルゴリズム
- **貪欲法 (Greedy Algorithm)**: 高速で良好な近似解を提供
- **網羅的探索 (Exhaustive Search)**: 最適解を保証（小規模な場合）
- **比較モード**: 両手法の結果を比較

### 3. パフォーマンス最適化
- **マルチコア並列処理**: ProcessPoolExecutorによる高速化
- **進捗表示**: リアルタイムで処理状況を確認
- **メモリ効率**: 大規模グラフにも対応

##  理論的背景

### 近接中心性 (Freeman 1979)
```
CC(v) = (n-1) / Σ d(v,u)
```
- ノードから他の全ノードへの最短経路距離の逆数
- Lightning Networkでは支払い送信効率を表す
- **外向き中心性**: ノードvから他のノードへの到達しやすさ

### 調和中心性 (Marchiori & Latora 2000)
```
HC(v) = Σ(u≠v) [1/d(v,u)] / (n-1)
```
- 非連結グラフでも安定（1/∞ = 0）
- 動的トポロジーに適している
- 連結成分では近接中心性と高相関（ρ > 0.95）

### 容量重み付き中心性（v3.1実装済み）
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

### 基本的な実行例

#### 1. 標準分析（重みなし、貪欲法）

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123...
```

この実行により以下が出力されます：
- トップ20の単一チャネル推奨
- 3チャネルの最適な組み合わせ（貪欲法）
- 近接中心性と調和中心性の改善率

#### 2. 容量重み付き分析

大容量チャネルを重視したトポロジー分析：

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --use-capacity
```

容量重み付きモードでは、大容量チャネルほど「近い」ノードとして評価されます。

#### 3. 調和中心性を優先

非連結グラフに強い調和中心性でソート：

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --sort-by harmonic
```

#### 4. 網羅的探索で最適解を求める

小規模な候補（デフォルト20ノード）から最適な組み合わせを探索：

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --method exhaustive \
    --combo-k 2
```

網羅的探索は計算量が多いため、`--combo-k 2`（2チャネル）を推奨します。

#### 5. 貪欲法と網羅的探索の比較

両手法を実行して結果を比較：

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --method both \
    --combo-k 2
```

実行時間と解の質を比較できます。

#### 6. カスタマイズされた分析

多くのチャネル候補と詳細な分析：

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --topk 50 \
    --combo-k 5 \
    --combo-top 10 \
    --n-jobs 4
```

- `--topk 50`: トップ50の単一チャネルを評価
- `--combo-k 5`: 5チャネルの組み合わせを探索
- `--combo-top 10`: トップ10の組み合わせを表示
- `--n-jobs 4`: 4つの並列ワーカーを使用

#### 7. 統計分析用データのエクスポート

限界効用データをCSV出力：

```bash
python ln_closeness_analysis.py \
    --pg-host localhost \
    --pg-port 5432 \
    --pg-db lightning_network \
    --pg-user readonly \
    --pg-pass 'password' \
    --target-node 02abc123... \
    --export-marginal-gains
```

`marginal_gains.csv`が追加出力され、各チャネル追加による中心性の変化を詳細に分析できます。

### オプション詳細

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--pg-host` | 必須 | PostgreSQLホスト |
| `--pg-port` | 5432 | PostgreSQLポート |
| `--pg-db` | 必須 | PostgreSQLデータベース名 |
| `--pg-user` | 必須 | PostgreSQLユーザー名 |
| `--pg-pass` | 必須 | PostgreSQLパスワード |
| `--target-node` | 必須 | 分析対象ノードのHex ID |
| `--topk` | 20 | 単一チャネル推奨のトップN（1-100） |
| `--combo-k` | 3 | 組み合わせチャネル数（1-10） |
| `--combo-top` | 5 | 表示する組み合わせ数（1-20） |
| `--n-jobs` | 3 | 並列ワーカー数（-1で全CPU、推奨は3-4） |
| `--method` | greedy | 最適化手法（greedy/exhaustive/both） |
| `--use-capacity` | False | 容量重み付き中心性を使用 |
| `--sort-by` | closeness | ソート基準（closeness/harmonic） |
| `--export-marginal-gains` | False | 限界効用データをCSV出力 |
| `--exhaustive-candidates` | 20 | 網羅的探索で考慮する候補数（1-50） |

### 推奨パラメータ設定

#### 一般的な分析
```bash
--topk 20 --combo-k 3 --method greedy --n-jobs 3
```

#### 詳細分析
```bash
--topk 50 --combo-k 5 --method greedy --n-jobs 4
```

#### 小規模ネットワーク（最適解を求める）
```bash
--topk 15 --combo-k 2 --method exhaustive
```

#### 大規模ネットワーク（メモリ節約）
```bash
--topk 10 --combo-k 2 --n-jobs 2
```

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
Mode: Capacity-weighted centrality

======================================================================
  Current Centrality Values
======================================================================
Node: ACINQ
Closeness Centrality:  0.353823
Harmonic Centrality:   0.412567

======================================================================
  Top 20 Recommendations (sorted by closeness)
======================================================================

Rank  Alias                    ΔCC %    ΔHC %    CC          HC          
---------------------------------------------------------------------------
1     LNBig.com                +2.10    +2.34    0.361245    0.421983    
2     Bitfinex                 +1.85    +2.01    0.360352    0.420873    
3     Kraken                   +1.68    +1.87    0.359771    0.420154    
4     ACINQ-2                  +1.52    +1.71    0.359183    0.419534    
5     Boltz                    +1.38    +1.58    0.358705    0.419087    
...

Correlation (ΔCC vs ΔHC): 0.9823

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
```

##  出力ファイルの説明

分析実行後、以下のCSVファイルが生成されます。

### 1. centrality_recommendations.csv

単一チャネル追加による中心性改善の推奨リスト。

#### カラム説明

| カラム名 | データ型 | 説明 |
|---------|---------|------|
| `rank` | int | 推奨順位（1から開始） |
| `node_id` | string | 推奨ノードのHex ID（66文字） |
| `alias` | string | ノードのエイリアス名 |
| `new_closeness` | float | チャネル追加後の近接中心性 |
| `new_harmonic` | float | チャネル追加後の調和中心性 |
| `delta_cc_abs` | float | 近接中心性の絶対変化量 |
| `delta_hc_abs` | float | 調和中心性の絶対変化量 |
| `delta_cc_pct` | float | 近接中心性の変化率（%） |
| `delta_hc_pct` | float | 調和中心性の変化率（%） |
| `capacity_weighted` | bool | 容量重み付きモードで実行されたか |

#### 活用方法

```python
import pandas as pd

df = pd.read_csv('centrality_recommendations.csv')

# トップ10の推奨ノード
top10 = df.head(10)
print(top10[['rank', 'alias', 'delta_cc_pct', 'delta_hc_pct']])

# 近接中心性改善率が3%以上のノード
high_impact = df[df['delta_cc_pct'] >= 3.0]

# 調和中心性でソート
df_sorted_hc = df.sort_values('delta_hc_pct', ascending=False)
```

### 2. optimal_combination.csv

分析で見つかった最適なチャネルの組み合わせ（ベスト1組）。

#### カラム説明

| カラム名 | データ型 | 説明 |
|---------|---------|------|
| `position` | int | 組み合わせ内の位置（1から開始） |
| `node_id` | string | 推奨ノードのHex ID |
| `alias` | string | ノードのエイリアス名 |
| `final_cc` | float | 全チャネル追加後の最終近接中心性 |
| `final_hc` | float | 全チャネル追加後の最終調和中心性 |
| `improvement_cc_pct` | float | 近接中心性の総改善率（%） |
| `improvement_hc_pct` | float | 調和中心性の総改善率（%） |
| `capacity_weighted` | bool | 容量重み付きモードで実行されたか |
| `method` | string | 使用された手法（greedy/best_found） |

#### 活用方法

このファイルには、分析で見つかった最も効果的なチャネルの組み合わせが記録されています。実際のチャネル開設の優先順位として活用できます。

```python
import pandas as pd

df = pd.read_csv('optimal_combination.csv')
print("推奨チャネル開設順:")
for _, row in df.iterrows():
    print(f"{row['position']}. {row['alias']} ({row['node_id'][:16]}...)")
print(f"\n期待される改善: CC +{df.iloc[0]['improvement_cc_pct']:.2f}%")
```

### 3. optimal_combination_greedy.csv

貪欲法による最適化の詳細結果（`--method greedy`または`both`実行時）。

#### カラム説明

| カラム名 | データ型 | 説明 |
|---------|---------|------|
| `iteration` | int | 選択の反復回数（1からk） |
| `node_id` | string | 選択されたノードのHex ID |
| `alias` | string | ノードのエイリアス名 |
| `marginal_gain_cc` | float | その時点での近接中心性の限界利得 |
| `marginal_gain_hc` | float | その時点での調和中心性の限界利得 |
| `marginal_gain_cc_pct` | float | 近接中心性の限界利得率（%） |
| `marginal_gain_hc_pct` | float | 調和中心性の限界利得率（%） |
| `cumulative_cc` | float | その時点までの累積近接中心性 |
| `cumulative_hc` | float | その時点までの累積調和中心性 |
| `cumulative_cc_improvement_pct` | float | 初期値からの累積改善率（CC、%） |
| `cumulative_hc_improvement_pct` | float | 初期値からの累積改善率（HC、%） |
| `capacity_weighted` | bool | 容量重み付きモードで実行されたか |

#### 活用方法

貪欲法の各ステップでの改善効果を可視化できます。

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('optimal_combination_greedy.csv')

# 限界利得の減少を可視化
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['marginal_gain_cc_pct'], marker='o', label='CC Marginal Gain')
plt.plot(df['iteration'], df['marginal_gain_hc_pct'], marker='s', label='HC Marginal Gain')
plt.xlabel('Iteration')
plt.ylabel('Marginal Gain (%)')
plt.title('Diminishing Returns in Channel Selection')
plt.legend()
plt.grid(True)
plt.show()

# 累積改善率の推移
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['cumulative_cc_improvement_pct'], marker='o', label='CC Cumulative')
plt.plot(df['iteration'], df['cumulative_hc_improvement_pct'], marker='s', label='HC Cumulative')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Improvement (%)')
plt.title('Cumulative Centrality Improvement')
plt.legend()
plt.grid(True)
plt.show()
```

### 4. optimal_combinations_exhaustive.csv

網羅的探索による上位組み合わせ（`--method exhaustive`または`both`実行時）。

#### カラム説明

| カラム名 | データ型 | 説明 |
|---------|---------|------|
| `rank` | int | 組み合わせの順位（1から開始） |
| `position` | int | 組み合わせ内のノード位置（1からk） |
| `node_id` | string | ノードのHex ID |
| `alias` | string | ノードのエイリアス名 |
| `combination_cc` | float | その組み合わせでの近接中心性 |
| `combination_hc` | float | その組み合わせでの調和中心性 |
| `delta_cc_abs` | float | 近接中心性の絶対変化量 |
| `delta_hc_abs` | float | 調和中心性の絶対変化量 |
| `delta_cc_pct` | float | 近接中心性の変化率（%） |
| `delta_hc_pct` | float | 調和中心性の変化率（%） |
| `capacity_weighted` | bool | 容量重み付きモードで実行されたか |

#### 活用方法

複数の組み合わせを比較して、代替案を検討できます。

```python
import pandas as pd

df = pd.read_csv('optimal_combinations_exhaustive.csv')

# 各組み合わせの概要
for rank in df['rank'].unique():
    combo = df[df['rank'] == rank]
    print(f"\n組み合わせ #{rank}:")
    print(f"  ノード: {', '.join(combo['alias'].values)}")
    print(f"  CC改善: +{combo.iloc[0]['delta_cc_pct']:.2f}%")
    print(f"  HC改善: +{combo.iloc[0]['delta_hc_pct']:.2f}%")
```

### 5. marginal_gains.csv（オプション）

`--export-marginal-gains`指定時のみ出力される統計分析用データ。

#### カラム説明

| カラム名 | データ型 | 説明 |
|---------|---------|------|
| `iteration` | int | 選択の反復回数 |
| `node_id` | string | 選択されたノードのHex ID |
| `alias` | string | ノードのエイリアス名 |
| `marginal_gain_cc` | float | 近接中心性の限界利得（絶対値） |
| `marginal_gain_hc` | float | 調和中心性の限界利得（絶対値） |
| `base_cc` | float | 初期の近接中心性 |
| `base_hc` | float | 初期の調和中心性 |
| `cumulative_cc` | float | その時点での近接中心性 |
| `cumulative_hc` | float | その時点での調和中心性 |
| `capacity_weighted` | bool | 容量重み付きモードで実行されたか |

#### 活用方法

限界利得の減少（diminishing returns）を統計的に分析できます。

```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('marginal_gains.csv')

# 限界利得の減少率を検証
gains = df['marginal_gain_cc'].values
iterations = df['iteration'].values

# 線形回帰で減少傾向を分析
slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, gains)
print(f"減少率: {slope:.6f} per iteration")
print(f"R^2: {r_value**2:.4f}")
print(f"p-value: {p_value:.4f}")

# 劣モジュラ性の検証
is_diminishing = all(gains[i] >= gains[i+1] for i in range(len(gains)-1))
print(f"限界利得は減少している: {is_diminishing}")
```

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

### 4. 中心性の方向性（v3.1修正済み）

**外向き中心性の正しい実装**:
- **外向き（Outgoing）**: ノードから他のノードへの到達性
  - グラフ`G`をそのまま使用
  - Lightning Networkでの支払い送信能力を表す
  
- **内向き（Incoming）**: 他のノードからの到達性
  - グラフ`G.reverse()`を使用
  - Lightning Networkでの支払い受信能力を表す

**重み付き分析での考慮**:
- 容量重み付きの場合：`nx.single_source_dijkstra_path_length`使用
- 重みなしの場合：`nx.single_source_shortest_path_length`（BFS）使用

## 注意事項

### 1. 容量重み付き分析について
- **容量 ≠ 実際の残高**: 容量は理論的上限であり、実際のルーティング能力とは異なります
- **構造的重要性**: ネットワークトポロジーにおける位置の重要性を示します
- **柔軟な選択**: `--use-capacity`オプションで重み付きモードを切り替え可能

### 2. グラフの方向性（v3.1で修正済み）
- **v3.1での修正**: 外向き近接中心性の計算が修正されました
- **正しい実装**: 外向きでは`G`、内向きでは`G.reverse()`を使用
- **重み付き時の処理**: 重み付き関数でも同じ方向性ルールを適用

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
- `alias` (text): ノードのエイリアス名
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

- **v3.2** (2025-10-30): READMEの使い方と出力ファイル説明を大幅拡充
  - 7つの実行例を追加（基本から統計分析まで）
  - 推奨パラメータ設定ガイドを追加
  - 全CSVファイルの詳細な説明とカラム定義を追加
  - Pythonコードによるデータ活用例を追加
- **v3.1** (2025-10-27): 外向き中心性の修正、容量重み付き実装、CSV出力改善
  - 外向き中心性の計算を修正（G.reverse()の誤使用を修正）
  - 容量重み付き中心性を`--use-capacity`オプションで実装
  - `marginal_gains.csv`を`--export-marginal-gains`オプション化
  - `optimal_combination.csv`で最適な組み合わせを統一出力
- **v3.0** (2025-10-27): 容量重み付き中心性オプションを追加（READMEのみ）
- **v2.5** (2025-10-20): 貪欲法と網羅的探索の実装、調和中心性の追加
- **v2.1** (2025-10-13): 外向き近接中心性計算の修正（初回試行）
- **v2.0** (2025-10-10): マルチコア並列処理の実装
- **v1.0** (2025-10-01): 初版リリース

---

**作成者**: taipp-rd  
**ライセンス**: MIT  
**最終更新**: 2025年10月30日（v3.2）
