### 1. レコメンドシステムを実現する際に有用と思われるアルゴリズムや手段をひとつ挙げ、説明
#### Alternating-Least-Squares with Weighted-λ-Regularization (ALS-WR)
- λで正則化した交互最小二乗法
- ユーザの嗜好、アイテムの特徴の背景に潜在的な因子が関係していると考え、user-factor vector と　item-factor vector を定義する。
- すべてのユーザに対して user-factor vector と item-factor vector を掛けあわせた結果 元の rating matrix を近似的に再現するように、user-factor vector と item-factor vector を計算する。
- 上の計算は、user-factor vector と item-factor vector のドット積で得られるユーザu のアイテムi に対する予測値と rating matrix の対応する要素の二乗誤差をすべてのユーザについて求めた総和に正則化項を加えた式全体の値を最小とするように計算する　(コスト関数の最小化)。
- コスト関数を最小にする user-factor vector と item-factor vector は、互いに他方を固定することで解析的に計算することができる。つまり、item-factor vector を固定して user-factor vector を計算、また、user-factor vector を固定して item-factor vector を計算することができる。
- この片方を固定して残りを計算するという操作を交互に繰り返す (user-factor vector と item-factor vector のドット積による予測値と元の行列との誤差の総和が一定の値に収束するまで)。
- こうして求められた　user-factor vector と item-factor vector を掛けあわせることで、ユーザがまだ評価していないアイテムへの評価値の予測をすることができる。
- 並列化して計算可能である。
  - あるユーザの因子ベクトルは他のユーザの因子ベクトルとは独立して計算することができるし、あるアイテムの因子ベクトルは他のアイテムの因子ベクトルとは独立して計算することができるため、この計算を並列化して求めることが可能である。
- 訓練データとなる行列が sparse ではない時 (implicit feedback を扱う時など) に良いパフォーマンスを上げることができる。
  - 2つの因子ベクトルを求める別の方法である Stochastic Gradient Descent (確率的勾配降下法) は、計算時に訓練データセットの各データを1つずつループするため、sparse でないデータを扱うと計算量が膨大になり、現実的な手段とならない。

### 2. 1.で挙げた手法を実装
- https://github.com/Fujiki-Nakamura/MovieLensRecommenderSystem

### 3. 2.で実装した手法の改善点を挙げ、説明
- 計算量が多いため、計算リソースに限りがあり並列化が困難な環境では、現実的な手法とならない可能性がある。
  - 計算に時間がかかり、モデルの更新を頻繁におこなうことができないという制約につながる。
