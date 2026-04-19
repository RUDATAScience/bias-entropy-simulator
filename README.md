# Informational Health Validation: エントロピーの減衰とデータ前処理の病理

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本リポジトリは、アンケートや大規模データ収集における「社会的望ましさバイアス（忖度）」が情報システムに与える不可逆的な破壊プロセスを検証するPythonシミュレーションです。

現代の機械学習やデータサイエンスにおいて「標準的な前処理」とされている外れ値除去（データクレンジング）が、バイアス下においてはむしろ「情報の腐敗」を加速させるという致命的なパラドックスを証明します。

## 📌 背景と問題意識 (Background)

データ分析の現場では、極端なマイノリティの意見（評価1など）は「ノイズ」や「外れ値」として学習前に除去されるのが一般的です。しかし、回答者の意思決定に「同調圧力（忖度）」が作用している場合、この処理は致命的な結果を招きます。

本シミュレーションは以下の3つの事実を定量的に証明します。

1. **シグナル消失の相図**: 回答者の確信度と忖度の干渉により、真実のシグナルが「相転移的」に蒸発する境界を特定します。
2. **情報の熱的死**: 忖度の増大に伴い、システムの多様性（シャノン・エントロピー）が急激に減衰し、硬直化するプロセスを示します。
3. **データ前処理の病理**: 実務家が良かれと思って行う「外れ値除去（2〜3シグマ基準など）」が、残された唯一の警告シグナルを組織的に抹殺し、データ全体をさらに「偽の合意点」へと吸い寄せる構造的欠陥を暴きます。

## 🧮 数理モデル (Mathematical Model)

個人の最終的な効用（U_total）を、内発的な「本音（U_true）」と外発的な「忖度（U_target）」の線形結合として定義し、Softmax関数を通じて選択確率を算出します。

U_total = (1 - v2) * U_true + v2 * U_target

* v2: 社会的望ましさ（忖度）の重み。0で完全な本音、1で完全な同調。
* Beta: 回答者の確信度。Softmax関数の鋭敏さを制御。

## 📊 出力される分析結果 (Outputs)

スクリプトを実行すると `rigorous_validation_results` ディレクトリが作成され、以下の高解像度グラフ（PNG）と生データ（CSV）が生成されます。

* **Fig A: Phase Diagram of Signal Evaporation (Heatmap)**
  * 確信度（Beta）と忖度（v2）の二次元パラメータ空間において、警告シグナルが消滅する「崖」の境界線を可視化した相図。
* **Fig B: Information Entropy Decay**
  * 忖度の増大に伴い、データの多様性を示す「シャノン・エントロピー」が急激に減衰し、システムが均質化（Homogenization）していく推移グラフ。
* **Fig C: How 'Data Cleaning' Amplifies Bias**
  * 生データ（Raw）と、統計的外れ値除去を行ったデータ（Filtered）の比較。データクレンジングを行うことで、平均値が真実からさらに遠ざかり、バイアスが増幅されるパラドックスの証明。

## 🚀 実行方法 (Usage)

本コードは **Google Colaboratory** での実行に最適化されています。

1. `validation_sim.py`（または Jupyter Notebook形式）を Google Colab にアップロードします。
2. 全てのセルを実行します。
3. 計算完了後、グラフとCSVを格納した `rigorous_validation_archive.zip` が自動的にダウンロードされます。

### ローカル環境での実行に関する注意
ローカルのPython環境（VSCode, JupyterLab等）で実行する場合は、Colab固有のモジュールを無効化してください。
スクリプト先頭の `from google.colab import files` および、末尾の `files.download(...)` の行を削除（またはコメントアウト）することで、作業ディレクトリにZIPファイルが生成されます。

```bash
# 依存ライブラリのインストール
pip install -r requirements.txt

# スクリプトの実行
python validation_sim.py
