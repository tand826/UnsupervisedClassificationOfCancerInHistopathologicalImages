# 機械学習を用いた臓器横断的な癌の検出の試み

# 手順

## 環境構築

- Ubuntu16.04
- python3.8.6
- CUDA
- gdc-client -> [URL](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool#:~:text=Binary%20Distributions)
  - バイナリを patch.py と同じディレクトリに置くか、パスを通してください
- wsiprocess -> [URL](https://github.com/tand826/wsiprocess#installation)
- requirements
  ```
  pip install -r requirements.txt
  ```

## 画像データの準備

- 注意：3TB 程度のデータを処理するため、保存先ディレクトリの残り容量を確認してください。
- WSI のディレクトリ
  - 1.3TB 程度の空き容量が必要です
  - HDD でも構いません
- パッチのディレクトリ
  - 1.5TB 程度の空き容量が必要です
  - 頻繁にアクセスするため SSD をおすすめします

## 設定の書き込み【重要】

- 保存先ディレクトリなどを<u><strong>コンフィグファイルに書き込む必要があります</strong></u>
- フルパスで書き込むする必要があります
  - OK: /home/user/wsi
  - NG: user/wsi
- <u>config/phase/autoencoder.yaml</u>に書き込んでください
  - WSI の保存ディレクトリを、の 3 行目の「wsi:」の後に
  - パッチの保存ディレクトリを、4 行目の「patch:」の後に
  - トレーニング結果・クラスタリング結果の保存ディレクトリを、5 行目の「result:」の後に

## WSI の準備

```bash
python patch.py phase=patch dir.suffix=patch
```

## autoencoder

```bash
python train.py phase=autoencoder dir.suffix=autoencoder
```

## k-means

- トレーニング結果のディレクトリを確認したうえで、config/phase/kmeans.yaml に書き込んでください
  - トレーニング結果のディレクトリを、3 行目の「saved_to:」の後に

```bash
python kmeans.py phase=kmeans dir.suffix=kmeans
```
