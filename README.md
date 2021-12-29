# CNNでの分類実験
- ResNet-18で転移学習
- データセット：Potato leaf（Healthy, Early blight, Late blightそれぞれ300枚->合計900枚）
  - Training complete in 1m 16s
  - Best val Acc: 0.976667
　  - Test accuracy(未知データ) :0.83

かなり高い性能が出ている

# その他の分類器
- 1位：SGDClassifier 
   - Train : 1.0, Validation: 0.91, Test :0.586
- 2位：XGBoost　
  - Train : 0.998, Validation: 0.86, Test, 0.34

上記分類機でも性能は出るが未知データセットへの汎化性能は低い

# Todo
- ハイパーパラメータチューニング
- CNN推論速度の検証
- TensorRTの利用
- ROC曲線
- Object detectionとの連携
- Mobileへの実装 (Tensorflow lite, Pytorch live)
-  GANによる画像生成