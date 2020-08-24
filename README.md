# asr-transformer
Asr task use transformer encoder and decoder.

```shell script
docker run -p 8500:8500 -p 8501:8501 --rm --mount type=bind,source=$(pwd)/saved_models,target=/models/asrtransfer -e MODEL_NAME=asrtransfer -t tensorflow/serving
```

整体模型已经完成,并且
用aishell-1的训练集训练,再用它的测试集上测试,效果很好.
但是用自己录制的声音去试却完全不行...我估计应该是训练数据泛化能力不行....

约束搜索还没实现,后续更新上..