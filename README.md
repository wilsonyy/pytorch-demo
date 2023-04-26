# pytorch-demo
最基本的训练-推理示例，展示以下GPU炼丹流程：
* 自定义数据集加载：本例以手写数据集为代替，已提前转化为图片
* 网络模型定义
* 训练
* 推理

# 环境准备（自行安装）
* conda：python3.8
* cudatoolkit：11.1.1
* cudnn: 8.1.0
* GPU：GTX 1050Ti / cuda 11.2

# 要点
本例label使用one-hot编码，可以模拟多特征分类的现象，本例手写数字简单分成10类，由于损失函数需要最终指定类别值，并不是one-hot编码格式，所以需要进行两者间的转化

# 代码构成
```
pytorch-demo
├── data   
       ├─ prediction       // 推理集图片
       ├─ test             // 测试集图片(只放100张)
       ├─ train            // 训练集图片(只放600张)
├── dataset.py             // 数据预处理转换
├── model.py               // 自定义网络模型
├── main.py                // 训练
├── prediction.py          // 预测
```
