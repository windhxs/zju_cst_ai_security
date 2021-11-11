# zju_cst_ai_security

# 介绍

浙大软院2021人工智能安全作业，用CNN实现CIFAR10分类任务

# 数据集

[CIFAR10](http://www.cs.toronto.edu/~kriz/cifar.html)

下载并解压到`data/`目录下，速度可能比较慢，可使用知乎搜索相关内容解决

# 模型

简化版的[ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)


# 运行

## 训练：

单GPU:

```
python main.py --cuda --gpuid 0 --train --model_path MODEL_DIR
```

多GPU：
```
python main.py --cuda --gpuid [gpuid list] --train --model_path MODEL_DIR
```

## 测试

```
python main.py --cuda --gpuid 0 --model_path MODEL_DIR
```
测试请勿使用多GPU运行。 

# 结果

acc 84.39%
