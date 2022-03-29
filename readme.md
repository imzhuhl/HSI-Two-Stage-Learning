Y. Qian, H. Zhu, L. Chen and J. Zhou, "Hyperspectral Image Restoration With Self-Supervised Learning: A Two-Stage Training Approach," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022.

https://ieeexplore.ieee.org/document/9658507


Our implementation based on PyTorch.


## 数据准备
CAVE 数据集，划分出训练集和测试集。归一化后加噪声，相应代码在 `preprocess_cave.py` 中。


## 预训练
`train_rn.py` 包含预训练代码。

* 开始训练：`python train_rn.py`
* `CUDA_ID` 控制使用的 GPU
* 相关训练配置在 `zcfg.py` 中


## 自监督微调
`finetune_rn.py` 包含自监督微调代码。

