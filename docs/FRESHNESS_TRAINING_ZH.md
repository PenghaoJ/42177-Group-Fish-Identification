# 鱼眼新鲜度训练说明

你负责的任务是 **FFE 鱼眼图片的新鲜度三分类**。输入为鱼眼近照，输出为
`highly_fresh`、`fresh`、`not_fresh`。鱼种只作为数据来源的元信息，不是这个模型的预测目标，
也不需要与 AutoFish 的五个鱼种配对。这里使用现成鱼眼裁剪图，不训练眼睛检测器。

## 数据与标签

使用已经整理好的 `42177_selected_data` 文件夹，读取 `manifests/ffe3/` 的固定清单。
不再随机拆分，不把验证或测试图重新放进训练集。

| 标签编号 | 标签 | 训练 | 验证 | 测试 |
|---|---|---:|---:|---:|
| 0 | highly_fresh | 1,235 | 266 | 263 |
| 1 | fresh | 924 | 199 | 197 |
| 2 | not_fresh | 915 | 197 | 194 |
| 合计 | | 3,074 | 662 | 654 |

来源：[FFE，Mendeley Data，第 1 版](https://data.mendeley.com/datasets/xzyx7pbr3w/1)。
数据许可为 CC BY 4.0。发布页标称 4,392 张；本次使用的已核验清单实际包含 4,390 张，报告按实际清单计数。
标签按原研究冰藏条件的三个阶段解释（第 1–2、3–4、5–6 天），不是适用于所有储存环境的天数预测。
它反映图片与数据集标签的对应关系，不能作为食品安全或可食用性结论。

引用：Prasetyo, E., Adityo, R. D., Suciati, N., & Fatichah, C. (2022).
*The Freshness of the Fish Eyes Dataset* (Version 1) [Data set]. Mendeley Data.
https://doi.org/10.17632/xzyx7pbr3w.1

清单已按已知来源和近重复组隔离。FFE 没有可靠的鱼个体编号，无法验证同一条鱼不同时间的照片
是否完全不跨集合。报告必须称为 **provisional source-grouped split**，不能声称
**specimen-independent split**。三类数据来自八个来源鱼种，来源鱼种不是模型输入。

## 本次比较方案

| 模型 | 输入与训练 | 验证集选择 |
|---|---|---|
| 颜色/纹理＋SVM | 221 维 HSV、Lab、LBP、GLCM；仅训练集拟合 StandardScaler | 24 组 linear/RBF、C、gamma、class_weight 参数，按 Macro-F1 选择 |
| ResNet-18 | ImageNet 预训练；224×224 等比例补边；更换为三分类头 | 保存验证集 Macro-F1 最好的轮次 |
| YOLOv8n-cls | ImageNet 预训练的分类模型；同样输入尺寸、三分类头 | 同上 |

两种 CNN 使用同一训练程序：seed=42177、batch=32、最多 24 轮、前 2 轮只训练分类头，
之后微调整个网络；AdamW，主干学习率 0.0001、分类头 0.001、weight decay=0.0001，
dropout=0.2、label smoothing=0.05。验证成绩连续 6 轮无改善后早停（至少运行 7 轮）。
学习率按验证 Macro-F1 平台期下降。平分时先比较准确率，再比较较小验证损失。

仅训练图做水平/垂直翻转和 ±12° 小角度旋转；不随机裁掉眼睛、不改变色调或饱和度。
ResNet 使用 ImageNet 均值和标准差，YOLO 使用 RGB [0,1]。
CUDA 训练使用 bfloat16，验证与推理使用 float32。硬件和版本不同可能产生微小差异。
这是每种 CNN **一个随机种子**的初始比较，不能写成多次重复实验均值。

最终模型先根据验证集选定并写入 `selection_before_test.json`，再统一运行测试。
报告记录 Accuracy、Macro-F1、各类 Precision/Recall/F1、混淆矩阵和错误预测。
SVM 的 score 为 null；CNN softmax 分数未经概率校准。这里不使用测试时增强。

## 在当前电脑运行

在仓库根目录打开 PowerShell。训练专用环境为 `.training/venv`；网站使用 `.venv`。

```powershell
$trainPython = '.training/venv/Scripts/python.exe'
$dataRoot = '../outputs/42177_selected_data'

& $trainPython -m fish_freshness.data --data-root $dataRoot --output .training/ffe_audit.json
& $trainPython -m fish_freshness.train_svm --data-root $dataRoot --audit .training/ffe_audit.json --output .training/runs/svm

& $trainPython -m fish_freshness.train_cnn --family resnet18 --pretrained .training/pretrained/resnet18-f37072fd.pth --data-root $dataRoot --audit .training/ffe_audit.json --output .training/runs/resnet18 --device cuda
& $trainPython -m fish_freshness.train_cnn --family yolov8n_cls --pretrained .training/pretrained/yolov8n-cls.pt --data-root $dataRoot --audit .training/ffe_audit.json --output .training/runs/yolov8n_cls --device cuda

& $trainPython -m fish_freshness.evaluate --data-root $dataRoot --audit .training/ffe_audit.json --runs .training/runs --output .training/evaluation --device cuda
```

每一步成功后再执行下一步。已经完成的结果不需要重新训练；程序拒绝覆盖非空训练目录。
若做新的验证集实验，使用新的输出目录并提前记录变更；不要根据已看到的测试集成绩反复选参数。
中断训练保留已保存的最佳权重和 history.csv，但当前版本不支持恢复优化器状态。

## 组员在新电脑配置

先安装 Python 3.12，取得这个代码版本，再将数据包解压并设置 `$dataRoot`。
RTX 50 系列可以使用下列 PyTorch 官方 CUDA 13.0 包。其他硬件按
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择版本。

```powershell
py -3.12 -m venv .training/venv
$trainPython = '.training/venv/Scripts/python.exe'
& $trainPython -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
& $trainPython -m pip install -r requirements.txt
& $trainPython training/download_freshness_pretrained.py
& $trainPython -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

没有 NVIDIA GPU 的电脑可安装官方 CPU 版并在训练命令中使用 `--device cpu`，速度会更慢。
预训练下载脚本验证固定 SHA-256。模型文件只能从可信的组员交付包加载。

## 你要交给小组的内容

1. 三个模型的权重：SVM `model.joblib`，两个 CNN 各自的 `best.pt`。
2. 各模型的 `config.json`、`summary.json`、验证预测、SVM 候选参数表和 CNN 训练曲线。
3. `evaluation/`：测试比较表、各类指标、混淆矩阵、逐图预测和按来源鱼种的诊断表。
4. `ffe_audit.json` 和三个固定清单，用来解释数据划分与无法验证个体独立性的限制。
5. 训练与推理代码，以及使用的 Python/依赖版本。

可以在分工表写：**FFE dataset verification, training and evaluation of SVM, ResNet-18 and
YOLOv8n-cls for three-class fish-eye freshness classification; model packaging and inference interface.**

推理入口已经提供 PIL → label/score/model_version 接口。单图测试示例：

```powershell
& $trainPython -m fish_freshness.predict --model .training/runs/resnet18/best.pt --image path/to/eye.png --device cpu
```

训练步骤本身不会更改网站配置。选定的 ResNet-18 现已完成单独的网站接入，安装与启动方法见
[鱼眼模型接入说明](FRESHNESS_MODEL.md)。其他两个模型保留在训练交付包用于比较。
