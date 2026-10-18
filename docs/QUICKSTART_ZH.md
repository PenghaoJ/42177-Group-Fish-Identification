# 组员本地运行指南

鱼种识别支持五类鱼，网页默认使用 ResNet-18。SVM 和 YOLOv8n-cls 也已接入。
FFE 鱼眼新鲜度也使用独立训练的 ResNet-18，支持非常新鲜、新鲜、不新鲜三类。

## 需要下载的文件

1. 项目代码：使用包含鱼种和鱼眼接入的代码版本，安装说明应包含 `scripts.install_freshness_model`。
2. 模型包：[completed_species_models.zip](https://drive.google.com/file/d/1ILFvrXrha-Rv-SMK_8XcLaDa_I5N5_W9/view)。
   该文件按原有 Google Drive 权限访问；如果无法下载，请让文件所有者向你的组员账号共享。
   无需重新训练，也不必下载完整图片数据集才能使用网站。
3. 鱼眼模型包：从训练负责同学取得 `42177_FFE_freshness_baseline_2026-10-18.zip`。

模型权重、数据集、`.venv` 和 `.runtime` 都不提交到 Git。

## Windows

建议使用 Python 3.12（64 位），将项目 ZIP 完整解压，再进入项目文件夹。

先双击 `Start-FishLens.cmd` 创建环境并安装依赖。首次需要联网，下载可能较慢。
在相应模型尚未安装时，该任务会报告模型不可用。

在项目文件夹打开 PowerShell，运行以下命令，将路径替换为模型 ZIP 的实际位置：

```powershell
.venv\Scripts\python.exe -m scripts.install_species_models --archive "C:\Users\你的用户名\Downloads\completed_species_models.zip"
.venv\Scripts\python.exe -m scripts.install_freshness_model --archive "C:\Users\你的用户名\Downloads\42177_FFE_freshness_baseline_2026-10-18.zip"
```

脚本会校验权重的 SHA-256、类别顺序和推理配置，安装三套鱼种权重与选定的鱼眼权重。
不需要手动解压、挑选或重命名模型文件。

然后双击 `Stop-FishLens.cmd`，再双击 `Start-FishLens.cmd`。
浏览器通常打开 <http://127.0.0.1:8765>；若端口被占用，以启动器显示的地址为准。
选择 **Fish species**，上传一张完整鱼身 JPG/PNG，点击 **Analyze image**。
选择 **Eye freshness** 时，请上传清楚的鱼眼近照；当前没有自动定位或裁剪眼睛的功能。
结果应显示 **Model output**。关闭浏览器不会停止服务，停止请双击 `Stop-FishLens.cmd`。

## macOS / Linux

在项目文件夹终端运行（需要已安装 Python 3.12）：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.install_species_models --archive "$HOME/Downloads/completed_species_models.zip"
.venv/bin/python -m scripts.install_freshness_model --archive "$HOME/Downloads/42177_FFE_freshness_baseline_2026-10-18.zip"
.venv/bin/python -m backend.server
```

打开 <http://127.0.0.1:8765>，在终端按 Ctrl+C 停止。默认使用 CPU，无需 CUDA。
Windows 是本次完成联调的环境；macOS/Linux 使用相同 Python 接口，尚未在本次工作中实测。

## 常见问题

- 下载代码后仍不能识别：确认三个模型已安装在 `models/species/`，然后重启服务。
- 依赖安装失败：检查网络、Python 版本和启动器的具体错误信息。
- 获取更多诊断：查看 `.runtime/server.stderr.log`，并访问 `/api/health` 检查注册模型。
- 想使用 SVM 或 YOLO：把 `backend/config.py` 的 `species` 改为 `svm` 或 `yolov8n_cls`，再重启。
- 鱼眼任务报告模型不可用：确认 `models/freshness/resnet18/` 已安装，再重启服务。
- 鱼眼仍显示演示：确认代码版本与 `backend/config.py` 已更新，重启服务并刷新页面。

完整接入与测试说明见 [SPECIES_MODELS.md](SPECIES_MODELS.md) 和 [FRESHNESS_MODEL.md](FRESHNESS_MODEL.md)。
