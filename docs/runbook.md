# 人脸识别 baseline 运行手册

## 服务器信息

- SSH: `49.232.174.36:2006`
- User: `cx`
- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 3060 12GB

## 环境

建议使用独立 Conda 环境：

```bash
conda create -n face310 python=3.10
conda activate face310
python -m pip install -U pip setuptools wheel
python -m pip install insightface onnxruntime-gpu opencv-python-headless numpy pillow scikit-learn
```

## 验证依赖

```bash
python - <<'PY'
import insightface, onnxruntime, cv2, numpy as np
print(insightface.__version__)
print(onnxruntime.__version__)
print(cv2.__version__)
print(np.__version__)
print(onnxruntime.get_available_providers())
PY
```

## 跑 demo

1. 准备一张人脸图片，例如 `lena.jpg` 或自己的测试照片。
2. 运行：

```bash
python scripts/face_demo.py --image /path/to/img1.jpg
python scripts/face_demo.py --image /path/to/img1.jpg --image2 /path/to/img2.jpg
```

## 说明

- 若 `CUDAExecutionProvider` 不可用，脚本会自动退回 CPU。
- 先跑通即可，不必一开始追求最优速度。
