# 猫脸个体识别实验

本目录保存猫脸个体识别的训练、评测和分析代码。大体积数据、embedding、模型权重与运行报告位于服务器，不复制进本地 Git 工程。

## 当前结果

- 数据清洗后保留 164,100 个猫身份、601,641 张图片。
- 冻结 DINOv2 ViT-B/14 reg4 基线：测试集 Top-1 `39.14%`。
- 监督式对比学习适配器：验证集 Top-1 `59.87%`，测试集 Top-1 `59.79%`。
- 固定验证身份与查询图片时，使用 1、2、3 张登记照的 Top-1 分别为 `61.36%`、`71.03%`、`76.49%`。

测试集只用于已经完成的最终评测；后续实验和参数选择继续使用验证集。

## 目录结构

```text
cat-recognition-system/
├── README.md
├── scripts/
│   ├── train_metric_adapter.py       # 训练轻量特征适配器
│   ├── evaluate_metric_adapter.py    # 独立验证/最终测试入口
│   ├── analyze_gallery_sizes.py      # 比较多图登记效果
│   ├── analyze_validation_errors.py  # 统计验证集错误
│   └── make_error_contact_sheet.py   # 生成错误图片联系表
└── tests/
    └── test_metric_adapter.py
```

## 服务器位置

```text
项目：/home/firecom/yjr/cat-recognition-system
清洗数据：/home/firecom/yjr/cat-recognition-system/data_clean_stream_v2
特征与报告：/home/firecom/yjr/cat-recognition-system/run_artifacts
最佳权重：/home/firecom/yjr/cat-recognition-system/run_artifacts/adapter_supcon_v1/best.pt
```

服务器使用 Conda 环境 `transformer`。核心依赖为 PyTorch、NumPy、FAISS 和 Pillow。

## 主要产物

```text
run_artifacts/adapter_supcon_v1/
├── best.pt
├── history.jsonl
├── test_metrics.txt
├── validation_gallery_sizes.json
├── validation_error_summary.json
├── validation_hard_errors.csv
└── validation_hard_errors_contact_sheet.png
```

## 下一步

1. 目视检查高置信错误联系表，区分标签、姿态、主体大小和背景问题。
2. 只在验证集上比较猫脸裁剪与原图特征。
3. 为实际登记流程采用至少 2 张、推荐 3 张不同姿态或场景照片。

