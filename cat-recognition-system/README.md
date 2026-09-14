# 猫脸个体识别实验

本目录保存猫脸个体识别的训练、评测和分析代码。大体积原始数据、embedding 和模型权重当前仍位于服务器；已下载的小型审计报告和错误联系表保存在本目录的 `run_artifacts/`。

## 当前结果

- 数据清洗后保留 164,100 个猫身份、601,641 张图片。
- 冻结 DINOv2 ViT-B/14 reg4 基线：测试集 Top-1 `39.14%`。
- 监督式对比学习适配器：验证集 Top-1 `59.87%`，测试集 Top-1 `59.79%`。
- 固定验证身份与查询图片时，使用 1、2、3 张登记照的 Top-1 分别为 `61.36%`、`71.03%`、`76.49%`。
- validation 跨身份感知哈希审计发现 349 组近重复候选，涉及 418 个身份；前 20 个高置信错误身份对中有 10 对被直接命中。
- 普通中心裁剪使冻结 DINOv2 validation Top-1 从约 `38.73%` 降至 `37.66%`。
- OpenCV 猫脸级联检测覆盖率为 `68.10%`；在相同检测成功子集上，猫脸裁剪 Top-1 `44.80%`，低于原图 `46.99%`，当前不采用。

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

1. 对 349 组跨身份近重复候选做分组复核，生成不改原图的排除/合并候选清单。
2. 在清理后的固定 validation 协议上重新计算基线和适配器指标。
3. 若继续研究裁剪，应换用经过猫脸标注训练的检测器；当前中心裁剪和传统级联方案均不采用。
4. 为实际登记流程采用至少 2 张、推荐 3 张不同姿态或场景照片。
