# ArcFace V2 与 Triplet 残差适配器公平对比

评测日期：2026-09-16

## 结论

在同一份原始 validation、同一张量预处理、同一路径排序和同一余弦检索协议下：

- 只有 1 张登记照时，当前 Triplet 残差适配器更好，Top-1 高 1.38 个百分点，Top-3 高 2.10 个百分点。
- 有 2 张登记照时，Triplet 仍略好，Top-1 高 0.67 个百分点，Top-3 高 0.83 个百分点。
- 有 3 张登记照时，ArcFace V2 略好，Top-1 高 0.13 个百分点，Top-3 高 0.15 个百分点；差距很小。
- 校园 MVP 初期很可能只有 1～2 张可靠登记照，因此当前应继续使用 Triplet 残差适配器作为默认检索模型；ArcFace V2 保留为多登记照候选模型。

## 公平评测协议

- 数据：原始 identity-disjoint validation，共 60,292 张图片、15,917 个身份。
- 每个身份按 `relative_path` 排序，前 K 张取归一化中心作为登记特征，其余作为查询。
- K=1/2/3 时分别有 15,498/12,083/8,283 个可评测身份。
- 指标：闭集 Top-1 与 Top-3；本轮不评估未知猫拒识阈值。
- ArcFace V2 的训练身份仅来自原始 train，未使用原始 validation。
- 最终 test 保持冻结，本轮没有用 test 选型或重新评测。

## 标准结果

| 登记照数 | 查询数 | Triplet Top-1 | ArcFace V2 Top-1 | Triplet Top-3 | ArcFace V2 Top-3 | 胜出 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44,375 | 60.59% | 59.21% | 70.46% | 68.36% | Triplet |
| 2 | 28,877 | 70.69% | 70.02% | 79.48% | 78.65% | Triplet |
| 3 | 16,794 | 76.99% | 77.12% | 84.59% | 84.74% | ArcFace V2（微弱） |

不同 K 会改变可评测身份和查询集合。作为敏感性复核，在固定 8,283 个身份、固定 16,794 个查询的 matched cohort 中，Triplet 在 K=1/2 的 Top-1 分别为 62.08%/71.63%，ArcFace V2 为 60.06%/70.58%；K=3 与上表相同。

## 复现信息

- ArcFace V2 checkpoint：`arcface_v2_trial/arcface_v2_epoch3.pt`
- checkpoint SHA-256：`f24068b8dff56de7bcff34b322bb9b8c31ca75fe33596556f8e6430ceaca6108`
- 特征提取脚本 SHA-256：`eb28fa4c05e6973d4166cbfadc188b37c4e0c79a0e80cf2e1886aabfb33bbf04`
- ArcFace V2 原始结果：`run_artifacts/arcface_v2_original_validation/validation_gallery_sizes.json`
- Triplet 原始结果：`run_artifacts/adapter_triplet_w020_v1/validation_gallery_sizes.json`

完整性检查：ArcFace V2 共生成 60,292 条特征，路径无重复，特征维度为 512，向量范数范围为 0.999930～1.000067。
