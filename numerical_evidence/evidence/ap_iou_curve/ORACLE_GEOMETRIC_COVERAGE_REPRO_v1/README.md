# Oracle geometric coverage / 完整候选池几何覆盖分析

比较 CDW1630 Test 上正式 DEIM 与 GCCS checkpoint（seeds 0/1/2）。
每个 GT 在全部最终层 query 框中独立取最佳 IoU；不筛类别、分数或 Top-K，
不做 NMS、一对一匹配或框裁切。报告最佳 IoU 的均值、中位数和 >=0.75 的 GT 覆盖比例。
同一个预测框可以为多个 GT 提供最佳 IoU，所以该覆盖比例不是实际检测召回率。
负宽/高或零面积预测仍保留在全 Q 候选池中，IoU 定为 0，不重排边界来生成有效框；
非有限值则停止计算。统计发现的这类候选数会保留在汇总 JSON 中。此处理不改捕获和权重。

本子包包含六组正式 checkpoint 的独立几何分析结果。
六组捕获均使用 FP32、EMA、batch 8 和官方 Test 预处理。
捕获文件仅保存几何分析需要的全框，不保存庞大的 decoder hidden 特征。

## 运行

在本子包目录内、主宿主 Python 环境中执行。核对已保存结果不需要 GPU：

```bash
python -B -m unittest discover -s tests -v
python -B -u run_analysis.py --stage summarize
```

迁移目录时可通过 `--release-root` 指定 `HiLoR-DETR` 根目录。
`--stage summarize` 核对捕获绑定、捕获文件及其 checkpoint 身份、Test 标注、
全部图像 ID、每图 300 框及坐标转换，然后重新计算。它不执行当前检测器或价值头，
因此不要求运行检测器源码。`bindings.json` 和捕获收据记录这六组捕获的来源。

如需从 checkpoint 重新导出全部框（需要 GPU），必须指定包外新工作目录：

```bash
python -B -u run_analysis.py --stage all --output-dir /your/new/oracle_geometry
```

也可先运行 `--stage preflight --output-dir /your/new/oracle_geometry`，再用同一
`--output-dir` 和 `--stage capture --model DEIM --seed 0` 分组运行。
新捕获严格核对源码、数据、checkpoint、CCVR 产物与配置；使用独立绑定。
不会覆盖发布包中的捕获或结果。生成新目录的配置不等同于运行模型。

## 输出

- `bindings.json`、`protocol.json`：输入与计算口径。
- `captures/{DEIM,GCCS}/seed{0,1,2}/full_query_boxes.npz`：每图全部 Q 个最终层框。
- 同目录 `CAPTURE_SUMMARY.json`：strict EMA 加载、数量、精度、文件 SHA；GCCS 含 CCVR 坐标不变性检查。
- `results/PER_GT_PAIRED.csv`：逐 GT 最佳 IoU、argmax query 和宿主内差值。
- `results/PER_SEED_METRICS.csv`：三个种子各自的三项指标。
- `results/THREE_SEED_SUMMARY.csv` 与 `SUMMARY.json`：均值、样本标准差及同种子差值。
- `results/REPORT.md`：实测结果与解释边界。未完成时没有结果结论。
- `MANIFEST.sha256.json`：本子包文件、完整捕获及结果校验值。

已有结果时，`--stage summarize` 会用 CPU 重新计算并逐项比较全部逐 GT 行和汇总数字，
完全一致才报告 `OFFLINE REPLAY PASS`，不会覆盖已保存的实测结果。
此命令也不刷新发布包的文件清单；发布资产始终只读。

CCVR 不生成新框。本分析在同一 GCCS forward 上运行该种子正式冻结的 CCVR，
检查全体原始框逐值不变，以及最终输出框均来自未改坐标的候选池。
不另将 CCVR 的 Top-300 框用于 oracle 覆盖统计。
