# 条件彩票 CPT 混合模型估计

## 目录结构

```
estimation/            库：每个模块只负责一件事
  config.py            路径、彩票集合、参数边界、种子、容差
  lotteries.py         实验中的 16 个彩票树
  lottery_paths.py     彩票树 → 收益流；给定历史后的剩余彩票
  data.py              读取与清洗 gamedata.csv；样本过滤（顶端选择者）
  cpt.py               标量 CPT（参考实现，用于复算与测试）
  spec.py              ModelSpec、每类参数布局、自由参数向量与边界
  valuation.py         参考点与向量化 CE 预测
  prepare.py           观测 → 条件状态数组
  likelihood.py        混合似然、后验、噪声更新、BIC
  em.py                单次 EM
  multistart.py        并行多起点 EM 与断点续算
  jobs.py              144 个模型配置与估计顺序
  warm_starts.py       已保存结果 → 起点
  store.py             原子写入、运行协议与指纹、保存模型
  diagnostics.py       概率权重单调性
  profile.py           分量 profile likelihood
  report/              离线 HTML 报告（汇总、复算、图、页面、存储、翻译）
scripts/               命令行入口，每个功能一个脚本
  run_grid.py          估计完整 144 模型网格
  run_exclusion.py     排除顶端选择者的稳健性检验
  run_profile.py       profile likelihood
  export_report.py     导出 HTML 报告
  organize_reports.py  整理已有报告
  consolidate.py       结果目录的 BIC 表与参数图
slides/make_figures.py 幻灯片图
legacy/                旧管线（冻结，见 legacy/README.md）
model_spec.py          旧 pickle 引用的模块名（别名）
tests/                 python -m unittest
```

所有脚本在仓库根目录运行，例如 `python scripts/run_grid.py --list`。

## 运行协议与指纹

每个结果目录的 `run_protocol.json`（profile 为 `protocol.json`）记录估计的科学输入：
数据文件、估计样本与彩票集合的哈希、参数边界、噪声边界、R^SQ 设定、估值版本、种子、容差和样本过滤。
指纹只由这些输入决定；代码版本（git commit、包哈希）记录在 `provenance` 中，不进入指纹。
指纹相同的目录可以续算；导出报告时，数据、样本、彩票、R^SQ 设定与估值版本一致即复算。代码一致性由复算 LL 检验（`tests/test_reproduction.py` 要求所有已保存结果逐位复现）。
重构前的结果目录使用旧指纹，可读取、导出和作为起点，但不能在原目录续算。

## 测试

```bash
python -m unittest
```

## 条件彩票 Grid EM 结果报告

```bash
python scripts/export_report.py
```

脚本扫描 `results_grid_conditional` 中的所有模型与起点 pickle，生成独立报告
`reports/grid_em_conditional_explorer/index.html`。默认复算所有文件的条件 CE、LL、后验和实际决策权重，保留完整原始字段、个人参数、迭代与优化器日志；不执行估计、不修改输入结果。

使用 `--language zh`（默认）或 `--language en` 选择中英文；`--lang` 是简写。
英文版默认输出到 `reports/grid_em_conditional_explorer_en/index.html`，与中文版分别保存，两个输出目录均在 `.gitignore` 中忽略。
页面、交互图表、诊断说明、参数 CSV 状态和导出图表随语言切换，原始 pickle 字段完整保留。

总览支持模型筛选、排序、最多四个文件并排比较，以及点击进入每个 pickle 的完整详情。所有记录共用 `detail.html?id=…`，原始数据按字节分块装入 `data`，不截断字段、参与者、后验或日志，不对数值取整。图表包括 BIC、起点 LL 差距、收敛、概率权重、参考点组合、金额依赖、分阶段拟合与个人参数分布。表格可下载 CSV，详情可下载完整可读 JSON；另生成用于导出的 Matplotlib SVG/PNG。

打开模型或勾选比较时，同时绘制每个类别的效用函数：以 `z=x−R` 为横轴，收益侧为 `z^α⁺`，损失侧为 `−λ(−z)^α⁻`。详情页中的金额控件同时调整效用图的正负绘图范围；每个模型结果另生成默认 `±£100` 范围的 SVG/PNG，可从详情页下载。

类别总览原来的 `A(100)` 改为 `lim A(d), d→0⁺`。对于 `A(d)=λd^(α⁻−α⁺)`，`α⁻>α⁺` 时极限为 `0`，`α⁻=α⁺` 时为 `λ`，`α⁻<α⁺` 时为 `+∞`；按估计值精确比较，不使用边界警示阈值近似判断。

生成后直接用浏览器打开 `index.html`，无需联网或启动服务器。移动或分享时保留整个报告文件夹中的 `assets`、`data`、`figures` 等目录。新增估计结果后再次运行脚本即可更新快照。

导出会在临时目录构建、校验后替换旧报告。所有语言及其他报告合计不超过 10,000 个文件，单个文件不超过 **25,000,000 字节**。数据包和下载分块预留余量，最大 20,000,000 字节。大 CSV 按完整行拆分，各分块都带表头；`catalog.json` 是小型分块目录，完整模型摘要位于它列出的 `catalog-part-*.json` 中。浏览器中的超大下载会提供 `.part` 文件和 `.parts.json` 清单，按清单顺序以二进制连接各分块即可无损还原。

整理已有报告并保留当时的全部复算结果：

```bash
python scripts/organize_reports.py
python scripts/organize_reports.py --check  # 检查整个 reports 的文件数与大小
```

整理脚本会保留未列入旧目录的详情记录，逐条校验原始字段、元数据、诊断及复算结果，再替换旧目录；其他分析报告及生成脚本也计入总限制。

可选命令：

```bash
python scripts/export_report.py --language en   # 英文版，使用独立输出目录
python scripts/export_report.py --language zh   # 中文版（默认）
python scripts/export_report.py --audit models   # 只复算模型，起点仍完整导出
python scripts/export_report.py --audit none     # 只读取保存诊断
python scripts/export_report.py --models-only    # 只导出模型 pickle
python scripts/export_report.py --language en --input-dir results_grid_conditional --output-dir reports/my_report
```

若当前代码或数据与估计协议不一致，仍导出参数和原始内容，但明确标记无法复算的诊断。达到 LL 容差、概率检查通过或较低 BIC，都不构成全局最优、机制识别或人口比例的证明；标准误等未保存信息不会被补造。
