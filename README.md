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
  run_summary.py       运行目录 → summary.txt（只读已保存文件）
  profile.py           分量 profile likelihood
  report/              离线 HTML 报告（汇总、复算、图、页面、存储、翻译）
scripts/               命令行入口，每个功能一个脚本
  run_grid.py          估计：任选样本（剔除规则）、spec 与 test
  catalog_results.py   生成 results/ 的索引（README.md、catalog.csv）
  summarize_run.py     为已有运行目录写 summary.txt
  run_profile.py       profile likelihood
  export_report.py     导出 HTML 报告
  organize_reports.py  整理已有报告
  consolidate.py       结果目录的 BIC 表与参数图
slides/make_figures.py 幻灯片图
results/               全部估计结果，按 样本/运行 组织（见下文与 results/README.md）
legacy/                旧管线（冻结，见 legacy/README.md）
model_spec.py          旧 pickle 引用的模块名（别名）
tests/                 python -m unittest
```

所有脚本在仓库根目录运行，例如 `python scripts/run_grid.py --list`。

## 估计

`scripts/run_grid.py` 是唯一的估计入口，三类选项相互独立：

| 选什么 | 选项 | 默认 |
|---|---|---|
| 剔除谁 | `--exclude-endpoint 0.5`（顶端作答比例 ≥ 0.5）、`--exclude-file 名单.csv`（含 `participant_label` 列，规则在模型之外决定，例如 page time） | 不剔除 |
| 哪些 spec | `--discounting exp qh`、`--curvature commonalpha sepalpha`、`--memory commem sepmem`、`--pw prelec tk` | 全部 |
| 哪些 test | `--tests test1 test2 test3`、`--clusters 1 2 3 4`（Test 1，可大于 4）、`--rp SQ PA LE FE`（Test 2） | 全部 |
| 效用函数 | `--utility power` 或 `--utility exponential`（每个输出目录只用一种） | power |
| 或直接点名 | `--only test3_exp_sepalpha_commem_tk test1_exp_sepalpha_commem_tk_C4` | |

边界与起点：`--lambda-upper`、`--gamma-lower`（TK 在 γ ≥ 0.2792 时单调）、`--warm-dirs`。
先用 `--list` 查看样本与模型清单，不会估计。

```bash
# 全样本、TK、只跑 Test 3
python scripts/run_grid.py --tests test3 --discounting exp --curvature sepalpha --memory commem --pw tk \
    --lambda-upper 20 --gamma-lower 0.28 --output-dir results/sample_stricter/tk_test3

# 同一协议下追加 Test 1 的 C=1..4（同一输出目录，模型会累积）
python scripts/run_grid.py --tests test1 --discounting exp --curvature sepalpha --memory commem --pw tk \
    --lambda-upper 20 --gamma-lower 0.28 --output-dir results/sample_stricter/tk_test3

# 按外部名单剔除
python scripts/run_grid.py --tests test3 --exclude-file my_exclusions.csv --output-dir results/sample_my_rule/test3
```

### 效用函数

| `--utility` | 收益（z=x−R≥0） | 损失（z<0） | α⁺、α⁻ 的含义与边界 |
|---|---|---|---|
| `power`（默认） | z^α⁺ | −λ(−z)^α⁻ | 幂指数，`config.BOUNDS` |
| `exponential` | (1−e^(−α⁺z))/α⁺ | −λ(1−e^(α⁻z))/α⁻ | 绝对风险厌恶系数（1/£），`config.CARA_BOUNDS` = [−0.3, 0.3] |

指数（CARA）效用满足 u(R)=0、u′(R⁺)=1、u′(R⁻)=λ，所以 λ 仍是参考点处的损失厌恶。α>0 时收益凹、损失凸，α<0 时相反，α=0 为线性。
α⁺、α⁻ 按风险偏好解读：α⁺>0 为收益域风险厌恶，α⁻>0 为损失域风险寻求。报告详情页的金额表和 `summary.txt` 都给出每类的风险溢价：在 0 与 ±£x 之间各半的赌局，期望值减确定性等价，占期望值绝对值的比例，只由效用函数决定（不含概率加权，λ 相互抵消），正值为风险厌恶，负值为风险寻求（`cpt.risk_premium`）。
边界不对称，因为两侧的数值限制不同（数据中最大的 |x−R| 是 £2155）：
α>0 时效用有上界 1/α（损失侧下界 −λ/α⁻），α·|z| 超过约 37 时浮点饱和、CE 无法反解，上界 0.01 使 α·|z|≤21.6；
α<0 时效用无界、不饱和，只在 |α|·|z|>709 时溢出，即 |α|<0.33，下界 −0.3 时彩票价值已接近其最好结果。
指数模型的 config ID 在 pw 之后加 `_cara`（如 `test1_exp_sepalpha_commem_tk_cara_C4`），不使用 `legacy_unconditional` 的幂效用起点。
它们的边界与幂效用不同，所以必须写入新的输出目录；协议多一项 `utility`。幂效用目录的指纹不变。

```bash
python scripts/run_grid.py --tests test3 --discounting exp --curvature sepalpha --memory commem --pw tk \
    --utility exponential --lambda-upper 20 --gamma-lower 0.28 --output-dir results/sample_stricter/tk_test3_cara
```

模型选择不进入指纹，所以同一输出目录可以分几次累积模型；样本、边界、种子、容差或 `--warm-dirs` 改变时须换目录。
剔除的被试与原因写入输出目录的 `excluded_participants.csv`。`--output-dir` 必须显式给出。

### 文本摘要 summary.txt

每保存一个模型，`run_grid.py` 都会在输出目录重写 `summary.txt`（与 `summary.csv` 并列）：样本与剔除、边界、起点数、所有模型的拟合表（k、LL、BIC、ΔBIC、ξ 中位数、退出原因、命中最优的起点数）、Test 1 按 C 与 Test 2 按参考点的 BIC 比较，以及每个模型逐类的参数（π、众数人数、RP 权重、δ、r、α⁺、α⁻、λ、γ/α、A(5)/A(20)/A(50)、£20/£100/£500 的收益域与损失域风险溢价 Premium±(x)、TK 单调性；† 表示在边界上）。它只读取已保存的文件，不重新估计；旧格式的协议和 `legacy_unconditional` 也能读。已有目录用：

```bash
python scripts/summarize_run.py tk_lam20_g028_main            # 运行名，或路径、<样本>/<运行>
python scripts/summarize_run.py --all                         # results/ 下所有运行
python scripts/summarize_run.py tk_lam20_g028_main --stdout   # 只打印
```

## 结果目录

所有估计结果都在 `results/<样本>/<运行>/`，profile likelihood 在 `results/<样本>/profiles/`：

```
results/
  README.md, catalog.csv        索引：每次运行一行（样本、剔除、边界、模型数、最低 BIC）
  legacy_unconditional/         原始网格（旧估值），只作 warm start
  sample_original/              1,131 人：grid_20seeds/、grid_100seeds/
  sample_stricter/              1,081 人：grid/、profiles/
  sample_stricter_no_endpoint/  851 人：prelec_lam6/、prelec_lam20/、tk_lam20/、tk_lam20_g028_c456/、tk_lam20_g028_main/
```

新样本建一个新的 `results/sample_<名称>/`，每次运行一个子目录；在样本或运行目录里放一行 `about.txt` 说明用途，然后运行 `python scripts/catalog_results.py` 更新索引。结果数据不进 git，索引与 `about.txt` 进 git。
旧目录名（如 `results_grid_conditional_stricter`）仍出现在已保存的协议与 profile 记录中，只作历史记录。

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

脚本默认扫描 `results/sample_original/grid_20seeds` 中的所有模型与起点 pickle（`--input-dir` 可换成任一运行目录），生成独立报告
`reports/grid_em_conditional_explorer/index.html`。默认复算所有文件的条件 CE、LL、后验和实际决策权重，保留完整原始字段、个人参数、迭代与优化器日志；不执行估计、不修改输入结果。

使用 `--language zh`（默认）或 `--language en` 选择中英文；`--lang` 是简写。
英文版默认输出到 `reports/grid_em_conditional_explorer_en/index.html`，与中文版分别保存，两个输出目录均在 `.gitignore` 中忽略。
页面、交互图表、诊断说明、参数 CSV 状态和导出图表随语言切换，原始 pickle 字段完整保留。

总览支持模型筛选、排序、最多四个文件并排比较，以及点击进入每个 pickle 的完整详情。所有记录共用 `detail.html?id=…`，原始数据按字节分块装入 `data`，不截断字段、参与者、后验或日志，不对数值取整。图表包括 BIC、起点 LL 差距、收敛、概率权重、参考点组合、金额依赖、分阶段拟合与个人参数分布。表格可下载 CSV，详情可下载完整可读 JSON；另生成用于导出的 Matplotlib SVG/PNG。

打开模型或勾选比较时，同时绘制每个类别的效用函数：以 `z=x−R` 为横轴，收益侧为 `z^α⁺`，损失侧为 `−λ(−z)^α⁻`（指数效用模型按上文 CARA 公式绘制，A(d)、M(d) 同样按 CARA 计算）。详情页中的金额控件同时调整效用图的正负绘图范围；每个模型结果另生成默认 `±£100` 范围的 SVG/PNG，可从详情页下载。

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
python scripts/export_report.py --language en --input-dir results/sample_stricter/grid --output-dir reports/my_report
```

若当前代码或数据与估计协议不一致，仍导出参数和原始内容，但明确标记无法复算的诊断。达到 LL 容差、概率检查通过或较低 BIC，都不构成全局最优、机制识别或人口比例的证明；标准误等未保存信息不会被补造。
