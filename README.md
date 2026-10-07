# 条件彩票 Grid EM 结果报告

在本项目的 Python 环境中运行：

```bash
python export_conditional_html.py
```

脚本扫描 `results_grid_conditional` 中的所有模型与起点 pickle，生成独立报告
`reports/grid_em_conditional_explorer/index.html`。默认复算所有文件的条件 CE、LL、后验和实际决策权重，保留完整原始字段、个人参数、迭代与优化器日志；不执行估计、不修改输入结果。

总览支持模型筛选、排序、最多四个文件并排比较，以及点击进入每个 pickle 的完整详情。图表包括 BIC、起点 LL 差距、收敛、概率权重、参考点组合、金额依赖、分阶段拟合与个人参数分布。表格可下载 CSV，详情可下载完整可读 JSON；另生成用于导出的 Matplotlib SVG/PNG。

生成后直接用浏览器打开 `index.html`，无需联网或启动服务器。移动或分享时保留整个报告文件夹中的 `assets`、`details` 等目录。新增估计结果后再次运行脚本即可更新快照。

可选命令：

```bash
python export_conditional_html.py --audit models   # 只复算模型，起点仍完整导出
python export_conditional_html.py --audit none     # 只读取保存诊断
python export_conditional_html.py --models-only    # 只导出模型 pickle
python export_conditional_html.py --input-dir results_grid_conditional --output-dir reports/my_report
```

若当前代码或数据与估计协议不一致，仍导出参数和原始内容，但明确标记无法复算的诊断。达到 LL 容差、概率检查通过或较低 BIC，都不构成全局最优、机制识别或人口比例的证明；标准误等未保存信息不会被补造。
