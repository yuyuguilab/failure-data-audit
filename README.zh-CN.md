[English](README.md) | **简体中文**

# 成功给方向，失败给精度：大模型训练中失败数据的结构化审计

*余玉桂 · 余玉桂战略研究室 · they4897@gmail.com*

**[阅读中文版论文 (PDF)](paper_zh.pdf) · [English paper](paper.pdf) · [编码手册](docs-codebook.md) · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355930.svg)](https://doi.org/10.5281/zenodo.23355930)**

## 摘要

当前大模型的训练管线几乎完全建立在"成功"之上：拒绝采样微调只保留验证通过的轨迹，蒸馏只发布通过的 rollout，偏好优化把失败压缩成偏好对的 rejected 边。而独立证据（困难负样本对比学习、过程监督错误标注、错误-修正微调如 LEMA）一致表明：显式标注并加以反思的失败数据，恰好改进当前模型最弱的两个维度——**泛化与校准**。

本文审计**两层级 33 个单位**（公开训练数据集与开源配方 28 个 + 2026 年秋季五家旗舰管线：GPT-6 Astra、Claude Opus 5.5、GLM-5.3、MiMo-V2.6、DeepSeek-V4.1-Flash），使用五级序数生命周期编码：**L0 丢弃 / L1 影子 / L2 瞬时 / L3 组件 / L4 一等公民**（定位+归因+反思三要素规模化）。

## 核心发现

1. **L4 层空缺：0/33。** 无任何已披露的管线把失败数据当作方法论级的一等资产，也无任何前沿实验室公开失败标注数据集（旗舰开放性 0/5）。
2. **主流开源配方垫底**：0/15 达到 L3；SFT 层 7/7 纯 L0（丢弃）。
3. **2026 秋季旗舰出现代际转变**：4/5 达到 L3（序数中位数 L1→L3）。DeepSeek-V4.1（员工提交失败案例→系统性失败重放环境→定向 RL，且进入能力域）与 MiMo-V2.6（reward-hacking 失败→保留原错误的反思修订数据）最接近 L4。
4. **失败信号载体正在迁移**：从偏好对（主流配方 6/15）迁移到 RL 环境与审计闭环（旗舰机制 0/14）。

文末给出失败数据一等公民化路线图：采集协议、毒性清洗、"错误类型+根因+修正"三段式 schema、开放失败语料建议。

## 仓库结构

```
paper.pdf / paper_zh.pdf      # 英文/中文论文（编译产物）
paper.tex / paper_zh.tex      # LaTeX 源码
paper.bbl / references.bib    # 45 条参考文献
04_manuscript.md              # 英文版（Markdown）
04_manuscript_zh.md           # 中文版（Markdown）
docs-codebook.md              # 审计协议 + L0-L4 编码手册
figures/                      # 图 1 生命周期框架、图 2 分布对比
analysis/coding_data.py       # 全部 44 条机制级编码记录 + 可复现表格/图表
analysis/output/              # CSV 编码表
```

## 复现

```bash
cd analysis && python3 coding_data.py   # 从内嵌编码记录重建全部表格与图
tectonic paper.tex                      # 重编译英文 PDF
tectonic paper_zh.tex                   # 重编译中文 PDF
```

`analysis/coding_data.py` 中每条编码决策都附支撑引句、来源 URL 与置信度分级（一手/转引）。所有结论限定于*已披露实践*：未披露 ≠ 不存在。

## 引用

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355930.svg)](https://doi.org/10.5281/zenodo.23355930)

```bibtex
@article{yu2026failuredata,
  title   = {Success Provides Direction, Failure Provides Precision:
             A Structured Audit of Failure Data in LLM Training},
  author  = {Yu, Yugui},
  journal = {Yu Yugui Strategy Lab, preprint},
  year    = {2026},
  month   = {10},
  doi     = {10.5281/zenodo.23355930},
  url     = {https://github.com/yuyuguilab/failure-data-audit}
}
```

## 许可

- **代码**（`analysis/`）：MIT
- **论文、编码手册、图表与编码数据**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## 联系方式

余玉桂 · 余玉桂战略研究室 · they4897@gmail.com
