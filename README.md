# Success Provides Direction, Failure Provides Precision: A Structured Audit of Failure Data in LLM Training

**成功给方向，失败给精度：大模型训练中失败数据的结构化审计**

*Yu Yugui · Yu Yugui Strategy Lab · they4897@gmail.com*

**[Read the paper (English PDF)](paper.pdf) · [中文版论文](paper_zh.pdf) · [Codebook / 编码手册](docs-codebook.md)**

---

## Abstract (EN)

Training pipelines for large language models are overwhelmingly built on success: rejection-sampling fine-tuning keeps only verified-correct trajectories, distillation releases only passing rollouts, and preference optimization reduces failure to the rejected edge of a preference pair. Meanwhile, independent evidence—from hard-negative contrastive learning, process-supervision error annotation, and error-correction fine-tuning (e.g., LEMA)—shows that failure data, when explicitly annotated and reflected upon, improves precisely the two weakest dimensions of current models: generalization and calibration.

This paper asks whether the industry is closing that gap. We audit **33 units across two layers**—public training datasets and open recipes (28 units), plus the five flagship pipelines disclosed in autumn 2026 (GPT-6 Astra, Claude Opus 5.5, GLM-5.3, MiMo-V2.6, DeepSeek-V4.1-Flash)—using a five-level ordinal lifecycle codebook: **L0 discarded / L1 shadow / L2 transient / L3 component / L4 first-class citizen** (error localization + attribution + reflection at scale).

**Key findings.**

1. **The L4 tier is vacant: 0/33 units.** No disclosed pipeline treats failure data as a first-class, methodology-level asset, and no failure-annotated dataset is publicly released by any frontier lab (openness 0/5 among flagships).
2. **Mainstream open recipes sit at the bottom**: 0/15 units reach L3; the SFT layer is 7/7 pure L0 (discard).
3. **A generational shift is underway in the 2026 autumn flagships**: 4/5 reach L3 (ordinal median L1→L3), with DeepSeek-V4.1 (employee-submitted failure cases → systematic replay environments → targeted RL, capability domain) and MiMo-V2.6 (reward-hacking failures → reflection-revision data with the original error retained) closest to L4.
4. **The carrier of failure signals is migrating** from preference pairs (6/15 mainstream units) to RL environments and audit loops (0/14 flagship mechanisms).

We conclude with a roadmap for making failure data a first-class citizen: collection protocols, toxicity cleaning, the three-part "error type + root cause + correction" schema, and an open failure-corpus recommendation.

## 摘要（中文）

当前大模型的训练管线几乎完全建立在"成功"之上：拒绝采样微调只保留验证通过的轨迹，蒸馏只发布通过的 rollout，偏好优化把失败压缩成偏好对的 rejected 边。而独立证据（困难负样本对比学习、过程监督错误标注、错误-修正微调如 LEMA）一致表明：显式标注并加以反思的失败数据，恰好改进当前模型最弱的两个维度——**泛化与校准**。

本文审计**两层级 33 个单位**（公开训练数据集与开源配方 28 个 + 2026 年秋季五家旗舰管线：GPT-6 Astra、Claude Opus 5.5、GLM-5.3、MiMo-V2.6、DeepSeek-V4.1-Flash），使用五级序数生命周期编码：**L0 丢弃 / L1 影子 / L2 瞬时 / L3 组件 / L4 一等公民**（定位+归因+反思三要素规模化）。

**核心发现：** ①L4 空缺——0/33，无一家前沿实验室公开失败标注数据集（旗舰开放性 0/5）；②主流开源配方垫底——0/15 达 L3，SFT 层 7/7 纯 L0；③2026 秋季旗舰出现代际转变——4/5 达 L3（中位数 L1→L3），DeepSeek-V4.1（员工提交失败案例→失败重放环境→定向 RL，且进入能力域）与 MiMo-V2.6（reward-hacking 失败→保留原错误的反思修订数据）最接近 L4；④失败信号载体正从偏好对（6/15）迁移到 RL 环境与审计闭环（0/14）。

文末给出失败数据一等公民化路线图：采集协议、毒性清洗、"错误类型+根因+修正"三段式 schema、开放失败语料建议。

## Repository Structure / 仓库结构

```
paper.pdf / paper_zh.pdf      # English & Chinese manuscripts (compiled)
paper.tex / paper_zh.tex      # LaTeX sources
paper.bbl / references.bib    # 45 references
04_manuscript.md              # English manuscript (Markdown)
04_manuscript_zh.md           # 中文版（Markdown）
docs-codebook.md              # Audit protocol + L0-L4 codebook (审计协议+编码手册)
figures/                      # Fig 1 lifecycle framework, Fig 2 distributions
analysis/coding_data.py       # All 44 mechanism-level coding records + reproducible tables/figures
analysis/output/              # CSV coding tables
```

## Reproduce / 复现

```bash
cd analysis && python3 coding_data.py   # regenerates all tables + figures from embedded coding records
tectonic paper.tex                      # rebuild English PDF
tectonic paper_zh.tex                   # rebuild Chinese PDF
```

Every coding decision in `analysis/coding_data.py` carries its supporting quotation, source URL, and confidence tier (primary/secondary). All conclusions are limited to *disclosed practice*: non-disclosure ≠ non-existence.

## Citation

```bibtex
@article{yu2026failuredata,
  title   = {Success Provides Direction, Failure Provides Precision:
             A Structured Audit of Failure Data in LLM Training},
  author  = {Yu, Yugui},
  journal = {Yu Yugui Strategy Lab, preprint},
  year    = {2026},
  month   = {10},
  url     = {https://github.com/yuyuguilab/failure-data-audit}
}
```

## License

- **Code** (`analysis/`): MIT
- **Manuscript, codebook, figures, and coding data**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Contact

Yu Yugui (余玉桂) · Yu Yugui Strategy Lab (余玉桂战略研究室) · they4897@gmail.com
