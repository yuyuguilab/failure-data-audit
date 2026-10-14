**English** | [简体中文](README.zh-CN.md)

# Success Provides Direction, Failure Provides Precision: A Structured Audit of Failure Data in LLM Training

*Yu Yugui · Yu Yugui Strategy Lab · they4897@gmail.com*

**[Read the paper (PDF)](paper.pdf) · [中文版论文](paper_zh.pdf) · [Codebook](docs-codebook.md) · [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355930.svg)](https://doi.org/10.5281/zenodo.23355930)**

## Abstract

Training pipelines for large language models are overwhelmingly built on success: rejection-sampling fine-tuning keeps only verified-correct trajectories, distillation releases only passing rollouts, and preference optimization reduces failure to the rejected edge of a preference pair. Meanwhile, independent evidence—from hard-negative contrastive learning, process-supervision error annotation, and error-correction fine-tuning (e.g., LEMA)—shows that failure data, when explicitly annotated and reflected upon, improves precisely the two weakest dimensions of current models: generalization and calibration.

This paper asks whether the industry is closing that gap. We audit **33 units across two layers**—public training datasets and open recipes (28 units), plus the five flagship pipelines disclosed in autumn 2026 (GPT-6 Astra, Claude Opus 5.5, GLM-5.3, MiMo-V2.6, DeepSeek-V4.1-Flash)—using a five-level ordinal lifecycle codebook: **L0 discarded / L1 shadow / L2 transient / L3 component / L4 first-class citizen** (error localization + attribution + reflection at scale).

## Key Findings

1. **The L4 tier is vacant: 0/33 units.** No disclosed pipeline treats failure data as a first-class, methodology-level asset, and no failure-annotated dataset is publicly released by any frontier lab (openness 0/5 among flagships).
2. **Mainstream open recipes sit at the bottom**: 0/15 units reach L3; the SFT layer is 7/7 pure L0 (discard).
3. **A generational shift is underway in the 2026 autumn flagships**: 4/5 reach L3 (ordinal median L1→L3), with DeepSeek-V4.1 (employee-submitted failure cases → systematic replay environments → targeted RL, capability domain) and MiMo-V2.6 (reward-hacking failures → reflection-revision data with the original error retained) closest to L4.
4. **The carrier of failure signals is migrating** from preference pairs (6/15 mainstream units) to RL environments and audit loops (0/14 flagship mechanisms).

The paper closes with a roadmap for making failure data a first-class citizen: collection protocols, toxicity cleaning, the three-part "error type + root cause + correction" schema, and an open failure-corpus recommendation.

## Repository Structure

```
paper.pdf / paper_zh.pdf      # English & Chinese manuscripts (compiled)
paper.tex / paper_zh.tex      # LaTeX sources
paper.bbl / references.bib    # 45 references
04_manuscript.md              # English manuscript (Markdown)
04_manuscript_zh.md           # Chinese manuscript (Markdown)
docs-codebook.md              # Audit protocol + L0-L4 codebook
figures/                      # Fig 1 lifecycle framework, Fig 2 distributions
analysis/coding_data.py       # All 44 mechanism-level coding records + reproducible tables/figures
analysis/output/              # CSV coding tables
```

## Reproduce

```bash
cd analysis && python3 coding_data.py   # regenerates all tables + figures from embedded coding records
tectonic paper.tex                      # rebuild English PDF
tectonic paper_zh.tex                   # rebuild Chinese PDF
```

Every coding decision in `analysis/coding_data.py` carries its supporting quotation, source URL, and confidence tier (primary/secondary). All conclusions are limited to *disclosed practice*: non-disclosure ≠ non-existence.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23355930.svg)](https://doi.org/10.5281/zenodo.23355930)

```bibtex
@article{yu2026failuredata,
  title   = {Success Provides Direction, Failure Provides Precision:
             A Structured Audit of Failure Data in LLM Training},
  author  = {Yugui Yu},
  journal = {Yu Yugui Strategy Lab, preprint},
  year    = {2026},
  month   = {10},
  doi     = {10.5281/zenodo.23355930},
  url     = {https://github.com/yuyuguilab/failure-data-audit}
}
```

## License

- **Code** (`analysis/`): MIT
- **Manuscript, codebook, figures, and coding data**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Contact

Yu Yugui · Yu Yugui Strategy Lab · they4897@gmail.com
