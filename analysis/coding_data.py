# -*- coding: utf-8 -*-
"""
coding_data.py — Failure-Data Lifecycle Audit: coding records + tables + figures
================================================================================
Paper : Failure-Data Training Methodology — position paper + structured evidence audit
Agent : statistical-analyst (03_analysis_report.md companion code)

WHAT THIS IS
------------
Mechanism-level coding records applying the codebook of 02_experiment_protocol.md
(Sect. 3) to the two locked corpora:

  Corpus A (dataset layer, Audit I)  : 28 units  -> 30 mechanism records
  Corpus B (pipeline  layer, Audit II):  5 units -> 14 mechanism records

Every record carries: main lifecycle level (L0-L4), the six sub-dimensions,
a verbatim/anchored evidence quote, source tier (P=primary / S=secondary),
confidence, and codebook rule notes. Data source = evidence tables of
00_input.md (Sect. 3 & 4), verified numbers per 01/02 (DAPO=arXiv:2503.14476,
PRS=arXiv:2606.26027, embodied 9440/78623 dataset REMOVED as unverifiable).
Access date for all sources: 2026-10-13.

HARD CONSTRAINTS IMPLEMENTED HERE (per 02 handoff notes)
-------------------------------------------------------
* L0-L4 is ORDINAL: no means / interval statistics. Medians and distributions only.
* "no-disclosure" is kept as a separate category, never merged into L0 (rule CR5).
* Fisher exact tests are EXPLORATORY only (saturated purposive sample, n=5 in
  corpus B); the paper's main narrative uses distributions + exemplar quotes.
* Unit main code = highest mechanism level (rule CR1); multi_level records all
  coexisting levels.

REPRODUCE
---------
    python3 _workspace/analysis/coding_data.py
Outputs: CSVs in  _workspace/analysis/output/
         figures in _workspace/figures/
"""

import csv
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output")
FIG = os.path.join(WS, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

ACCESS = "2026-10-13"

# ---------------------------------------------------------------------------
# Orderings (ordinal scales — NEVER averaged)
# ---------------------------------------------------------------------------
LEVELS = ["L0", "L1", "L2", "L3", "L4", "no-disclosure"]
LEVEL_ORD = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
GRAN_ORD = {"none": 0, "trajectory": 1, "step": 2, "causal": 3}
ATTR_ORD = {"none": 0, "binary": 1, "root-cause": 2}
REFL_ORD = {"none": 0, "correction-pair": 1, "full-report": 2}
LEVEL_NAMES = {"L0": "Discard", "L1": "Shadow", "L2": "Transient",
               "L3": "Component", "L4": "First-class"}


def max_ord(values, order):
    """Highest disclosed form, ignoring 'nd' (not disclosed)."""
    vals = [v for v in values if v in order]
    if not vals:
        return "nd"
    return max(vals, key=lambda v: order[v])


# ---------------------------------------------------------------------------
# MECHANISM-LEVEL CODING RECORDS (data dictionary = 02 Sect. 5)
# ---------------------------------------------------------------------------
R = []  # list of dicts

def rec(record_id, unit_id, unit_name, corpus, stage, mech, level, gran, attr,
        refl, dom, openn, veh, quote, source, arxiv, tier, veranchor, conf, notes,
        multi=""):
    R.append(dict(
        record_id=record_id, unit_id=unit_id, unit_name=unit_name, corpus=corpus,
        stage=stage, mechanism_desc=mech, level_main=level, multi_level=multi,
        dim_granularity=gran, dim_attribution=attr, dim_reflection=refl,
        dim_domain=dom, dim_openness=openn, dim_vehicle=veh,
        evidence_quote=quote, source_url=source, arxiv_id=arxiv,
        source_tier=tier, access_date=ACCESS, version_anchor=veranchor,
        confidence=conf, notes=notes))

# ===== CORPUS A — dataset layer (Audit I) ==================================

# --- Pre-training ------------------------------------------------------------
rec("A-01-m1", "A-01", "Dolma", "A", "pretraining",
    "Web-corpus quality filtering; no notion of model failure in curation",
    "L0", "none", "none", "none", "nd", "not-released", "none",
    "「预训练（Dolma/FineWeb 等）：纯正向语料，无失败概念」(00_input §3.1 审计摘录)",
    "AI2 Dolma paper / data card", "", "P", "Dolma 技术报告", "medium",
    "BC-01 边界：质量过滤非模型失败判定，按 L0 判据①『过滤后发布』广义适用；domain 概念不适用记 nd")
rec("A-02-m1", "A-02", "FineWeb", "A", "pretraining",
    "Web-corpus quality filtering; no notion of model failure in curation",
    "L0", "none", "none", "none", "nd", "not-released", "none",
    "「纯正向语料，无失败概念」(00_input §3.1 审计摘录)",
    "HF HuggingFaceFW/FineWeb data card", "", "P", "FineWeb 技术报告", "medium",
    "BC-01 边界（同 A-01）")

# --- SFT / distillation -------------------------------------------------------
rec("A-03-m1", "A-03", "DeepSeek R1 (SFT stage)", "A", "SFT",
    "Stage-3 rejection-sampling SFT: only correct+language-consistent+high-quality trajectories kept; all failure trajectories discarded",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「第三阶段拒绝采样微调：只保留正确答案+语言一致+高质量轨迹（约 60 万推理+20 万非推理），失败轨迹全部丢弃」(00_input §3.2)",
    "arxiv.org/abs/2501.12948", "2501.12948", "P", "R1 paper 2025-01", "medium",
    "codebook L0 正例；RL 阶段失败处置另列 A-16")
rec("A-04-m1", "A-04", "Llama 2 (SFT)", "A", "SFT",
    "Best-of-N rejection sampling: only highest-scoring generations enter SFT",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「Llama 2/3：best-of-N 拒绝采样只取最高分入 SFT」(00_input §3.2)",
    "arxiv.org/abs/2307.09288", "2307.09288", "P", "Llama 2 paper 2023-07", "medium", "")
rec("A-05-m1", "A-05", "Llama 3 (SFT)", "A", "SFT",
    "Best-of-N rejection sampling for SFT data; failures filtered pre-training",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「best-of-N 拒绝采样只取最高分入 SFT」(00_input §3.2)",
    "arxiv.org/abs/2407.21783", "2407.21783", "P", "Llama 3 paper 2024-07", "medium",
    "DPO 阶段机制在 A-10 单独编码（CR1）")
rec("A-06-m1", "A-06", "OpenR1-Math-220k", "A", "SFT",
    "Answer verification then discard of incorrect trajectories in community reproduction",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「答案验证后丢弃错误轨迹」(00_input §3.2)",
    "github.com/huggingface/OpenR1", "", "P", "OpenR1 repo README", "medium",
    "原始轨迹另行发布（A-26 OpenR1-Math-Raw）")
rec("A-07-m1", "A-07", "Bespoke-Stratos-17k", "A", "SFT",
    "Only verification-passing trajectories released",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「只发布验证通过轨迹」(00_input §3.2)",
    "HF bespokelabs/Bespoke-Stratos-17k", "", "P", "HF data card", "medium", "")
rec("A-08-m1", "A-08", "OpenMathReasoning", "A", "SFT",
    "Only majority-vote passing solutions retained",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「只保留 majority-vote 通过解」(00_input §3.2)",
    "HF nvidia/OpenMathReasoning", "", "P", "HF data card", "medium", "")
rec("A-09-m1", "A-09", "OpenMathInstruct-2", "A", "SFT",
    "Verification-filtered release; expected_answer field enables correctness judgment but no failure instances retained",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「带 expected_answer 字段可判对错（验证过滤为主）」(00_input §3.4)",
    "HF nvidia/OpenMathInstruct-2", "", "P", "HF data card", "medium",
    "BC-02 边界：codebook L0 边界判例——含可判对错字段≠保留失败实例→判 L0")

# --- Preference alignment ------------------------------------------------------
rec("A-10-m1", "A-10", "Llama 3 DPO", "A", "preference",
    "Preference pairs built from score gap (mu+2sigma vs mu-2sigma); failure exists only as rejected side",
    "L1", "trajectory", "none", "none", "capability", "not-released", "preference-pair",
    "「按分数差构造偏好对（μ+2σ vs μ−2σ）做 DPO」(00_input §3.3)",
    "arxiv.org/abs/2407.21783", "2407.21783", "P", "Llama 3 paper 2024-07", "medium", "")
rec("A-11-m1", "A-11", "Qwen3 (RM)", "A", "preference",
    "Verifier-rejected answers used as rejected side for RM training",
    "L1", "trajectory", "none", "none", "capability", "not-released", "preference-pair",
    "「验证器判错的回答作 rejected 训 RM」(00_input §3.3)",
    "arxiv.org/abs/2505.09388", "2505.09388", "P", "Qwen3 report 2025-05", "medium", "单元主码 L1")
rec("A-11-m2", "A-11", "Qwen3 (RM)", "A", "preference",
    "Rejected answers discarded (alternative branch of the same pipeline)",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「验证器判错的回答作 rejected 训 RM 或丢弃」(00_input §3.3)",
    "arxiv.org/abs/2505.09388", "2505.09388", "P", "Qwen3 report 2025-05", "medium",
    "BC-07 边界：『或丢弃』分支按 CR1 拆分机制条目")
rec("A-12-m1", "A-12", "Tulu 3", "A", "preference",
    "On-policy multi-response sampling, PairRM ranking, best=chosen worst=rejected (~270k preference prompts)",
    "L1", "trajectory", "none", "none", "capability", "not-released", "preference-pair",
    "「on-policy 从 SFT checkpoint 采样多响应，PairRM 排序，最优 chosen 最差 rejected（约 27 万偏好 prompt）」(00_input §3.3)",
    "arxiv.org/abs/2411.15124", "2411.15124", "P", "Tulu 3 paper 2024-11", "medium", "")
rec("A-13-m1", "A-13", "HH-RLHF", "A", "preference",
    "~170k chosen/rejected pairs (helpful 138k + harmless 43k); failure only as rejected side",
    "L1", "trajectory", "none", "none", "both", "released", "preference-pair",
    "「约 17 万 chosen/rejected 对（helpful 13.8 万+harmless 4.3 万）」(00_input §3.3)",
    "HF Anthropic/hh-rlhf", "", "P", "HF data card", "medium",
    "codebook L1 正例；BC-12：rejected 随偏好对公开（非独立失败数据集）→ openness=released")
rec("A-14-m1", "A-14", "SmolLM2 (DPO)", "A", "preference",
    "UltraFeedback DPO: GPT-4 low-score responses serve as rejected",
    "L1", "trajectory", "none", "none", "capability", "not-released", "preference-pair",
    "「UltraFeedback DPO（GPT-4 低分响应为 rejected）」(00_input §3.3)",
    "HF HuggingFaceTB/smoltalk", "", "P", "HF data card", "medium",
    "底层 UltraFeedback 原始发布物在 A-23 编码")
rec("A-15-m1", "A-15", "GLM-4.5 MIPO", "A", "preference",
    "Graded mixing of negative/refusal samples by strength (mixed-strength negative preference)",
    "L1", "none", "none", "none", "both", "not-released", "preference-pair",
    "「不同强度负样本/拒答样本分级混合——各家披露中最接近系统性保留失败信号的设计（但 5.x 已弃用）」(00_input §3.3)",
    "arxiv.org/abs/2508.06471", "2508.06471", "P", "GLM-4.5 report 2025-08", "medium",
    "BC-03 边界：强度分级不属 L3 判据①列举组件形态（无定位/归因/反思）→ L1 上界；CR3：GLM-5.x 弃用（偏好载体衰落证据，见 B-03）")

# --- RL ------------------------------------------------------------------------
rec("A-16-m1", "A-16", "GRPO (R1 instantiation)", "A", "RL",
    "Failure rollouts contribute negative advantage to gradient; discarded after use, never released",
    "L2", "trajectory", "none", "none", "capability", "not-released", "RL-environment",
    "「RL（GRPO/PPO）：失败 rollout 负优势参与梯度，用完即弃、从不发布」(00_input §3.1)",
    "arxiv.org/abs/2501.12948", "2501.12948", "P", "R1 paper 2025-01", "medium",
    "codebook L2 正例")
rec("A-17-m1", "A-17", "DAPO (in-group failures)", "A", "RL",
    "Within-group failure rollouts carry negative advantage",
    "L2", "trajectory", "none", "none", "capability", "not-released", "RL-environment",
    "「组内失败 rollout 负优势（L2 特征）」(00_input §3.2 DAPO 条)",
    "arxiv.org/abs/2503.14476", "2503.14476", "P", "DAPO paper 2025-03", "medium",
    "编号按 01/02 核证基线 2503.14476（00_input 误记 2412.19337）")
rec("A-17-m2", "A-17", "DAPO (dynamic sampling)", "A", "RL",
    "Dynamic sampling drops all-failure prompt groups (zero gradient framed as compute waste)",
    "L0", "trajectory", "none", "none", "capability", "not-released", "none",
    "「DAPO 动态剔除全失败组（零梯度=算力浪费）——当前范式视失败数据为成本而非资产」(00_input §3.2)",
    "arxiv.org/abs/2503.14476", "2503.14476", "P", "DAPO paper 2025-03", "medium",
    "BC-06 边界：codebook L2 判例——单元主码 L2，multi_level=L0+L2；『剔除全失败组』=丢弃制度化的算法级证据")

# --- Academic exceptions ---------------------------------------------------------
rec("A-18-m1", "A-18", "PRM800K", "A", "academic",
    "Step-level human ratings (-1 marks error step) + found_error termination; 800k step labels",
    "L3", "step", "none", "none", "capability", "released", "static-dataset",
    "「逐步骤人工 rating（-1=错误步）+found_error 终止；80 万步级标注」(00_input §3.4)",
    "github.com/openai/prm800k", "", "P", "OpenAI repo/HF（HF 版已 404）", "medium",
    "L4 不满足判据①：有定位、无归因、无反思")
rec("A-19-m1", "A-19", "Math-Shepherd", "A", "academic",
    "Monte-Carlo automatic step-level correctness probability labels (~400k samples)",
    "L3", "step", "none", "none", "capability", "released", "static-dataset",
    "「蒙特卡洛自动步级正确概率标签；~40 万样本」(00_input §3.4)",
    "HF peiyi9979/Math-Shepherd", "", "P", "HF data card", "medium", "")
rec("A-20-m1", "A-20", "LEMA", "A", "academic",
    "Wrong solutions + GPT-4 error-pointing-and-correction pairs (~24k-50k pairs)",
    "L3", "step", "none", "correction-pair", "capability", "released", "static-dataset",
    "「错误解答+GPT-4 指错与修正对（约 2.4–5 万对）」(00_input §3.4；LEMA 微调 LLaMA-2-70B GSM8K 83.5%)",
    "arxiv.org/abs/2310.20689", "2310.20689", "P", "LEMA paper 2023-10", "medium",
    "codebook L4 边界判例：三要素基本齐（定位+修正、归因弱）但万级规模、未进主流配方→判据②不满足→L3；语料 A 最接近 L4 者")
rec("A-21-m1", "A-21", "ETO (Trial-and-Error)", "A", "academic",
    "Failure trajectories paired with successful ones for contrastive training (WebShop/ALFWorld/SciWorld, tens of thousands of trajectories)",
    "L3", "trajectory", "none", "correction-pair", "capability", "released", "static-dataset",
    "「失败轨迹与成功轨迹配对对比训练」(00_input §3.4)",
    "arxiv.org/abs/2403.02502", "2403.02502", "P", "ETO paper 2024-03", "medium",
    "载体=含对比结构的静态数据（非 DPO 型偏好对）")
rec("A-22-m1", "A-22", "UltraInteract_pair", "A", "academic",
    "Tree-structured trajectories with erroneous actions + reflections composed into preference pairs (28k instructions)",
    "L3", "step", "none", "correction-pair", "capability", "released", "preference-pair",
    "「树状轨迹含错误动作+反思组成偏好对；2.8 万指令」(00_input §3.4)",
    "arxiv.org/abs/2404.02078", "2404.02078", "P", "UltraInteract paper 2024-04", "medium",
    "BC-05 边界：codebook L1 判例——rejected 附带显式反思组件→multi L1+L3，主码 L3")
rec("A-23-m1", "A-23", "UltraFeedback (original release)", "A", "academic",
    "GPT-4 five-dimension 1-5 scores + per-response critique (low score + critique = failure analysis); 64k prompts x 4 responses",
    "L3", "trajectory", "none", "correction-pair", "capability", "released", "static-dataset",
    "「GPT-4 五维 1-5 评分+逐条 critique（低分+批评=失败分析）；6.4 万 prompt×4 回答」(00_input §3.4)",
    "HF openbmb/UltraFeedback", "", "P", "HF data card", "medium",
    "BC-04 边界：原始发布物=批判性失败分析公开（L3）；下游 DPO 用法=影子（L1，见 A-14）——单元按原始发布物编码，multi L1+L3")
rec("A-24-m1", "A-24", "ProcessBench", "A", "academic",
    "Benchmark locating the first erroneous step (3.4k problems)",
    "L3", "step", "none", "none", "capability", "released", "static-dataset",
    "「定位首个错误步骤；3400 题」(00_input §3.4)",
    "HF PRIME-RL/ProcessBench", "", "P", "HF data card", "medium",
    "BC-10：unit 层=评测侧参照（02 §2.1 排除标准注）——错误定位数据存在但用途为评测非训练")
rec("A-25-m1", "A-25", "PRMBench", "A", "academic",
    "Six-dimension error-type attribution benchmark (~11k samples)",
    "L3", "step", "root-cause", "none", "capability", "released", "static-dataset",
    "「六维错误类型归因；~1.1 万样本」(00_input §3.4)",
    "PRMBench 论文", "", "P", "PRMBench paper", "medium",
    "BC-10：评测侧参照；语料 A 中唯一 root-cause 归因")
rec("A-26-m1", "A-26", "OpenR1-Math-Raw", "A", "academic",
    "Raw trajectories including verifier-failing ones (~1M); only large-scale release retaining failures in the mainstream ecosystem",
    "L3", "trajectory", "none", "none", "capability", "released", "static-dataset",
    "「原始轨迹含未通过验证者（主流生态唯一保留失败的发布）；~100 万轨迹」(00_input §3.4)",
    "HF open-r1/OpenR1-Math-Raw", "", "P", "HF data card", "medium",
    "codebook L3 判例：三要素不齐→不升 L4；全语料最高开放度的 L3（百万级）")
rec("A-27-m1", "A-27", "NPO forgetting sets", "A", "academic",
    "Forget sets as DPO negative side (TOFU 4k / WMDP / MUSE)",
    "L1", "trajectory", "none", "none", "alignment", "released", "preference-pair",
    "「forget 集作 DPO 负侧（TOFU 4k/WMDP/MUSE）」(00_input §3.4)",
    "arxiv.org/abs/2404.05868", "2404.05868", "P", "NPO paper 2024-04", "medium",
    "遗忘目标数据以公开基准形式存在；失败侧=负偏好形态（L1）")
rec("A-28-m1", "A-28", "SPAR f26", "A", "academic",
    "Human-annotated agent failure traces, 200-300 traces (open-source plan for 2026-10)",
    "L3", "nd", "nd", "nd", "capability", "not-released", "static-dataset",
    "「人工标注 agent 失败 trace；200-300 条（2026-10 开源计划）」(00_input §3.4)",
    "sparai.org/projects/f26/recPcCQjmdWCEjPJL", "", "P", "SPAR project page", "low",
    "BC-11（CR4 证据不足）：粒度/归因/反思未披露→nd；开源计划截至访问日未兑现→not-released；需补查项目页与发布物")

# ===== CORPUS B — pipeline layer (Audit II) ================================

rec("B-01-m1", "B-01", "GPT-6 Astra (2026-09-03)", "B", "OpenAI",
    "CoT misalignment monitoring covering frontier RL training runs + misalignment reporting framework: failure signals routed to monitoring, not training data",
    "L0", "trajectory", "none", "none", "alignment", "not-released", "audit-loop",
    "「CoT 失对齐监控覆盖 frontier RL training runs；失对齐报告框架（09-16）；失败信号流向监控/评测而非训练」(00_input §4)",
    "deploymentsafety.openai.com/gpt-6-astra", "", "P", "GPT-6 Astra system card 2026-09-03", "medium",
    "BC-08：显式路由陈述（失败不进训练）→按 CR1『入评估集=L0』判例赋 L0；监控为监督侧保留（vehicle=audit-loop）")
rec("B-01-m2", "B-01", "GPT-6 Astra (2026-09-03)", "B", "OpenAI",
    "Production low-quality responses routed to evaluation sets only",
    "L0", "trajectory", "none", "none", "alignment", "not-released", "static-dataset",
    "「生产劣质回复仅入评测集」(00_input §4)",
    "deploymentsafety.openai.com/gpt-6-astra", "", "P", "system card", "medium",
    "同 CR1 GLM badcase 判例")
rec("B-01-m3", "B-01", "GPT-6 Astra (2026-09-03)", "B", "OpenAI",
    "RL penalizes cheating and unauthorized actions (failure rollouts transiently enter gradient as penalty)",
    "L2", "trajectory", "none", "none", "alignment", "not-released", "RL-environment",
    "「RL 'penalize cheating and unauthorized actions'」(00_input §4，系统卡原文片段)",
    "deploymentsafety.openai.com/gpt-6-astra", "", "P", "system card", "high",
    "训练侧唯一失败使用=瞬时惩罚；五家中唯一 <L3（『最远』判定的量化依据）")
rec("B-02-m1", "B-02", "Claude Opus 5.5 (2026-09-22)", "B", "Anthropic",
    "AIR (Alignment Intelligence Report): 7200+ agent failure scenarios drive rapid model improvements during training; ~2000-scenario automated behavior audits",
    "L3", "trajectory", "binary", "none", "alignment", "not-released", "audit-loop",
    "「AIR 7200+ agent 失败场景驱动 'rapid model improvements during training'；近 2000 场景自动行为审计」(00_input §4)",
    "anthropic.com 官方系统卡（直链待登记，02 §2.2 执行项）", "", "P", "Claude Opus 5.5 发布材料 2026-09-22", "medium",
    "失败→训练闭环但限对齐域；系统卡直链与原文引句需补查")
rec("B-02-m2", "B-02", "Claude Opus 5.5 (2026-09-22)", "B", "Anthropic",
    "Automatic generation of failure scenarios into safety training data",
    "L3", "trajectory", "binary", "none", "alignment", "not-released", "static-dataset",
    "「失败场景自动生成入安全训练」(00_input §4)",
    "anthropic.com 官方系统卡（直链待登记）", "", "P", "同上", "medium",
    "归因佐证：『flawed environments are a major source of misaligned behavior』→剔除缺陷环境=环境/模型二元归因（BC-13：环境治理不单列机制，仅作 attribution 证据）")
rec("B-03-m1", "B-03", "GLM-5.3 (2026 H2)", "B", "Zhipu",
    "SFT erroneous-segment retention with loss masking",
    "L3", "step", "binary", "none", "capability", "not-released", "static-dataset",
    "\"Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors\" (GLM-5 报告 §3.1)",
    "arxiv.org/abs/2602.15763", "2602.15763", "P", "GLM-5 report §3.1", "high",
    "codebook L3 正例（逐字引句）")
rec("B-03-m2", "B-03", "GLM-5.3 (2026 H2)", "B", "Zhipu",
    "Error-notebook-style RL task selection: tasks filtered by persistent failure ('GLM-4.7 几乎不会做/持续失败') + failure-attribution removal (model error vs environment collapse)",
    "L3", "trajectory", "binary", "none", "capability", "not-released", "RL-environment",
    "「RL 选题按 'GLM-4.7 几乎不会做/持续失败' 过滤（错题本式选题）；失败归因剔除（区分模型错/环境崩）；solver 轨迹关闭 reward shortcuts」(00_input §4)",
    "arxiv.org/abs/2602.15763", "2602.15763", "P", "GLM-5 report", "medium",
    "02 §3.3 判据示例：GLM-5.3 失败归因剔除=binary；reward-shortcut 关闭=verifier 完整性治理（C5c 素材）；原文引句待补")
rec("B-03-m3", "B-03", "GLM-5.3 (2026 H2)", "B", "Zhipu",
    "Badcases routed to evaluation sets only",
    "L0", "trajectory", "none", "none", "capability", "not-released", "static-dataset",
    "「badcase 仅入评估集」(00_input §4)",
    "arxiv.org/abs/2602.15763", "2602.15763", "P", "GLM-5 report", "medium",
    "CR1 判例原文：掩码 L3+badcase 入评估集 L0→主码 L3；CR3：MIPO 在 5.x 弃用（后训练链 SFT→Reasoning RL→Agentic RL→General RL→OPD 无偏好阶段）")
rec("B-04-m1", "B-04", "MiMo-V2.6 (2026-09-22)", "B", "Xiaomi",
    "Mid-Training Alignment Data: training samples synthesized from real reward-hacking failures — model reflects on and revises the erroneous reasoning",
    "L3", "step", "binary", "full-report", "alignment", "not-released", "static-dataset",
    "\"keeps the original error recognizable and makes the correction explicit\" (§4.2.6)＋「从真实 reward-hacking 失败案例合成训练样本——模型对错误推理反思、修订」(00_input §4)",
    "arxiv.org/abs/2610.11959", "2610.11959", "P", "MiMo-V2.6 report §4.2.6", "high",
    "codebook L4 边界判例：三要素齐但限 reward-hacking 域、规模未披露→判据②不满足→L3；最标准错题本形态")
rec("B-04-m2", "B-04", "MiMo-V2.6 (2026-09-22)", "B", "Xiaomi",
    "Rollout-based auditing: 4 rollouts per task with full audit of failures, labeled false pos/neg",
    "L3", "trajectory", "binary", "none", "alignment", "not-released", "audit-loop",
    "「rollout-based auditing（§4.2.1）每任务 4 条 rollout 含失败全审计标 false pos/neg」(00_input §4)",
    "arxiv.org/abs/2610.11959", "2610.11959", "P", "MiMo-V2.6 report §4.2.1", "high", "")
rec("B-04-m3", "B-04", "MiMo-V2.6 (2026-09-22)", "B", "Xiaomi",
    "GAR: online grader jointly inspects in-group successful + failed trajectories; hacking-passing trajectories get reward zeroed and treated as failures",
    "L3", "trajectory", "binary", "none", "alignment", "not-released", "audit-loop",
    "「GAR（§4.3.2）在线 grader 联合审视组内成功+失败轨迹，确认 hacking 的通过轨迹 reward 置零视为失败」(00_input §4)",
    "arxiv.org/abs/2610.11959", "2610.11959", "P", "MiMo-V2.6 report §4.3.2", "high",
    "假成功/真作弊清洗=C5b 毒性清洗素材")
rec("B-04-m4", "B-04", "MiMo-V2.6 (2026-09-22)", "B", "Xiaomi",
    "Penalty: hierarchical credit assignment removes infrastructure failures; segment-level errors penalized more",
    "L3", "step", "binary", "none", "alignment", "not-released", "RL-environment",
    "「Penalty（§6.1）分层 credit assignment 剔除基础设施失败、segment 级错误加重惩罚」(00_input §4)",
    "arxiv.org/abs/2610.11959", "2610.11959", "P", "MiMo-V2.6 report §6.1", "high",
    "segment 级=step 粒度佐证；基础设施失败剔除=binary 归因佐证；BC-14：Table 2 公开 5 条 reward-hacking 失败轨迹原文=exemplar 级披露，不构成数据集开放")
rec("B-05-m1", "B-05", "DeepSeek-V4.1-Flash (2026-09-09)", "B", "DeepSeek",
    "Organized failure collection (internal employee-submitted negative feedback & model failure cases at scale) -> single/multi-turn environments -> systematic replay of failures + targeted RL against weaknesses",
    "L3", "trajectory", "root-cause", "none", "capability", "not-released", "RL-environment",
    "\"we collect negative feedback and model failure cases submitted by internal employees at scale\"＋\"systematic replay of failures and targeted RL against observed model weaknesses\" (报告 §5.1.1, 00_input §4)",
    "arxiv.org/abs/2609.19969", "2609.19969", "P", "DeepSeek-V4.1-Flash report §5.1.1", "high",
    "L4 判据③④不满足：未理论化（『数据管线工程』名义：'the marginal return of engineering the data and environment pipeline substantially exceeds that of algorithmic novelty'）且不公开→L3（『最近』但未达一等公民）；02 §3.3 判据示例：员工提交失败分类构建环境=root-cause")
rec("B-05-m2", "B-05", "DeepSeek-V4.1-Flash (2026-09-09)", "B", "DeepSeek",
    "Coding-agent data deliberately retains tasks where the model performs poorly; QC agent flags + repair agent corrects; reward hacking / environment collapse recorded as failure trajectories feeding repercussion signals back to RL",
    "L3", "trajectory", "binary", "none", "capability", "not-released", "RL-environment",
    "「coding agent 数据刻意保留『模型表现差的任务』，质检 agent 标记+修复 agent 纠错；reward hacking/环境崩溃记为失败轨迹向 RL 框架回传 repercussion 信号」(00_input §4)",
    "arxiv.org/abs/2609.19969", "2609.19969", "P", "DeepSeek-V4.1-Flash report §5.1.1", "medium",
    "CR3 代际证据：R1 四阶段（含拒绝采样 L0）在 4.x 消失——同厂 2025 L0 → 2026 L3")

# ---------------------------------------------------------------------------
# UNIT-LEVEL AGGREGATION (rule CR1: unit main = max mechanism level)
# ---------------------------------------------------------------------------
# Primary vehicle per unit: characteristic carrier of the unit's SIGNATURE
# failure mechanism (vehicle is nominal, no max rule; choices documented in
# notes; mechanism-level vehicles reported separately).
UNIT_PRIMARY_VEHICLE = {
    # corpus A — L0 discards have no failure carrier
    **{uid: "none" for uid in
       ["A-01", "A-02", "A-03", "A-04", "A-05", "A-06", "A-07", "A-08", "A-09"]},
    "A-10": "preference-pair", "A-11": "preference-pair", "A-12": "preference-pair",
    "A-13": "preference-pair", "A-14": "preference-pair", "A-15": "preference-pair",
    "A-16": "RL-environment", "A-17": "RL-environment",
    "A-18": "static-dataset", "A-19": "static-dataset", "A-20": "static-dataset",
    "A-21": "static-dataset", "A-22": "preference-pair", "A-23": "static-dataset",
    "A-24": "static-dataset", "A-25": "static-dataset", "A-26": "static-dataset",
    "A-27": "preference-pair", "A-28": "static-dataset",
    # corpus B — signature mechanisms
    "B-01": "audit-loop",        # failure signals routed to monitoring (00_input: 流向监控/评测)
    "B-02": "audit-loop",        # AIR failure-scenario loop
    "B-03": "static-dataset",    # masked erroneous segments in SFT data (signature)
    "B-04": "static-dataset",    # synthesized reflect-and-revise samples
    "B-05": "RL-environment",    # systematic replay of failures
}

UNITS = {}
for r in R:
    u = UNITS.setdefault(r["unit_id"], dict(
        unit_id=r["unit_id"], unit_name=r["unit_name"], corpus=r["corpus"],
        stage=r["stage"], mechanisms=[], levels=set(), gran=[], attr=[],
        refl=[], dom=[], openn=[], veh=[], conf={}, quotes=[]))
    u["mechanisms"].append(r["record_id"])
    u["levels"].add(r["level_main"])
    for k, key in (("gran", "dim_granularity"), ("attr", "dim_attribution"),
                   ("refl", "dim_reflection"), ("dom", "dim_domain"),
                   ("openn", "dim_openness"), ("veh", "dim_vehicle")):
        u[k].append(r[key])
    u["conf"][r["level_main"]] = r["confidence"]
    u["quotes"].append(r["evidence_quote"])

UNIT_ROWS = []
for uid, u in sorted(UNITS.items()):
    lvls = [l for l in u["levels"] if l in LEVEL_ORD]
    main = max(lvls, key=lambda l: LEVEL_ORD[l]) if lvls else "no-disclosure"
    multi = "+".join(sorted(u["levels"], key=lambda l: LEVEL_ORD.get(l, 99))) \
        if len(u["levels"]) > 1 else ""
    # openness / domain / vehicle: take from the highest-level mechanism(s)
    top = [r for r in R if r["unit_id"] == uid and r["level_main"] == main]
    UNIT_ROWS.append(dict(
        unit_id=uid, unit_name=u["unit_name"], corpus=u["corpus"], stage=u["stage"],
        n_mech=len(u["mechanisms"]), level_main=main, multi_level=multi,
        granularity=max_ord(u["gran"], GRAN_ORD),
        attribution=max_ord(u["attr"], ATTR_ORD),
        reflection=max_ord(u["refl"], REFL_ORD),
        domain=top[0]["dim_domain"] if len({t["dim_domain"] for t in top}) == 1
               else "/".join(sorted({t["dim_domain"] for t in top})),
        openness="released" if any(t["dim_openness"] == "released" for t in top)
                 else "not-released",
        vehicle=UNIT_PRIMARY_VEHICLE[uid],
        confidence=top[0]["confidence"]))

# Cohorts (pre-specified per 02 handoff: 2025 gen = R1/Llama-3-era mainstream
# recipes = corpus A SFT+preference+RL layers; 2026 gen = five flagships)
MAINSTREAM = [u for u in UNIT_ROWS if u["corpus"] == "A"
              and u["stage"] in ("SFT", "preference", "RL")]
ACADEMIC = [u for u in UNIT_ROWS if u["corpus"] == "A" and u["stage"] == "academic"]
PRETRAIN = [u for u in UNIT_ROWS if u["corpus"] == "A" and u["stage"] == "pretraining"]
FLAGSHIP = [u for u in UNIT_ROWS if u["corpus"] == "B"]


def level_counts(rows):
    c = {l: 0 for l in LEVELS}
    for u in rows:
        c[u["level_main"]] += 1
    return c


def median_level(rows):
    """Ordinal median: middle value of sorted levels (even n -> 'between X-Y')."""
    lv = sorted((LEVEL_ORD[u["level_main"]] for u in rows
                 if u["level_main"] in LEVEL_ORD))
    if not lv:
        return "n/a"
    name = lambda v: [k for k, x in LEVEL_ORD.items() if x == v][0]
    n = len(lv)
    if n % 2 == 1:
        return name(lv[n // 2])
    lo, hi = lv[n // 2 - 1], lv[n // 2]
    return name(lo) if lo == hi else "between %s-%s" % (name(lo), name(hi))


def pct(x, n):
    return "%.1f" % (100.0 * x / n) if n else "0.0"


# ---------------------------------------------------------------------------
# Fisher exact (exploratory only)
# ---------------------------------------------------------------------------
def fisher2(a, b, c, d, alt="two-sided"):
    st, p = fisher_exact([[a, b], [c, d]], alternative=alt)
    return p


def fisher_group_higher(s1, n1, s2, n2):
    """One-sided p for 'group 1 has a HIGHER rate of successes than group 2'.
    Orients the table so scipy's 'greater' (OR>1) tests the intended direction."""
    return fisher_exact([[s1, n1 - s1], [s2, n2 - s2]], alternative="greater")[1]


def fmt_p(p):
    return "<.001" if p < 0.001 else ("%.3f" % p).lstrip("0") if p < 0.9995 else "1.000"


# ===========================================================================
# BUILD ALL OUTPUT
# ===========================================================================
L = []

def say(s=""):
    L.append(s)
    print(s)

say("=" * 78)
say("FAILURE-DATA LIFECYCLE AUDIT — coding results (access date %s)" % ACCESS)
say("=" * 78)
say()

# ---- Table 1: corpus A level x stage --------------------------------------
say("TABLE 1 — Corpus A (dataset layer): lifecycle level distribution")
say("-" * 78)
stages = ["pretraining", "SFT", "preference", "RL", "academic"]
hdr = ["level (name)"] + stages + ["total A", "share of 28"]
say("| " + " | ".join(hdr) + " |")
cA = level_counts([u for u in UNIT_ROWS if u["corpus"] == "A"])
rows_t1 = []
for l in ["L0", "L1", "L2", "L3", "L4", "no-disclosure"]:
    cells = [sum(1 for u in UNIT_ROWS if u["corpus"] == "A"
                 and u["stage"] == s and u["level_main"] == l) for s in stages]
    row = ["%s (%s)" % (l, LEVEL_NAMES.get(l, "-"))] + \
          [str(x) for x in cells] + [str(cA[l]), pct(cA[l], 28) + "%"]
    rows_t1.append(row)
    say("| " + " | ".join(row) + " |")
say("| total | " + " | ".join(str(sum(1 for u in UNIT_ROWS if u["corpus"] == "A"
    and u["stage"] == s)) for s in stages) + " | 28 | 100% |")
say()
say("  L4 = 0 across ALL 33 units (explicit vacancy — supports C4).")
say("  Mainstream-recipe subset (SFT+preference+RL, n=%d): %s ; median=%s ; >=L3: %d (%s%%)"
    % (len(MAINSTREAM), {l: level_counts(MAINSTREAM)[l] for l in ["L0", "L1", "L2", "L3", "L4"]},
       median_level(MAINSTREAM),
       sum(1 for u in MAINSTREAM if u["level_main"] in ("L3", "L4")),
       pct(sum(1 for u in MAINSTREAM if u["level_main"] in ("L3", "L4")), len(MAINSTREAM))))
say("  SFT layer: %d/%d units at L0 (100%%)."
    % (level_counts([u for u in MAINSTREAM if u["stage"] == "SFT"])["L0"],
       len([u for u in MAINSTREAM if u["stage"] == "SFT"])))
say()

# ---- Table 2: corpus B matrix ----------------------------------------------
say("TABLE 2 — Corpus B (five 2026 flagships): level x sub-dimension matrix")
say("-" * 78)
say("| model | level | multi | granularity | attribution | reflection | domain | openness | vehicle | conf |")
for u in sorted(FLAGSHIP, key=lambda x: x["unit_id"]):
    say("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        u["unit_name"], u["level_main"], u["multi_level"] or "-",
        u["granularity"], u["attribution"], u["reflection"], u["domain"],
        u["openness"], u["vehicle"], u["confidence"]))
say()
cB = level_counts(FLAGSHIP)
say("  Corpus B: L2 x1 (GPT-6 Astra), L3 x4 ; >=L3 = 4/5 (80%%) ; L4 = 0 ; median = %s"
    % median_level(FLAGSHIP))
say("  Mechanism-level vehicles in B (n=%d): %s ; preference-pair = 0."
    % (sum(1 for r in R if r["corpus"] == "B"),
       {v: sum(1 for r in R if r["corpus"] == "B" and r["dim_vehicle"] == v)
        for v in ["audit-loop", "static-dataset", "RL-environment"]}))
say()

# ---- Table 3: sub-dimension distributions ----------------------------------
say("TABLE 3 — Sub-dimension distributions (unit level), Corpus A vs Corpus B")
say("-" * 78)
def dim_table(key):
    vals = sorted({u[key] for u in UNIT_ROWS})
    say("| %s | " % key + " | ".join(vals) + " |")
    say("| Corpus A (n=28) | " + " | ".join(
        str(sum(1 for u in UNIT_ROWS if u["corpus"] == "A" and u[key] == v))
        for v in vals) + " |")
    say("| Corpus B (n=5) | " + " | ".join(
        str(sum(1 for u in UNIT_ROWS if u["corpus"] == "B" and u[key] == v))
        for v in vals) + " |")
    say()
for key in ["granularity", "attribution", "reflection", "domain", "openness", "vehicle"]:
    dim_table(key)
say("  Three-element pass rates (locate=step/causal; attribute=binary/root-cause;")
say("  reflect=correction-pair/full-report), L3 units (n=14) vs non-L3 (n=19):")
for label, ok in (("locate  ", lambda u: u["granularity"] in ("step", "causal")),
                  ("attribute", lambda u: u["attribution"] in ("binary", "root-cause")),
                  ("reflect ", lambda u: u["reflection"] in ("correction-pair", "full-report"))):
    l3 = sum(1 for u in UNIT_ROWS if u["level_main"] == "L3" and ok(u))
    nl3 = sum(1 for u in UNIT_ROWS if u["level_main"] != "L3" and u["level_main"] != "no-disclosure" and ok(u))
    say("    %s: L3 %d/14 (%s%%) vs non-L3 %d/19 (%s%%)"
        % (label, l3, pct(l3, 14), nl3, pct(nl3, 19)))
say()

# ---- Table 4: generational contrast ----------------------------------------
say("TABLE 4 — Generational contrast (2025 mainstream recipes vs 2026 flagships)")
say("-" * 78)
c25, c26 = level_counts(MAINSTREAM), level_counts(FLAGSHIP)
say("| cohort | n | L0 | L1 | L2 | L3 | L4 | >=L3 | median |")
say("| 2025 mainstream (corpus A: SFT+pref+RL) | %d | %d | %d | %d | %d | %d | %d (%s%%) | %s |"
    % (len(MAINSTREAM), c25["L0"], c25["L1"], c25["L2"], c25["L3"], c25["L4"],
       sum(1 for u in MAINSTREAM if u["level_main"] in ("L3", "L4")),
       pct(sum(1 for u in MAINSTREAM if u["level_main"] in ("L3", "L4")), len(MAINSTREAM)),
       median_level(MAINSTREAM)))
say("| 2026 flagships (corpus B) | %d | %d | %d | %d | %d | %d | %d (%s%%) | %s |"
    % (len(FLAGSHIP), c26["L0"], c26["L1"], c26["L2"], c26["L3"], c26["L4"],
       sum(1 for u in FLAGSHIP if u["level_main"] in ("L3", "L4")),
       pct(sum(1 for u in FLAGSHIP if u["level_main"] in ("L3", "L4")), len(FLAGSHIP)),
       median_level(FLAGSHIP)))
say()
a = sum(1 for u in MAINSTREAM if u["level_main"] in ("L3", "L4"))  # 2025 >=L3
b = len(MAINSTREAM) - a
c = sum(1 for u in FLAGSHIP if u["level_main"] in ("L3", "L4"))    # 2026 >=L3
d = len(FLAGSHIP) - c
p2 = fisher2(a, b, c, d)
p1 = fisher_group_higher(c, len(FLAGSHIP), a, len(MAINSTREAM))
say("  EXPLORATORY Fisher exact, >=L3 x generation, [[%d,%d],[%d,%d]]: p(two-sided)=%s, p(one-sided, 2026>2025)=%s"
    % (a, b, c, d, fmt_p(p2), fmt_p(p1)))
say("  (Exploratory only: saturated purposive sample, n=5 flagships, ordinal scale.)")
say()

# ---- Table 5: L x three elements -------------------------------------------
say("TABLE 5 — Lifecycle level x three elements (all 33 units)")
say("-" * 78)
say("| level | n | locate | attribute | reflect |")
for l in ["L0", "L1", "L2", "L3", "L4"]:
    us = [u for u in UNIT_ROWS if u["level_main"] == l]
    say("| %s | %d | %d | %d | %d |" % (
        l, len(us),
        sum(1 for u in us if u["granularity"] in ("step", "causal")),
        sum(1 for u in us if u["attribution"] in ("binary", "root-cause")),
        sum(1 for u in us if u["reflection"] in ("correction-pair", "full-report"))))
say()
say("  EXPLORATORY Fisher exact (L3 vs non-L3), per element:")
for label, ok in (("locate", lambda u: u["granularity"] in ("step", "causal")),
                  ("attribute", lambda u: u["attribution"] in ("binary", "root-cause")),
                  ("reflect", lambda u: u["reflection"] in ("correction-pair", "full-report"))):
    x1 = sum(1 for u in UNIT_ROWS if u["level_main"] == "L3" and ok(u))
    x2 = 14 - x1
    y1 = sum(1 for u in UNIT_ROWS if u["level_main"] != "L3" and u["level_main"] != "no-disclosure" and ok(u))
    y2 = 19 - y1
    say("    %s: [[%d,%d],[%d,%d]] p(two-sided)=%s" % (
        label, x1, x2, y1, y2, fmt_p(fisher2(x1, x2, y1, y2))))
say()

# ---- Sensitivity ------------------------------------------------------------
say("SENSITIVITY CHECKS")
say("-" * 78)
# S1: drop low/medium-confidence boundary units
drop_ids = {"A-28", "A-01", "A-02", "A-15"}  # SPAR(low), pretraining x2, MIPO boundary
MS_s = [u for u in MAINSTREAM if u["unit_id"] not in drop_ids]
a1 = sum(1 for u in MS_s if u["level_main"] in ("L3", "L4"))
say("  S1 exclude boundary/low-conf units {%s}:" % ", ".join(sorted(drop_ids)))
say("     2025 cohort n=%d, >=L3 = %d (%s%%) — conclusion unchanged (still 0)."
    % (len(MS_s), a1, pct(a1, len(MS_s))))
c3 = sum(1 for u in FLAGSHIP if u["unit_name"].startswith("Claude") is False
         and u["level_main"] in ("L3", "L4"))
say("     2026 cohort: dropping the one medium-conf flagship (Claude) still gives")
say("     3/4 >=L3; Fisher p(one-sided) = %s"
    % fmt_p(fisher_group_higher(3, 4, 0, len(MS_s))))
# S2: cohort definition variants
v1 = MAINSTREAM + PRETRAIN
a2 = sum(1 for u in v1 if u["level_main"] in ("L3", "L4"))
say("  S2 cohort variant (pretraining included in 2025 gen): n=%d, >=L3=%d; Fisher p(one-sided)=%s"
    % (len(v1), a2, fmt_p(fisher_group_higher(4, 5, a2, len(v1)))))
aA = sum(1 for u in UNIT_ROWS if u["corpus"] == "A" and u["level_main"] in ("L3", "L4"))
say("  S3 whole-corpus contrast (all 28 A vs 5 B): A >=L3 = %d/28; Fisher p(one-sided)=%s"
    % (aA, fmt_p(fisher_group_higher(4, 5, aA, 28))))
say()
say("  Ordinal discipline: no means computed anywhere; medians and distributions only.")
say()

# ---- write CSVs --------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", path)

write_csv(os.path.join(OUT, "audit_records_mechanism.csv"),
          list(R[0].keys()), [[r[k] for k in r] for r in R])
write_csv(os.path.join(OUT, "audit_units.csv"),
          list(UNIT_ROWS[0].keys()), [[u[k] for k in u] for u in UNIT_ROWS])
write_csv(os.path.join(OUT, "table1_corpusA_levels.csv"), hdr, rows_t1)

# log to file
with open(os.path.join(OUT, "analysis_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

# ===========================================================================
# FIGURES (English text; 300 dpi)
# ===========================================================================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 1: lifecycle framework ---------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.set_xlim(0, 105)
ax.set_ylim(0, 42)
ax.axis("off")
box_info = [
    ("L0 — Discard", "Failures filtered out\nbefore training\n(rejection sampling)", "n=9", "#d9d9d9"),
    ("L1 — Shadow", "Failure only as the\nrejected side of a\npreference pair", "n=7", "#c6dbef"),
    ("L2 — Transient", "Failure rollouts enter\ngradient as negative\nadvantage, then discarded", "n=3", "#9ecae1"),
    ("L3 — Component", "Failures retained or\nprocessed: masked\nsegments, audit labels,\nreplay envs, reflections", "n=14", "#6baed6"),
    ("L4 — First-class", "Locate + attribute +\nreflect, at scale,\ntheorized, released", "n=0\n(vacant)", "#ffffff"),
]
BW, XSTEP = 18.4, 20.72
for i, (title, desc, n, color) in enumerate(box_info):
    x = 0.8 + i * XSTEP
    dashed = title.startswith("L4")
    box = FancyBboxPatch((x, 10), BW, 24,
                         boxstyle="round,pad=0.4,rounding_size=1.2",
                         linewidth=2.0 if dashed else 1.0,
                         linestyle="--" if dashed else "-",
                         edgecolor="#08519c", facecolor=color)
    ax.add_patch(box)
    ax.text(x + BW / 2, 30.5, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#08306b")
    ax.text(x + BW / 2, 22.5, desc, ha="center", va="center", fontsize=7.2, color="#333333")
    ax.text(x + BW / 2, 13.2, n, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="#08519c")
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + BW + 0.2, 22), (x + XSTEP - 0.2, 22),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color="#08519c", linewidth=1.4))
ax.text(52.5, 38.5, "Failure-Data Lifecycle Framework (training-data treatment of failures)",
        ha="center", fontsize=12, fontweight="bold")
ax.text(31, 4.6, "33 audited units (28 dataset-layer + 5 flagship pipelines); access date 2026-10-13",
        ha="center", fontsize=7.5, style="italic", color="#555555")
# vacancy annotation: vertical pointer to the BOTTOM EDGE of the L4 box
ax.annotate("L4 vacant across the entire audit\n(key finding: supports gap claim C4)",
            xy=(90.5, 9.9), xytext=(90.5, 3.0), fontsize=7.5, color="#a50f15",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#a50f15", lw=1.1,
                            shrinkA=2, shrinkB=0))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_lifecycle_framework.png"), dpi=300)
plt.close(fig)
print("wrote fig1")

# ---- Fig 2: level distributions by cohort ------------------------------------
cohorts = [("Mainstream recipes 2023–25\n(corpus A: SFT + pref + RL, n=15)", MAINSTREAM, "#f16913"),
           ("Academic exceptions\n(corpus A, n=11)", ACADEMIC, "#3182bd"),
           ("2026 flagships\n(corpus B, n=5)", FLAGSHIP, "#31a354")]
lvls = ["L0", "L1", "L2", "L3", "L4"]
xpos = range(len(lvls))
w = 0.26
fig, ax = plt.subplots(figsize=(8.6, 4.6))
for i, (lab, rows, col) in enumerate(cohorts):
    c = level_counts(rows)
    n = len(rows)
    vals = [100.0 * c[l] / n for l in lvls]
    bars = ax.bar([x + (i - 1) * w for x in xpos], vals, width=w * 0.92,
                  label=lab, color=col, edgecolor="white", linewidth=0.5)
    for bb, v, l in zip(bars, vals, lvls):
        cnt = c[l]
        if cnt > 0:
            ax.text(bb.get_x() + bb.get_width() / 2, v + 1.6, "%d" % cnt,
                    ha="center", fontsize=8, color="#333333")
ax.set_xticks(list(xpos))
ax.set_xticklabels(["L0\nDiscard", "L1\nShadow", "L2\nTransient", "L3\nComponent", "L4\nFirst-class"])
ax.set_ylabel("Share of units (%)")
ax.set_ylim(0, 100)
ax.set_title("Failure-data lifecycle levels by cohort: 2025 mainstream vs 2026 flagships", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.text(4.0, 50, "0%\nin all\ncohorts", ha="center", va="center", fontsize=8,
        color="#a50f15", fontweight="bold")
ax.text(0.99, 0.02, "Numbers above bars = unit counts. L4 = 0 across all 33 units (explicit vacancy).\n"
        "Pre-training units (n=2) excluded (failure concept not applicable). Ordinal scale — no means.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#555555")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_level_distribution.png"), dpi=300)
plt.close(fig)
print("wrote fig2")

print("\nDONE.")
