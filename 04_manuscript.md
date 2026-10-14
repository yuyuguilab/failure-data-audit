# Success Provides Direction, Failure Provides Precision: A Structured Audit of Failure Data in LLM Training

<!-- Editorial note (for submission-preparer; remove in camera-ready).
Primary title: "Success Provides Direction, Failure Provides Precision: A Structured Audit of Failure Data in LLM Training"
Alternate 1: "The Vacant Tier: Auditing the Lifecycle of Failure Data from Open-Source Recipes to 2026 Flagship Pipelines"
Alternate 2: "Learning from Mistakes, Except When Training: A Two-Layer Audit of Failure Data Across LLM Post-Training Practice"
-->

**Yugui Yu (YYG Strategy Lab)**

<!-- TODO(author): final author list, ORCID, and affiliations to be confirmed before submission. -->

*Position paper with a structured evidence audit. Draft of 2026-10-13; evidence window closes 2026-10-13. Target: arXiv preprint, cs.CL primary / cs.LG secondary.*

---

## Abstract

A durable intuition from human learning—students keep error notebooks because mistakes, once explained, sharpen far more than repeated successes—has no systematic counterpart in large language model (LLM) training data. We formalize this intuition as a position: success examples provide direction, while attributed failure examples provide precision, because each explained failure contracts the space of plausible policies. To measure how far current practice stands from this ideal, we audited how training pipelines actually treat failure data, coding 33 evidence units (28 open-source datasets and recipes; five flagship pipelines released in autumn 2026) with a five-level lifecycle codebook—L0 discard, L1 shadow, L2 transient, L3 component, L4 first-class—plus six sub-dimensions. The L4 tier, failure data combining error annotation, causal attribution, and reflective correction at scale under a public methodology, is vacant: 0/33 units. Mainstream 2023–25 recipes top out at transient gradients: 0/15 reach L3, and all seven SFT-stage units are pure L0. By contrast, four of five 2026 flagships operate explicit failure-data components (cohort median L3), and the failure-signal vehicle is migrating from preference pairs (6/15 mainstream units) to RL environments and audit loops (0/14 flagship mechanisms). Failure data with all three elements exists in exactly one unit (1/33). We argue that, within the audited disclosure layer, failure data is an unoccupied high ground, and we specify the components required to make it a first-class citizen.

**Keywords**: failure data; learning from mistakes; error-annotated data; post-training; RLVR; data curation

---

## 1. Introduction

Human learners have long institutionalized the study of failure: the error notebook, in which a student records each mistake, diagnoses its cause, and writes out the corrected reasoning, is a standard artifact of mathematics education in China and elsewhere. In July 2026, the vice president of Fudan University, Yungang Jiang, publicly argued at the World Artificial Intelligence Conference that failure data is as important for model capability as success data, explicitly invoking the error-notebook metaphor for foundation-model corpus construction [1]. The metaphor resonates because it names something the field practices in private and discards in public: every reinforcement learning (RL) run generates failures, every verifier flags them, and almost none of them survive as training assets.

This paper defends a precise version of that intuition. Our position is that **success provides direction, while failure provides precision**. In Bayesian terms, a correct demonstration confirms that some policy works; an *attributed* failure eliminates a region of the hypothesis space. Accumulated, diverse, well-explained failures contract the posterior toward the correct policy in a way that additional successes cannot replicate, since successes in the same region are largely redundant. The claim comes with quality caveats that we carry throughout: failures must be diverse and information-dense rather than homogeneous; they must be detoxified, since fake successes such as reward hacking are actively harmful; and coverage of failure modes, not any fixed success-to-failure ratio, is the operative variable.

The intuition has non-trivial cognitive antecedents, which we invoke as motivation rather than proof. The description–experience gap shows that judgments formed from sampled experience, including experienced losses, systematically differ from those formed from described probabilities, suggesting that experienced outcomes carry signal that descriptions do not [2]. Studies of the Iowa Gambling Task indicate that losses can be encoded implicitly, as somatic markers that precede conscious insight, and still guide choice [3]. We do not claim these results transfer to gradient descent; they motivate the hypothesis that explained failures carry information that successes do not.

Why does this matter now? Because generalization and calibration are currently among the weakest dimensions of frontier models, and the evidence points at training-data practice. Reinforcement learning with verifiable rewards (RLVR) appears to reweight existing capability rather than expand it: pass@1 rises while pass@k at large k does not surpass the base model [4]. Hallucination has been argued to be a structural property of the pretraining–posttraining interface, with evaluations that reward confident guessing reinforcing miscalibration [5]. Post-training itself damages calibration, with aligned models measurably more overconfident than their pretrained checkpoints [6], and RL-style objectives (PPO, GRPO, DPO) inducing overconfidence through reward exploitation [7]. Failure-driven learning is a candidate training signal for precisely these dimensions—learning where the boundary of one's competence lies—and controlled evidence already exists: fine-tuning on machine-annotated error–correction pairs lifted LLaMA-2-70B to 83.5% on GSM8K, surpassing GPT-4 at the time [8], and process reflection supervision turned a collapsing tool-use RL run into a +22.25-point gain with nonzero scores on double-unseen out-of-distribution scenarios where an SFT baseline scored exactly zero [9].

What is missing is an account of practice. Scattered technical-report disclosures—rejection sampling that discards failed trajectories, preference pairs whose rejected side silently carries failure, RL environments that replay observed weaknesses—have never been systematically compared, for want of a shared vocabulary for "how this pipeline treats failure data." That gap, between proven methodology and adopted methodology, is our subject.

We contribute four things:

- **A lifecycle codebook** that grades how a training pipeline treats failure data on an ordinal scale from L0 (discarded) to L4 (first-class), with six orthogonal sub-dimensions (Section 3), making "what happens to failures" comparable across papers and vendors for the first time.
- **A two-layer audit** applying this codebook to 33 evidence units—28 open-source datasets and mainstream recipes, and five 2026 flagship pipelines—yielding 44 mechanism-level coding records anchored to primary sources (Section 4).
- **A quantified generational shift**: mainstream 2023–25 recipes contain no unit at L3 or above (0/15) while four of five autumn-2026 flagships operate explicit failure components, with the failure-signal vehicle migrating from preference pairs to RL environments and audit loops (Section 5).
- **A roadmap** specifying the components—tripartite data schema, collection protocols, toxicity cleaning, verifier-dependence handling, open corpora, and evaluation—that first-class failure data still requires, with falsifiable monitoring indicators (Section 7).

The paper proceeds as follows. Section 2 organizes prior work into five families of failure signals plus the 2026 industry disclosure line, and states the gap. Section 3 presents the codebook. Section 4 details the audit method. Section 5 reports results against our four research questions. Section 6 discusses implications and limitations. Section 7 gives the roadmap; Section 8 concludes.

## 2. Background: Five Families of Failure Signals

We organize prior work not by paper but by the *role the failure signal plays in training*. Five families exist, plus a 2026 industry disclosure line; each holds a partial proof of the value of failure data, and none completes it.

### 2.1 Contrastive negatives: failure as decision boundary

Negative sampling is the oldest systematic use of "what is wrong." The survey literature reports that the hardest negatives carry the most information gain, with as little as 5% of the hardest negatives supporting high-precision contrastive learning [10]. Theory confirms that more negatives do not necessarily hurt downstream generalization [11], and negative-dominant objectives—upweighting negatives under imbalance—achieve the best Friedman ranks in domain-generalization studies (4.80, best among compared methods) [12]. This family demonstrates that failure signals can be the *dominant* learning signal, but it has remained in representation learning and never entered LLM post-training.

### 2.2 Process-supervision error annotation: failure as step-level label

A second family annotates *where* reasoning fails. PRM800K provides 800,000 step-level human ratings on math solutions, with erroneous steps explicitly marked [13]; Math-Shepherd derives step-correctness probabilities automatically at a scale of roughly 400,000 samples [14]. The evaluation side extends this: ProcessBench tests whether models can locate the first erroneous step among 3,400 solutions [15], and PRMBench probes six error types across about 11,000 samples [16]. This family solves localization but produces labels, not teaching material.

### 2.3 Error–correction fine-tuning: failure as teaching material

The family closest to the error notebook converts failures into lessons. LEMA fine-tunes on erroneous solutions paired with GPT-4-generated identification and correction of the mistakes—roughly 24k–50k pairs—and shows that such error–correction data is not homogeneous with chain-of-thought data, with larger gains for weaker models [8]. Process Reflection Supervision (PRS) is, to our knowledge, the strongest recent demonstration: it diagnoses why multi-step tool-use RL collapses, then has an auxiliary analyzer (gpt-5-mini) generate root-cause analyses and improvement plans for failed trajectories, trained jointly with an SFT loss; on BFCL-V3, PRS reaches an average of 25.75 versus 3.50 for vanilla GRPO (+22.25 points, with GRPO collapsing to zero in some settings), and on ACEBench scenarios unseen in both content and format, PRS scores 12.00 where the SFT baseline scores 0.00—the hardest existing evidence that reflective failure data produces *generalization rather than memorization* [9]. In embodied domains, AHA trains a vision–language model to detect and reason over robotic manipulation failures [17], and Foresight targets failure detection in long-horizon manipulation [18]; both stop at detection. The family's limitation is scale and adoption: tens of thousands of pairs, no mainstream recipe.

### 2.4 The rejected side of preference pairs: failure as relative signal

In preference alignment, failure survives only as the *rejected* member of a chosen–rejected pair, without independent semantics. Llama 3 constructs pairs by score gaps [19]; Qwen3 routes verifier-rejected responses to the rejected side or discards them [20]; Tulu 3 builds roughly 270,000 on-policy preference prompts [21]; HH-RLHF contributes about 170,000 human-written pairs [22]; UltraFeedback couples low scores with per-aspect critiques [23]; SmolLM2 performs DPO over low-rated responses [24]. MIPO, the mixed preference optimization used in GLM-4.5's expert-merging stage, mixed preference data of differing strengths including graded negative samples—the closest the preference era came to systematic negative-signal retention [25]. Our audit codes MIPO as L1 (Section 5.1); its retirement in GLM 5.x (Section 5.2) marks the vehicle's decline. NPO formalizes unlearning as negative preference optimization, repurposing "forget" sets as negatives [26]. Across the family, failure never appears as an absolute, teachable signal.

### 2.5 RL failure signals: failure as gradient component

In RL, failed rollouts contribute negative advantage and are then discarded—used transiently, never released. This is the GRPO pattern in DeepSeek-R1 [27]. More telling, discarding has been institutionalized at the algorithm level: DAPO's dynamic sampling explicitly removes prompt groups in which *all* rollouts fail, framing zero-gradient groups as compute waste [28]. On the agent side, ETO pairs failed and successful trajectories for contrastive training [29], and UltraInteract packs error-containing tree trajectories with reflections into preference pairs [30]. The family shows failure as a live but unarchived signal.

### 2.6 The 2026 industry disclosure line

Between August and October 2026, first-party disclosures of five flagship pipelines revealed explicit failure-data mechanisms: GPT-6 Astra routes failure signals to monitoring and evaluation rather than training data, with RL penalties on cheating as the only training-side use [31]; Claude Opus 5.5's Alignment Intelligence Report drives training improvements from more than 7,200 collected agent failure scenarios, within the alignment domain [32]; GLM-5.3 retains erroneous trajectory segments under loss masks and selects RL tasks that the previous generation persistently fails [33]; MiMo-V2.6 synthesizes reflection-and-revision training samples from real reward-hacking failures [34]; and DeepSeek-V4.1-Flash collects employee-submitted failure cases at scale and replays them through targeted RL [35]. Policy-side attention exists [1], but the corpus establishment has not followed: the Chinese foundational corpus release of September 2026 contains no failure-corpus track [36].

### 2.7 The gap

Each family holds a piece of the proof: contrastive negatives establish that failure can dominate learning (but only in representation space); process supervision establishes localization (but not correction); error–correction fine-tuning establishes teaching value (but at 10^4–10^5 scale, outside mainstream recipes); the rejected side institutionalizes failure in its weakest form; RL uses failure transiently and discards it. The 2026 disclosure line shows industrial components arriving, unnamed and unreleased. No family, recipe, or vendor—as of 2026-10-13 and within disclosed practice—treats failure data combining error localization, causal attribution, and reflective correction as a first-class, scaled, public training methodology. The audit below measures exactly this gap.

## 3. A Lifecycle Codebook for Failure Data

The **unit of coding** is a failure-handling mechanism in a training pipeline or dataset, anchored to a citable primary source; a unit with multiple mechanisms (e.g., DAPO) receives one mechanism record per mechanism, and its main code is the highest level among them. By *failure* we mean any sample or trajectory that the system itself judges erroneous, unverified, or rejected. The **main code** grades the unit on an ordinal lifecycle scale, with criteria in universal-quantifier form (all listed criteria must hold):

- **L0 — Discard.** Failed trajectories are filtered before training (verification, rejection sampling, or filtered release); the release contains no failure instances; no failure processing exists. *Example:* DeepSeek-R1's third-stage rejection sampling keeps only correct, language-consistent, high-quality trajectories [27]. *Boundary:* OpenMathInstruct-2 ships an `expected_answer` field that enables right/wrong judgment; a field enabling judgment is not a retained failure instance, so the unit codes L0 [37].
- **L1 — Shadow.** Failure exists only as the rejected side of preference pairs, with no independent error semantics, and is not separately released. *Example:* HH-RLHF pairs [22]. *Boundary:* UltraInteract's rejected side carries explicit reflection text, which satisfies an L3 criterion; the unit is coded multi-level L1+L3 with main code L3 [30].
- **L2 — Transient.** Failed rollouts contribute gradient (negative advantage or penalty) during training, are discarded afterward, and are never released. *Example:* GRPO's failed rollouts [27]. *Boundary:* DAPO exhibits in-group negative advantage (L2) *and* explicit removal of all-failure groups (an L0 feature); the mechanisms are split, and the unit's main code is L2 [28].
- **L3 — Component.** Failure is explicitly retained or processed into a recognizable component—loss masks, audit labels, replay environments, synthetic correction samples, or reflection reports—but the component is localized or domain-limited, and is not released or theorized as public methodology. *Example:* GLM-5.3's error-segment masking [33]. *Boundary:* OpenR1-Math-Raw releases roughly one million raw trajectories including unverified ones—the highest-openness L3 in our corpus—but the three elements are incomplete, so it does not reach L4 [38].
- **L4 — First-class.** Error annotation, causal attribution, and reflective correction are all present; the data operates at scale (10^5+ samples or as an industrial-pipeline core); it is explicitly named and theorized as methodology; and the data or methodology is public. The codebook anticipated that L4 would be empty; the vacancy, once confirmed, is itself a finding rather than a coding failure.
- **No-disclosure.** For flagship pipelines, absence of failure-data disclosure is coded as *no-disclosure*, not L0: L0 requires an explicit statement of filtering. In the final corpus, zero units fell into this category—all 33 units could be leveled from the locked corpus.

Six **sub-dimensions** are coded independently at the highest verifiable form: error granularity (none / trajectory / step / causal), attribution (none / binary model-versus-environment / root-cause), reflection (none / correction-pair / full reflective report), domain (capability / alignment), openness (released / not-released), and vehicle (static dataset / preference pair / RL environment / audit loop). The first three—localize, attribute, reflect—constitute the *three elements* whose joint presence defines the error-notebook form. Figure 1 renders the framework with unit counts; the L4 tier is drawn vacant.

**Figure 1.** The failure-data lifecycle framework (L0–L4). Unit counts from the audit of 33 units (28 dataset-layer + 5 flagship pipelines); the L4 tier is vacant across the entire audit—itself the paper's key finding. (`figures/fig1_lifecycle_framework.png`)

Two coding disciplines apply throughout. First, L0–L4 is ordinal: we report frequencies and medians, never means. Second, "not disclosed" never implies "does not exist": disclosure asymmetry is treated as a validity threat, not evidence (Section 6.5).

## 4. Audit Method

### 4.1 Design and corpora

The study is a document audit of publicly available sources; it involves no human subjects. Evidence was assembled in two layers, locked as audit-v1.0 on 2026-10-13.

**Corpus A (dataset layer, n=28)** covers open training recipes and datasets across five stages, each anchored to a primary source: pre-training—Dolma [39], FineWeb [40]; SFT/distillation—DeepSeek-R1 [27], Llama 2 [41], Llama 3 [19], OpenR1-Math-220k [42], Bespoke-Stratos [43], OpenMathReasoning [44], OpenMathInstruct-2 [37]; preference alignment—Llama 3 DPO [19], Qwen3 [20], Tulu 3 [21], HH-RLHF [22], SmolLM2 [24], GLM-4.5 MIPO [25]; RL—GRPO as implemented in R1 [27], DAPO [28]; and academic exceptions—PRM800K [13], Math-Shepherd [14], LEMA [8], ETO [29], UltraInteract [30], UltraFeedback [23], ProcessBench [15], PRMBench [16], OpenR1-Math-Raw [38], NPO's unlearning sets [26], and SPAR f26 [45]. Inclusion required a verifiable primary document (paper, official data card, or official repository), community adoption or methodological representativeness, and disclosure no later than 2026-10-13. Units supported only by media reports were excluded; one widely circulated embodied failure-analysis dataset could not be located in primary sources after multiple targeted searches and was removed from the evidence chain entirely—embodied failure reasoning is instead cited through verifiable work [17].

**Corpus B (pipeline layer, n=5)** covers flagships released after Jiang's July 2026 remarks, each anchored to first-party documentation: GPT-6 Astra's system card [31], Claude Opus 5.5's system materials [32], the GLM-5 technical report (anchor for GLM-5.3) [33], MiMo-V2.6 [34], and DeepSeek-V4.1-Flash [35]. This is saturation purposive sampling—the five are the recognized first tier of closed and open flagships—not probability sampling; conclusions are accordingly limited to *disclosed practice*. A literature corpus (the five families above) and an arXiv re-check over August–October 2026 complete the materials.

### 4.2 Procedure

Coding followed the pipeline: locate → include/exclude → primary-source verification → mechanism-level coding → reliability recheck → data lock. Each of the 44 mechanism records (30 in Corpus A, 14 in Corpus B) carries a verbatim or excerpt-level quote with section anchor, source tier, confidence rating, and conflict-rule citations. Five conflict rules govern judgments: multi-mechanism units take the highest level as main code (CR1); primary sources override secondhand ones (CR2); versions of the same vendor are split into separate records (CR3); undecidable criteria yield low confidence and never inference (CR4); and no-disclosure is never coded L0 (CR5). For Corpus B, every code at L2 or above is anchored to a verbatim quote of at most 50 words.

### 4.3 Verification corrections

Because our upstream evidence chain contained two propagating errors, we enforced a verification gate on all load-bearing numbers. PRS is cited from the verified primary source (arXiv:2606.26027), with the analyzer model corrected to gpt-5-mini and gains reported as +22.25 points (25.75 versus 3.50), never converted into a ratio multiple [9]. DAPO is cited under its correct identifier (arXiv:2503.14476) [28]. All headline numbers in this paper trace to the locked analysis and were re-checked against that gate.

### 4.4 Reliability and quality gates

All 33 units were coded by a single coder under the codebook's universal-quantifier criteria. To mitigate single-coder risk, the codebook, the boundary-case registry (14 registered cases), and the full mechanism-level coding table are released with the paper, so every judgment can be independently re-adjudicated; a blind intra-rater recoding of a stratified 20% sample (six Corpus-A units, one full flagship, one literature family) is scheduled for 2026-10-27, with raw agreement and Cohen's κ to be reported (target κ ≥ .70; **κ = [to be inserted after 2026-10-27]**). Secondary-tier evidence (news reporting, e.g., [1]) is confined to motivation and never supports a claim.

### 4.5 Validity threats

We flag five threats—disclosure bias in both directions, the single coder, the ordinal scale, the truncated time window, and excerpt-level quotes—and treat them in Section 6.5.

## 5. Results

Across 33 units and 44 mechanism records, the full-corpus distribution is L0 ×9 (27.3%), L1 ×7 (21.2%), L2 ×3 (9.1%), L3 ×14 (42.4%), L4 ×0, with zero units in no-disclosure and an ordinal median of L2. We report results by research question: dataset layer (RQ2), pipeline layer (RQ3), and the vacancy scan (RQ4); RQ1 (effectiveness) is literature-based and was treated in Sections 2.1–2.3.

### 5.1 The dataset layer: systematic demotion of failure data

**Table 1. Corpus A (dataset layer): failure-data lifecycle level × training stage (n=28 units).**

| Level | Pre-train | SFT | Preference | RL | Academic | Total (%) |
|---|---|---|---|---|---|---|
| L0 Discard | 2 | 7 | 0 | 0 | 0 | **9 (32.1%)** |
| L1 Shadow | 0 | 0 | 6 | 0 | 1 | **7 (25.0%)** |
| L2 Transient | 0 | 0 | 0 | 2 | 0 | **2 (7.1%)** |
| L3 Component | 0 | 0 | 0 | 0 | 10* | **10 (35.7%)** |
| L4 First-class | 0 | 0 | 0 | 0 | 0 | **0 (0.0%)** |
| No-disclosure | 0 | 0 | 0 | 0 | 0 | **0** |
| **Total** | 2 | 7 | 6 | 2 | 11 | **28** |

\* Includes two evaluation-side reference units (ProcessBench, PRMBench); excluding them, L3 = 8/28.

**Figure 2.** Distribution of failure-data lifecycle levels by cohort: mainstream recipes 2023–25 (n=15), academic exceptions (n=11), and 2026 flagships (n=5). Numbers above bars are unit counts; L4 = 0 in all cohorts. Pre-training units excluded (failure concept not applicable). Ordinal scale — no means reported. (`figures/fig2_level_distribution.png`)

Table 1 and Figure 2 show a clean stratification by stage. **All seven SFT/distillation units are L0**: rejection-sampling-style discarding of failed trajectories is the default operation of 2023–25 SFT data production (R1, Llama 2/3, OpenR1-Math, Bespoke-Stratos, OpenMathReasoning, OpenMathInstruct-2). The preference stage is uniformly L1—six of six units, including MIPO, which grades negative-sample strength but performs no localization, attribution, or reflection, and is therefore best described as the upper bound of the shadow tier rather than a component design. The RL stage reaches L2; DAPO records multi-level L0+L2, with its all-failure-group removal constituting algorithm-level evidence of institutionalized discarding. Restricting to the mainstream subset (SFT + preference + RL, n=15): **0/15 units reach L3, and the subset's ordinal median is L1**. Every L3 unit sits in the academic-exception layer (10/11, with SPAR f26 coded at low confidence)—the community demonstrably knows how to build failure-rich datasets, and just as demonstrably does not ship them in mainstream recipes. Two boundary readings deserve note: the two evaluation-side L3 units evidence that localization- and attribution-type data exist, but for evaluation, not training—we code data form, not use equivalence (the OpenMathInstruct-2 boundary reading appears in Section 3).

### 5.2 The pipeline layer: a generational shift, unnamed

**Table 2. Corpus B (five 2026 flagships): lifecycle level × sub-dimensions (n=5).**

| Model (release) | Main | Multi | Granularity | Attribution | Reflection | Domain | Openness | Vehicle | Conf. |
|---|---|---|---|---|---|---|---|---|---|
| GPT-6 Astra (09-03) | **L2** | L0+L2 | trajectory | none | none | alignment | not-released | audit-loop | high† |
| Claude Opus 5.5 (09-22) | **L3** | — | trajectory | binary | none | alignment | not-released | audit-loop | medium |
| GLM-5.3 (2026 H2) | **L3** | L0+L3 | **step** | binary | none | capability | not-released | static-dataset | high |
| MiMo-V2.6 (09-22) | **L3** | — | **step** | binary | **full-report** | alignment | not-released | static-dataset | high |
| DeepSeek-V4.1-Flash (09-09) | **L3** | — | trajectory | **root-cause** | none | capability | not-released | RL-environment | high |

† Unit confidence from the main-code mechanism (RL penalty, verbatim from the system card); monitoring/evaluation-routing mechanisms are medium (excerpt-level quotes).

Table 2 locates each flagship. **Four of five units are L3; the cohort median is L3.** GPT-6 Astra is the sole sub-L3 flagship: its only training-side use of failure is a transient RL penalty—"penalize cheating and unauthorized actions" [31]—while failure signals are explicitly routed to monitoring and evaluation (coded as L0 mechanisms under CR5, since the routing is stated). GLM-5.3 masks rather than deletes errors: "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors" [33], alongside notebook-style RL task selection against persistently-failed tasks and binary attribution separating model error from environment crash. MiMo-V2.6 is the only unit with all three elements, synthesizing training samples from real reward-hacking failures in a form that "keeps the original error recognizable and makes the correction explicit" [34]. DeepSeek-V4.1-Flash is structurally closest to L4: "we collect negative feedback and model failure cases submitted by internal employees at scale," feeding "systematic replay of failures and targeted RL against observed model weaknesses" [35]—organized collection, root-cause attribution, and capability-domain scope. Claude Opus 5.5 drives training improvements from 7,200+ collected agent failure scenarios, but within the alignment domain only [32] (excerpt-level evidence; paraphrased here).

**Table 3. Sub-dimension distributions (unit level), Corpus A vs Corpus B.**

| Sub-dimension | Values | Corpus A (n=28) | Corpus B (n=5) |
|---|---|---|---|
| Granularity | none / trajectory / step / causal / n.d. | 3 / 18 / 6 / **0** / 1 | 0 / 3 / 2 / **0** / 0 |
| Attribution | none / binary / root-cause / n.d. | 26 / **0** / 1 / 1 | 1 / 3 / 1 / 0 |
| Reflection | none / correction-pair / full-report / n.d. | 23 / 4 / **0** / 1 | 4 / 0 / **1** / 0 |
| Domain | capability / alignment / both / n.d. | 23 / 1 / 2 / 2 | 2 / 3 / 0 / 0 |
| Openness | released / not-released | **11 / 17** | **0 / 5** |
| Vehicle | none / pref-pair / static / RL-env / audit-loop | 9 / 8 / 9 / 2 / 0 | 0 / **0** / 2 / 1 / 2 |

Table 3 shows the *vehicle migration*, our second structural finding. In the 2025 mainstream cohort, 6/15 units carry failure via preference pairs; across the 14 mechanism records of the five flagships, **preference-pair carrying is 0** (audit-loop ×4, static-dataset ×5, RL-environment ×5). The same-vendor record is decisive on version dynamics: MIPO's graded negatives (L1) were retired in GLM 5.x, whose post-training chain contains no preference-optimization stage [25,33]; and DeepSeek's own trajectory runs from R1's rejection sampling (2025-01, L0) to V4.1-Flash's failure replay (2026-09, L3), with the rejection-sampling stage gone [27,35].

**Table 4. Generational contrast: 2025 mainstream recipes vs 2026 flagships.**

| Cohort | n | L0 | L1 | L2 | L3 | L4 | ≥L3 | Median |
|---|---|---|---|---|---|---|---|---|
| 2025 mainstream (SFT+pref+RL) | 15 | 7 (46.7%) | 6 (40.0%) | 2 (13.3%) | 0 | 0 | **0 (0.0%)** | L1 |
| 2026 flagships | 5 | 0 | 0 | 1 (20%) | 4 (80%) | 0 | **4 (80.0%)** | L3 |

Table 4 formalizes the shift; an exploratory Fisher exact test gives *p* = .001 (two-sided; identical one-sided). We stress the caveats: saturated purposive sampling, n = 5, ordinal data, no randomization—the distribution and the anchored quotes, not the test, carry the argument. Sensitivity checks bound the claim honestly: dropping low-confidence and boundary units leaves 0/14 at ≥L3 for the mainstream cohort and 3/4 for flagships (one-sided *p* = .005); adding pre-training units to the 2025 cohort changes nothing (*p* < .001); but comparing *all* 28 Corpus-A units against the flagships yields 10/28 at ≥L3 and *p* = .089 (one-sided)—the generational contrast's strength depends on the "mainstream recipe" framing, since the academic-exception layer is by construction a counterexample layer. We therefore claim a generational shift in *mainstream disclosed practice*, not a population estimate.

### 5.3 The vacancy: three elements and openness

**Table 5. Lifecycle level × three elements (all 33 units).**

| Level | n | Locate (step/causal) | Attribute (binary/root-cause) | Reflect (pair/full-report) |
|---|---|---|---|---|
| L0 | 9 | 0 | 0 | 0 |
| L1 | 7 | 0 | 0 | 0 |
| L2 | 3 | 0 | 0 | 0 |
| L3 | 14 | 8 (57.1%) | 5 (35.7%) | 5 (35.7%) |
| L4 | 0 | — | — | — |

Table 5 shows that the three elements co-vary with level—strictly concentrated in L3 (e.g., localization 8/14 vs 0/19 elsewhere; exploratory Fisher *p* = .00022; attribution and reflection 5/14 vs 0/19, *p* = .008)—but are jointly satisfied by **exactly one unit of 33 (MiMo-V2.6)**, and even within L3 by one of fourteen. The elements remain individually scarce: causal-granularity localization is absent corpus-wide (0/33); root-cause attribution exists in one evaluation-side unit in Corpus A (PRMBench) and one flagship (DeepSeek); full reflective reports exist in one unit (MiMo). Openness splits the corpus along an institutional fault line: the 11 units whose failure data is released are *all* on the academic side (10 academic exceptions plus HH-RLHF's released rejected sides), while the flagships release none (0/5)—MiMo's publication of five reward-hacking failure trajectories is exemplar-level disclosure, and its open-sourcing of roughly 7,000 RL environments is not a release of failure data.

The L4 scan across both corpora returns **0/33**, with the four criteria failing in a structured way: criterion (i), the three elements, is met only by MiMo; criterion (ii), scale, is undisclosed for MiMo; criterion (iii), explicit theorization as methodology, is met by no vendor—DeepSeek frames the entire apparatus as engineering: "the marginal return of engineering the data and environment pipeline substantially exceeds that of algorithmic novelty" [35]; and criterion (iv), public release, is met by no flagship. The nearest misses are LEMA (elements largely present and released, but at 10^4 scale and outside mainstream recipes) and DeepSeek-V4.1-Flash (organized scale and root-cause attribution, but unnamed and unreleased). Finally, the procedural corroboration: the August–October 2026 arXiv re-check surfaced no large-scale failure-dataset or industrial failure-reflection methodology paper; adjacent hits concern failure detection [18].

## 6. Discussion

### 6.1 A vacancy that is a finding

The L4 tier was expected to be empty when the codebook was written, and the audit confirms it systematically. The situation is therefore not "failure data does not work"—the methodological families supply controlled positive evidence (Section 2)—nor "nobody wants it": a senior policy figure named it publicly in July 2026 [1], and four of five flagship pipelines now contain failure components. Within the audited disclosure layer and the 2026-10-13 evidence window, first-class failure data is an *unoccupied high ground*: proven at small scale, named in public discourse, demanded by the weakest model dimensions, and claimed by no one as methodology. The vacancy is falsifiable and time-stamped: a public release within twelve months that satisfies all four L4 criteria would refute it, and we propose exactly that as a monitoring indicator.

### 6.2 The best vehicle may be environments, not datasets

The vehicle migration (Table 3) suggests the field has voted with its infrastructure. The preference-pair vehicle—the shadow form—did not survive its era: MIPO, its most systematic design, was retired rather than scaled [25]. The three flagship carriers each embody one element: MiMo's reflection-revision data carries reflection, DeepSeek's replay environments carry attribution-driven targeting, GLM's masked segments carry localization. It is plausible that the terminal form of failure data is not a static annotated corpus but a *closed loop*—collect, attribute, replay, retrain—as DeepSeek's disclosures imply [35]. We hold a dual position. For industry pipelines, environments plus replay may indeed dominate: they keep failure data on-policy and current. But closed loops are private by construction; if the community is ever to study failure-data methodology cumulatively, static, openly licensed, tripartite-annotated corpora remain the only shareable form—and PRS demonstrates that reflective *text* data, trainable with a plain SFT loss, already works [9]. Both forms, and a bridge between them, belong in the roadmap.

### 6.3 Implications for the weakest dimensions

Our audit measures practice, not effects, and we resist causal inference beyond the evidence. The alignment of incentives is nonetheless notable: generalization and calibration, where current training is weakest, are precisely where failure data is a candidate signal, by the posterior-contraction argument of Section 1. RLVR appears to sharpen pass@1 without expanding pass@k [4]; hallucination behaves as a calibration failure reinforced by confident-guessing rewards [5]; post-training measurably damages confidence calibration [6,7]. Training on attributed failures is, formally, training on the *boundary of competence*—the region where a well-calibrated model must decline. The strongest existing hint is PRS's double-unseen result (12.00 vs 0.00) [9]; establishing the generalization–calibration payoff of failure data at scale is an open experimental program, which our roadmap's evaluation component is designed to seed.

### 6.4 Verifier dependence

Failure data inherits its definition from verifiers, and verifiers can be gamed. Reward hacking is a *fake success*—toxic to train on as either positive or negative without causal attribution. The flagships' components are largely answers to exactly this problem: MiMo's online grader jointly inspects successful and failed rollouts and zeroes the reward of confirmed-hacked passes, converting them to failures [34]; GLM closes reward shortcuts in solver trajectories and separates model error from environment collapse [33]; DeepSeek records hacking and environment crashes as failure trajectories fed back to the RL framework [35]. The open problem is the no-verifier domain: where success cannot be checked automatically, failure attribution requires new instruments, without which the methodology cannot leave verifiable domains such as mathematics and code.

### 6.5 Limitations

Five limitations bound our claims. (1) *Disclosure bias, both directions*: undisclosed failure practice would make our L-levels underestimates; vendor embellishment would make components overstated. All conclusions are therefore scoped to disclosed practice, and we release the anchoring quotes so readers can re-adjudicate. (2) *Single coder*: the audit is one coder's application of a public codebook; the intra-rater κ is pending (Section 4.4), and no inter-rater reliability exists. (3) *Ordinal discipline*: L0–L4 supports medians and distributions only; we report no means or interval statistics. (4) *Time truncation*: the corpus is an autumn-2026 snapshot of a moving target; units are version-anchored, and the generational contrast is cohort-framed (the full-corpus sensitivity at *p* = .089 marks the boundary of the statistical claim). (5) *Quote confidence*: verbatim quotes anchor all high-confidence Corpus-B codes; three Corpus-B mechanisms and most Corpus-A anchors rest on excerpt-level evidence, which we paraphrase rather than quote.

## 7. Roadmap: Making Failure Data a First-Class Citizen

The audit's vendor mapping (Table 6) shows which components exist and which are missing; none of the five vendors assembles them all, and the open-corpus component is vacant across the entire industry.

**Table 6. Roadmap components × flagship coverage (from the audit's sub-dimension records).**

| Component | MiMo | DeepSeek | GLM-5.3 | Claude | GPT-6 |
|---|---|---|---|---|---|
| (a) Tripartite collection/annotation protocol | partial (reflection-revision samples, hacking domain only) | partial (employee submission + classification, no reflective text) | partial (masked retention, no reflection) | none | none |
| (b) Toxicity cleaning & causal attribution | yes (hacked passes zeroed; infrastructure failures removed) | yes (hacking/crash recorded as failures) | partial (binary model-vs-environment removal) | partial (flawed-environment attribution) | not disclosed |
| (c) Verifier-dependence handling | yes (joint success/failure grading) | partial (inspection agent flags) | yes (reward shortcuts closed) | not disclosed | not disclosed |
| (d) Open corpus & shared protocols | none (open environments ≠ open failure data) | none | none | none | none |

From this mapping we specify the missing infrastructure:

1. **Tripartite data schema.** Failure records as first-class objects with three mandatory fields—error type (localized at step or causal granularity), root cause, and correction text—together with provenance (task, policy version, verifier verdict). The schema is vehicle-neutral: it applies to static datasets, preference-pair extensions, and environment logs.
2. **Collection and annotation protocols.** Who collects failures (users, employees, rollouts), how they are sampled for diversity rather than volume, and which analyzer annotates them—with the analyzer model disclosed, a detail our verification chain shows to be load-bearing [9].
3. **Toxicity cleaning and causal attribution.** A mandatory pre-training stage that separates fake successes (reward hacking) from infrastructure failures from valuable model failures, following the flagship precedents (Section 6.4).
4. **Verifier-dependence handling.** Protocols for dual-grading passed-but-suspect trajectories, and a research program for failure attribution in no-verifier domains.
5. **Open corpus and shared protocols.** The ecosystem's single large-scale release retaining failures is OpenR1-Math-Raw (~10^6 raw trajectories, L3 with openness=released but elements incomplete) [38]; we propose community tripartite annotation layers over exactly such raw seeds, with licenses and provenance conventions—activity that no vendor currently undertakes (Table 6d).
6. **An evaluation benchmark.** Controlled comparisons of models trained with and without matched failure data, scored on out-of-distribution generalization (double-unseen designs, following [9]) and on calibration (confidence–correctness alignment), plus "failure literacy" tasks measuring localization and attribution (extending [15,16]).

Falsifiable predictions follow from the audit: no L4 release within twelve months would extend the vacancy; a continued preference-pair share near zero would confirm the vehicle migration; and the first vendor to theorize its failure pipeline as methodology would occupy the vacant tier.

## 8. Conclusion

We audited how LLM training practice treats failure data, and found a structured paradox. The methodology is proven at small scale and the discourse has named it, yet across 33 audited units the highest lifecycle tier is empty (0/33), the 2023–25 mainstream never rose above transient failure gradients (0/15 at ≥L3; all SFT units pure discard), and the 2026 flagships—which do operate failure components (4/5 at L3)—keep them local, unnamed, and unreleased, having already abandoned the preference-pair vehicle that once carried failure implicitly (0/14 flagship mechanisms). Success gives direction; failure gives precision; current practice, within everything it discloses, throws the precision away. The audit instruments—a lifecycle codebook, a boundary-case registry, and open coding records—are released with this paper so that the vacancy can be monitored, and refuted, in public.

*Audit materials availability.* The codebook, the 14-case boundary registry, the mechanism-level coding table (44 records), the reproduction script, and both figures are packaged as supplementary materials accompanying this manuscript.

## References

*(Numbered in order of first citation; exported in BibTeX as `references.bib`. arXiv identifiers follow the verified citation chain of Section 4.3.)*

1. Yungang Jiang, remarks at the WAIC 2026 "Corpus Foundations" forum, reported by People's Daily Online, 2026-07-19. http://sh.people.com.cn/n2/2026/0719/c134768-41643408.html (secondary-tier source; motivation only).
2. R. Hertwig and I. Erev. The description–experience gap in risky choice: A review. *Trends in Cognitive Sciences*, 13(12):517–523, 2009.
3. A. Bechara, H. Damasio, D. Tranel, and A. R. Damasio. The Iowa Gambling Task and the somatic marker hypothesis. *Journal of the International Neuropsychological Society*, 11(1):37–42, 2005. *(Bibliographic details unverified; to be confirmed.)*
4. Y. Yue, Z. Chen, R. Lu, et al. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? *NeurIPS*, 2025. arXiv:2504.13837.
5. A. T. Kalai, O. Nachum, S. S. Vempala, and E. Zhang. Why language models hallucinate. arXiv:2509.04664, 2025.
6. G. He, P. Cui, and J. Chen. Investigating uncertainty calibration of aligned language models under the multiple-choice setting. arXiv:2310.11732, 2023.
7. X. Xie, X. Liu, and B. Yao. Know when you're wrong: Aligning confidence with correctness for LLM error detection. arXiv:2603.06604, 2026.
8. S. An, Z. Ma, Z. Lin, et al. Learning from mistakes makes LLM better reasoner. arXiv:2310.20689, 2023.
9. Y. Hao, Z. Jin, H. Liao, K. Liu, and J. Zhao. Why multi-step tool-use reinforcement learning collapses and how supervisory signals fix it. arXiv:2606.26027, 2026. (Key Laboratory of Cognition and Decision Intelligence Complex Systems, Institute of Automation, CAS.)
10. Z. Yang, M. Ding, T. Huang, et al. Does negative sampling matter? A review with insights into its theory and applications. arXiv:2402.17238, 2024. *(TPAMI version per upstream records; volume unverified.)*
11. P. Awasthi, N. Dikkala, and P. Kamath. Do more negative samples necessarily hurt in contrastive learning? *ICML*, 2022. arXiv:2205.01789.
12. M. Cao, et al. Negatives-dominant contrastive learning for generalization in imbalanced domains. *ICML*, 2026. arXiv:2601.21999. *(Full author list to be completed.)*
13. H. Lightman, V. Kosaraju, Y. Burda, et al. Let's verify step by step. arXiv:2305.20050, 2023. (PRM800K.)
14. P. Wang, L. Li, Z. Shao, et al. Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. *ACL*, 2024. arXiv:2312.08935.
15. C. Zheng, Z.-R. Zhang, B. Zhang, et al. ProcessBench: Identifying process errors in mathematical reasoning. *ACL*, 2025. arXiv:2412.06559.
16. M. Song, et al. PRMBench: A fine-grained and challenging benchmark for process-level reward models. arXiv:2501.03124, 2025. *(Full author list to be completed.)*
17. J. Duan, W. Pumacay, N. Kumar, et al. AHA: A vision-language-model for detecting and reasoning over failures in robotic manipulation. arXiv:2410.00371, 2024.
18. H. Zhang, Y. Lu, B. Wang, et al. Foresight: Failure detection for long-horizon robotic manipulation with action-conditioned world model latents. arXiv:2606.23085, 2026.
19. A. Grattafiori, et al. The Llama 3 herd of models. arXiv:2407.21783, 2024.
20. Qwen Team. Qwen3 technical report. arXiv:2505.09388, 2025.
21. N. Lambert, et al. TÜLU 3: Pushing frontiers in open language model post-training. arXiv:2411.15124, 2025.
22. Anthropic. HH-RLHF: Human preference data regarding helpfulness and harmlessness. Dataset, 2022. https://huggingface.co/datasets/Anthropic/hh-rlhf
23. OpenBMB. UltraFeedback. Dataset, 2023. https://huggingface.co/datasets/openbmb/UltraFeedback
24. HuggingFaceTB. SmolTalk (SmolLM2 post-training corpus). Dataset, 2024. https://huggingface.co/datasets/HuggingFaceTB/smoltalk
25. Z.ai. GLM-4.5 technical report (MIPO: mixed preference optimization with graded negatives). arXiv:2508.06471, 2025.
26. Z. Zhang, et al. Negative preference optimization: From catastrophic collapse to effective unlearning. arXiv:2404.05868, 2024.
27. DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv:2501.12948, 2025.
28. Q. Yu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv:2503.14476, 2025.
29. Y. Song, D. Yin, X. Yue, et al. Trial and error: Exploration-based trajectory optimization for LLM agents (ETO). arXiv:2403.02502, 2024.
30. S. Yuan, et al. UltraInteract. arXiv:2404.02078, 2024.
31. OpenAI. GPT-6 Astra system card. deploymentsafety.openai.com/gpt-6-astra, accessed 2026-10-13.
32. Anthropic. Claude Opus 5.5 system card and Alignment Intelligence Report materials. anthropic.com, accessed 2026-10-13. *(Direct URL to be registered at submission.)*
33. Z.ai. GLM-5 technical report. arXiv:2602.15763, 2026. (Anchor for GLM-5.3 disclosures.)
34. Xiaomi MiMo Team. MiMo-V2.6 technical report. arXiv:2610.11959, 2026.
35. DeepSeek-AI. DeepSeek-V4.1-Flash technical report. arXiv:2609.19969, 2026.
36. Cyberspace Administration of China. Chinese foundational corpus 4.0 release. cac.gov.cn, 2026-09-16. (Secondary-tier source; background only.)
37. NVIDIA. OpenMathInstruct-2. Dataset, 2024. https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
38. Hugging Face open-r1. OpenR1-Math-Raw. Dataset, 2025. https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw
39. Allen Institute for AI. Dolma. Dataset, 2024. https://huggingface.co/datasets/allenai/dolma
40. HuggingFaceFW. FineWeb. Dataset, 2024. https://huggingface.co/datasets/HuggingFaceFW/fineweb
41. H. Touvron, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv:2307.09288, 2023.
42. Hugging Face. OpenR1 (OpenR1-Math-220k). Repository, 2025. https://github.com/huggingface/OpenR1
43. Bespoke Labs. Bespoke-Stratos-17k. Dataset, 2025. https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k
44. NVIDIA. OpenMathReasoning. Dataset, 2025. https://huggingface.co/datasets/nvidia/OpenMathReasoning
45. SPAR. SPAR f26: Human-annotated agent failure traces. sparai.org/projects/f26/recPcCQjmdWCEjPJL, accessed 2026-10-13. (Open-release plan; low-confidence unit.)
