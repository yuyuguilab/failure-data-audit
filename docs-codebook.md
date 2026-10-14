# 02 Experiment Protocol — 结构化证据审计协议 + 编码手册（Codebook）

> experiment-manager 产出 · 2026-10-13 · 上游：01_research_design.md · 本协议无人类被试、无新实验，"数据收集"= 对公开文档的检索、核证与编码

## 1. 设计概要（来自 01，一段话）

本文是 position paper + 结构化证据审计：不做新实验，而是用统一的**失败数据生命周期编码框架（L0–L4 + 6 副维度）**对两层语料做系统编码——审计 I（数据集层：开源训练配方与数据集如何处置失败数据，回答 RQ2/C2）与审计 II（管线层：五家 2026 秋季旗舰的一手披露，回答 RQ3/C3），文献语料支撑 RQ1，全域扫描支撑 C4。协议目标：任何人都可按本文复现语料组装、抽取与编码，无需追问。流程：**检索 → 纳入/排除 → 一手源核证 → 编码 → 信度复检 → 数据锁定**。核证基线（来自 01 附录，全协议强制执行）：PRS=arXiv:2606.26027（BFCL-V3 25.75 vs GRPO 3.50，即 +22.25 分，**禁写"25 倍"**；ACEBench 双未见 12.00 vs SFT 0.00；分析器为 gpt-5-mini，**非 GPT-4o-mini**）；DAPO=arXiv:2503.14476；具身"9440/78623"数据集不可核证，**整体移出语料**；MiMo=arXiv:2610.11959、DeepSeek-V4.1-Flash=arXiv:2609.19969、GLM-5=arXiv:2602.15763。

## 2. 审计语料（Materials）

### 2.1 语料 A（数据集层，审计 I）

**纳入标准（满足①且满足②或③）**：① 失败数据的处置方式可从公开一手文档核证（官方论文原文 / 官方数据卡 / 官方 GitHub README）；② 该配方/数据集被社区广泛使用（主流模型后训练组件或高引用学术数据集），或具有方法代表性（某家族的典型形态）；③ 披露时间 ≤ 2026-10-13。
**排除标准**：仅有媒体转述、无原始出处的条目（如具身 9440/78623，已确认不可核证）；纯评测基准中不含训练用途披露的（ProcessBench/PRMBench 保留，因其错误定位/归因维度构成 L3 判据参照，标注 unit 层=评测侧）。

**清单（28 单元，锚定来源即编码证据入口）**：

| 层 | 单元 | 锚定 |
|---|---|---|
| 预训练 | Dolma；FineWeb | AI2 / HF 官方数据卡 |
| SFT/蒸馏 | DeepSeek R1；Llama 2；Llama 3；OpenR1-Math-220k；Bespoke-Stratos-17k；OpenMathReasoning；OpenMathInstruct-2 | arXiv:2501.12948；2307.09288；2407.21783；github.com/huggingface/OpenR1；HF bespokelabs/Bespoke-Stratos-17k；HF nvidia/OpenMathReasoning；HF nvidia/OpenMathInstruct-2 |
| 偏好对齐 | Llama 3 DPO；Qwen3；Tulu 3；HH-RLHF；SmolLM2；GLM-4.5 MIPO | arXiv:2407.21783；2505.09388；2411.15124；HF Anthropic/hh-rlhf；HF HuggingFaceTB/smoltalk；arXiv:2508.06471 |
| RL | GRPO（R1 实现形态）；DAPO | arXiv:2501.12948；arXiv:2503.14476 |
| 学术例外 | PRM800K；Math-Shepherd；LEMA；ETO；UltraInteract_pair；UltraFeedback；ProcessBench；PRMBench；OpenR1-Math-Raw；NPO 遗忘集；SPAR f26 | github.com/openai/prm800k；HF peiyi9979/Math-Shepherd；arXiv:2310.20689；2403.02502；2404.02078；HF openbmb/UltraFeedback；HF PRIME-RL/ProcessBench；PRMBench 论文；HF open-r1/OpenR1-Math-Raw；arXiv:2404.05868；sparai.org/projects/f26/recPcCQjmdWCEjPJL |

### 2.2 语料 B（管线层，审计 II）

五家 2026-07（WAIC 喊话）后发布的旗舰，**仅收官方一手文档**，访问日期统一为 2026-10-13，编码前先下载存档（PDF 或快照）并记录版本日期：

| 模型（发布日） | 一手文档 | 定位说明 |
|---|---|---|
| GPT-6 Astra（09-03） | deploymentsafety.openai.com/gpt-6-astra（系统卡）+ 09-16 失对齐报告框架官方页面（URL 执行时定位并登记） | 失败信号流向监控/评测 |
| Claude Opus 5.5（09-22） | anthropic.com 官方系统卡/发布文（含 AIR；具体 URL 执行时定位并登记） | 00_input 未给直链，属执行项非设计缺口 |
| GLM-5.3（2026 下半年） | GLM-5 技术报告 arXiv:2602.15763（§3.1 为关键引句来源）；执行时确认 GLM-5.3 是否有独立报告，有则以其为准 | 错误段掩码、RL 选题 |
| MiMo-V2.6（09-22） | arXiv:2610.11959（§4.2.1/§4.2.6/§4.3.2/§6.1/Table 2） | 反思修订数据 |
| DeepSeek-V4.1-Flash（09-09） | arXiv:2609.19969（§5.1.1） | 失败重放环境 |

### 2.3 文献语料（RQ1）

五家族各 2–5 篇代表作（清单=01 §3，不重复），加 2026-08~10 复核检索（程序见 §4 第 2 步）。

## 3. 编码手册（Codebook，方法学核心）

### 3.1 证据单元与编码粒度

**编码单元**：一个"训练管线/数据集中失败数据的处置机制"，锚定到可引用来源。语料 A 单元=数据集或配方；语料 B 单元=旗舰管线中的披露机制。**机制级条目**：同一单元含多个机制时（如 DAPO），每个机制单独成行，单元主码=各机制的**最高生命周期等级**，多级情况记入 `multi_level` 字段。

### 3.2 主编码：失败数据生命周期等级（ordinal，L0–L4）

判据采用全称量词式（所列判据**全部满足**才赋码）；"失败"= 被系统判定为错误/未通过验证/被拒绝的样本或轨迹。

| 级 | 操作判据（全部满足） | 正例 | 边界案例与判定规则 |
|---|---|---|---|
| **L0 丢弃** | ① 失败轨迹在进入训练前被显式过滤（验证/拒绝采样/过滤后发布）；② 发布物不含失败实例；③ 无任何失败信号加工形式 | R1 第三阶段拒绝采样：仅保留正确+语言一致+高质量轨迹（arXiv:2501.12948） | **OpenMathInstruct-2**：带 expected_answer 字段可判对错。规则：含"可判定对错的字段"≠"保留失败实例"——发布物中不存在被标注保留的失败样本 → 判 L0 |
| **L1 影子** | ① 失败仅以偏好对 rejected 侧存在；② 无独立错误语义（不标注错在哪）；③ 失败侧不单独发布 | HH-RLHF（约 17 万 chosen/rejected 对） | **UltraInteract_pair**：树状轨迹含错误动作+反思文本构成偏好对。规则：若 rejected 侧附带显式修正/反思组件则已具备 L3 判据① → 多级 L1+L3，主码 L3（副维度：载体=偏好对，开放性=发布） |
| **L2 瞬时** | ① 失败 rollout 在训练中产生梯度贡献（负优势/惩罚）；② 训练后即弃；③ 从不发布 | GRPO 失败 rollout 负优势参与梯度（arXiv:2501.12948） | **DAPO**：组内失败 rollout 有负优势（L2 特征），但 dynamic sampling 显式剔除全失败组（零梯度，L0 特征）。规则：按机制分立条目——"组内负优势"=L2、"全失败组剔除"=L0，单元主码取最高级=L2，`multi_level`=L0+L2，备注注明"剔除全失败组"为丢弃制度化的算法级证据 |
| **L3 组件** | ① 失败被显式保留或加工为**可识别组件**（损失掩码/审计标签/重放环境/合成修正样本/反思报告）；② 局部化或限域（限 reward hacking、对齐域等，或规模未披露/限于万级）；③ 不作为公开方法论核心外发 | GLM-5.3 错误段掩码："Erroneous segments within trajectories are retained but masked out in the loss function"（GLM-5 报告 §3.1，arXiv:2602.15763） | **OpenR1-Math-Raw**：保留未通过验证的原始轨迹且公开发布（~100 万）。规则：判据①满足、开放性=发布，但三要素（定位/归因/反思）不齐 → 不升 L4，判 L3（开放性=发布）——全语料中最高开放度的 L3 |
| **L4 一等公民** | ① 错误标注+因果归因+反思修正三要素齐备；② 规模化（≥十万级样本或工业管线核心组件）；③ 作为方法论核心被显式命名/理论化；④ 失败数据或方法论公开 | **预期不存在**——L4 空缺本身是审计发现（支撑 C4） | **MiMo-V2.6 反思修订数据**：三要素齐（"keeps the original error recognizable and makes the correction explicit"，§4.2.6），但限 reward hacking 域、规模未披露 → 判据②不满足 → L3。**LEMA**：三要素基本齐且公开发布，但万级规模、未进主流配方 → 判据②不满足 → L3（语料 A 中最接近 L4 者） |

### 3.3 副维度（各自独立标注，均取披露物中可核证的最高形态）

1. **错误粒度 granularity**：none / trajectory（整轨迹判错）/ step（步级标注或掩码）/ causal（根因级分析）。判据：披露的最细错误定位单位。例：R1=trajectory；PRM800K=step（-1 标错误步）；PRS=causal（根因分析文本）。
2. **归因 attribution**：none / binary（区分模型错 vs 环境崩）/ root-cause（失败根因分类并驱动数据决策）。例：GLM-5.3 失败归因剔除=binary；DeepSeek 员工提交失败分类构建环境=root-cause。
3. **反思 reflection**：none / correction-pair（错-改对配对）/ full-report（含错误类型+根因+修正的完整反思文本）。例：LEMA=correction-pair；PRS/MiMo=full-report。
4. **域 domain**：capability / alignment / both。例：DeepSeek=capability；GPT-6 Astra=alignment（监控）；Claude AIR=alignment。
5. **开放性 openness**：released / not-released。判据：失败数据本身（非成功数据）是否公开可得。
6. **载体 vehicle**：static-dataset / preference-pair / RL-environment / audit-loop。例：LEMA=static-dataset；MIPO=preference-pair（5.x 已弃用）；DeepSeek 失败重放=RL-environment；Claude AIR=audit-loop。

### 3.4 编码冲突解决规则

- **CR1 多机制并存**：机制分立条目，主码=最高级，`multi_level` 记录全部（如 DAPO：L0+L2 主码 L2；GLM-5.3：掩码 L3+badcase 入评估集 L0 → 主码 L3）。
- **CR2 一手 vs 转引冲突**：以一手为准；转引仅记入 provenance 并将 confidence 降为 low。
- **CR3 跨版本演化**：同一厂商按版本分立条目（GLM-4.5 MIPO 与 GLM-5.x 分立），主码以截至 2026-10-13 的最新版本计，历史条目保留作载体迁移证据（MIPO 之死）。
- **CR4 证据不足**：任一判据无法从文档判定 → confidence=low + 写入边界案例登记表，禁止推断（"未披露≠不存在"，见 01 §8 披露偏差条）。
- **CR5 未披露≠L0**：语料 B 中"无失败数据披露"编码为"无披露证据"，不得直接赋 L0——L0 仅当文档**显式陈述**过滤/剔除行为时使用。

## 4. 检索与抽取程序（Procedure，可执行步骤）

1. **语料 A 定位**：逐条打开 §2.1 锚定来源，下载论文 PDF / 数据卡快照，登记 URL、版本日期、访问日期（2026-10-13）。域名级 URL 逐条核证至具体页面。
2. **2026 窗口复核**（支撑 C4）：arXiv 检索，日期窗 2026-08-01~10-13，检索式（英文为主）：`"failure data" training`、`"learning from mistakes" LLM`、`"error correction" fine-tuning`、`"failure reflection" dataset`、`"negative demonstrations"`；中文：失败数据 / 错题本 训练语料。记录命中清单（预期：无大规模失败数据集论文；相邻命中如 Foresight arXiv:2606.23085、AHA arXiv:2410.00371 记为背景）。
3. **语料 B 定位与存档**：按 §2.2 获取五家一手文档，存档 PDF/快照，登记版本锚（报告日期+版本号）；补齐 GPT-6 失对齐框架与 Claude 系统卡的具体 URL。
4. **引用句定位**：在每份文档内全文检索关键词（failure, error, discard, reject, filter, mask, replay, hacking, attribution），定位披露段落；**逐字引用**（英文原文、≤50 词、附章节号/页码/Table 号），禁止转述替代引句。
5. **编码**：按 §3 codebook 逐机制填写审计记录表（§5 数据字典）；边界判定必须引用 CR 规则编号。
6. **边界案例登记**：编码中每遇 CR1/CR4 触发即登记（单元/冲突描述/引用规则/最终判定/理由）。
7. **信度复检**：编码完成 2 周后，按 §6.1 执行 20% 重编码。
8. **数据锁定**：审计表 + 登记表 + 存档快照统一版本号（audit-v1.0），冻结后交 statistical-analyst；此后任何改动须新增版本并在论文中列出 deviation。

## 5. 数据字典与存储计划

存储：CSV（主表）+ JSON（同构备份），UTF-8，git 版本管理，与快照同库。一个观测=一条机制级编码记录。

| 字段 | 类型 | 定义 |
|---|---|---|
| record_id | str | 机制条目唯一号（如 A-03-m1） |
| unit_id / unit_name | str | 所属单元（A-01 Dolma…；B-01 GPT-6 Astra…） |
| corpus | enum | A（数据集层）/ B（管线层） |
| stage | enum | 预训练/SFT/偏好/RL/学术例外；B 层填厂商名 |
| mechanism_desc | str | 失败处置机制的一句话描述 |
| level_main | enum | L0/L1/L2/L3/L4/no-disclosure（主码） |
| multi_level | str | 并存等级列表（CR1 时填写） |
| dim_granularity / dim_attribution / dim_reflection / dim_domain / dim_openness / dim_vehicle | enum | §3.3 各维度取值 |
| evidence_quote | str | 逐字引句（≤50 词）+ 章节定位 |
| source_url / arxiv_id | str | 一手来源 |
| source_tier | enum | P（一手）/ S（转引） |
| access_date / version_anchor | date/str | 访问日期；文档版本锚 |
| confidence | enum | high（一手+引句）/ medium（一手+推断性表述）/ low（转引或判据不全） |
| notes | str | CR 规则引用、边界判定理由 |

## 6. 质量控制与验证门（Quality Gates）

- **G1 纳入门**：无一手文档 → 排除（语料 B）或降级为 S 级背景（语料 A 需满足§2.1①才可入）。
- **G2 一手/转引分级**：P=论文原文、官方系统卡/技术报告/数据卡；S=媒体报道、二手综述。**规则：S 级证据不得支撑任何核心 claim（C1–C5）**，仅可作动机背景（如姜育刚 WAIC 论断，人民网 2026-07-19 属 S 级，只进引言不进证据链）。
- **G3 引句门**：语料 B 中 level≥L2 的每个编码必须附逐字引句；无引句的编码打回重抽。
- **G4 数字核证门**：全部关键数字以 §1 核证基线为准（PRS +22.25 分而非 25×、gpt-5-mini、DAPO=2503.14476）；发现新数字必须回原文核对后才可入表。
- **G5 信度门（见 6.1）**；**G6 锁定门**：deviation 清单随数据锁定一并交付。

### 6.1 信度子协议（单编码者限制的缓解）

① Codebook 全文随论文公开（附录/OSF）；② 编码完成 2 周后，编码者对**分层随机 20%**样本（语料 A 抽 ~6 单元、语料 B 抽 1 家全机制、文献语料抽 1 家族代表）在"盲"条件下（不查看首次编码表）重编码，报告主码原始一致率与 Cohen's κ，**目标 κ≥.70**（方法论参考线）；未达标 → 修订 codebook 判据、全量重编码主码；③ 边界案例登记表全程维护并在论文附录公开，使读者可独立复议每个边界判定。

## 7. 伦理与合规

- **无人类被试声明**：研究仅审计公开文档与数据集，不采集人类被试数据，consent/IRB 不适用（若目标平台要求伦理声明，由 submission-preparer 按"document audit, no human subjects"措辞处理，列为 user action item 确认）。
- **合理使用**：厂商文档逐字引用每处 ≤50 词、注明出处与章节，仅用于学术分析与批评性评论，符合 fair use 惯例；不复制整节或 redistributе 报告原文。
- **数据许可**：引用 HF 数据集时登记其数据卡 license 字段；本文不重新分发数据，仅编码与引用。
- **利益冲突**：审计对象为五家厂商公开披露，编码基于判据而非厂商立场；局限性（披露偏差、单编码者）在论文中如实声明（接 01 §8）。

## 8. 试编码与修订标准（Pilot）

正式编码前对 3 个代表单元试编码：R1（L0 预期）、HH-RLHF（L1）、MiMo-V2.6（L3 多机制）。**修订触发条件**：试编码中出现 codebook 无法判定的案例 ≥2 个，或 3 个中任一与预期等级不符且原因属判据缺陷（而非证据新颖）→ 修订 §3.2 判据后重跑 pilot 再全量；若属证据新颖（如发现 L4 候选）→ 直接登记边界案例并按 CR 规则处理。

## 9. Handoff Notes

**→ statistical-analyst（03_analysis_report.md）**：
- 数据结构：~28 语料 A 单元 + 5 语料 B 单元，机制级条目数预计 35–45（含多机制拆分）；字段见 §5。
- 必算表：审计 I / II 各等级频数表（含 no-disclosure 单列）；L 等级 × 三要素（定位/归因/反思达标与否）交叉表；L 等级 × 载体交叉表（呈现偏好对→RL 环境的迁移）；2025 代（R1/Llama 3 系）vs 2026 代（五旗舰）等级分布对比（Fisher 精确检验可作探索性脚注，主叙事用分布与 exemplar 引句）。
- **序数尺度硬约束**：L0–L4 禁求均值/间距统计；可用中位数、分布图、有序趋势描述。信度：报告 20% 重编码原始一致率 + κ（G5 结果）。**DAPO 口径提醒**：主码 L2（multi_level L0+L2）——01 §6 表格"DAPO 全失败组剔除=L0"指单一机制条目，两口径并存不矛盾，制表时用机制级条目即可同时呈现；"无披露证据"单列，不计入 L0 频数（CR5）。
- C4 的检索复核结果（§4 第 2 步命中清单）作为"缺位"判定的程序性证据。

**→ paper-writer（04_manuscript.md）**：
- Methods 章节素材：§2 语料表（压缩为论文 Table 1）、§3 codebook（Table 2 主等级判据 + 副维度表）、§4 程序叙事、§6.1 信度声明（单编码者 + 20% 重编码 κ + codebook 公开）、§7 伦理声明。
- 逐字引句库（evidence_quote 字段）可直接用于正文 exemplar：GLM-5 §3.1 掩码句、MiMo §4.2.6 "keeps the original error recognizable…"、DeepSeek §5.1.1 "we collect negative feedback…"——引用时保留原文并给出处。
- 强制核证（重申）：PRS 写"+22.25 分 / BFCL-V3 25.75 vs 3.50 / ACEBench 12.00 vs 0.00"，禁"25×"与"GPT-4o-mini"；DAPO=2503.14476；具身 9440/78623 不得出现；G2 规则下姜育刚论断只进引言背景。

**发现的 01 需澄清处（非设计修订）**：① DAPO 编码口径（见上，已由 CR1+机制级条目解决，建议 03 制表时注意）；② 01 §9 handoff 要求"每级 2 正例 1 边界例"，本协议按任务指令采用"≥1 正例+1 边界例"并全部取自 00_input 已核证证据，信息量等同；③ Claude Opus 5.5 与 GPT-6 失对齐框架的一手 URL 未在 00_input 给出直链，已列为执行步骤（§4 第 3 步），不阻塞设计。
