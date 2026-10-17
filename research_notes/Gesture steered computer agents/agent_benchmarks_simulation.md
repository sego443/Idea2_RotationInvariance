# Agent benchmarks for an offline "agent proposes 4, wristband user picks 1" simulation

Scope: which computer-use / GUI / web agent benchmarks support simulating a 4-gesture wristband (~96% per-stroke accuracy) user who picks one of k=4 agent-proposed actions per step (or rejects/undoes), and what is known about agent step-level and top-k accuracy. Research date: 2026-10-17. Numbers are as reported by the cited source; leaderboard aggregators are marked as such and are not primary sources.

## Q1. Offline datasets with human trajectories and per-step ground truth / candidate sets

### Takeaway
Mind2Web (and its screenshot variant Multimodal-Mind2Web) is the only major offline dataset that natively ships a ranked per-step candidate set with ground truth and a published multiple-choice protocol (5 options + None), which maps directly onto a "pick 1 of 4" interface. AndroidControl and GUI-Odyssey give per-step ground-truth actions (and AndroidControl a low-level instruction per step) but no candidate sets; candidates would have to come from sampling the agent.

### Cited Findings
**Mind2Web (Deng et al., NeurIPS 2023 D&B)**
- 2,350 tasks (61 of 2,411 discarded in verification) from 137 websites in 31 domains; avg 7.3 actions per task — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Splits: Train 1,009 tasks / 73 sites; Test Cross-Task 252 tasks / 69 sites; Cross-Website 177 tasks (10 sites per top-level domain); Cross-Domain 912 tasks / 73 sites (Information and Service domains held out) — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Action space: element + operation (Click, Type, Select Option), with typed/selected value — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Candidate generation: a fine-tuned DeBERTa-base (86M) ranker gives Recall@50 of 88.9% / 85.3% / 85.7% (Cross-Task / Website / Domain); top-50 is the candidate pool for all downstream experiments. HTML cleaning reduces elements from 1,135 to 580 on average while keeping 94.7% target recall — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- MindAct casts action prediction as multiple choice: "up to 5 candidate elements within each input, together with a None option"; top-k candidates are split into groups of five and run as a tournament until one remains or all are rejected — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Evaluation is teacher-forced: "Each step of the task is evaluated independently with the ground truth action history provided"; a step succeeds only if element and operation are both correct; a task only if all steps succeed — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Best original MindAct: 52.0% step SR Cross-Task, 38.9% / 39.6% Cross-Website / Cross-Domain; whole-task SR stays low because "the agent often commits at least one error step in most cases" — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Licenses: code MIT; training and test data CC BY 4.0 (test data password-protected, "mind2web") — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)

**Multimodal-Mind2Web (screenshots aligned to Mind2Web), step-level numbers from UI-TARS Table 7 (Jan 2025)**
- UI-TARS-72B: Ele.Acc / Op.F1 / Step SR = 74.7 / 92.5 / 68.6 (Cross-Task), 72.4 / 91.2 / 63.5 (Cross-Website), 68.9 / 91.8 / 62.1 (Cross-Domain) — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- UI-TARS-7B Step SR 67.1 / 61.7 / 60.5; Aguvis-72B 64.0 / 56.5 / 58.2; Claude computer-use 53.5 / 47.7 / 56.4; GPT-4o (pure screenshot agent) 4.3 / 3.9 / 4.5; GPT-4V + OmniParser 39.4 / 36.5 / 42.0 — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- A 2026 paper reports macro element accuracy of 48.1% on Multimodal-Mind2Web for its setting (setup-dependent, not comparable to the above) — [Latent World Action Model for GUI Agents](https://arxiv.org/html/2610.14228) (search snippet only; not read in full)

**WebLINX (Lù, Kasner, Reddy; ICML 2024) — conversational, human-in-the-loop turns**
- 2,337 expert demonstrations, 155 real websites, >100K actions and utterances; each action has DOM, screenshot and video frames; dialogue between an "instructor" and a "navigator" — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- Turn-level metrics: intent match, element IoU, text F1, micro-averaged into an overall score; splits TEST_IID plus OOD splits (WEB, CAT, GEO, VIS = instructor without screen) — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- Candidate retrieval (DMR, fine-tuned MiniLM) Recall@10: 74.27 in-domain, 59.73 VIS, 50.95 GEO, 44.05 CAT, 52.75 WEB, 51.87 OOD overall. Downstream models use top-10 candidates (vs. Mind2Web's 50) to avoid 13 generation rounds; DMR takes 186 ms/turn vs. 916 ms for the Mind2Web DeBERTa ranker — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- Fine-tuned LLaMA-2-13B overall scores on OOD splits: 27.0 (WEB), 24.3 (CAT), 25.9 (GEO), 25.0 (VIS); intent match is ~79–85% but element IoU only ~21–26% — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- Smaller fine-tuned decoders beat zero-shot GPT-4V and fine-tuned multimodal models — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- License: not checked in this session (gap).

**AndroidControl (Li et al., Google, NeurIPS 2024 D&B)**
- 15,283 demonstrations, 833 apps, 40 app categories; average ~5.5 steps per episode — [AndroidControl repo](https://github.com/google-research/google-research/tree/master/android_control); [M2-Miner](https://arxiv.org/html/2602.05429v1) (for the 5.5 figure)
- Each step: screenshot, accessibility tree, a human low-level instruction, the high-level goal and a JSON action; 8 action types (click, long_press, scroll, open_app, input_text, navigate_home, navigate_back, wait); Apache 2.0 — [AndroidControl repo](https://github.com/google-research/google-research/tree/master/android_control)
- "High" setting = goal only (agent plans); "Low" = per-step human instruction — [benchmarklist AndroidControl](https://benchmarklist.com/benchmarks/android_control_high_em/)
- UI-TARS Table 8 (SR = step success): UI-TARS-72B 91.3 (Low) / 74.7 (High); UI-TARS-7B 90.8 / 72.5; OS-Atlas-7B 85.2 / 71.2; Qwen2-VL-7B 82.6 / 69.7; Claude computer-use 19.4 / 12.5 and GPT-4o 19.4 / 20.8 (zero grounding without a grounding model). UI-TARS evaluated on a random 1,000-example sample — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- ScaleTrack (2025) reports 86.6% Low / 77.9% High step SR — [ScaleTrack](https://arxiv.org/pdf/2505.00416) (search snippet)

**GUI-Odyssey (OpenGVLab, 2024)**
- 7,735 episodes, 6 devices, 201 apps, >1,400 app combinations, 6 cross-app task types; average episode 2.1x longer than AITZ (UI-TARS describes 15+ steps per task); CC BY 4.0 — [GUI-Odyssey paper](https://arxiv.org/pdf/2406.08451); [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- Step SR: UI-TARS-72B 88.6, UI-TARS-7B 87.0, OS-Atlas-7B 62.0, Claude computer-use 3.1, GPT-4o 3.3 — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)

**AITW (Android in the Wild, NeurIPS 2023 D&B)**
- 715k episodes, 30k unique instructions, Android 10–13, 8 device types — [AITW NeurIPS page](https://nips.cc/virtual/2023/poster/73496)
- Known to be noisy; AITZ is a cleaned/annotated subset (AITZ step-accuracy gains cited in [Latent World Action Model](https://arxiv.org/html/2610.14228)). AITZ size/license not verified here.

**OmniACT (CMU + Writer, ECCV 2024)**
- 9,802 instances: macOS 4,258, Windows 2,247, Linux 1,134, Web 2,163; splits 6,789 / 992 / 2,021; output is a PyAutoGUI script; 10 action types (click 63.7%, press 16.3%, write 11.7%); metric "Action Score"; MIT license — [OmniACT paper](https://arxiv.org/html/2402.17553v1); [HF card](https://huggingface.co/datasets/Writer/omniact/blob/5e4bcb26745afc8d58ba9519ea9c9fb832386d26/README.md)
- It is single-screenshot-to-script, not step-by-step with intermediate screens, so it is a poor fit for per-step selection.

**Mind2Web-Live / WebCanvas (iMean AI, 2024)**
- 542 tasks with 2,439 intermediate "key node" evaluation states on live websites; Step Score from key nodes; best model at release 23.1% task SR — [WebCanvas](https://arxiv.org/abs/2406.12373v2) (via search summary)

**ScreenSpot family (grounding only)**
- Single-step instruction-to-click-target; UI-TARS reports 38.1 on ScreenSpot-Pro (SOTA at Jan 2025) and 91.6 for UI-TARS-7B on ScreenSpot-v2 — [UI-TARS paper](https://arxiv.org/pdf/2501.12326). Useful only to model grounding errors of a chosen candidate, not selection.

### Inferences
- For a "pick among 4" simulation, Mind2Web's pipeline (ranked candidate pool + MCQ with None) is the closest published analogue; "None of these" maps naturally to a reject gesture.
- Teacher-forced offline replay (ground-truth history) means a rejected/wrong pick cannot be followed into a new state; recovery and undo can only be modelled analytically (e.g., wrong pick -> forced undo cost) rather than observed.
- Offline step accuracy for fine-tuned models on in-domain data (AndroidControl-Low ~91%, GUI-Odyssey ~89%) is much higher than for zero-shot frontier APIs in the same table (often <20% when no grounding model is used), so the choice of proposer model dominates the simulation outcome.

### Gaps
- WebLINX, AITZ and Multimodal-Mind2Web licenses not verified in this session.
- No 2026 offline-dataset leaderboard from a primary source was found; the most comprehensive step-SR table read is UI-TARS (Jan 2025).

## Q2. Online environments: current best success, cost, human-in-the-loop

### Takeaway
Online benchmarks (OSWorld-Verified, WebArena, AndroidWorld) are approaching or exceeding the human baselines on aggregator leaderboards by late 2026, but those numbers come from secondary sites and system-level scaffolds; primary sources verified here are mostly from 2024–2025. Online runs give true task success and real irreversible effects but cost far more than offline replay.

### Cited Findings
**OSWorld / OSWorld-Verified (XLANG)**
- 369 tasks (361 excluding Google Drive tasks); human success 72.36%; best model at original release 12.24% — [OSWorld site](http://osworld-v1.xlang.ai/)
- OSWorld-Verified (blog dated 2025-07-28): ~300 issues fixed; AWS migration cut evaluation "from 10+ hours to just 20 minutes" with up to 50 parallel environments. Verified results at publication: CoACT-1 60.76%, Agent S2.5 + o3 56.0%, GTA1 + o3 53.1%, Claude 4 Sonnet 43.9% — [XLANG blog](https://xlang.ai/blog/osworld-verified)
- UI-TARS-72B (Jan 2025): 24.6 (50 steps) / 22.7 (15 steps) vs Claude 22.0 / 14.9 — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- Aggregator, unverified: as of 2026-10-13, Qwen3.8 Max 86.1%, Claude Fable 5 85.0%, Claude Mythos 5 85.0%, Claude Mythos Preview 85.4%, Claude Opus 4.8 83.4% on OSWorld-Verified — [BenchLM](https://benchlm.ai/benchmarks/osworld-verified); "Intelligence-Indeed Agent" 90.19% — [presenc.ai](https://presenc.ai/research/osworld-computer-use-leaderboard-2026). These should be checked against the official XLANG leaderboard (it loads dynamically and could not be read here).
- OSWorld 2.0 exists (2026); a search summary attributes cost estimates at a 150-step budget of ~$1,875 (GPT-5.5) and ~$1,030 (Claude Opus 4.7), and a human median of ~1.6 h per task — [Epoch AI OSWorld 2.0](https://epoch.ai/benchmarks/osworld-2); [Yutori OSWorld 2.0](https://yutori.com/leaderboards/osworld-2.md). Unverified: it is unclear whether these figures are per task or per full run; I did not open the pages.

**WebArena / VisualWebArena (CMU)**
- Aggregator, unverified: top WebArena system WebTactix (DeepSeek v3.2) 74.3%, OpAgent 71.6% as of June 2026; another snapshot (2026-05-22) lists Claude Mythos Preview 68.7%, GPT-5.4 Pro 65.8%, Claude Opus 4.6 64.5%. Scores reflect scaffolding, not just models — [Steel leaderboard](https://leaderboard.steel.dev/leaderboards/webarena.md); [BenchLM WebArena](https://benchlm.ai/md/benchmarks/webarena.md)
- VisualWebArena 2026 numbers: not verified (gap).

**AndroidWorld (Google)**
- 116 parameterised tasks over 20 apps; human 80%; best at release M3A + GPT-4 Turbo 30.6% — [AndroidWorld summary](https://themoonlight.io/review/androidworld-a-dynamic-benchmarking-environment-for-autonomous-agents) (secondary; consistent with the paper)
- UI-TARS-72B 46.6 vs GPT-4o 34.5 (Jan 2025) — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- Aggregator, unverified: 2026-07-29 snapshot Qwen3.7 Plus 81.0%, Holo3.1-35B-A3B 79.3%; other listings show 97.4% pass@1 for several systems — [BenchLM AndroidWorld](https://benchlm.ai/benchmarks/androidworld). A Feb 2026 paper is titled "Achieving Perfect Accuracy on AndroidWorld Through Task Decomposition" — [arXiv 2602.07787](https://arxiv.org/pdf/2602.07787), suggesting the benchmark is near saturation.

**WindowsAgentArena (Microsoft, 2024)**
- 150+ Windows tasks adapted from the OSWorld framework; Navi 19.5% vs unassisted human 74.5% — [WAA arXiv](https://arxiv.org/html/2409.08264v1)

**WorkArena (ServiceNow)**: not retrieved in this session (gap).

**Human-in-the-loop online evaluations**
- CowPilot (CMU, 2025): agent proposes next steps; user can pause, reject or take over. On 27 WebArena tasks (easy/medium/hard, 5 sites): GPT-4o autonomous 48%, copilot 95%, human-only 89%; LLaMA-8B autonomous 4%, copilot 81%. With GPT-4o humans performed 15.2% of steps (~1.1 human steps per task); evaluations by three study authors, no timing data — [CowPilot](https://arxiv.org/html/2501.16609v2)
- Magentic-UI (Microsoft, 2025): six mechanisms (co-planning, co-tasking, multi-tasking, action guards, long-term memory); evaluated with autonomous benchmarks, simulated-user testing, qualitative user studies and safety tests — [Magentic-UI arXiv](https://arxiv.org/abs/2507.22358)

### Inferences
- CowPilot is the closest published precedent for "agent proposes, human approves/overrides"; its 48% -> 95% jump with ~1 human step per task is a useful upper anchor, but it used the authors as users (not naive users) and a small task set.
- Given near-saturation of AndroidWorld and OSWorld-Verified on aggregators, a steering interface's added value is more visible with weaker/cheaper proposers or harder sets (OSWorld 2.0), which is an argument to report results across proposer strength.

### Gaps
- No primary source verified for any 2026 leaderboard number; official XLANG and AndroidWorld leaderboards could not be read.
- Per-task API cost for OSWorld/WebArena/AndroidWorld from primary sources was not found.

## Q3. How often is the correct next action in an agent's top-1 vs top-4?

### Takeaway
No paper found reports top-4 action recall for a frontier agent directly. Available proxies: retrieval Recall@k for element candidates (Mind2Web, WebLINX), step SR as top-1, and Best-of-N / pass@k sampling curves (UI-TARS, a 2026 world-model paper), which show that sampled candidate sets contain the correct action noticeably more often than the greedy action. Top-4 recall must be measured by the project itself.

### Cited Findings
- Top-1 proxies (step SR): Multimodal-Mind2Web 60–69% for UI-TARS; AndroidControl-High ~72–78%; AndroidControl-Low ~87–91%; GUI-Odyssey ~87–89% — [UI-TARS paper](https://arxiv.org/pdf/2501.12326); [ScaleTrack](https://arxiv.org/pdf/2505.00416)
- Element retrieval top-k: Mind2Web Recall@50 85–89% — [Mind2Web paper](https://arxiv.org/pdf/2306.06070); WebLINX Recall@10 74% in-domain, ~52% OOD — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- UI-TARS Best-of-N with N = 1, 16, 64 on Multimodal-Mind2Web, AndroidControl and GUI-Odyssey (1,000-example samples): at N=1 system-2 (thinking) is slightly worse than system-1; at N=16/64 system-2 gains a clear advantage because "the diversity of candidates increases the likelihood that the correct action is among the sampled outputs". Exact values are only in Figure 8 (not tabulated) — [UI-TARS paper](https://arxiv.org/pdf/2501.12326)
- A 2026 mobile world-model paper samples k=3 candidate actions per step and reports Qwen3-VL-8B Pass@3 step accuracy of 69.55% on AITZ (and 66.04% for Pass@9 as extracted, which is non-monotonic and may be a different configuration; treat with caution); it concludes posterior selection "is limited by both world-model fidelity and the agent's proposal ability" — [How Mobile World Model Guides GUI Agents?](https://arxiv.org/html/2605.10347v1)
- Search summaries of the same paper say Pass@k increases with k for action-type and step accuracy — [arXiv 2605.10347](https://arxiv.org/html/2605.10347v1)

### Inferences
- Pass@k with an oracle selector equals the success rate of a perfect human picker; the simulated wristband user is "oracle selector x gesture confusion (~96%) x user's own judgement error". The key quantity to measure is Pass@4 (correct action anywhere in 4 proposals) vs Pass@1 for the chosen proposer.
- Sampled candidates are often near-duplicates (same element, different reasoning); deduplicating by (action type, target element, value) before showing 4 options matters, otherwise effective k < 4.
- Estimate (not reported): if top-1 step accuracy is p1 and top-4 is p4, per-step success under steering ~= p4 x 0.96 (gesture) x human-judgement accuracy; with ~7 steps (Mind2Web average) the gap between p1^7 and (p4 x 0.96)^7 is the headline effect.

### Gaps
- No primary source reports top-4 action recall for GPT/Claude/Gemini computer-use agents, nor calibrated per-candidate probabilities.
- UI-TARS BoN numbers are figure-only.

## Q4. User-simulation methodology, KLM timing, pitfalls

### Takeaway
User simulation is an established evaluation method (information retrieval, dialogue, τ-bench), and KLM provides standard operator times for time-per-task estimates. Main pitfalls: simulator realism, teacher forcing that hides error recovery, and false negatives when several actions are valid.

### Cited Findings
- Balog & Zhai survey (Foundations and Trends in IR, 2024; arXiv 2306.08550) reviews frameworks for user simulators used in evaluation of search, recommendation and conversational assistants, with links to ML, dialogue, user modelling and economics — [arXiv](https://arxiv.org/abs/2306.08550v1); [NOW Publishers](https://www.nowpublishers.com/article/Details/INR-098)
- τ-bench (Sierra, 2024): LLM-simulated user + domain APIs + policy; introduces pass^k (success in all k trials); gpt-4o <50% task success and pass^8 <25% in retail — [τ-bench](https://export.arxiv.org/pdf/2406.12045); [Sierra blog](https://sierra.ai/blog/benchmarking-ai-agents)
- Magentic-UI includes "simulated user testing of its interaction capabilities" — [Magentic-UI](https://arxiv.org/abs/2507.22358)
- KLM operators (Card, Moran, Newell 1980): K = 0.2 s keystroke/click, P = 1.1 s point, H = 0.4 s homing, M = 1.35 s mental preparation, plus system response R; total time = sum of operators — [KLM-GOMS](https://www.wikipedia.com/wiki/KLM-GOMS); [LMU KLM slides](https://www.medien.informatik.uni-muenchen.de/lehre/ws0708/mmi1/KLM.pdf)
- WebLINX notes that keeping a human's flow uninterrupted needs total delay below ~1 s (citing Carroll & Rosson) — [WebLINX paper](https://arxiv.org/pdf/2402.05930v2)
- CowPilot logs human vs agent step shares and is offered as a data-collection and evaluation tool — [CowPilot](https://arxiv.org/html/2501.16609v2)

### Inferences
- Time model for the wristband: per step = R (agent latency for 4 proposals) + M (read 4 options, ~1.35 s nominal; likely more for 4 items) + one gesture stroke time (measured on the project's hardware) + expected correction cost (P(misclassify) x (undo stroke + re-pick)). The gesture time and confusion matrix should come from the project's measured data; KLM values substitute only for the reading/decision part.
- Simulated-user pitfalls to state up front: (1) teacher-forced offline steps overstate success because errors never propagate; (2) single ground truth undercounts alternative valid actions; (3) the simulated user is assumed to always recognise the correct option, so report results with a "human judgement error" parameter swept; (4) irreversible actions are not labelled in Mind2Web/AndroidControl, so an irreversible-error rate requires a rule-based or manual label set (e.g., submit, purchase, send, delete).

### Gaps
- No primary source found on simulated users with explicit gesture/BCI confusion matrices for agent steering; closest analogues are BCI/AAC selection-interface simulations, not searched here.
- No validated KLM extension for reading k=4 agent proposals.

## Q5. Cheapest, most defensible first simulation

### Takeaway
Start offline on Mind2Web (Cross-Task/Website/Domain test splits, CC BY 4.0) or Multimodal-Mind2Web with teacher forcing, using the published candidate pool and asking the proposer for 4 distinct actions per step; then validate on a small online WebArena subset (as CowPilot did with 27 tasks) or AndroidWorld (116 tasks) to observe error propagation and undo.

### Cited Findings
- Mind2Web provides ground truth per step, ranked candidates (Recall@50 85–89%), an MCQ-with-None protocol, step-level independent evaluation and CC BY 4.0 data — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- Mind2Web's own GPT-4 evaluation used only 50 tasks per split with top-10 candidates "due to limited budget", a precedent for small-budget subsets — [Mind2Web paper](https://arxiv.org/pdf/2306.06070)
- AndroidControl (Apache 2.0) gives per-step low-level instructions, allowing a "Low" (user intent known) vs "High" (goal only) contrast — [AndroidControl repo](https://github.com/google-research/google-research/tree/master/android_control)
- CowPilot's 27-task WebArena subset is a published precedent for human-agent co-pilot evaluation — [CowPilot](https://arxiv.org/html/2501.16609v2)
- OSWorld-Verified runs in ~20 minutes with 50 parallel AWS environments, but frontier-model API cost per run is high — [XLANG blog](https://xlang.ai/blog/osworld-verified)

### Inferences
- Proposed first protocol (design, not executed): for each Mind2Web test step, the proposer returns 4 deduplicated candidate actions (or uses the top-4 MindAct-style options); the simulated user picks the ground-truth one if present, passed through the measured 4x4 gesture confusion matrix, and uses reject/undo when absent. Metrics: step success, task success (all steps), strokes per task, KLM+measured-stroke time per task, and the fraction of steps where a misclassified pick hits a rule-labelled irreversible action. Report true picker vs. random picker and vs. agent-alone top-1 as controls.
- Cost estimate (not reported anywhere; rough): Mind2Web test ~1,341 tasks x ~7.3 steps ~= 9,800 steps; one proposer call per step at a few thousand input tokens gives on the order of tens of millions of tokens for the full test set, so a 50–100 task subset per split keeps the first run cheap. Verify with the chosen model's pricing before running.
- Caveats for defensibility: Mind2Web is from 2023 and is in many models' training data (contamination), and UI-TARS-style models train on it, so use a zero-shot API proposer and report in-domain vs cross-domain separately; offline results support only "selection accuracy given correct history", not end-to-end task success.

### Gaps
- No existing study combines a gesture/BCI selector with a GUI-agent benchmark; this would be novel, and no prior numbers exist to compare against.
- Irreversibility labels for any of these datasets were not found.
