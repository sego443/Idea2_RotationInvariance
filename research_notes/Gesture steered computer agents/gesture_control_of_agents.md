# Gesture and minimal-input control of LLM / computer-use agents, and human-in-the-loop oversight interfaces (2022 to Oct 2026)

Scope note: research pass done 2026-10-17 with web search and fetch. "Verified" means the claim was read from the primary page (arXiv abstract or HTML, vendor docs) or a search snippet that quotes that page during this pass. Items marked UNVERIFIED come from background knowledge or secondary press only and need checking before they are cited. Product announcements are kept apart from peer-reviewed or preprint research results. Numbers are reported as the source states them; nothing here is estimated.

## Q1. Wearable or gesture input to commercial AI assistants and agents

### Takeaway
Shipping wearables (Meta Neural Band, Apple Watch double tap / wrist flick, Android XR pinch) use tiny gesture vocabularies for navigation, confirm/dismiss and media. They do not steer a general computer-use agent. Agent delegation on wearables (glasses, rings) is still started by voice, and gestures are limited to activation, scrolling or acknowledgement.

### Cited Findings
**Meta Ray-Ban Display + Meta Neural Band (product, announced Sept 2025)**
- $799 glasses with a 600x600 colour display in the right lens. The Neural Band is an sEMG wristband that can pick up muscle signals even when the movement is "visually imperceptible" — [eWeek](https://www.eweek.com/news/meta-ray-ban-display-ai-glasses-launch/); [Meta help page](https://www.meta.com/help/ai-glasses/1864740664283499/)
- Meta AI shows step-by-step responses on the display, and "thumb swipes or subtle hand gestures on the band move through each stage". Wrist rotation controls volume. Handwriting by finger movement was announced for later — [eWeek](https://www.eweek.com/news/meta-ray-ban-display-ai-glasses-launch/); [techpoint.africa guide](https://techpoint.africa/guide/meta-ray-ban-display-glasses/)
- Press reports say the decoder was trained on data from "nearly 200,000 research participants" (company claim) — [eWeek](https://www.eweek.com/news/meta-ray-ban-display-ai-glasses-launch/)
- Gesture vocabulary described from the Nature paper: index-finger handwriting, wrist rotation for a 1-D cursor, thumb swipe along the side of the index finger, and thumb tap/hold — [UploadVR](https://www.uploadvr.com/meta-semg-wristband-gestures-nature-paper/)
- Research result (Kaifosh & Reardon et al., Nature 2025, "A generic non-invasive neuromotor interface for human-computer interaction"). Closed-loop median performance without per-user training: 0.66 target acquisitions/s in continuous navigation, 0.88 gesture detections/s in a discrete-gesture task, and handwriting at 20.9 WPM. Personalisation improves handwriting by 16%. More than 100 h of sEMG data released — [CSHL repository PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf); [IDEAS listing (Nature v645 i8081)](https://ideas.repec.org/a/nat/nature/v645y2025i8081d10.1038_s41586-025-09255-w.html); [UploadVR](https://www.uploadvr.com/meta-semg-wristband-gestures-nature-paper/)
- No source found showing Neural Band gestures driving a multi-step computer-use agent. Gestures move through Meta AI answers and app UI.

**Apple Watch (product)**
- watchOS 26 adds Wrist Flick, a quick turn of the wrist over and back that dismisses notifications and calls, silences timers and alarms, and returns to the watch face. It uses the accelerometer, the gyroscope and an ML model, and runs on Series 9, Series 10 and Ultra 2. Double tap is the existing companion gesture — [heise](https://heise.de/-10491847); [MacTrast](https://www.mactrast.com/?p=169815); [9to5Mac Feb 2026](https://9to5mac.com/2026/02/04/watchos-26-added-a-new-apple-watch-trick-that-ive-been-loving/)
- The search results did not link Wrist Flick to Siri or Apple Intelligence agent actions — [heise](https://heise.de/-10491847). UNVERIFIED: whether double tap can confirm or advance Siri / Apple Intelligence responses. This was not confirmed from an Apple primary page in this pass.

**Google / Android XR (product)**
- Galaxy XR (Oct 2025) is the first Android XR headset. Gemini is integrated at the system level and is aware of what the user sees. Input is voice, hand tracking (pinch to tap) and eye tracking — [Samsung support](https://ushl.samsung.com/us/support/answer/ANS10007511); [Gigazine](https://wbgsv0a.gigazine.net/gsc_news/en/20251022-samsung-galaxy-xr)
- Press reports Galaxy Ring could become a controller for Galaxy XR and glasses. This is a report or speculation, not a shipped feature — [Croma](https://www.croma.com/unboxed/galaxy-ring-xr-controller-galaxy-xr-smart-glasses)

**Rings and voice wearables (products)**
- Natura "Interface", a $99 ring announced 8 Oct 2026, gives "one press" access to multiple AI agents with voice task control — [TechRepublic](https://www.techrepublic.com/article/news-natura-interface-ai-smart-ring-agents-price-features/); [PYMNTS](https://www.pymnts.com/?p=4291426)
- Sandbar Stream Ring ($249) captures voice notes, and the AI asks clarifying questions — [CXOToday](https://cxotoday.com/news-analysis/stream-ring-from-sandbars-the-wearable-that-brings-ai-at-your-fingertips-in-2026/). WIZPR ring is a whisper interface to assistants — [MobileSyrup](https://mobilesyrup.com/2024/04/07/whispr-ring-ai-assisntant-on-your-fingertips/)
- Humane AI Pin and Rabbit R1 ("Large Action Model"): UNVERIFIED in this pass. From background knowledge, both were voice-first, and Humane shut the Pin down in early 2025. Neither offered gesture-driven agent selection. Check before citing.

**Research prototypes**
- "AI at your Fingertips: Wearable Ring as a Low-Friction Interface for Agentic AI" (Columbia + Microsoft Research, CHI '26 Extended Abstracts / demo). Tap to activate, whisper a command (e.g., "draft slides"), and get a haptic signal when the agent finishes. The agent pipeline recovers from failures by itself ("fire-and-forget"). Touch is used only for activation, and the command is voice — [MSR PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/chi26d-sub1595-cam-i61.pdf); [ACM DL](https://dl.acm.org/doi/full/10.1145/3772363.3798736)
- VisionClaw (Liu, Lee, Gonzalez, Gonzalez-Franco, Suzuki; arXiv 2604.03486). An always-on agent on Meta Ray-Ban glasses that combines egocentric perception with OpenClaw agents for tasks such as adding to cart, notes, calendar and IoT. Input is speech plus camera, with no gestures. Lab N=12 and longitudinal N=5. Reported faster completion and lower interaction overhead than baselines, and users delegated more over time — [arXiv](https://arxiv.org/abs/2604.03486)
- Satori, a proactive AR assistant with BDI user modelling (CHI 2025), was surfaced by search but not read in this pass — [search context only; see UploadVR/IMU search results]
- GestureGPT (arXiv 2310.12821) uses LLM agents to *interpret* free-form hand gestures zero-shot. The direction is gesture to LLM understanding, not gesture steering an agent — [arXiv](https://arxiv.org/pdf/2310.12821)

### Inferences
- Every shipped wearable-AI product found uses gestures for navigation and acknowledgement (next, back, dismiss, select, volume) and voice for intent. Selecting among agent-proposed actions with gestures is not offered by any product found.
- Meta's 0.88 discrete detections/s with cross-user generalisation is the strongest published benchmark for wrist gesture throughput. A 4-swipe piezo band is a cheaper, lower-vocabulary point in the same design space, and comparisons should be framed that way, not as a competitor in accuracy.

### Gaps
- Could not verify from Apple primary documents whether double tap interacts with Siri / Apple Intelligence.
- Could not find official Meta documentation of the exact gesture-to-Meta-AI mapping, or of any agentic (multi-step, cross-app) action on Ray-Ban Display.
- Google glasses (Android XR AI glasses) gesture specs were not verified.

## Q2. Research on minimal-input / mixed-initiative GUI agents ("agent proposes, human selects") and how often users intervene

### Takeaway
The "agent proposes the next step, human accepts, rejects or overrides" pattern is well established (CowPilot, Magentic-UI, Morae, PlowPilot), and intervention-rate data now exist (about 15 to 22% of steps). All of these systems assume full keyboard, mouse or screen-reader input. None restricts the human to a 4-symbol noisy channel or offers K candidate actions for one-shot selection.

### Cited Findings
- **CowPilot** (Huq, Wang, Xu et al., CMU; NAACL 2025 demo; arXiv 2501.16609). The agent proposes next steps, and the user can pause, reject, or take an alternative action, then hand control back. Across case studies on 5 websites, collaborative mode reached a 95% success rate with humans doing 15.2% of steps, and the agent alone drove up to half of task success — [arXiv](https://arxiv.org/abs/2501.16609); [HF papers](https://huggingface.co/papers/2501.16609)
- **CowCorpus / "Modeling Distinct Human Interaction in Web Agents"** (Huq, Wang et al., CMU + Duke; arXiv 2602.17588, v4 Jul 2026). 400 real-user trajectories with more than 4,200 interleaved human and agent actions. Four styles were identified: takeover, hands-on, hands-off and collaborative. The three motivations for intervening were error correction, preference refinement and assistive takeover on complex UI. Intervention intensity was 21.63% on standard tasks and 16.06% on free-form tasks. Fine-tuned intervention predictors improved accuracy by 61.4 to 63.4% over base LMs. The deployed PlowPilot was rated "36.8% higher in usefulness" in the HTML version read — [arXiv HTML](https://arxiv.org/html/2602.17588). An abstract version found by search says "26.5% increase in user-rated usefulness". The two versions conflict, so cite the specific version — [arXiv PDF](https://arxiv.org/pdf/2602.17588)
- **"Here, Let Me Help": An Empirical Study of User Interventions in Human–Web Agent Collaboration** (CHI '26). N=30, 12 structured tasks on live shopping, travel and information sites. It separates explicit interventions (halt or override) from implicit ones (guiding or preparing the environment without stopping). Users did not want continuous control. They wanted to step in near sensitive actions, when ambiguity rose, or when the agent's confidence seemed unjustified. Agents asked for unnecessary confirmations too often — [ACM DL](https://dl.acm.org/doi/full/10.1145/3772318.3791536) (full text returned 403, so findings come from the search snippet)
- **Morae: Proactively Pausing UI Agents for User Choices** (Peng, Li, Bigham, Pavel; UIST 2025; arXiv 2508.21456). For blind and low-vision users, a large multimodal model detects choice points from UI code and screenshots and pauses to ask the user. Verification questions take 4 answers: yes, no, unanswerable-and-proceed, not-important-and-proceed. In a field study, BLV users completed more tasks and picked options that matched their preferences better than with baselines that included OpenAI Operator. N was not given in the abstract — [arXiv](https://arxiv.org/abs/2508.21456)
- **Magentic-UI** (Mozannar et al., Microsoft Research; arXiv 2507.22358, Jul 2025). Open-source human-in-the-loop web agent with co-planning (a plan editor before execution), co-tasking, multi-tasking, action guards (approval before potentially irreversible actions, with user-settable approval frequency) and long-term memory. It was evaluated on agentic benchmarks, with simulated users, in qualitative user studies and in safety tests. No numbers appear in the abstract — [arXiv](https://arxiv.org/abs/2507.22358); [MSR blog](https://www.microsoft.com/en-us/research/blog/magentic-ui-an-experimental-human-centered-web-agent)
- "When Users Change Their Mind: Evaluating Interruptible Agents in Long-Horizon Web Navigation" (arXiv 2604.00892) covers addition, revision and retraction interruptions. Only the snippet was read — [arXiv](https://arxiv.org/html/2604.00892.pdf)
- Accessibility-oriented LLM GUI agents found are voice- or text-driven. Examples: DexAssist (voice, dual LLM, accessible web navigation) — [arXiv 2411.12214](https://arxiv.org/pdf/2411.12214), and a multimodal GUI architecture for LLM assistants — [arXiv 2510.06223](https://arxiv.org/html/2510.06223v2). LLM-accelerated communication for severe motor impairment (switch/gaze AAC abbreviation expansion) — [arXiv 2312.01532](https://arxiv.org/pdf/2312.01532)
- Agent-internal candidate-action sets: WebChallenger lists candidate actions as a numbered list and the model picks an index — [arXiv 2606.10423](https://arxiv.org/pdf/2606.10423). ANCHOR generates multiple plausible next actions at branch points — [arXiv 2602.07153](https://arxiv.org/pdf/2602.07153). PrecogUI generates candidate actions and scores their reliability — [arXiv 2609.36923](https://arxiv.org/html/2609.36923v1). In all three the *model* selects, not a human.

**Commercial agents: takeover and confirmation**
- OpenAI Operator (Jan 2025) has "takeover mode" for credentials and payments, during which it stops capturing the screen. It asks for confirmation before final actions such as purchases or sending email. "Watch mode" is required on sensitive sites such as email and finance — [Axios](https://www.axios.com/2025/01/23/openai-operator-agent-preview-chatgpt); [Operator system card PDF](https://cdn.openai.com/operator_system_card.pdf) (the PDF fetch failed with a TLS error, so policy details come from search snippets; system-card confirmation and prompt-injection metrics were NOT verified in this pass). The o3 CUA system card also exists — [PDF](https://cdn.openai.com/pdf/4375e605-f9a6-438d-bcc8-190599c183a6/o3_cua_system_card.pdf)
- ChatGPT agent (2025) has similar confirmation and takeover behaviour. UNVERIFIED in this pass.
- Google Project Mariner works only in the active tab, asks for human confirmation before sensitive actions such as purchases, and lets the user stop and take over at any time — [DeepMind page](https://deepmind.google/technologies/project-mariner/); [Tom's Guide](https://www.tomsguide.com/ai/google-gemini/google-reveals-project-mariner-a-new-browser-agent-that-can-automate-your-digital-life)
- Anthropic computer-use docs (verbatim): "Asking a human to confirm decisions that might result in meaningful real-world consequences and any tasks requiring affirmative consent, such as accepting cookies, completing financial transactions, or agreeing to terms of service." The docs also recommend VM isolation, domain allowlists, and not giving the model credentials. Prompt-injection classifiers exist, but "the precautions above remain important" — [Claude platform docs](https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/computer-use-tool)

### Inferences
- CowPilot's 15.2% human-step share and CowCorpus's 16 to 22% intervention intensity suggest that, with keyboard and mouse, users touch roughly 1 in 5 to 1 in 7 steps. A gesture-select interface would turn many of these overrides into a choice among K alternatives. This is a hypothesis to test, not a finding.
- Morae's 4-response verification scheme (yes / no / skip / not important) is the closest published analogue to a 4-symbol response vocabulary. It was designed for screen-reader users, not for a noisy gesture channel.

### Gaps
- No paper found that measures agent success when the human can only pick from K agent-proposed actions (top-K coverage of the correct next action) on WebArena, OSWorld or Mind2Web-type tasks.
- Morae's N and per-task numbers were not extracted.
- The "Here, Let Me Help" quantitative intervention rates were not obtained (paywall / 403).

## Q3. Interaction design of approving / rejecting agent actions: confirmation fatigue, risk gating, levels of autonomy

### Takeaway
Recent controlled evidence shows that per-action confirmation is a weak safeguard. Users blocked only about 21% of problematic actions that reached execution, whatever the oversight strategy, and planning-level strategies help mainly by preventing problems upstream. Autonomy-level frameworks (operator, collaborator, consultant, approver, observer) treat approval as a design choice, but they do not model the reliability of the approval channel itself.

### Cited Findings
- **Comparing Human Oversight Strategies for Computer-Use Agents** (Chen, Zhang, et al.; arXiv 2604.04918). N=48, 192 live web sessions, 6 domains. Four strategies were compared: Risk-Gated (agent escalates selectively), Action Confirmation (per-action approval), Supervisory Co-Execution (plan level) and Structurally Enriched (plan plus action). Users blocked only 29 of 138 problematic actions that reached execution (21%), with no reliable difference between strategies. Plan-based strategies reduced problematic-action occurrence by 76% (OR=0.24) through upstream prevention, not better detection. Action Confirmation had lower workload on low-consequence tasks and a higher burden on high-consequence ones. Failure modes included timing misalignment (approval requested after the risky state already existed, or delegated approvals that carried over into unanticipated steps). Users judged procedural correctness rather than whether the action should happen, and plan approval reduced scrutiny during execution — [arXiv HTML](https://arxiv.org/html/2604.04918v2)
- **Levels of Autonomy for AI Agents** (Feng, McDonald, Zhang; UW; arXiv 2506.12469, Jun 2025). L1 user as operator, L2 collaborator, L3 consultant, L4 approver (engaged only in risky or pre-specified situations), L5 observer. Autonomy is framed as a deliberate design decision separate from capability, and the paper proposes "AI autonomy certificates" — [arXiv](https://arxiv.org/abs/2506.12469v1)
- The CHI '26 interventions study reports over-frequent unnecessary confirmations and a heavy oversight burden — [ACM DL](https://dl.acm.org/doi/full/10.1145/3772318.3791536)
- "Characterizing Unintended Consequences of GUI Agents for Web Browsing" (CHI '26) was found but not read — [ACM DL](https://dl.acm.org/doi/10.1145/3772318.3790696)
- Magentic-UI lets users set how often approvals are needed (a configurable action-guard frequency) — [MSR blog](https://www.microsoft.com/en-us/research/blog/magentic-ui-an-experimental-human-centered-web-agent)
- Automated steering: "From Uncertainty to Action: Learning to Steer LLM Agents" (Li, Duan, et al.; arXiv 2610.09115, 6 Oct 2026). Uncertainty can flag failing trajectories, but no single signal locates the step where steering helps. VoS (Value of Steering) learns intervention points from about 82k counterfactual continuations, for +7.8 points on average. There are no human users — [arXiv](https://arxiv.org/abs/2610.09115). Related: "Learning to Steer LLM Agents and Multi-Agent Systems" (arXiv 2610.22316) — [arXiv](https://arxiv.org/html/2610.22316)

### Inferences
- The 21% block rate means reversibility (undo) and upstream plan or option shaping matter more than per-step confirm dialogs. This supports reserving an undo gesture and moving confirmation to irreversible actions only.
- With a noisy gesture channel, a false "confirm" adds to the user's own low catch rate. No study found quantifies this compounding.

### Gaps
- No study found on confirmation fatigue where the confirm action itself is error-prone, as with a gesture misclassification.
- No standard risk taxonomy of irreversible GUI actions was verified across vendors. Each vendor lists examples (purchases, email, ToS, cookies).

## Q4. Agent as probabilistic prior over intent, with user input as a noisy low-bandwidth channel

### Takeaway
The noisy-channel / Bayesian-prior framing is mature in AAC and BCI text entry (Nomon, Dasher, LM-fused BCI spellers) and is now being extended to LLMs, but only for *text* communication. No work was found that treats a computer-use agent's action distribution as the prior and a wearable gesture classifier's confusion matrix as the likelihood for GUI control.

### Cited Findings
- **Nomon** (performance evaluation, CHI 2022; arXiv 2204.01619). A single-switch selection interface with a probabilistic selection method. It learns each user's click-time distribution nonparametrically, does not need options arranged in a grid, and can be used for web browsing, games and drawing. Over 10 sessions with a webcam switch, users typed faster than with row-column scanning, and the gain was larger for picture selection — [arXiv](https://arxiv.org/abs/2204.01619); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/146286)
- Transformer LMs adapted for BCI predictive typing, with robustness tested on noisy histories (GPT-2 best on clean text) — [arXiv 2305.03819](https://arxiv.org/html/2305.03819v1)
- LLM priors fused with neural evidence in BCI decoding to reduce keystrokes (review) — [Sensors 2025, 25(13):3987](https://www.mdpi.com/1424-8220/25/13/3987); [arXiv 2412.07355](https://arxiv.org/pdf/2412.07355v1)
- Huffman scanning uses LMs inside grid scanning — summarised in [ResearchGate review](https://www.researchgate.net/publication/302031802_Integrating_language_models_into_classifiers_for_BCI_communication_A_review) (secondary)
- LLM abbreviation expansion for severe motor impairment reduces scans and selections — [arXiv 2312.01532](https://arxiv.org/pdf/2312.01532)
- VoS (above) uses agent uncertainty to decide *when* to steer, but steering is automated, not done by a human channel — [arXiv](https://arxiv.org/abs/2610.09115)

### Inferences
- The framework transfers directly. P(action | screen, history) from the agent is the prior, P(gesture | intended slot) is the confusion matrix (for example 96% diagonal), and the posterior over the intended action is computed after each swipe. Confirmation is triggered when the posterior is below a threshold or the action is irreversible. Information-theoretically, 4 symbols at 96% accuracy carry less than 2 bits per selection (about 1.7 bits under a symmetric-error assumption; this is my own calculation, not a published figure). So the agent's top-4 coverage is the bottleneck.

### Gaps
- No paper found applying Nomon/Dasher-style Bayesian selection to LLM agent action choice.
- No paper found reporting top-K next-action coverage of LLM agents as an interface metric.

## Q5. What has not been done (gap statement)

### Takeaway
Based on this pass, no published system or study was found that combines (a) a tiny-vocabulary (around 4-gesture) wearable, (b) a general computer-use / GUI agent that proposes K candidate next actions for one-gesture selection, and (c) explicit modelling of gesture-recognition uncertainty in confirmation or undo policy. Each piece exists on its own.

### Cited Findings
- Wearable plus agent: activation and voice only (ring tap plus whisper; glasses plus speech) — [MSR ring PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2026/03/chi26d-sub1595-cam-i61.pdf); [VisionClaw](https://arxiv.org/abs/2604.03486)
- Shipped gesture vocabularies are navigation and dismissal only — [eWeek/Meta](https://www.eweek.com/news/meta-ray-ban-display-ai-glasses-launch/); [heise/Apple](https://heise.de/-10491847)
- Agent-proposes / human-selects exists only with full input — [CowPilot](https://arxiv.org/abs/2501.16609); [Morae](https://arxiv.org/abs/2508.21456); [Magentic-UI](https://arxiv.org/abs/2507.22358)
- Oversight via confirmation is weak even with reliable input (21% block rate) — [arXiv 2604.04918](https://arxiv.org/html/2604.04918v2)
- Noisy-channel selection exists only for text entry and switch AAC — [Nomon](https://arxiv.org/abs/2204.01619)

### Inferences
- Defensible novelty claims, all to be framed as design or hypothesis until tested: (1) a K=4 action-proposal interface matched to a 4-gesture wearable; (2) posterior-based confirmation that combines classifier confusion with agent confidence and action reversibility; (3) evaluation metrics such as top-4 coverage, swipes per task, error-induced irreversible actions, and comparison to CowPilot-style full-input baselines.
- Risk to novelty: Meta's Neural Band platform could add agentic Meta AI flows with swipe-to-select at any time. That would be a product precedent, though probably without published uncertainty modelling.

### Gaps
- Search coverage was limited to about 25 queries and fetches. ASSETS/IMWUT papers on switch-access LLM agents and on EMG/IMU control of LLM chat may exist and were not found. A targeted ACM DL search is recommended.
- Unverified: Humane/Rabbit details, ChatGPT agent confirmation specifics, Operator system-card numbers, Apple double-tap with Siri, Android XR glasses gestures, Satori (CHI 2025) content.
