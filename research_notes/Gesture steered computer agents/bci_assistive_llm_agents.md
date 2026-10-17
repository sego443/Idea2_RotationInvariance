# BCI and assistive low-bandwidth input systems combined with LLMs / AI agents (2022 to October 2026)

Scope note: research done 2026-10-17 with about 20 web tool calls. Several primary pages (Business Wire, Nature, MassDevice, MobiHealthNews) returned HTTP 403 or redirects, so some company claims are cited from secondary press. Every item is labeled **[peer-reviewed]**, **[preprint]**, **[company demo/press]**, or **[unverified]**. Numbers are reported values unless marked "inference".

## Q1. Synchron + OpenAI (2024), Apple BCI HID (2025), NVIDIA Holoscan / Chiral: what the user selects, how options are generated, any evaluation

### Takeaway
Synchron's OpenAI feature is an LLM "auto-complete" that generates candidate replies the Stentrode user picks with a point-and-click (switch-like) input. The Apple BCI HID work routes that same low-bandwidth signal through iOS Switch Control. The NVIDIA GTC 2025 demo showed smart-home control in Vision Pro. None of these has a published quantitative evaluation of the LLM layer: they are company demos.

### Cited Findings
- **[company demo/press]** July 2024: Synchron announced a chat feature powered by OpenAI generative AI in its BCI platform. — [ForkLog](https://forklog.com/en/synchron-integrates-openais-ai-into-brain-computer-interface/); [Silicon UK press release](https://www.silicon.co.uk/press-release/synchron-announces-brain-computer-interface-chat-feature-powered-by-openai)
- **[company demo/press]** Mechanism: the feature "can now generate automated prompts for texting and chatting, categorized by contextual inputs, including the user's emotion". Before, each word had to be typed; now the AI suggests response options. — [ForkLog](https://forklog.com/en/synchron-integrates-openais-ai-into-brain-computer-interface/)
- **[company demo/press]** Mark (Jackson), 64, with ALS, used it in a video to answer text messages from his doctor and to interact with ChatGPT. — [ForkLog](https://forklog.com/en/synchron-integrates-openais-ai-into-brain-computer-interface/)
- **[company demo/press]** Synchron says it does not share BCI brain data with OpenAI. — [MobiHealthNews/Vive summary via search](https://www.mobihealthnews.com/news/synchron-will-enable-thought-control-apple-devices-brain-implant); [OpenAI community thread](https://community.openai.com/t/openais-ai-powered-chat-feature-for-synchrons-brain-computer-interface-bci/870427)
- **[company demo/press]** Input modality: an endovascular Stentrode over motor cortex detects motor intent and transmits it wirelessly, giving "hands-free point-and-click". — [ForkLog](https://forklog.com/en/synchron-integrates-openais-ai-into-brain-computer-interface/)
- **[company demo/press]** May 13, 2025: Apple unveiled a BCI Human Interface Device (BCI HID) profile, and Synchron said it was the first BCI to integrate it natively, for iPhone, iPad and Vision Pro. — [MobiHealthNews](https://www.mobihealthnews.com/news/synchron-will-enable-thought-control-apple-devices-brain-implant); [designboom](https://www.designboom.com/technology/synchron-brain-implant-users-iphones-ipads-minds-apple-05-15-2025/)
- **[company demo/press]** Aug 2025: first public demo of a thought-controlled iPad. Mark (COMMAND study, ALS) used Apple's built-in **Switch Control** with the Stentrode to go through the home screen, open apps and compose text. — [MobiHealthNews](https://www.mobihealthnews.com/news/als-patient-controls-ipad-thought-using-synchron-and-apple-bci-tech); [AppleInsider](https://appleinsider.com/articles/25/08/04/als-sufferer-uses-ipad-in-demonstration-of-thought-control); [MassDevice](https://www.massdevice.com/synchron-debuts-bci-controlled-ipad-apple/)
- **[company demo/press]** March 2025, NVIDIA GTC: Synchron announced "Chiral", a "foundation model of human cognition" pre-trained generatively on neural data (called "Cognitive AI"). It also showed its BCI running on NVIDIA Holoscan and displayed in Apple Vision Pro. Rodney Gorham (ALS, implanted 2020) adjusted lights, music, temperature and appliances with thought control. — [eWEEK](https://www.eweek.com/de/news/synchron-chiral-brain-ai-medical-test/); [Business Wire (403 on fetch)](https://www.businesswire.com/news/home/20250319964709/en); [Fierce Biotech](https://www.fiercebiotech.com/medtech/synchron-and-nvidia-set-sights-ai-model-trained-direct-brain-activity)

### Inferences
- Synchron's design is the closest industrial analogue to our idea: a very-low-bandwidth discrete selector plus LLM-generated candidates. Its scope is communication (replies) and OS navigation through Switch Control (scanning). It is not an agent that proposes multi-step computer actions.
- Switch Control scanning means the user picks among *UI elements* sequentially. It does not pick among *agent-proposed semantic actions*. That is the gap our 4-option agent menu targets.

### Gaps
- No found publication reports how many LLM suggestions Synchron shows per turn, its acceptance rates, its timing, or its error/undo handling.
- Chiral: no paper, benchmark or evaluation found. Whether Chiral is used in any agentic loop is unverified.
- Business Wire primary releases could not be fetched (403), so exact press wording is unconfirmed.

## Q2. Neuralink, Precision, Paradromics, Blackrock: BCI + AI/LLM computer and agent control

### Takeaway
Neuralink is the only company with a documented LLM integration: Grok/ChatGPT generate candidate replies that ALS participant Brad Smith selects by brain-click. No evaluation has been published. No found peer-reviewed paper from any of these companies covers LLM-agent control.

### Cited Findings
- **[press, MIT Tech Review]** Brad G. Smith (third Neuralink recipient, first with ALS, first nonverbal, implanted Nov 2024) uses Grok. Neuralink engineers integrated LLMs, including ChatGPT and Grok, to "serve up a selection of relevant replies to questions, as well as options for things he could say in conversations going on around him". He selects by brain-clicking, which is faster than typing a full answer (that can take minutes). He also gave Grok his progress notes to draft answers. — [MIT Technology Review, 2025-05-07](https://www.technologyreview.com/2025/05/07/1116139/this-brain-implant-gets-a-boost-from-generative-ai/)
- **[press]** The article gives no typing-speed or with/without-AI numbers. It raises authorship and authenticity concerns ("Where does Smith end and Musk's ecosystem begin?"). — [MIT Technology Review](https://www.technologyreview.com/2025/05/07/1116139/this-brain-implant-gets-a-boost-from-generative-ai/); [Futurism](https://futurism.com/nonverbal-neuralink-patient-grok-answers)
- **[secondary press, weak]** By Jan 2026, 21 Neuralink participants were reported. — [Technology.org](https://www.technology.org/2026/01/29/neuralink-brain-chips-connect-21-minds-to-machines/). The claim "participants type 40 WPM mentally" comes from low-quality aggregators and is **[unverified]**.
- **[company press]** Sept 2026: Paradromics reported that its first Connexus participant (Connect-One EFS, FDA IDE) used the system for real-time speech and text communication. Paradromics claims ITR above 200 bit/s in benchmarking. No LLM-agent use is described. — [Business Wire 2026-09-14](https://www.businesswire.com/news/home/20260914434545/en/Paradromics-Achieves-Real-Time-Speech-with-Connexus-Brain-Computer-Interface)
- **[peer-reviewed]** Academic intracortical work with Blackrock arrays (BrainGate / UC Davis) reached long-term independent home use. An ALS participant used speech and cursor decoders for over 3,800 hours over nearly 2 years, sent more than 180,000 sentences, and ran his own PC (messages, browsing, video calls, full-time work). An LLM agent layer is not described as the control mechanism. — [Nature Medicine via PubMed](https://pubmed.ncbi.nlm.nih.gov/42297978/); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13375540/); [UC Davis news 2026-06](https://health.ucdavis.edu/news/headlines/brain-computer-interface-enables-independent-accurate-communication-for-man-living-with-als/2026/06)
- **[press]** Blackrock/BrainGate history: cursor, robotic-arm and imagined-handwriting text demos. Precision Neuroscience: a surface (cortical film) array implanted in humans for safety and data gathering. No LLM-agent demos found. — [MIT Tech Review 2024](https://www.technologyreview.com/2024/04/19/1091505/companies-brain-computer-interfaces/)

### Inferences
- High-bandwidth implants (speech BCIs, high-ITR arrays) reduce the need for option-selection amplification. The "LLM proposes, user picks" pattern appears mainly where bandwidth is tiny (click/switch). That matches our 2-bit/stroke regime.

### Gaps
- No published Neuralink paper on Grok/LLM integration. Its claimed "Telepathy" agentic features are unverified.
- No Precision, Paradromics or Blackrock LLM-agent demos found, as of Oct 2026.
- Speech-BCI LLM rescoring (e.g., Willett et al. 2023 Nature; Card et al. 2024 NEJM) is relevant background but was not re-verified in this session **[unverified here]**.

## Q3. Academic papers 2023 to 2026: EEG/ECoG/intracortical/EMG BCIs with LLMs or AI copilots for action selection / agentic control

### Takeaway
Peer-reviewed work splits into three groups. First, AI copilots for continuous control (Lee et al. 2025 NMI, vision/RL copilot rather than an LLM). Second, LLM-augmented spellers (ChatBCI). Third, EEG-selected skill libraries for robots (NOIR). Work on LLM *agents* downstream of BCIs is mostly 2025 to 2026 preprints and position papers. Two of them (route-safety audit; inference vs. control) argue that decoder uncertainty must be enforced by an external gate, not left to the LLM. That directly supports our "uncertainty-aware confirmation" component.

### Cited Findings
- **[peer-reviewed]** Lee, Lee, Mishra et al. (Kao lab, UCLA), "Brain–computer interface control with artificial intelligence copilots", *Nature Machine Intelligence* 7, 1510–1523 (2025). An EEG decoder is paired with a camera-based AI copilot that infers the user's goal under shared autonomy. — [UCLA Samueli](https://samueli.ucla.edu/ai-co-pilot-boosts-noninvasive-brain-computer-interface-by-interpreting-user-intent/); [Nature (redirect)](https://www.nature.com/articles/s42256-025-01090-y)
  - 4 participants: 3 without impairment, 1 with spinal cord injury (paralysis below the waist). — [UCLA](https://samueli.ucla.edu/ai-co-pilot-boosts-noninvasive-brain-computer-interface-by-interpreting-user-intent/)
  - Cursor task: 8 targets with a 0.5 s hold. The paralyzed participant's performance was 3.9 times higher with the copilot. — [UCLA](https://samueli.ucla.edu/?p=84351); [SingularityHub](https://singularityhub.com/2025/09/08/this-wearable-brain-reading-device-allowed-people-to-control-a-robotic-arm/)
  - Robotic-arm block task: the paralyzed participant finished in about 6.5 min with the copilot and could not finish without it. — [UCLA](https://samueli.ucla.edu/ai-co-pilot-boosts-noninvasive-brain-computer-interface-by-interpreting-user-intent/)
  - The copilot is not an LLM in the found descriptions. Whether the paper discusses LLM copilots as future work is unverified (full text paywalled).
- **[preprint; arXiv 2411.15395]** ChatBCI (Hong, Wang, Najafizadeh, 2024): a P300 speller where GPT-3.5 word suggestions appear as extra keys. Classifier: SWLDA. N=7 online. Task 1 (copy-spelling): −62.14% time, −53.22% keystrokes, +198.96% ITR. Task 2 (improvised): 80.68% keystroke savings, 8.53 characters/min. — [arXiv](https://arxiv.org/abs/2411.15395)
- **[peer-reviewed, CoRL 2023; arXiv 2311.01454]** NOIR (Stanford): SSVEP selects the object (CCA), motor imagery selects skill and parameters (CSP+QDA), and the result maps to 14 parameterized robot skills (Pick, Place, Push...). It covers 20 household tasks. The robot learns the user's object/skill/parameter choices few-shot to cut decoding effort. It is a skill library rather than an LLM. — [arXiv](https://arxiv.org/pdf/2311.01454); [project page](https://noir-corl.github.io/)
- **[preprint, medRxiv Sept 2026]** Gorenshtein et al., "AI agents at the brain-computer interface: separating inference from control". Computational benchmark: 1,065 episodes from 47 people with ALS using a P300 speller (bigP3BCI, with reconstructed decoder confidence). Five LLMs were given tools read_buffer / lookup_action / execute / abstain. Advisory condition: confidence goes in the prompt. Enforced condition: a deterministic gate controls execution.
  - No LLM reduced unfaithful execution below a deterministic threshold at matched coverage.
  - Apparent "22-point safety gains" came from acting less often, not from better decisions.
  - Two models declined through parse failures (54.3 and 30.3 pp).
  - A hybrid (the LLM proposes semantic corrections, an external gate decides) extended coverage in 5 of 10 models with no errors observed.
  — [medRxiv](https://www.medrxiv.org/content/10.64898/2026.09.13.26362955v1.full)
- **[preprint, arXiv 2606.09315]** Tai (Anhui Univ.), "Brain-Prompt Injection: A Route-Safety Audit for BCI–LLM Agents". Decoded EEG commands become instructions that a tool-using LLM routes to actions such as "move the cursor", "send a message" or "confirm a transfer". Data: binary left/right fist execution/imagery, EEGMMI (60 subjects, 5,400 events), offline.
  - An independent EEG confirmation window with split-conformal calibration cut the attack route rate from 1.000 to 0.178 (FAR=0.05) and 0.091 (FAR=0.01).
  - Provenance tagging blocked context injection (1.000→0.000).
  - Dual-decoder agreement alone was bypassed (1.000) unless confirmation was added (→0.000).
  — [arXiv](https://arxiv.org/html/2606.09315v1)
- **[position paper, NeurIPS 2025 position track; arXiv 2510.22095]** Chen et al. (incl. Philip S. Yu), "Embracing Trustworthy Brain-Agent Collaboration...". Proposes moving from BCI to "Brain-Agent Collaboration": agents as active partners that make up for low ITR and calibration burden. No system evaluation. — [arXiv](https://arxiv.org/abs/2510.22095)
- **[preprints, tangential]** BrainAgent (arXiv 2606.25400), EasyBCI Agent (arXiv 2607.29007) and NS-Copilot (arXiv 2609.01971) use LLM agents to *analyze* neural data, not to act for the user. — [BrainAgent](https://arxiv.org/html/2606.25400); [EasyBCI Agent](https://arxiv.org/pdf/2607.29007)
- **[preprint, arXiv 2606.16413]** An augmented-reality brain-robot interface for generalist robot-arm manipulation (2026). Details not verified. — [arXiv](https://arxiv.org/pdf/2606.16413)
- **[peer-reviewed, PMC]** Levels of shared autonomy in brain-robot interfaces combining EEG, EMG and eye tracking, with three autonomy levels (assisted teleoperation, shared autonomy, full automation). — [PMC12852366](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12852366/)

### Inferences
- Across the field, the selection step is either a fixed skill grammar (NOIR), word suggestions (ChatBCI), or reply suggestions (Synchron/Neuralink). No found system has an LLM computer-use agent propose next *GUI actions* that a tiny discrete BCI/gesture alphabet selects.
- The two 2026 safety preprints give a ready argument for our design: keep the confirmation threshold deterministic and calibrated (e.g., conformal), outside the LLM.

### Gaps
- No published paper uses agent/task context to produce **pseudo-labels for recalibrating the input decoder** after rewearing or session change. The closest related idea is NOIR's few-shot learning of user choices, which adapts the *policy*, not the decoder (inference).
- No found paper optimizes **option-set composition for information gain** (e.g., expected-entropy-maximizing 4-option menus) in an LLM+BCI setting. Classic BCI-speller language-model work (e.g., Nomon, RSVP Keyboard) exists but was not re-verified here **[unverified]**.
- EMG-wristband + LLM agent: no paper found. (Meta's sEMG wristband work, Nature 2025, is a likely neighbor but was not searched or verified here **[unverified]**.)

## Q4. Eye-gaze, switch and AAC users with LLMs or agents operating computers/phones

### Takeaway
The strongest peer-reviewed AAC+LLM evidence is SpeakFaster (Nature Communications 2024), and it is about text, not actions. LLM GUI agents for motor-impaired users exist mainly as voice-driven prototypes (DexAssist). No found study evaluates switch or gaze users steering a general computer-use agent by selecting among agent-proposed actions.

### Cited Findings
- **[peer-reviewed, Nature Communications 2024]** SpeakFaster (Google): fine-tuned LLMs plus conversational context expand heavily abbreviated input (word initials plus extra letters/words when needed) into full phrases. Offline simulation: 57% more motor actions saved than traditional predictive keyboards. Two eye-gaze AAC users with ALS (lab and field): text-entry rates 29–60% above baseline. Mobile pilot with 19 non-AAC participants: motor savings matched simulation. The authors propose extending the approach to switch scanning. — [PMC11530652](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530652/); [Google Research](https://research.google/pubs/using-large-language-models-to-accelerate-communication-for-eye-gaze-typing-users-with-als/); [arXiv 2312.01532](https://arxiv.org/pdf/2312.01532)
- **[preprint, arXiv 2411.12214]** DexAssist: a voice-enabled dual-LLM framework for web navigation, aimed at users with fine motor impairments (Parkinson's, cerebral palsy, dyspraxia). — [arXiv](https://arxiv.org/pdf/2411.12214)
- **[demo]** Nomon: a single-switch selection interface for noisy switch users, built with a switch user and AAC experts. It is a probabilistic selection method, not an LLM. — [Michigan Tech](https://digitalcommons.mtu.edu/michigantech-p/16108)
- **[peer-reviewed]** Multimodal access (eye-tracking + switch-scanning) gave fewer total errors than eye-tracking alone in severe motor impairment. — [PMC9576815](https://pmc.ncbi.nlm.nih.gov/articles/PMC9576815/)
- **[company demo]** Synchron/Apple Switch Control is the commercial path from a one-bit BCI "switch" to full iOS control (see Q1). — [AppleInsider](https://appleinsider.com/articles/25/08/04/als-sufferer-uses-ipad-in-demonstration-of-thought-control)

### Inferences
- In AAC, LLMs have been used to *expand* input (abbreviations to sentences, suggestions to replies). They have not been used to *propose actions* for a GUI agent. Our "4 candidate next actions" concept carries the SpeakFaster logic (LLM context buys back keystrokes) from text into action space.

### Gaps
- Google Project Gameface (face-gesture mouse, open-sourced 2023, Android 2024) and "Talk to me"-style accessibility agents were not verified in this session **[unverified]**. Neither is known to combine an LLM agent with action selection.
- No CHI/ASSETS/UIST 2025–2026 paper was found with motor-impaired participants evaluating an LLM computer-use agent through switch or gaze selection of proposed actions. A deeper ACM DL search is recommended.

## Q5. Cross-cutting comparison table and what has not been done

### Takeaway
No found system combines all of these: (a) a cheap non-invasive wearable, (b) a tiny fixed gesture alphabet (4 symbols), (c) a general computer-use agent that proposes the next actions, (d) reserved meta-symbols (reject/more/undo), (e) uncertainty-aware confirmation of irreversible actions, and (f) context-derived pseudo-labels to recalibrate the decoder. Components (c) and (e) appear separately in reply-suggestion demos and 2026 safety preprints.

### Cited Findings (summary table; all values as reported)

| System | Evidence type | Input / bandwidth | What LLM/AI generates | Options per selection | Error handling | Evaluation |
|---|---|---|---|---|---|---|
| Synchron + OpenAI (2024) | company demo | Stentrode motor intent, point-and-click | Reply prompts categorized by context incl. emotion | not reported | not reported | 1 user video (Mark); no metrics — [ForkLog](https://forklog.com/en/synchron-integrates-openais-ai-into-brain-computer-interface/) |
| Synchron + Apple BCI HID (2025) | company demo | Stentrode as switch via Switch Control | none (OS scanning) | scanning UI elements | OS-level | iPad demo, no metrics — [AppleInsider](https://appleinsider.com/articles/25/08/04/als-sufferer-uses-ipad-in-demonstration-of-thought-control) |
| Synchron Chiral / Holoscan (2025) | company demo | Stentrode | foundation model (no details) | n/a | n/a | smart-home demo — [eWEEK](https://www.eweek.com/de/news/synchron-chiral-brain-ai-medical-test/) |
| Neuralink + Grok/ChatGPT (2025) | press | intracortical cursor + brain-click | Candidate replies / conversation contributions | "a selection" (not quantified) | not reported | none published — [MIT TR](https://www.technologyreview.com/2025/05/07/1116139/this-brain-implant-gets-a-boost-from-generative-ai/) |
| Lee et al. NMI 2025 | peer-reviewed | EEG continuous decoding | vision/RL copilot infers goal (not LLM) | continuous | shared autonomy | N=4; 3.9x cursor; arm 6.5 min vs fail — [UCLA](https://samueli.ucla.edu/ai-co-pilot-boosts-noninvasive-brain-computer-interface-by-interpreting-user-intent/) |
| ChatBCI 2024 | preprint | P300 speller | GPT-3.5 word suggestions as keys | extra keys in matrix | not reported | N=7; −62% time, 80.7% keystroke saving, 8.53 char/min — [arXiv](https://arxiv.org/abs/2411.15395) |
| NOIR 2023 | peer-reviewed (CoRL) | SSVEP + MI EEG | fixed 14-skill library (not LLM); few-shot learning of choices | object/skill/param menus | human confirmation steps (details unverified) | 20 tasks — [arXiv](https://arxiv.org/pdf/2311.01454) |
| SpeakFaster 2024 | peer-reviewed | eye-gaze typing | LLM abbreviation expansion | top candidate phrases | user picks or refines | 2 ALS users, +29–60% rate; 19 pilot — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11530652/) |
| Gorenshtein et al. 2026 | preprint (offline) | P300 speller logs | LLM agent with execute/abstain tools | n/a | enforced deterministic gate beats LLM judgment | 47 ALS users, 1,065 episodes — [medRxiv](https://www.medrxiv.org/content/10.64898/2026.09.13.26362955v1.full) |
| Tai 2026 | preprint (offline) | EEG MI binary | tool-using LLM routes | binary | conformal-calibrated EEG confirmation window | EEGMMI 60 subjects — [arXiv](https://arxiv.org/html/2606.09315v1) |

### Inferences (gaps relative to our idea)
- **Cheap wearable + tiny gesture set + general computer-use agent:** not found. Existing LLM integrations depend on implants (Synchron, Neuralink) or lab EEG. Their agent role is reply drafting, not GUI action proposal.
- **Information-optimal option sets:** no found system designs the k=4 candidate set to maximize expected information or task progress per stroke. Our 4-way, ~96%-accurate channel (~1.7–2 bits/stroke after errors; inference) makes this a natural contribution.
- **Uncertainty-aware confirmation:** supported in principle by two 2026 offline preprints (deterministic or conformal gates beat LLM self-judgment). No online user study combines decoder uncertainty with *action reversibility* to decide when to confirm.
- **Context-based decoder recalibration (agent context → pseudo-labels after rewearing):** no prior instance found. NOIR adapts the action policy, not the signal decoder. This looks novel, but it needs a dedicated literature check on "error-related potential / task-context self-calibration" in classic BCI, which this session did not search.
- **Evaluation norms to adopt:** report task success, time per task, selections per task, and confirmation/undo rates. Peer-reviewed comparators report speedup ratios (3.9x), keystroke savings (53–81%), and rate gains (29–60%).

### Gaps
- Primary press releases (Business Wire, Nature full text) were not accessible. Exact wording and any supplementary metrics are unverified.
- This session did not cover: ErrP-based implicit feedback with LLMs, Meta sEMG wristband + AI, Apple/Google OS-level AI accessibility agents (e.g., Gemini-powered Voice Access), or 2025–2026 CHI/ASSETS LLM-for-AAC papers beyond SpeakFaster. These should be checked before claiming novelty.
