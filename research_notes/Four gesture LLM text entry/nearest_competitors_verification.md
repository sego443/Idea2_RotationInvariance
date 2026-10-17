# Nearest competitors to 4-direction piezo-wristband text entry with LM decoding (verified, as of 2026-10-17)

Scope note: our idea = 4 in-air wrist swipes (UP/DOWN/LEFT/RIGHT) on a 4-channel piezoelectric wristband, one stroke per letter over 4 letter groups, LM/LLM decoding from soft per-stroke posteriors, grouping optimized against the recognizer confusion matrix, and LM-based unsupervised resolution of the cluster-to-direction mapping after rewear plus pseudo-label recalibration. Everything below is from primary sources unless marked UNVERIFIED. "Reported" = number stated in the source; "Inference" = my reading.

## Q1. LowKeyEMG (arXiv 2507.19736)

### Takeaway
LowKeyEMG is the closest wearable competitor: 4 letter-group gesture keys (+1 space, +2 system gestures) decoded from 64-channel forearm sEMG, with an RWKV-169M LM doing word beam search with passage context; 3 able-bodied participants, 23.3 WPM expert. It uses per-day supervised calibration, a corpus-collision-optimized grouping (not confusion-matrix-aware), assumes homoscedastic gesture errors, does not state that soft classifier probabilities are passed to the decoder, and does not address rewear or label-free calibration. Still an arXiv preprint (v1, 26 Jul 2025); no later venue found.

### Cited Findings
- Title "LowKeyEMG: Electromyographic typing with a reduced keyset"; authors Johannes Y. Lee, Derek Xiao, Shreyas Kaasyap, Nima R. Hadidi, John L. Zhou, Jacob Cunningham, Rakshith R. Gore, Deniz O. Eren, Jonathan C. Kao; all University of California, Los Angeles; arXiv:2507.19736v1 [cs.HC], 26 July 2025 — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Venue: only arXiv v1 found; searches for a conference/journal version returned only arXiv — [arXiv abs](https://arxiv.org/abs/2507.19736v1)
- Gesture set: 7 classes = 4 alphabetic/punctuation "gesture keys" (thumb, index, ring, little finger presses), 1 space key (wrist ulnar deviation), 2 system keys (lift little finger = undo; lift middle+ring = select) — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Hardware: 2 custom saline-sponge electrode arrays, 32 electrodes each (64 channels), 2000 Hz, forearm (one array distal, one proximal) — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Grouping (4-optimized layout, reported): {a,c,d,u,v,w,x}, {g,j,k,l,m,o,p,y}, {b,e,r,t,z,comma}, {f,h,i,n,q,s,period,question mark} — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Grouping optimization: "minimized the expected number of collisions between words" using Brown-corpus word frequencies, with a Gumbel-softmax relaxation and gradient descent on word-word collisions (i.e., text-only; no recognizer confusion matrix) — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Decoder: RWKV LM, checkpoint "RWKV/rwkv-4-169m-pile"; word beam search adapted to a tokenized LM, beam width K=30; best condition feeds previous sentences of the passage as context ("completion+context") — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Error modeling: authors state the word filter and character model "did not account for varying error distributions ... instead assuming homoscedastic gesture errors" — [arXiv HTML](https://arxiv.org/html/2507.19736v1)
- Soft vs hard: the paper as read does not explicitly state that classifier softmax probabilities enter the beam search; the extraction indicates gesture labels drive decoding. Treat as "not stated / apparently hard labels" (see Gaps) — [arXiv HTML](https://arxiv.org/html/2507.19736v1)
- Classifier: EEGNet-style CNN, input 64 x 240 (120 ms); window-shift augmentation (16 strides) plus Gaussian noise — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Calibration: "On each day, we collected 80 events total per gesture, totaling 560 gesture events", split 80/10/10 (448/56/56) — per-day supervised calibration; rewear/electrode re-placement between days not discussed — [arXiv HTML](https://arxiv.org/html/2507.19736v1)
- Participants: 3 healthy participants, ages 25–33 (no motor-impaired users tested; motor impairment named as motivation) — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Online WPM (reported): base 14.4; completion 17.2; completion+context 19.5; "expert" (3rd repetition) 23.3 average, range 16.4–27.8 — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Gestures per character (reported): base 1.11, completion 0.95, completion+context 0.83 (17% gesture-efficiency gain) — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Offline/simulated word accuracy (reported): RWKV vs 4-gram word model, top-1 96.9% vs 66.6%, top-3 99.2% vs 85.8%, simulated on 6 public-domain book passages (877–1585 chars), optimized 4-key layout, passage context — [arXiv HTML](https://arxiv.org/html/2507.19736v1)
- Gesture decoding: >90% offline test accuracy; online error rates reported per participant as 5.8–9.5% (H1), 20.0–22.7% (H2), 10.4–13.9% (H3) — [arXiv HTML](https://arxiv.org/html/2507.19736)
- Selection: >=94.8% of selections were in the top 2 candidates — [arXiv HTML](https://arxiv.org/html/2507.19736v1)
- Future work mentions adding a personalized layer (e.g., n-gram) to the LM; no mention of soft-probability decoding or label-free recalibration — [arXiv HTML](https://arxiv.org/html/2507.19736v1)

### Inferences
- Differentiators available to us vs LowKeyEMG: (a) 4-channel low-cost piezo vs 64-channel wet sEMG; (b) in-air directional wrist swipes vs finger presses; (c) a dedicated space/undo/select gesture set is absent in our minimal 4-symbol scheme (we would need word segmentation by timing or another cue — a design gap we must address); (d) confusion-aware grouping vs text-only collision minimization; (e) soft-posterior noisy-channel decoding vs homoscedastic/label-based decoding; (f) rewear handling via LM-based mapping recovery and pseudo-labels, which LowKeyEMG does not attempt.
- LowKeyEMG's own reported per-participant online error rates (up to ~23%) show the decoding gap that confusion-aware soft decoding targets; this supports the motivation but is not our result.

### Gaps
- Whether LowKeyEMG's beam search consumes softmax probabilities could not be confirmed verbatim; the WebFetch summarization did not find an explicit statement. A direct read of the decoding section/appendix PDF is recommended before claiming "hard decisions only" in a paper.
- Number of days per participant and whether arrays were removed between days were not extracted.
- Error rate used in the simulated 99.2% top-3 test was not found.

## Q2. FlexType (Gaines, Baker & Vertanen, IUI 2023) and Vertanen/Gaines follow-ups 2023–2026

### Takeaway
FlexType is the closest grouping/decoding precedent: 4 character groups, entered by tapping 1–4 fingers on a smartphone touchscreen, decoded word-by-word by a VelociTap-based decoder (12-gram character + 4-gram word LM, previous-word context), with groups optimized against an LM-derived (not recognizer-derived) confusion matrix. The follow-ups found (PETRA 2024, EMNLP Findings 2025, BCI-meeting abstracts, CoLM/WiNLP 2026 workshops) stay on touchscreen / AAC / BCI LM work; none uses a wearable sensor or soft gesture posteriors for FlexType.

### Cited Findings
- FlexType: Dylan Gaines, Mackenzie Baker, Keith Vertanen (Michigan Technological University), IUI '23, Sydney, 27–31 Mar 2023, DOI 10.1145/3581641.3584077 — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Input: four groups so users "tap with between one and four fingers on a single hand to designate each group"; swipes for control (one-finger left = backspace char; right = end word; up/down = iterate n-best list). Paper suggests gloves, finger pressure sensors, or "surface electromyography via electrodes on the forearm" as possible sensors, and single-switch/BCI use, but none was implemented — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Optimization: corpus analysis with VelociTap-based decoder (12-gram character model, 4-gram word model), predicting each word given preceding words, top-100 predictions restricted to same-length words; mispredictions build a character-level confusion matrix; metrics "badgram clarity" and "WER clarity" (simulated decoding on 250 test phrases); n-opt local search — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Groups (reported): constrained (alphabetical) (a–e), (f–m), (n–r), (s–z, '); best badgram-clarity unconstrained (a,h,k,n,s,x), (b,e,f,g,j,m,q,u), (c,d,o,t,v,z,'), (i,l,p,r,w,y) [clarity 0.8191; WER clarity 0.8209]; best WER-clarity unconstrained (a,d,f,h,k,q,y,'), (b,c,e,i,j,n,x), (g,l,o,s,v,w), (m,p,r,t,u,z) [0.8542] — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Decoder input: simulated taps are exact group identities; no recognizer noise model for the gesture channel was described in the extracted text (touch input is discrete finger count) — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Study: 20 recruited, 16 completed (8 per condition, between-subjects), 8 one-hour sessions, university students/staff, fluent English; sighted users typing without visual feedback — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Results (sessions 5–8, reported): constrained 12.0 WPM, 2.03% CER; unconstrained 13.5 WPM, 1.81% CER; overall 12.8 WPM, 1.92% CER; no significant difference after ~4 h; unconstrained had more first-session errors — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Improving FlexType (Gaines & Vertanen, PETRA '24, DOI 10.1145/3652037.3652059): added a letter-entry mode; longitudinal study with legally blind users; 7.7 WPM, 7.0% CER after correction; slower than iOS onscreen Braille, no significant difference from VoiceOver keyboard — [keithv.com](https://keithv.com/pub/improveflextype); [Vertanen publications](https://www.keithv.com/pub/)
- Adapting LLMs for Character-based AAC (Gaines & Vertanen, Findings of EMNLP 2025, pp. 15273–15291; arXiv 2501.10582): algorithm for character predictions from subword LLMs, beating classification-layer, byte-level LLM and n-gram baselines; AAC domain adaptation. Not a reduced-keyboard or noisy-gesture paper — [ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-emnlp.826/); [arXiv](https://arxiv.org/html/2501.10582v3)
- Other group outputs 2023–2026 listed on Vertanen's page: pre-trained neural LM character predictions for BCI (BCI Meeting 2023); simulated online cBCI typing with different LMs (Memmott, Gaines et al., BCI Meeting 2025); RSVP keyboard with inquiry preview (J. Neural Eng. 2025); LLM personalization for predictive typing (Bahmani & Vertanen, CoLM workshop 2026; WiNLP 2026); AI-powered AAC interfaces (CHI '26 workshop); A Dataset of Noisy Typing on QWERTY Keyboards (IUI '23) — [Vertanen publications](https://www.keithv.com/pub/)

### Inferences
- FlexType already claims "first" LM-aware grouping optimization; our novelty must be framed as optimizing against the recognizer's physical confusion matrix (sensor-channel confusions), which FlexType does not do — its confusion matrix captures LM disambiguation confusions only.
- The group's BCI work (cBCI simulations, RSVP keyboard) is where probabilistic evidence fusion with LMs is standard; but none of it is a 4-group wearable gesture keyboard.

### Gaps
- Which of the two unconstrained sets was used in the FlexType user study was not extracted.
- Full text of Improving FlexType (PETRA '24) not read (NSF PAR fetch failed); its decoder (still VelociTap n-gram?) unverified.
- Gaines's PhD dissertation (MTU, 2023) not read; could contain unpublished wearable/EMG FlexType variants — UNVERIFIED.

## Q3. Kimura, "3-Key-Input" (arXiv 2606.11642)

### Takeaway
Single author Naoki Kimura (LY Corporation, Tokyo); paper states "Published in: ICASSP 2026". It is an offline, noise-free simulation (exact key sequences, 300 sentences) with Trie / GPT-2 / GPT-4o-family decoders. A 4-key condition was run (a 3→4 key significance test is reported) but Table 2 lists only 2, 3 and 5 keys, so no 4-key CER/WER is reported.

### Cited Findings
- Author/affiliation: Naoki Kimura, LY Corporation, Tokyo, Japan; arXiv:2606.11642v1 [cs.HC], 10 Jun 2026; PDF footer "Published in: ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing" — [arXiv PDF](https://arxiv.org/pdf/2606.11642); also listed at [LY Corp research](https://research.lycorp.co.jp/en/publications/2705)
- Setup: 2–5 keys; mappings layout-based (QWERTY left/middle/right or rows), frequency-balanced, intentionally worst-case; decoders Trie-only, Trie+GPT-2 beam search ("Trie + LM"), Trie+GPT-4o selection; 300 newly written sentences (100 business / conversational / technical) — [arXiv PDF](https://arxiv.org/pdf/2606.11642)
- Table 2 (GPT-4o rows, reported): 2 keys CER 0.233 / WER 0.337; 3 keys CER 0.095 / WER 0.122; 5 keys CER 0.054 / WER 0.068. Trie-only: 2 keys 0.385/0.562, 5 keys 0.112/0.168; 2 keys Trie+LM 0.274/0.413; 3 keys Trie-only 0.216/0.325, Trie+LM 0.142/0.213 — [arXiv PDF](https://arxiv.org/pdf/2606.11642)
- 4 keys: stats "3→4: n=300, dz=0.57, p<0.001"; text "With 4–5 keys accuracy further improves, but the gain is small"; and suggests designers "expose a 4-key mode" for technical vocabulary. No numeric 4-key CER/WER appears in the tables — [arXiv PDF](https://arxiv.org/pdf/2606.11642)
- Key-stream entropy: 2 keys 0.97, 3 keys 1.54, 5 keys 2.26 bits/char — [arXiv PDF](https://arxiv.org/pdf/2606.11642)
- Mapping effect (3 keys, GPT-4o): CER 0.091 for left-middle-right, top-middle-bottom and frequency-based; 0.096 worst-case (p=0.038) — [arXiv PDF](https://arxiv.org/pdf/2606.11642)
- 3-key model comparison (CER): GPT-4o 0.095, GPT-4o-mini 0.137, o1-mini 0.083, o1 0.066, o3-mini 0.071 — [arXiv PDF](https://arxiv.org/pdf/2606.11642)
- Explicit scope: removes "immediacy and device compute constraints"; offline evaluation; no user study; mentions EMG/eye/head/sip-and-puff switches only as AAC design implications; no key-press error/noise model found in the text (noise is mentioned only in related work, e.g., VelociTap) — [arXiv PDF](https://arxiv.org/pdf/2606.11642)

### Inferences
- Kimura's finding that mapping barely matters under a strong LM (noise-free input) is a direct counterpoint to our claim that confusion-aware grouping helps; our argument must rest on noisy-input conditions, where Kimura provides no evidence either way.
- Any 4-key number we attribute to Kimura would be an interpolation (between 3-key 0.095 and 5-key 0.054 CER); do not cite one.

### Gaps
- No 4-key CER/WER in the paper; the IEEE Xplore version of the ICASSP paper was not checked for differences.

## Q4. Targeted sweep 2023–Oct 2026 for closer matches

### Takeaway
No work found that combines (few-direction wrist gestures) + (wearable low-channel sensor) + (soft-posterior LLM decoding) + (confusion-aware grouping) + (LM-driven unsupervised mapping recovery after rewear). The closest prior art for the "calibration-free via language" component is old BCI work (Kindermans et al., NeurIPS 2012), not 2023–2026 gesture text entry. Calibration-free EMG work (Meta, 2025; Olsson et al., 2026) achieves zero-calibration by large-scale pretraining, not by language priors.

### Cited Findings
- Kindermans, Verschore, Verstraeten & Schrauwen, "A P300 BCI for the Masses: Prior Information Enables Instant Unsupervised Spelling" (NeurIPS 2012): unsupervised hierarchical probabilistic P300 speller using transfer from other subjects and a language model; unsupervised performance paralleled or surpassed supervised models without a training session — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2012/hash/c042f4db68f23406c6cecf84a7ebb0fe-Abstract.html)
- Meta sEMG Neural Band: 48 dry electrodes; generic model trained on data from >6,500 participants works without per-user calibration; handwriting text entry 20.9 WPM, improvable by personalization (secondary press coverage of the 2025 Nature paper; primary not re-read here) — [scimex](https://www.scimex.org/newsfeed/could-you-swap-your-mouse-and-keyboard-for-a-smart-bracelet); [heise](https://heise.de/-10500065)
- Meta rolled out neural handwriting for Ray-Ban Display in an Early Access Program (Aug 2026, press report) — [Ubergizmo](https://www.ubergizmo.com/2026/08/ray-ban-display-meta-rolls-out-neural-handwriting-for-silent-ai-interactions/)
- Calibration-free EMG motor-intent decoding via large-scale supervised pretraining (Olsson et al., Advanced Intelligent Systems, 2026) — [Wiley](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aisy.202500791)
- EMG-UP (arXiv 2509.21589, 2025): source-free unsupervised cross-user EMG gesture personalization with high-confidence pseudo-labels — no language model — [arXiv](https://arxiv.org/html/2509.21589v1)
- WrisText (pre-2023 baseline, CHI 2018): one-handed smartwatch text entry by whirling the wrist toward 6 directions, each a key of 4–5 alphabetical letters, with disambiguation — closest "directional wrist gesture ambiguous keyboard" precedent but 6 groups, smartwatch sensing, pre-LLM — [WrisText PDF](https://www.sfu.ca/~xingdong/papers/WrisText.pdf) (fetch failed on certificate; facts from search snippet — UNVERIFIED in full text)
- RingGesture (arXiv 2410.18100, 2024): ring-based mid-air gesture typing with deep-learning word prediction (word-gesture, not 4-group ambiguous) — [arXiv](https://arxiv.org/pdf/2410.18100)
- FineType (CHI 2025): wristband IMU + IR camera finger-combination tapping for text entry — different input paradigm — [ACM DL](https://dl.acm.org/doi/10.1145/3706598.3714278)
- No piezoelectric-wristband text-entry paper was found in the searches run — searches above (absence of evidence only).

### Inferences
- The LM-as-supervisor for mapping recovery is best framed as extending the Kindermans-style "unsupervised spelling with LM prior" to a wearable gesture keyboard with a permutation-unknown cluster-to-group mapping after rewear; this lineage should be cited to avoid an overclaim of novelty.
- The novelty claim that survives the sweep is the combination, not any single component: soft-posterior LLM decoding (known in BCI/touch decoding), confusion-aware grouping (FlexType/LowKeyEMG optimize from text only), and rewear mapping recovery via language (not found in 2023–2026 wearable text entry).

### Gaps
- Search budget limited (~20 calls); Semantic Scholar/ACM DL citation lists of LowKeyEMG and FlexType were not exhaustively checked; CHI 2026 / UIST 2026 / IMWUT 2026 proceedings were not scanned paper by paper.
- Meta Nature 2025 paper details (handwriting personalization gain, any reduced-gesture keyboard) not verified from the primary source.
- WrisText full text (sensing, decoder, WPM) not verified due to fetch failure.
