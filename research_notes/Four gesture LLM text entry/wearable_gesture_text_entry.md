# Wearable and Gesture-Based Text Entry: Sensing, Gesture Alphabets, Speeds, and Gaps (relevance: 4-swipe piezo wristband + LM decoding)

Research date: 2026-10-17. Verification legend: **[V-primary]** = number read in the primary paper text this session; **[V-secondary]** = number from a search snippet / secondary page this session, primary not opened; **[UNVERIFIED]** = from background knowledge only, not confirmed this session; treat as a lead. All WPM/CER values below are *reported* values from the cited source; anything labeled "Inference" is my own estimate or reasoning.

## Q1. Meta Reality Labs sEMG wristband (Nature 2025) and Neural Band text entry (2025-2026)

### Takeaway
Meta's Nature paper reports generic (no per-user calibration) sEMG handwriting at a median 20.9 WPM closed-loop, a nine-gesture discrete set at 0.88 detections/s, and a 16% handwriting improvement from 20 min of personalization data; decoding was greedy CTC with **no language model**. A productized "Neural Handwriting" feature for Ray-Ban Display launched in early access (CES, Jan 2026) and to all US owners by about Aug 2026, with no public WPM/accuracy figures.

### Cited Findings
- Bibliographic details: "A generic non-invasive neuromotor interface for human-computer interaction", CTRL-labs at Reality Labs (Kaifosh & Reardon are corresponding/lead), *Nature* 645, 18 Sept 2025, pp. ~702-711 (page 706 seen in text), DOI 10.1038/s41586-025-09255-w [V-primary] — [Nature](https://www.nature.com/articles/s41586-025-09255-w); [CSHL repository PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf); [RePEc record](https://ideas.repec.org/a/nat/nature/v645y2025i8081d10.1038_s41586-025-09255-w.html)
- Abstract numbers: closed-loop median 0.66 target acquisitions/s (continuous navigation), 0.88 gesture detections/s (discrete gestures), handwriting 20.9 WPM; personalization improves handwriting decoding by 16% [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Baselines in the same paper: open-loop handwriting on a surface without a pen was 25.1 WPM (n = 75 different participants) vs 20.9 WPM with the sEMG decoder; mobile phone keyboard ~36 WPM; Joy-Con controller 1.45 completions/s vs 0.88 with sEMG gestures; MacBook trackpad 0.68 s and mocap wrist 0.96 s vs 1.51 s sEMG wrist acquisition [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- **Language model: none.** Quote: "Experimentation presented in this work used 'greedy' CTC decoding, although beam decoding with language modelling in our decoders would have been possible." Character-level CTC with FastEmit regularization [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Online CER definition: per-trial Levenshtein / max(prompt length, output length), median over trials per block; WPM assumes 5 chars/word, timed first-to-last emitted character; the paper warns online CER is not comparable to its offline CER [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Online CER values: the bioRxiv preprint (v1, 2024) reported 17.0 adjusted WPM and median CER 0.048 and 0.052 in the last two blocks [V-secondary, via fetch of preprint]; one search summary claimed CER "~35% practice → ~25% evaluation", which I could not confirm and which conflicts with the preprint. Treat the Nature online-CER value as **unconfirmed** — [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v1.full)
- Personalization: fine-tuning with 20 min of user data gave a 16% median improvement in offline CER (Nature version); for a 25-participant generic model, 20 min of personalization ≈ 14,000 min of extra generic data, but only ~1 min for a 200-participant model. Personalization helped 88% of participants on themselves; personalizing on another person helped only 7% of pairs; gains were largest for users with high generic CER [V-primary]. (The preprint reported 8.35% ± 2.36% relative gain [V-secondary].) — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf); [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v1.full)
- Discrete gestures: nine gestures, including thumb swipes (left/right/up/down), thumb tap, index pinch-and-hold, middle pinch-and-hold; errors mix decoder and behavioral errors; preprint reported median first-hit probability 0.92 [V-primary for gesture set; V-secondary for 0.92] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Hardware: sEMG-RD, dry electrodes, 48 electrode pins forming 16 bipolar channels, 2 kHz sampling, four band sizes, adjustable gap; training corpus up to 6,527 participants for the largest handwriting model (60.2M parameters); ~20-24 naive test users per online task [V-primary / V-secondary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Donning variability: the paper explicitly addresses variability "across different participants and band donnings" and electrode displacement as a key barrier [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Handwriting dataset released (FRL Handwriting, sEMG) on NEMAR — [GitHub nm000106](https://github.com/nemarDatasets/nm000106)
- Product: Meta announced "Neural Handwriting" for Ray-Ban Display + Neural Band at CES 2026 (early access, US, English; WhatsApp and Messenger) — [Meta blog CES 2026](https://www.meta.com/blog/ces-2026-meta-ray-ban-display-teleprompter-emg-handwriting-garmin-unified-cabin-university-of-utah-tetraski/); [UploadVR](https://www.uploadvr.com/neural-band-handwriting-recognition-start/)
- Reported ~Aug 2026 rollout to all US Ray-Ban Display owners (in-air or on-surface finger writing into WhatsApp, Messenger, Instagram, native messaging); $799 glasses with band included; international rollout paused due to supply — [Ubergizmo, Aug 2026](https://www.ubergizmo.com/2026/08/ray-ban-display-meta-rolls-out-neural-handwriting-for-silent-ai-interactions/); [Digital Trends](https://digitaltrends.com/computing/metas-ray-ban-display-now-types-messages-from-your-finger-movements)
- UploadVR coverage gives no WPM, accuracy, personalization or LM details for the product [V-secondary] — [UploadVR](https://www.uploadvr.com/neural-band-handwriting-recognition-start/)

### Inferences
- The Nature system is the strongest calibration-free wrist text entry result, but it relies on 16-channel 2 kHz dry sEMG and data from thousands of people; it is not low-cost passive sensing. Its four thumb swipes are used for navigation, not text.
- Because Meta used greedy CTC without an LM, "add an LM to wrist decoding" is an open, acknowledged improvement path, not something Meta claimed.
- Product speed/accuracy is unpublished; any comparison to the product must be qualitative.

### Gaps
- Exact Nature online CER values (figure 3i) could not be extracted from the text layer.
- Orion (2024 prototype) text entry details not researched this session; no public metrics found.
- Whether the shipping product uses an LM/autocorrect is not stated in sources found.

## Q2. Directional / unistroke alphabets and smartwatch keyboards: which use only 4 directions; speeds and errors

### Takeaway
Pure 4-direction alphabets exist (MDITIM, UDLR, Wobbrock's 4-key EdgeWrite, H4-Writer with 4 buttons) but are deterministic multi-stroke codes (about 3 strokes per letter) without LM disambiguation, reaching ~5-16 WPM (H4-Writer ~20 WPM with practice). Smartwatch keyboards with statistical decoding reach 20-24 WPM but use touch, not wrist motion.

### Cited Findings
- **MDITIM** (Isokoski & Raisamo, 2000): letters built from strokes in four cardinal directions (up/down/left/right); KSPC 3.06; ~4.9 WPM and ~3% errors with arrow keys after ten 30-min sessions (stylus-trained) [V-primary via Wobbrock 2006 text] — [Wobbrock et al. CHI 2006 PDF](https://faculty.washington.edu/wobbrock/pubs/chi-06.02.pdf)
- **UDLR** (Evreinova, Evreinov & Raisamo 2004): every letter = 3 arrow-key presses (KSPC 3.00); ~13.5 WPM after a week, high confusion from arbitrary codes [V-primary via Wobbrock 2006] — [Wobbrock et al. CHI 2006](https://faculty.washington.edu/wobbrock/pubs/chi-06.02.pdf)
- **Wobbrock, Myers & Rothrock, "Few-key Text Entry Revisited: Mnemonic Gestures on Four Keys", CHI 2006**: EdgeWrite letters mapped to 4 keys (corner-sequence gestures, timeout segmentation); 5 subjects × 10 sessions; session-10 means 4-key 15.95 WPM, 4-key no visualization 16.86 WPM (vs 3-key 9.81, 5-key 12.86); best subject 24.06 WPM; uncorrected errors 4-key 1.69%, no-viz 2.00%; removing visualization caused no detriment (eyes-free) [V-primary] — [CHI 2006 PDF](https://faculty.washington.edu/wobbrock/pubs/chi-06.02.pdf)
- **H4-Writer** (MacKenzie, Soukoreff & Helga, UIST 2011, "1 Thumb, 4 Buttons, 20 Words Per Minute"): Huffman-coded key sequences on 4 buttons, eyes-free capable, ~20 WPM after practice per title [V-secondary] — [ACM DL](https://dl.acm.org/doi/10.1145/2047196.2047258); [York page](https://www.yorku.ca/mack/uist2011.html); gestural Huffman variant — [MobileHCI 2013](https://www.yorku.ca/mack/mhci2013e.html)
- **EdgeWrite** (Wobbrock, Myers & Kembel, UIST 2003): stylus unistroke along edges/diagonals of a square (corners); novices 18% more accurate than Graffiti, speed n.s. (6.6 vs 7.2 WPM); all motor-impaired subjects could use it [V-secondary] — [UIST 2003 PDF](https://faculty.washington.edu/wobbrock/pubs/uist-03.pdf)
- **Graffiti vs Unistrokes** (Castellucci & MacKenzie, CHI 2008): over 20 sessions, Graffiti 4.0 → 11.4 WPM, Unistrokes 4.1 → 15.8 WPM; correction rates Graffiti ~26%, Unistrokes 43% → 16% [V-secondary] — [CHI 2008 PDF](https://www.yorku.ca/mack/chi2008-p305.pdf)
- Unistrokes original: Goldberg & Richardson, "Touch-typing with a stylus", INTERCHI 1993 [UNVERIFIED this session]
- **Quikwriting** (Perlin, UIST 1998): two-gesture-per-letter navigation from center to perimeter zones [V-secondary description; numbers not found] — [Wobbrock 2006 / search](https://www.yorku.ca/mack/uist2011.html)
- **Smartwatch keyboards (reported WPM, as compiled in a survey/comparison)**: WatchWriter 24.0, DualKey 21.6, SwipeBoard 19.6, SplitBoard 15.9, COMPASS 12.5, ZoomBoard 9.3 [V-secondary; compiled numbers, conditions differ] — [SwipeRing paper](https://openreview.net/pdf?id=eVyaNs3sm8a); [Arif smartwatch survey](http://www.m.asarif.com/pub/Arif_HCII2016.pdf)
- SwipeBoard (Chen, Grossman & Fitzmaurice, UIST 2014): two swipes per letter (region then letter) — essentially a 2-level directional code [V-secondary] — [ResearchGate](https://www.researchgate.net/publication/288205295_Swipeboard_A_text_entry_technique_for_ultra-small_interfaces_that_supports_novice_to_expert_transitions)
- WatchWriter (Gordon, Ouyang & Zhai, CHI 2016, Google): tap and gesture typing on miniature QWERTY with statistical decoding (LM) — [ResearchGate](https://www.researchgate.net/publication/301931220_WatchWriter_Tap_and_Gesture_Typing_on_a_Smartwatch_Miniature_Keyboard_with_Statistical_Decoding)
- COMPASS (Yi et al., CHI 2017): bezel-rotation keyboard for non-touch watches, three cursors + button — [PDF](https://pi.cs.tsinghua.edu.cn/lab/papers/COMPASS_Xin%20Yi_CHI2017.pdf)
- Vulture (Markussen, Jakobsen & Hornbæk, CHI 2014): mid-air word-gesture keyboard with pinch delimiter; 20.6 WPM after 10 sessions; 28.1 WPM after training on a few phrases (59% of direct touch) [V-secondary] — [CHI 2014 PDF](https://www.kasperhornbaek.dk/papers/CHI2014_Vulture.pdf)
- Related eyes-free directional methods: QB-Gest (bimanual thumb drags toward Qwerty letter directions) — [York](https://www.yorku.ca/mack/hcii2020b.html); DuSK short/long directional swipes (2025) — [arXiv 2503.02133](https://arxiv.org/pdf/2503.02133)

### Inferences
- Only MDITIM, UDLR, 4-key EdgeWrite and H4-Writer (as 4 buttons) are true 4-symbol input alphabets; all decode deterministically (letter = fixed multi-symbol code). 8pen, Quikwriting and marking menus use more directions or 2-level selection.
- A 4-swipe + LM design changes the trade-off: 1 stroke per letter (KSPC ≈ 1 before corrections) rather than ~3, at the cost of ambiguity resolved by the LM — i.e., it is an ambiguous keyboard (T9-like) rather than a unistroke code.

### Gaps
- 8pen and marking-menu text entry numbers not retrieved. Original Goldberg & Richardson and Perlin numbers not verified.

## Q3. Wrist/forearm/hand/finger sensing for text entry and gestures

### Takeaway
Wrist IMU (WrisText, 6 directions, 9.9-15.2 WPM), finger-worn rings (TipText ~11.9-13.3 WPM; BiTipText ~23-25 WPM), surface/10-finger tapping (TapType 19 WPM) and EMG (Meta 20.9 WPM; Myo-based keyboards) dominate. Passive piezo/strain bands (WristFlex, Tomo) are mostly gesture recognizers, not text-entry systems.

### Cited Findings
- **WrisText** (Gong et al., UIST 2018 [venue UNVERIFIED]): one-handed smartwatch text entry by whirling the wrist toward **six directions**, each a key on a circular alphabetical keyboard; average 9.9 WPM across sessions, up to 15.2 WPM at the end; DTW + kNN recognition of wrist motion (watch IMU + sensors), with word-level disambiguation [V-secondary] — [WrisText PDF](https://www.sfu.ca/~xingdong/papers/WrisText.pdf)
- Wrist-IMU finger-tap sensing struggles to exceed ~10 gestures at high accuracy (claim in a related paper's summary) [V-secondary] — [FineType CHI 2025](https://dl.acm.org/doi/10.1145/3706598.3714278)
- **FineType** (CHI 2025): wristband IMU + IR camera detects tap gestures by finger combinations/postures; multitask CNN [numbers not retrieved] — [ACM DL](https://dl.acm.org/doi/10.1145/3706598.3714278)
- **TipText** (Xu et al., UIST 2019): thumb types on an invisible ambiguous keyboard on the index fingertip; ~11.9 WPM (some sources 13.3 WPM) [V-secondary; sources conflict] — [ResearchGate](https://www.researchgate.net/publication/336654303_TipText_Eyes-Free_Text_Entry_on_a_Fingertip_Keyboard); [FlexType related work, 13.3 WPM](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- **BiTipText** (CHI 2020): bimanual fingertip keyboard, ~23.4 WPM (0.03% uncorrected errors), ~25 WPM in another report [V-secondary] — [PDF](https://www.cs.dartmouth.edu/~xingdong/papers/BiTipText.pdf)
- **DRG-Keyboard** (IMWUT 2023): two IMU rings, thumb-to-index micro-gesture typing [numbers not retrieved] — [PDF](https://pi.cs.tsinghua.edu.cn/wp-content/uploads/2023/04/IMWUT-DRG-Keyboard.pdf)
- **RingGesture** (arXiv 2024): ring-based mid-air gesture typing with deep word prediction — [arXiv 2410.18100](https://arxiv.org/pdf/2410.18100)
- **picoRing** (2024): battery-free rings for thumb-to-index input — [arXiv 2411.13065](https://arxiv.org/pdf/2411.13065)
- **TapType** (2024): ten-finger typing on surfaces via Bayesian inference; 19 WPM at 0.6% CER, experts >25 WPM [V-secondary] — [arXiv 2410.06001](https://arxiv.org/pdf/2410.06001)
- **HiFinger**: pressure sensors on fingers, 6 ambiguous groups × 2-step selection for VR/AR [V-primary via FlexType related work] — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- **Myo EMG keyboard** ("Typing Everywhere with an EMG Keyboard", 2020): virtual keyboard from Myo armband while typing on any surface [numbers not retrieved] — [Springer](https://link.springer.com/content/pdf/10.1007/978-3-030-60245-1_17)
- **EMG + piezoelectric (PZT) silent text input** (2025, Sensors/PMC): combined sensors; best non-feature model 95.63% accuracy [task set not retrieved] — [PMC12031478](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12031478/)
- Tap Strap: commercial 10-finger chorded wearable; no WPM data found [V-secondary] — [TapType arXiv](https://arxiv.org/html/2410.06001)
- Skinput (Harrison et al., CHI 2010, bio-acoustic), WristFlex (Dementyev & Paradiso, UIST 2014, pressure-sensor wristband), Tomo (Zhang & Harrison, UIST 2015, EIT wristband), RingText, TypeRing, DigiTouch, Tap Strap WPM: **[UNVERIFIED this session]**; these are gesture-recognition or alternative-keyboard leads, not confirmed text-entry metrics.

### Inferences
- Among wrist-only (no rings, no camera) text entry, WrisText is the closest analogue: directional wrist motion + small key set + disambiguation; it uses 6 directions with a watch IMU rather than 4 swipes with piezo.
- I found no peer-reviewed piezoelectric/strain wristband *text-entry* study with WPM/CER; the 2025 EMG+PZT paper reports classification accuracy, not entry rate.

### Gaps
- Calibration requirements per system were mostly not retrievable in the snippets; WrisText calibration not confirmed.

## Q4. Mid-air / eyes-free gesture typing for AR/VR and smart glasses, 2022-2026, including LLM decoding

### Takeaway
Recent XR work uses neural or LLM decoders for word-gesture input (Gesture2Text, RingGesture, CHI 2025 FLAN-T5 Flexible Typing); none found pair an LLM with a wrist band that emits only a handful of discrete symbols.

### Cited Findings
- LLM decoder for keyboards (CHI 2025, "LLM Powered Text Entry Decoding and Flexible Typing on Smartphones"): fine-tuned FLAN-T5; 93.1% top-1 on user-drawn gestures, 95.4% on real tap data; users mixed gestures (35.9%), taps (29.0%), multi-stroke (6.1%), tap-gestures (29.0%) [V-secondary] — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12723528/); [ACM DL](https://dl.acm.org/doi/abs/10.1145/3706598.3714314)
- Gesture2Text (2024): generalizable XR word-gesture decoder via coarse trajectory discretization and pre-training — [arXiv 2410.18099](https://arxiv.org/pdf/2410.18099)
- One-Dimensional Handwriting on smart glasses (Yu et al., Tsinghua) — [PDF](https://pi.cs.tsinghua.edu.cn/lab/papers/p71-yu.pdf)
- STAR (2025): smartphone-analogous typing in AR — [arXiv 2511.21143](https://arxiv.org/html/2511.21143v1)
- TapGazer (CHI 2022): finger tapping + gaze word selection — [ACM DL](https://dl.acm.org/doi/fullHtml/10.1145/3491102.3501838)
- GestureGPT: LLM agents for zero-shot free-form gesture understanding (not text entry) — [arXiv 2310.12821](https://arxiv.org/pdf/2310.12821)
- HeadText: hands-free text entry with head gestures from an earpiece IMU — [arXiv 2205.09978](https://arxiv.org/pdf/2205.09978)

### Inferences
- LLM decoding in text entry so far consumes rich spatial signals (touch traces); applying an LLM to a 4-symbol ambiguous stream with soft posteriors is a less explored niche.

### Gaps
- Quantitative WPM for Gesture2Text, RingGesture, STAR not retrieved this session.

## Q5. Is "few discrete gestures (<=6) + language model" (especially 4 swipes + LM) already done? Closest matches

### Takeaway
The *interaction concept* — 4 ambiguous letter groups + a long-span language model — is published: **FlexType (Gaines, Baker & Vertanen, IUI 2023)**, 12.0 WPM at 2.03% CER, eyes-free, one hand, but input was 1-4-finger taps on a phone touchscreen, not wrist sensing. I found no published wrist-worn system using only 4 directional swipes + LM; closest wrist analogue is WrisText (6 directions). Novelty must rest on the sensing (low-cost piezo band), soft posteriors fed to the LM, short calibration, and rewearing robustness — not on the 4-group LM idea itself.

### Cited Findings
- **FlexType** (IUI '23, Sydney, DOI 10.1145/3581641.3584077): ambiguous keyboard with **four character groups**, groups optimized with a disambiguation algorithm using a long-span LM; decoder based on VelociTap with a 12-gram character LM and a 4-gram word LM (100K vocab), returns 6-best list; input = tap with 1-4 fingers on a black OnePlus 5T screen with text-to-speech feedback; swipes for backspace/word end/next candidate; 16 participants (between-subjects, 8 per condition) [V-primary] — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- FlexType results: recommended alphabetical groups {a-e}, {f-m}, {n-r}, {s-z,'}: 12.0 WPM, 2.03% CER after ~4 h practice, no visual feedback; no significant difference vs optimized unconstrained groups (aim: separate vowels) after 4 h [V-primary] — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- FlexType authors explicitly name eyes-free, single-switch, BCI and forearm-sensing applications for 4 groups ("four fingers... forearm [1]") [V-primary] — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Ambiguous-keyboard lineage cited by FlexType: Scanning Ambiguous Keyboard (MacKenzie & Felzer, 3 groups + space); Optimal-T9 smartwatch (9 keys); T18 (15.7 WPM); Lesher et al. grouping optimization; Lee et al. 5/10-key fingernail keyboards; FlexType states no prior grouping optimization used a long-span LM [V-primary] — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- WrisText: 6 wrist directions over a circular alphabetical keyboard, 9.9-15.2 WPM — closest *wrist-motion* ambiguous keyboard [V-secondary] — [WrisText PDF](https://www.sfu.ca/~xingdong/papers/WrisText.pdf)
- Meta Nature 2025: 4 thumb-swipe directions in the 9-gesture set, but used for navigation; text via handwriting without LM [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- 4-direction deterministic codes (MDITIM, UDLR, 4-key EdgeWrite) — no LM — [Wobbrock CHI 2006](https://faculty.washington.edu/wobbrock/pubs/chi-06.02.pdf)

### Inferences
- An evaluation of "4 swipes + LM" should benchmark against FlexType (12.0 WPM / 2.03% CER, same 4-group structure) and WrisText (6-direction wrist) as the two nearest baselines, and report KSPC, WPM, CER and word disambiguation accuracy.
- Estimate (not a reported number): with ~96% per-stroke accuracy, a 5-letter word has ~0.96^5 ≈ 82% chance of all strokes correct under hard decisions; passing soft posteriors to the LM is what lets the decoder recover some stroke errors — an explicit advantage over FlexType's exact-sequence matching (FlexType's decoder returned words "that matched a user's exact tap sequence").
- Swipes as letter input collide with swipes as editing commands (FlexType used swipes for backspace/word-end); a 4-swipe-only design needs a separate delimiter/correction strategy (e.g., timing, a fifth gesture, or LM-driven segmentation).

### Gaps
- I could not exhaustively rule out non-indexed theses/workshop papers combining wrist IMU/EMG 4-direction input with an LM; claim "not found" rather than "not done".

## Q6. Remaining gaps relevant to the idea

### Takeaway
Open gaps: (1) low-cost passive sensing for text entry with reported WPM/CER; (2) short or zero calibration without thousands of training users; (3) rewearing/donning robustness quantified in text entry; (4) LM/LLM decoding from soft per-gesture posteriors of a tiny gesture set.

### Cited Findings
- Electrode displacement and donning variability are recognized barriers; Meta solved them with large-scale data (thousands of participants), 16 channels at 2 kHz [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Personalization is valuable mostly when generic data are small (20 min ≈ 14,000 min generic data for a 25-participant model) — supports short user calibration for small-data systems [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- Meta's own decoder did not use an LM though they note it was possible [V-primary] — [CSHL PDF](https://repository.cshl.edu/id/eprint/42049/1/10.1038.s41586-025-09255-w.pdf)
- LLM text decoding work targets rich touch/gesture inputs (FLAN-T5, CHI 2025) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12723528/)
- Wrist IMU finger-gesture vocabularies tend to saturate around 10 gestures — [FineType](https://dl.acm.org/doi/10.1145/3706598.3714278)

### Inferences
- A low-cost 4-channel piezo band with 4-8-stroke calibration sits in a gap between Meta (high-channel, data-hungry, calibration-free) and FlexType (LM-based but touchscreen). Claims must stay within the project's evidence boundaries: per-stroke accuracy in presegmented same-session conditions does not establish continuous text entry WPM, rewearing or cross-person performance.

### Gaps
- No primary source found reporting text-entry WPM for piezoelectric/strain/FMG wristbands; no verified rewearing analyses for wrist text entry outside Meta.
