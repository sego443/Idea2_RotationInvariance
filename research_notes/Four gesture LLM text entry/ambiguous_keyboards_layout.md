# Ambiguous / reduced-key text entry with very few keys, and letter-grouping optimization (including recognizer-aware optimization)

Scope note: notes compiled 2026-10-17 by a research subagent. "Reported" means the number appears in the cited source; "estimate" or "inference" marks my own reasoning. Items I could not open in full text are marked UNVERIFIED. All numbers for offline simulations assume noise-free key input unless stated.

## 1. T9 / dictionary disambiguation: accuracy vs number of keys; English word-collision rates for 3, 4, 5 keys

### Takeaway
With 8 letter keys (phone keypad), dictionary disambiguation is nearly unambiguous (KSPC about 1.007; about 5% of words need NEXT). Ambiguity rises steeply below 5 keys: MacKenzie's tabulation puts one-key-disambiguation KSPC at 1.05 for a 4-letter-key design (TMK4) and 1.55 for a 2-letter-key design, and a 4-group LM-optimized keyboard (FlexType) gives about 79-85% top-1 word accuracy offline with a long-span LM. I found no clean published table of raw English word-collision rates for exactly 3, 4 and 5 keys.

### Cited Findings
- Phone keypad (T9-style) word-level dictionary disambiguation: KSPC = 1.0072; multitap 2.0342; LetterWise (prefix-based) 1.1500. "Only about 5% of words require the NEXT function" (unweighted). T9 falls to multitap-equivalent speed when about 15% of the least common words are missing from the dictionary. LetterWise reached 21.0 wpm vs multitap 15.5 wpm at session 20. MacKenzie, Kober, Smith, Jones & Skepner, "LetterWise: Prefix-based disambiguation for mobile text input", UIST 2001, pp. 111-120 — [MacKenzie et al. 2001](https://www.yorku.ca/mack/uist01.html)
- MacKenzie's Table 5-1 (Text Entry Systems book chapter, "Text Entry Using a Small Number of Buttons"): KSPC under multitap vs one-key-with-disambiguation, and conditional entropy, by number of keys (T factor includes space key): L1K (T=2): multitap 10.37, disambiguation 20.05, cond. entropy 6.36 bits; L2K (T=3): 5.15 / 1.55 / 2.43; TMK4 (T=5, i.e., 4 letter keys + space): 3.11 / 1.05 / 0.63; Phone (T=9): 2.02 / 1.01 / 0.12; ACD (T=9): 1.84 / 1.01 / 0.10; Qwerty (T=27): 1.00 / 1.00 / 0.00. The average candidate rank for TMK4 is 1.28; for L2K 3.97 — [MacKenzie, Chapter 5](https://www.yorku.ca/mack/chapter5.html)
- Textonym (collision) statistics for the standard T9 layout, from blog analyses (not peer-reviewed): 13,186 of 73,895 English words (17.8%) share a key sequence with another word; hill-climbing re-grouping reduced this to 12.7% — [gfredericks.com](https://gfredericks.com/blog/84). Another reported figure: about 85% of common English words map to a unique digit sequence — [MacKenzie UIST 2001 PDF](https://www.yorku.ca/mack/uist01.pdf) (snippet-level; exact wording UNVERIFIED).
- Dunlop (2004, MobileHCI, LNCS 3160, pp. 342-346), "Watch-top text-entry: can phone-style predictive text-entry work with only 5 buttons?": four alphabetic buttons plus a central space key; corpus-based technical evaluation of error rates plus user studies; "results, though in favour of the phone pad, are encouraging and show such a method is feasible." Numbers not visible on the repository landing page — [Strathprints 2554](https://strathprints.strath.ac.uk/2554/). Author list conflict: search snippet attributes it to Mark D. Dunlop alone; the fetched landing-page summary listed "Dunlop and Brewster" — treat co-authorship as UNVERIFIED.
- Dunlop's follow-up (2008) "Investigating five key predictive text entry with combined distance and keystroke modelling": predicted similar performance with bigram prediction on a five-key pad as on standard phones; novices on five-key pads performed similarly to novice nine-key users — [Strathprints 4830](https://strathprints.strath.ac.uk/4830/)
- FlexType offline (4 groups, noise-free taps, VelociTap decoder with 12-gram character + 4-gram word LM, word length known, left context available): best unconstrained grouping top-1 word accuracy ("WER clarity") 0.8542; badgram-optimized unconstrained set 0.8209; best alphabetically constrained set 0.7872 — [Gaines, Baker & Vertanen, IUI 2023](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- 3-Key-Input (Kimura, LY Corporation; arXiv 2606.11642, stated as Proc. ICASSP 2026 pp. 19392-19396): sentence-level decoding, noise-free keys. CER/WER: 2 keys Trie-only 0.385/0.562, Trie+GPT-2 0.274/0.413, GPT-4o 0.233/0.337; 3 keys Trie-only 0.216/0.325, Trie+LM 0.142/0.213, GPT-4o 0.095/0.122; 5 keys Trie-only 0.112/0.168, GPT-4o 0.054/0.068. Key-stream entropy at 3 keys 1.54 bits/char. 4-key results not tabulated in the fetched HTML — [arXiv 2606.11642](https://arxiv.org/html/2606.11642)

### Inferences
- Estimate: for 4 letter groups with isolated words and word-frequency-only disambiguation, top-1 word accuracy plausibly sits in the 70-85% range (bracketed by TMK4 average rank 1.28 and FlexType's 0.79-0.85 with context). This is my bracketing, not a reported figure; it should be computed directly on the project's corpus.
- Because the task here gives word boundaries but (one word per recording) no sentence context, the relevant baseline is isolated-word disambiguation, which is harder than FlexType's and 3-Key-Input's in-context settings.

### Gaps
- No primary source found giving raw English type/token collision rates for 3, 4 and 5 key groupings side by side. Lesher et al. (1998) likely report keystroke-efficiency numbers per key count, but I could not open the full text.
- 4-key CER/WER from 3-Key-Input not recovered (only 2/3/5 shown in the fetched view).

## 2. Classic reduced/few-key systems: year, venue, keys, method, KSPC, WPM, error, accuracy

### Takeaway
The few-key literature splits into (i) ambiguous keyboards with dictionary or LM disambiguation (Lesher 1998; Dunlop 2004; Tanaka-Ishii 2002; Gong & Tarasewich 2005; SAK 2010; FlexType 2023) and (ii) unambiguous multi-stroke codes (H4-Writer, MDITIM, EdgeWrite). Four-key methods typically reach 12-20 wpm after hours of practice.

### Cited Findings
- Lesher, Moulton & Higginbotham, "Optimal character arrangements for ambiguous keyboards", IEEE Transactions on Rehabilitation Engineering 6(4):415-423, Dec 1998. Optimizes assignment of N characters to M keys under exact keystroke efficiency; uses n-opt (two-opt to five-opt) local search over a character confusability matrix derived from a character-level predictor — [PubMed 9865889](https://pubmed.ncbi.nlm.nih.gov/9865889/); method description per [FlexType related work](https://www.keithv.com/pub/flextype/gaines-flextype.pdf). Per-key-count efficiency numbers: UNVERIFIED (full text not accessed).
- Tanaka-Ishii, Inutsuka & Takeichi, "Entering Text with a Four-Button Device", COLING 2002. Predictive, adaptive LM; English, French, Thai and Japanese; claimed at least as efficient as then-current phone text entry, fewer keystrokes than a full keyboard; one user reached a maximum of 68 wpm after adaptation — [ACL Anthology C02-1149](https://preview.aclanthology.org/menus/C02-1149/). Exact English letter grouping: UNVERIFIED.
- Gong & Tarasewich, "Alphabetically constrained keypad designs for text entry on mobile devices", CHI 2005, pp. 211-220. Eight letter keys, strict alphabetical order; groupings found by enumeration/genetic algorithm to minimize candidate-list length; argued alphabetic constraint reduces visual search — [ResearchGate](https://www.researchgate.net/publication/221515493_Alphabetically_constrained_keypad_designs_for_text_entry_on_mobile_devices); [MacKenzie Ch. 5](https://www.yorku.ca/mack/chapter5.html). ACD KSPC with disambiguation 1.01, multitap 1.84 ([MacKenzie Ch. 5](https://www.yorku.ca/mack/chapter5.html)).
- MacKenzie & Felzer, SAK: Scanning Ambiguous Keyboard for one-key text entry, ACM TOCHI 2010: three letter groups plus a space key, scanned; after space, candidates scanned in frequency order — [MacKenzie TOCHI 2010](https://www.yorku.ca/mack/TOCHI2010.html); description per [FlexType](https://www.keithv.com/pub/flextype/gaines-flextype.pdf). MacKenzie, "The one-key challenge", ASSETS 2009 — [yorku.ca/mack/assets2009](https://www.yorku.ca/mack/assets2009.html) (numbers not extracted).
- H4-Writer: MacKenzie, Soukoreff & Helga, "1 thumb, 4 buttons, 20 words per minute: Design and evaluation of H4-Writer", UIST 2011 (not 2009). Base-4 Huffman codes over 4 buttons, prefix-free (no segmentation needed), KSPC 2.321; 7.7 wpm session 1 to 20.4 wpm session 10 (six participants); mean error 0.69%; 86.3% of phrases error-free. Comparators reported in the paper: MDITIM KSPC 3.047, EdgeWrite (4-key) 4.356 (16.9 wpm), Quikwriting 2.929, multitap 2.032, T9 1.007 — [MacKenzie et al. UIST 2011](https://www.yorku.ca/mack/uist2011.html); [ACM DL](https://dl.acm.org/doi/10.1145/2047196.2047258)
- H4 descendants: Huffman Base-4 Text Entry Glove (H4-TEG), ISWC 2012 — [yorku.ca/mack/iswc2012](https://www.yorku.ca/mack/iswc2012.html); "Gestural text entry using Huffman codes", MobileHCI 2013 — [PDF](https://www.yorku.ca/mack/mhci2013-119.pdf); H4VR: one-handed gesture text entry in VR with a four-key keyboard, CHI 2023 — [yorku.ca/mack/chi2023b](https://www.yorku.ca/mack/chi2023b.html); gaze-controlled H4-Writer variant — [arXiv 1801.00184](https://arxiv.org/pdf/1801.00184). (Speeds for these not extracted.)
- FlexType (Gaines, Baker & Vertanen, IUI 2023): 4 groups, 1-4 finger taps, eyes-free (groups read aloud). Constrained groups (a-e)(f-m)(n-r)(s-z,'); unconstrained WER-optimized (a,d,f,h,k,q,y,')(b,c,e,i,j,n,x)(g,l,o,s,v,w)(m,p,r,t,u,z). Longitudinal study: constrained 12.0 wpm, 2.03% CER; unconstrained 13.5 wpm, 1.81% CER; no significant difference after about four hours, but more early errors with unconstrained groups. n-best navigation about 0.29-0.32 swipes/word in sessions 5-8 — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf); [project page](https://keithv.com/pub/flextype). Follow-up "Improving FlexType: Ambiguous Text Input for Users with Visual Impairments" — [keithv.com](https://keithv.com/pub/improveflextype) (contents not extracted).
- Other few-group systems summarized in FlexType Table 1: Tap123 (T6, eyes-free), Optimal-T9 smartwatch (Qin et al., T9, clarity + speed), T18 smartwatch (15.7 wpm), TipText (T6, 13.3 wpm), HiFinger (T6 with second action), FingerText (Lee et al., T5/T10, Pareto optimization of speed, accuracy, comfort, clarity) — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Arnott & Javed (1992, probabilistic character disambiguation for reduced keyboards), Foulds et al., Levine et al.: cited in [MacKenzie Ch. 5](https://www.yorku.ca/mack/chapter5.html) without numbers there; primary details UNVERIFIED. "The Ambiguous Keyboard" (AAC-related article) — [doi 10.1044/aac13.1.12](https://doi.org/10.1044/aac13.1.12) (not opened).

### Inferences
- Unambiguous 4-key Huffman codes cost about 2.3 strokes/char; ambiguous 1-stroke-per-letter 4-group schemes cost about 1.0-1.3 strokes/char plus candidate selection. With a per-stroke recognizer accuracy of 96%, the per-letter error exposure is roughly proportional to strokes per character, which favours the ambiguous 1-stroke scheme (inference).

### Gaps
- Lesher 1998, Arnott & Javed 1992, Foulds 1987, Levine 1987 detailed numbers not verified from primary text. Less-Tap, TongueWise and "1-Line keyboard" were not retrieved in this pass (Less-Tap KSPC 1.49 multitap appears in MacKenzie's Table 5-1 but its own paper was not opened).

## 3. Huffman / variable-length codes, scanning, Morse, Dasher for accessibility

### Takeaway
Language-model-driven coding (Huffman scanning, Dasher) roughly doubles throughput over unigram or row/column scanning; the Roark line explicitly builds codes that account for the probability of user selection errors.

### Cited Findings
- Roark, Fried-Oken & Gibbons, "Huffman and Linear Scanning Methods with Statistical Language Models", Augmentative and Alternative Communication 31(1):37-50, 2015, doi 10.3109/07434618.2014.997890. Calibrated scan rates, 16 non-disabled adults: row/column step 21.2 cpm (8.4 bits/char actual, 4.5 optimal), row/column auto 19.6 cpm, Huffman unigram 12.8 cpm (7.9 bpc), Huffman 8-gram 24.5 cpm (4.0 bpc actual, 2.6 optimal; error 4.2%), linear grid 8-gram 23.2 cpm, RSVP 8-gram 20.9 cpm. The 8-gram model nearly doubled Huffman scanning speed vs unigram. One participant with locked-in syndrome showed a similar pattern — [Roark et al. 2015 PDF](https://www.ohsu.edu/sites/default/files/2018-11/Huffman%20and%20Linear%20Scanning.pdf)
- Earlier: Roark et al., "Scanning methods and language modeling for binary switch typing", NAACL-HLT 2010 SLPAT workshop — [ACL W10-1304](https://aclanthology.org/W10-1304/). Search-snippet claim (not verified in full text): the algorithm "accounts for any probability of error in following a branch of the tree, and builds the optimal coding tree even when there is non-zero probability of taking a branch in error." Huffman scanning reportedly shipped as an option in Android accessibility — [ResearchGate summary](https://www.researchgate.net/publication/257267638_Huffman_scanning_Using_language_models_within_fixed-grid_keyboard_emulation) (UNVERIFIED).
- Dasher (Ward & MacKay, "Fast hands-free writing by gaze direction", Nature 418:838, 2002): up to 25 wpm with gaze after practice vs about 15 wpm for gaze on-screen keyboards; about five times fewer spelling errors — [arXiv cs/0204030](https://arxiv.org/pdf/cs/0204030); [Cambridge](https://cam.ac.uk/node/21505)
- Mid-air/glove Huffman base-4 codes: H4-TEG — [ISWC 2012](http://www.yorku.ca/mack/iswc2012.pdf).

### Inferences
- The Roark error-aware Huffman tree is the closest classical precedent for designing a code around a noisy input channel, but it is a code for a binary switch with an LM-adaptive tree, not a fixed letter-to-symbol grouping.

### Gaps
- Morse-code accessibility speeds were not retrieved in this pass.

## 4. Co-optimizing grouping with recognizer confusion / noise model (key gap question)

### Takeaway
I found no paper that jointly optimizes a fixed letter-to-gesture grouping with (a) the input recognizer's measured confusion matrix and (b) an LM decoder. Existing "confusion-aware" grouping work uses a disambiguation confusability matrix (which letters the LM confuses when they share a key), not sensor/classifier confusion. Neighbouring work optimizes layouts under a spatial noise model (touch or gesture typing) or builds error-aware codes (Roark), but not for discrete ambiguous groups with a measured classifier confusion.

### Cited Findings
- FlexType explicitly claims to be the first "language-model-aware optimization procedure" for ambiguous groups; its confusion matrix is built from LM mispredictions in noise-free simulation ("badgram clarity"), and its WER clarity simulates exact taps. Badgram vs WER clarity correlation r = 0.9019 over 1,000 random sets. No input noise is modeled in optimization — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- Lesher et al. 1998 and Lee et al. (FingerText) use corpus-derived confusability matrices (LM/disambiguation confusability), per FlexType's related-work section — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- LowKeyEMG (Lee, Xiao, Kaasyap et al., UCLA; arXiv 2507.19736, July 2025): sEMG, 7 gestures (4 letter groups + space, undo, select). Groups optimized to minimize expected word collisions under Brown-corpus word frequencies via Gumbel-softmax gradient descent: {a,c,d,u,v,w,x}, {g,j,k,l,m,o,p,y}, {b,e,r,t,z,comma}, {f,h,i,n,q,s,period,question}. The optimization did not model EMG classifier confusion. Offline gesture classifier >90% test accuracy; decoder RWKV-4-169M with word beam search and Levenshtein-tolerant character matching (some tolerance to gesture errors at decode time) — [arXiv 2507.19736](https://arxiv.org/html/2507.19736)
- Smith, Bi & Zhai, "Optimizing Touchscreen Keyboards for Gesture Typing" (CHI 2015, venue UNVERIFIED here): optimizes for "gesture clarity" (distinctiveness of word gestures under noise) plus speed and Qwerty similarity; error rate reductions of 52% (clarity only) and 37% (clarity + speed + Qwerty similarity) vs Qwerty — [Google Research](https://research.google/pubs/optimizing-touchscreen-keyboards-for-gesture-typing/); [Columbia project page](https://www.cs.columbia.edu/~brian/projects/optimizing_keyboards.html)
- Dunlop & Levine (CHI 2012, Pareto optimization of touchscreen keyboards on tap clarity, finger distance and familiarity) and Qin et al. (Optimal-T9, clarity + speed) — summarized in [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf). Tap clarity includes a spatial tap-error notion, but for non-ambiguous or spatial layouts.
- BCI spellers: layouts have been based on letter frequency or on minimizing typing mistakes w.r.t. a dictionary, and row-column matrices modified because most errors are wrong row/column — [BCI Spellers review, PMC5924393](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924393/); LM integration review — [PMC5495144](https://pmc.ncbi.nlm.nih.gov/articles/PMC5495144/). Region-based P300 speller compared alphabetical vs frequency-based letter distribution for accuracy/usability — [CMC 2021](https://www.techscience.com/cmc/v67n1/41181/html). None found that optimizes letter assignment against the measured EEG classifier confusion jointly with an LM.
- Recent LLM-BCI speller work (ChatBCI, P300 + LLM word prediction) uses LLMs for prediction, not for grouping design — [arXiv 2411.15395](https://arxiv.org/html/2411.15395); [Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-25660-7); [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.11.06.686984v1.full)
- 3-Key-Input reports that mapping choice changes CER by <0.5 pp with strong LM priors (worst-case mapping +0.5 pp), and models no input noise — [arXiv 2606.11642](https://arxiv.org/html/2606.11642)

### Inferences
- With noise-free input, a strong LM makes the grouping matter little (3-Key-Input). With a noisy recognizer, the relevant quantity becomes "LM-confusable letter pairs that are also recognizer-confusable symbols", which none of the found work optimizes. The physical confusion structure (e.g., UP/DOWN vs LEFT/RIGHT confusion in a wrist sensor) could be matched by placing letters that the LM easily separates on symbols the recognizer often confuses. This is a design hypothesis, not a result.
- A natural objective is expected top-1 word accuracy under P(observed symbols | word) = product of per-stroke confusion rows, marginalized over a word prior; FlexType's WER clarity is the noise-free special case (inference).

### Gaps
- Absence of evidence is from a limited search (about 20 queries/fetches); a targeted search of RSVP Keyboard (Oken/Erdogmus), Vertanen's later noisy-input ambiguous-keyboard work, and gesture-set recognizability literature (e.g., Long & Landay similarity, Vatavu) was not completed.

## 5. 2020-2026: ambiguous keyboards revisited with neural LMs / LLMs

### Takeaway
Two recent works directly pair few-key ambiguous input with neural LMs: LowKeyEMG (2025; 4 letter groups, RWKV LM, 96.9% top-1 vs 66.6% for a 4-gram baseline) and 3-Key-Input (2026; GPT-4o sentence decoding, 3-key CER 9.5%). Both assume either a real but unmodeled-in-optimization recognizer, or perfect keys; neither uses soft per-stroke posteriors from the recognizer in the LM search, as far as the fetched text shows.

### Cited Findings
- LowKeyEMG (arXiv 2507.19736, 2025): word top-1 accuracy 96.9% vs 66.6% for a 4-gram baseline; top-3 99.2% vs 85.8%; selection in top-2 at least 94.8% of the time. Entry: 14.4 wpm base, 17.2 with completion, 19.5 with completion + context; expert (third repetition) 23.3 wpm average (16.4-27.8). Gestures per character 1.11 / 0.95 / 0.83 — [arXiv 2507.19736](https://arxiv.org/html/2507.19736). Caution: whether the 96.9% figure uses sentence context and clean gestures was not confirmed in the fetched summary.
- 3-Key-Input (Kimura, 2026): GPT-4o selection over trie-generated candidates reduces 3-key CER from 0.216 (trie-only) to 0.095 and WER from 0.325 to 0.122; 2-key WER 0.562 to 0.337; 5-key WER 0.168 to 0.068. Technical sentences roughly twice the error of business sentences. Conclusion: 3 keys are a practical minimum for general English offline with strong LM priors — [arXiv 2606.11642](https://arxiv.org/html/2606.11642)
- FlexType (2023) used a 12-gram char + 4-gram word VelociTap decoder, not a neural LM — [FlexType PDF](https://www.keithv.com/pub/flextype/gaines-flextype.pdf)
- LLMs for character-based AAC (adapting LLMs for character prediction) — [arXiv 2501.10582](https://arxiv.org/html/2501.10582v2); LLMs to accelerate communication for users with severe motor impairments (abbreviation expansion) — [arXiv 2312.01532](https://arxiv.org/pdf/2312.01532). Related, not ambiguous-keyboard decoding.

### Inferences
- Relative gains from neural LMs vs n-gram/dictionary are large (LowKeyEMG +30 pp top-1; 3-Key-Input roughly halves WER vs trie). Much of this gain depends on sentence context; for one-word-per-recording, gains from an LLM are likely smaller because only a word prior is available (inference).

### Gaps
- No found paper reports LLM decoding of ambiguous key sequences for isolated words without context, nor with soft recognizer posteriors. 4-key LLM results not recovered from 3-Key-Input.

## 6. What remains unexplored for a 4-symbol, one-symbol-per-letter scheme decoded by a modern LLM with soft inputs

### Takeaway
The combination "4 directional wrist gestures + one symbol per letter + soft per-stroke posteriors + LLM/n-gram decoding + grouping co-optimized with measured recognizer confusion" does not appear in the found literature. Each piece exists separately.

### Cited Findings
- 4-group ambiguous keyboards with LM disambiguation exist (FlexType 2023; LowKeyEMG 2025; Dunlop 2004; Tanaka-Ishii 2002) — [FlexType](https://www.keithv.com/pub/flextype/gaines-flextype.pdf); [LowKeyEMG](https://arxiv.org/html/2507.19736); [Dunlop 2004](https://strathprints.strath.ac.uk/2554/); [Tanaka-Ishii 2002](https://preview.aclanthology.org/menus/C02-1149/)
- Grouping optimization so far targets LM confusability or word collisions under clean input — [FlexType](https://www.keithv.com/pub/flextype/gaines-flextype.pdf); [LowKeyEMG](https://arxiv.org/html/2507.19736); [Lesher 1998](https://pubmed.ncbi.nlm.nih.gov/9865889/)
- Noise-aware layout optimization exists for spatial gesture/touch keyboards — [Smith, Bi & Zhai](https://research.google/pubs/optimizing-touchscreen-keyboards-for-gesture-typing/); noise-aware coding exists for switch scanning — [Roark et al. 2010](https://aclanthology.org/W10-1304/)
- LLM sentence decoding of clean ambiguous sequences exists — [3-Key-Input](https://arxiv.org/html/2606.11642)

### Inferences (open gaps, stated as research opportunities, not results)
- Joint grouping optimization over (recognizer confusion matrix) x (LM word distribution), with expected top-k word accuracy as the objective, evaluated against matched baselines (alphabetical, frequency, collision-minimizing, FlexType groups).
- Soft-posterior decoding: integrating per-stroke class posteriors (not hard labels) into an LLM or n-gram word search, and quantifying the gain over hard decisions at ~96% per-stroke accuracy. For a 5-letter word, hard-decision all-strokes-correct probability is 0.96^5 ≈ 0.82 (arithmetic estimate), so error-tolerant decoding is needed.
- Isolated-word (no sentence context) decoding with LLM priors vs unigram/dictionary priors, including out-of-vocabulary/unseen-word handling (relevant to the project's open-vocabulary rule).
- Gesture-aware grouping constraints (e.g., assigning confusable gesture pairs to letter groups that the LM can separate) and learnability trade-offs (alphabetical constraint cost about 7 pp WER clarity in FlexType: 0.787 vs 0.854).
- Reporting with matched shuffled-label controls and per-class recall, which none of the HCI papers above report for their recognizers.

### Gaps
- My search was bounded (about 20 tool calls); Less-Tap, TongueWise, 1-Line keyboard, QWERH, Arnott & Javed, Foulds, Levine and Morse were not verified from primary sources. A dedicated search of Vertanen/Gaines follow-ups (2024-2026) and of RSVP Keyboard noise modeling is advisable before claiming novelty.
