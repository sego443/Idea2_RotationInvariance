# Piezoelectric / mechanical wristband gesture sensing and rewear/rotation robustness

Scope note: Searched 2026-10-17 with a capped budget (~17 tool calls). Many ACM/IEEE full texts were not fetched (403 or paywall), so most numbers come from abstracts, publisher pages or search-result snippets. Items marked **[unverified]** are recalled from background knowledge and were NOT confirmed in this session; do not cite them without checking.

## 1. Piezoelectric and mechanical (pressure/FMG/tendon) wrist gesture sensing: gesture sets, accuracy, users, cross-session

### Takeaway
Passive piezo/FSR/pressure wristbands routinely report 80-98% accuracy on 4-12 static gestures or finger taps, but almost always within a single wearing session, per user, on small cohorts (typically under 15 people). Rewear/donning results, where they exist, come from FMG and capacitive bands rather than piezo bands, and they show clear drops that need either training data from several wearings or some recalibration.

### Cited Findings
**Piezoelectric wristbands**
- Booth & Goldsmith, "A Wrist-Worn Piezoelectric Sensor Array for Gesture Input," J. Med. Biol. Eng. (2018): six piezoelectric sensors inside an adjustable wrist strap pick up vibrations and shape changes from muscle/tendon motion. Mean classification accuracy 97% for tap gestures made with each of the five fingers — [Springer](https://link.springer.com/article/10.1007/s40846-017-0303-8); [ResearchGate](https://www.researchgate.net/publication/318511313_A_Wrist-Worn_Piezoelectric_Sensor_Array_for_Gesture_Input)
- Earlier conference version: Booth & Goldsmith, "Detecting finger gestures with a wrist worn piezoelectric sensor array" — [Semantic Scholar](https://www.semanticscholar.org/paper/Detecting-finger-gestures-with-a-wrist-worn-sensor-Booth-Goldsmith/fded9fb163bf2f2c2d3566b7eef719edd19140da/figure/1)
- U. Calgary thesis, "Design and Testing of a Wristband with Piezoelectric Sensors for Finger Gesture Recognition": highest mean gesture-detection accuracy 93% across participants. Per-participant classifiers reached 93% (which finger) and 98% (gesture type). This is the snippet-level summary; the full thesis was not read — [U Calgary](https://ucalgary.scholaris.ca/items/5e34f101-7946-4afd-a9b3-96c6e38d994b)
- PVDF-based wearable controller for gesture applications, IEEE TBioCAS (2017). PVDF film on a forearm compression sleeve. A single wrist PVDF sensor separated hand open from hand closed — [DOI 10.1109/tbcas.2017.2683458](https://doi.org/10.1109/tbcas.2017.2683458); [Frontiers summary](https://www.frontiersin.org/articles/10.3389/fnbot.2019.00114/full)
- A 24-sensor high-density piezoelectric skin reported about 98% average accuracy for wrist gestures. This is a snippet citation; I could not identify the primary paper — [Frontiers Neurorobotics 2019 (related work)](https://www.frontiersin.org/articles/10.3389/fnbot.2019.00114/full)
- Gesture-recognition wristband with optimised piezoelectric energy harvesters (self-powered sensing concept) — [ResearchGate](https://www.researchgate.net/publication/348095782_Gesture_Recognition_Wristband_Device_with_Optimised_Piezoelectric_Energy_Harvesters)
- TapWristband (IEEE TMC 2025): wrist-vibration sensing that turns finger taps into a wearable keypad, i.e. vibration/piezo-like sensing used for text-like keypad input — [IEEE CSDL](https://www.computer.org/csdl/journal/tm/2025/04/10759301/221BudCLY3e)

**Pressure / FSR / FMG**
- WristFlex (Dementyev & Paradiso, UIST 2014, MIT Media Lab): an FSR array around the wrist separates subtle finger-pinch gestures from tendon movement with >80% accuracy. Sensor power is 60.7 µW, so a small LiPo could last more than a week. An accelerometer can add gestures. The user studies covered accuracy, speed and repeatability — [MIT Media Lab](https://www.media.mit.edu/publications/wristflex-low-power-gesture-input-with-wrist-worn-pressure-sensors/); [DSpace@MIT](https://dspace.mit.edu/handle/1721.1/103642)
- Piezoresistive array armband with a reduced number of sensors (Sensors, 2019/2020, open access) — [PMC6978746](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6978746/)
- Wearable FMG armband with FSRs for upper-limb gestures (2023, open access) — [PMC10708660](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10708660/)
- Survey statement: intra-wearing cross-session FMG tests (band not removed) range from 79% to 98%. Inter-session and inter-trial averages of about 92.2% and 88.9% have been reported. Doffing/donning displacement "may cause severe performance degradation" — [search-snippet summary of FMG literature, sources: PMC10708660](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10708660/); [IEEE TNSRE 2022](https://doi.org/10.1109/tnsre.2022.3212705)
- Flexible capacitive wristband with a triplet network for inter-day use (IEEE TNSRE 2022): 91.98% average accuracy across all sessions of all subjects, with sessions involving re-donning across days — [IEEE Xplore 9919169](https://ieeexplore.ieee.org/document/9919169/); [DOI](https://doi.org/10.1109/tnsre.2022.3212705)
- Jiang, Xiao, Menon and colleagues (SFU MENRVA): wrist FMG patterns for a gesture change with forearm position. In donning protocols the band was removed and re-donned after each trial, by the experimenter at first and by the subject in the final trial — [Cogent Eng. 2020, forearm rotation with high-density FMG](https://www.tandfonline.com/doi/full/10.1080/23311916.2020.1795051); [ResearchGate, Wrist-worn FMG and user anthropometry](https://www.researchgate.net/publication/342133462_Wrist-worn_wearables_based_on_force_myography_on_the_significance_of_user_anthropometry); [PMC7291722](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7291722/)
- Review: Xiao & Menon, "A Review of Force Myography Research and Development," Sensors 2019 — [ResearchGate](https://www.researchgate.net/publication/336693891_A_Review_of_Force_Myography_Research_and_Development)
- FMG benchmark dataset for gesture recognition and transfer learning (arXiv 2007.14918) — [arXiv](https://arxiv.org/pdf/2007.14918)
- pFMG soft armband for IR consumer-device teleoperation (2024) — [PMC11435868](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11435868/)
- Wrist-worn pressure array estimating full hand pose and contact forces (arXiv 2609.16518, 2026) — [arXiv](https://arxiv.org/pdf/2609.16518)

**Other wrist modalities (comparison points)**
- Tomo (Zhang & Harrison, UIST 2015): 8-electrode EIT band. Accuracy on gross hand gestures / thumb-finger pinches was 97% / 87% at the wrist and 93% / 81% on the arm — [FIGLAB](https://www.figlab.com/research/2015/tomo); [chrisharrison.net](https://chrisharrison.net/index.php/Research/Tomo). Re-wear numbers were not retrieved. **[unverified]** I recall the paper includes a cross-wearing evaluation that showed a substantial drop.
- SensIR (UIST 2017): an IR transmission + reflection bracelet, 12 discrete gestures at 93.3% — [ResearchGate](https://www.researchgate.net/publication/320568752_SensIR_Detecting_Hand_Gestures_with_a_Wearable_Bracelet_using_Infrared_Transmission_and_Reflection); [ACM](https://dl.acm.org/citation.cfm?id=3126604). Follow-up: "Improving Hand Gesture Recognition via Infrared Tomography of the Wrist over Multiple Wearing Sessions" (Springer, HCII 2023), which directly addresses multi-wearing — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-35596-7_33)
- WristWhirl (Gong, Yang, Irani, UIST 2016): wrist-as-joystick using 12 IR proximity sensors on the strap, plus a piezo vibration sensor that detects pinch sound through the skin to mark gesture start and end — [Dartmouth](http://home.cs.dartmouth.edu/~jungong/blog/wristwhirl); [U Manitoba](https://hci.cs.umanitoba.ca/Publications/details/wristwhirl-one-handed-continuous-smartwatch-input-using-wrist-gestures)
- Serendipity (Wen, Ramos Rojas, Dey, CHI 2016): smartwatch IMU, 5 fine finger gestures, average F1 87% — [ACM DOI 10.1145/2858036.2858466](https://dl.acm.org/doi/10.1145/2858036.2858466)

### Inferences
- Piezo wrist sensing is established for finger taps and static gestures. In-air directional wrist swipes recognized by a 4-element circumferential piezo ring look less explored. I found no paper with exactly this configuration, but the search was limited.
- Published piezo-band accuracies (93-98%) are within-session, so the user's own within-session numbers should be compared with these and not with rewear numbers.
- Rewear evidence comes mostly from FMG, capacitive, IR and EIT bands. The user's ~0.55 zero-calibration rewear macro-F1 on 4 classes fits the literature's qualitative finding of "severe degradation" after re-donning.

### Gaps
- No piezo-wristband paper with an explicit rewear or donning protocol and numbers was found.
- Tomo cross-wear numbers, WristFlex per-gesture results and participant counts were not retrieved from full texts.

## 2. Rotation / rewear / electrode-shift robustness (EMG/FMG): what works zero-shot, how much calibration

### Takeaway
Electrode shift is a long-standing, well-quantified problem in EMG. Accuracy drops sharply once displacement exceeds the electrode pickup area (Hargrove et al. 2008). Proposed fixes include multi-position training, rotation estimation and channel re-mapping (often only at discrete 45° steps on 8-channel armbands), spatial-transformer CNNs, and unsupervised or pseudo-label self-calibration. Large-scale generic models (Meta, Nature 2025) reach usable zero-shot cross-user performance only with thousands of training participants and a high-density dry-electrode band, and personalization still adds 16-30%.

### Cited Findings
- Hargrove, Englehart & Hudgins, "A training strategy to reduce classification degradation due to electrode displacements in pattern recognition based myoelectric control," Biomed. Signal Process. Control 3(2):175-180 (2008). Accuracy falls significantly when displacement exceeds the EMG pickup area. Training on multiple displaced positions mitigates this — [Google Scholar record](https://scholar.google.com/scholar?q=Hargrove+Englehart+Hudgins+training+strategy+electrode+displacements+2008); [discussion in PMC4134107](https://pmc.ncbi.nlm.nih.gov/articles/PMC4134107/)
- Electrode shift effects on wavelet features in myoelectric PR — [PMC4134107](https://pmc.ncbi.nlm.nih.gov/articles/PMC4134107/)
- Common spatial patterns improve HD-EMG robustness to electrode shift — [PMC4668610](https://pmc.ncbi.nlm.nih.gov/articles/PMC4668610/)
- "Advanced Hand Gesture Prediction Robust to Electrode Shift with an Arbitrary Angle," Sensors 20(4):1113 (2020). Prior methods only corrected rotation in discrete 45° steps. This work identifies shift by interpolated peak location (IPL), rearranges the signal matrix, trains on three standard shifts and tests on multiple sessions — [PMC7070560](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7070560/); [DOI 10.3390/s20041113](https://doi.org/10.3390/s20041113)
- Extended Spatial Transformer CNN for gesture recognition and self-calibration with sparse sEMG electrodes. Spatial-transformer layers learn the rotation angle for coarse correction — [PubMed 36378801](https://pubmed.ncbi.nlm.nih.gov/36378801/)
- Fast, low-impact embedded orientation-correction algorithm for gesture-recognition armbands (2025, open access) — [PMC11991082](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11991082/)
- Masked autoencoder + Swin transformer to mitigate HD-EMG electrode shift — [arXiv 2410.17261](https://arxiv.org/pdf/2410.17261); HD-EMG electrode subsets to tackle shift — [arXiv 2401.02773](https://arxiv.org/html/2401.02773)
- Unsupervised domain-adversarial self-calibration (Côté-Allard et al.) uses pseudo-labels to cope with shift and non-stationarity without labeled recalibration — [arXiv 1912.11037](https://arxiv.org/html/1912.11037); [ResearchGate](https://www.researchgate.net/publication/344567141_Unsupervised_Domain_Adversarial_Self-Calibration_for_Electromyography-Based_Gesture_Recognition)
- Effect of sEMG electrode (re)placement and feature-set size on movement recognition — [arXiv 2005.02105](https://arxiv.org/pdf/2005.02105)
- Myo armband practice: the stock software calibrated from a single repetition of each of 5 gestures. Position and rotation on the arm affect accuracy, so recalibration per session is typical — [J. Mobile Multimedia 2017](https://www.riverpublishers.com/journal/journal_articles/RP_Journal_1550-4646_1234.pdf); [EUSIPCO 2017](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347665.pdf). (A "fails if shifted >5 mm" claim appeared only in a low-quality aggregator snippet and is excluded.)
- Côté-Allard et al., "Deep Learning for Electromyographic Hand Gesture Signal Classification Using Transfer Learning," IEEE TNSRE 27(4):760-771 (2019). Pretraining across users and then fine-tuning reached 98.31% offline (7 gestures, 17 participants, CWT-ConvNet) and 68.98% (18 gestures, 10 participants, raw-EMG ConvNet) — [arXiv 1801.07756](https://arxiv.org/abs/1801.07756v5); [UiO abstract](https://mn.uio.no/ifi/english/people/aca/kyrrehg/pubs/allard-tnsre2019-abstract.html). **[unverified detail]** I recall that the target users supplied few cycles (about 1-4) for fine-tuning.
- Big-data multi-user myoelectric models (bioRxiv 2024) — [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.07.11.603119v1.full.pdf)
- Meta Reality Labs, "A generic non-invasive neuromotor interface for human-computer interaction," Nature 645 (2025), DOI 10.1038/s41586-025-09255-w. sEMG wristband, training data from thousands of participants. Closed-loop medians: 0.66 target acquisitions/s (navigation), 0.88 gesture detections/s (discrete), handwriting 20.9 WPM. Personalization improves handwriting decoding by 16% — [RePEc](https://ideas.repec.org/a/nat/nature/v645y2025i8081d10.1038_s41586-025-09255-w.html); [CSHL repository PDF](https://repository.cshl.edu/id/eprint/42049/). The 2024 bioRxiv preprint reports different headline numbers (17.0 adjusted WPM handwriting; personalization "up to 30%") — [bioRxiv v2](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v2). Cite the Nature numbers. The amount of personalization data and any band-rotation analysis were not retrieved. **[unverified]** I recall that the band is meant to be donned with a variable rotation and that training spans many donnings.

### Inferences
- Treating a physical band rotation as a cyclic channel permutation works best when there are many channels (8-16+) and fields are spatially smooth. With 4 sensors 90° apart and localized tendon or bone mechanics, sub-quadrant rotations fall between sensors, which helps explain the user's finding that rotation does not act as a permutation. The IPL and arbitrary-angle work motivates interpolation, but only on dense EMG arrays.
- Zero-shot robustness in the literature generally comes from (a) training across many donnings or positions, or (b) massive multi-user data. Neither is cheap for a single-participant, low-cost piezo band. Short labeled recalibration is the standard practical answer (Myo used 1 repetition per gesture).

### Gaps
- No rotation-invariant method was found that was validated specifically on piezoelectric or FMG bands with 4 channels.
- I did not retrieve Pan et al. (electrode-shift-invariant features) or "wearing-independent" gesture recognition papers. **[unverified]** whether these exist under those names.

## 3. Few-shot / fast calibration: prototypical networks, Riemannian alignment, budgets

### Takeaway
Riemannian re-centering (Zanini et al. 2018) and Riemannian Procrustes Analysis (Rodrigues et al. 2019) are the standard covariance-domain alignment tools for cross-session/subject BCI. Re-centering needs only unlabeled data from the new session (a reference mean), while RPA adds stretching and rotation, which need some labeled target data. In EMG/FMG, typical calibration budgets are 1-5 repetitions per gesture. The user's 4-8 labeled strokes after donning (1-2 per class) sit at the low end of this range.

### Cited Findings
- Zanini, Congedo, Jutten, Said, Berthoumieu, "Transfer Learning: A Riemannian Geometry Framework With Applications to Brain-Computer Interfaces," IEEE TBME (2018). Each domain's covariances are re-centred to a reference (the Riemannian mean of resting trials) so that domains share the identity. This is effective for between-session and between-subject variability — [PubMed 28841546](https://pubmed.ncbi.nlm.nih.gov/28841546/); [Frontiers review](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.779231/full)
- Rodrigues, Jutten, Congedo, "Riemannian Procrustes Analysis: Transfer Learning for Brain-Computer Interfaces," IEEE TBME (Aug 2019). The method re-centers, stretches and rotates target covariances to match the source — [PubMed 30596565](https://pubmed.ncbi.nlm.nih.gov/30596565/); [Semantic Scholar](https://www.semanticscholar.org/paper/Riemannian-Procrustes-Analysis:-Transfer-Learning-Rodrigues-Jutten/c3a52bf2930adf3c28b89fa61fa7f83d1af45d46)
- Euclidean alignment revisited for EEG transfer (2025) — [arXiv 2502.09203](https://arxiv.org/html/2502.09203v2); Riemannian transfer for shorter c-VEP calibration — [Sci. Rep. 2022](https://www.nature.com/articles/s41598-022-14026-y); covariance-based harmonization of M/EEG datasets — [PMC12007539](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007539/)
- Recalibration budgets in EMG: Myo uses 1 repetition per gesture; research practice often uses about 5 repetitions per gesture for training — [J. Mobile Multimedia 2017](https://www.riverpublishers.com/journal/journal_articles/RP_Journal_1550-4646_1234.pdf)
- The triplet (metric-learning) network for an inter-day capacitive wristband is a metric-learning relative of prototypical networks for wearable rewear (91.98% across sessions) — [IEEE TNSRE 2022](https://doi.org/10.1109/tnsre.2022.3212705)

### Inferences
- Unsupervised re-centering on the first unlabeled seconds after donning could act as a "zero-label" baseline between zero calibration (0.55) and 4-8 labeled strokes (0.93-0.96). RPA's rotation step is a natural fit for donning rotation, but it acts on the covariance eigenbasis, not on physical angle.

### Gaps
- Prototypical-network EMG/FMG papers with explicit shots-vs-accuracy curves were not retrieved in this session.

## 4. Lagged-covariance / Riemannian tangent-space classification on piezo/FMG wristband signals

### Takeaway
Covariance plus tangent-space classification is established for sEMG gesture decoding (e.g., J. Neural Eng. 2024). I found no published application to piezoelectric or FMG wristband signals, so this looks like an open niche, with the caveat that the search was limited.

### Cited Findings
- "Topology of surface electromyogram signals: hand gesture decoding on Riemannian manifolds," J. Neural Eng. (2024): covariance matrices projected to the tangent space outperform standard EMG features for gesture classification — [IOPscience](https://iopscience.iop.org/article/10.1088/1741-2552/ad5107); [arXiv 2311.08548](https://arxiv.org/pdf/2311.08548)
- Riemannian features + RNNs for high-dimensional finger-motion decoding from EMG (arXiv 2026) — [arXiv 2604.22499](https://arxiv.org/html/2604.22499v1)
- Riemannian re-centering is "the most commonly used" SPD transfer method and avoids tangent-projection errors when source and target means differ — [PMC12007539](https://pmc.ncbi.nlm.nih.gov/articles/PMC12007539/)
- A targeted search for "force myography covariance Riemannian tangent space FSR" returned only EEG and shape-analysis papers and no FMG or piezo applications — [search: Tangent space spatial filters, arXiv 1909.10567](https://arxiv.org/pdf/1909.10567)

### Inferences
- Lagged (Hankel/time-delay-embedded) covariance on a 4-channel piezo band would yield SPD matrices of modest size (e.g., 4×L by 4×L), which suits Riemannian methods with few calibration samples. This would be a novel application. Novelty should be stated as "not found in our search", not as "first".

### Gaps
- I did not search time-delay-embedded covariance for EMG specifically (common in EEG SSVEP/c-VEP, e.g. [Sci. Rep. 2022](https://www.nature.com/articles/s41598-022-14026-y)).

## 5. Piezo/mechanical wristband text entry; overall gaps

### Takeaway
Wrist-based text entry exists for IMU/IR wrist-joystick (WrisText), sEMG handwriting (Meta, 20.9 WPM) and vibration-sensing keypads (TapWristband). I found no text-entry system built on a low-cost passive piezoelectric wristband with a few directional gestures plus LM decoding, nor one that evaluates rewear with very short calibration.

### Cited Findings
- WrisText (Gong et al., SFU/Dartmouth, UIST 2018): one-handed smartwatch text entry using wrist joystick motion (IR proximity sensing derived from WristWhirl) — [PDF](https://www.sfu.ca/~xingdong/papers/WrisText.pdf). **[unverified]** WPM figure (I recall roughly 5-15 WPM after practice).
- TapWristband: wrist-vibration-sensed keypad (IEEE TMC 2025) — [IEEE CSDL](https://www.computer.org/csdl/journal/tm/2025/04/10759301/221BudCLY3e)
- Meta sEMG wristband handwriting at 20.9 WPM (Nature 2025) — [RePEc](https://ideas.repec.org/a/nat/nature/v645y2025i8081d10.1038_s41586-025-09255-w.html)
- Pressure-based XR text input citing an EMG wristband T9 keyboard at 15.7 WPM average — [arXiv 2507.20741](https://arxiv.org/pdf/2507.20741)
- Ring-based mid-air gesture typing with deep-learning word prediction (RingGesture) — [arXiv 2410.18100](https://arxiv.org/pdf/2410.18100); generalizable word-gesture decoder with trajectory coarse discretization (Gesture2Text) — [arXiv 2410.18099](https://arxiv.org/pdf/2410.18099)

### Inferences
- Positioning: (i) low-cost passive piezo, 4 channels; (ii) explicit rewear and physical-rotation evaluation; (iii) a very short post-donning calibration of 1-2 strokes per class; (iv) a 4-symbol ambiguous alphabet resolved by an LM (T9-like). Each element has precedent, but I found no work that combines them. WrisText and EMG-T9 are the closest text-entry analogues.
- Our results come from a single participant and same-day sessions, so the claim should be limited to "feasible for this participant/hardware". The literature's multi-user rewear numbers (e.g. 91.98% inter-day capacitive) are not directly comparable.

### Gaps
- WPM and error rates for WrisText and TapWristband were not retrieved.
- No source was found on combining ambiguous 4-key input with LLM decoding on a wristband (the T9-style EMG result is the closest).
