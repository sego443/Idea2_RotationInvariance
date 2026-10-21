# Prior art covers every part of `rec1_borrowed`, but not on a piezo band

Written 2026-10-19. This is a literature note only; no experiment was run for it. It reviews prior work for the `rec1_borrowed` recipe in `src/w02_cov_fewshot.py`, the "first prefix stroke of each class (4 strokes)" row of the [W02 exploratory development record](../docs/results/2026-10-15-w02-exploratory-development.md). The recipe uses lagged-covariance features mapped to tangent space at the recording's prefix mean. Within-class covariance is borrowed from other wears (PCA 30 + Ledoit-Wolf) and gives a Mahalanobis distance. One labeled stroke per class sets the initial class means. Two soft-assignment passes over the recording's unlabeled body strokes then refine those means. Only 4 strokes are labeled, but the recipe also needs the other 4 prefix strokes (for the tangent reference) and the recording's 12 later strokes (for refinement), so it is not a 4-action startup procedure. The result with only 4 startup actions is nearest centroid at 0.873.

Every one of these ingredients has published precedent, mostly in EEG brain–computer interfaces and sEMG. No source found here combines them, and none applies covariance or Riemannian methods to a piezoelectric or force-myography wristband. A reviewer would read the recipe as known components applied to a new sensor, not as a new algorithm. The project figure quoted here (macro-F1 0.933 on the nine W02 development wears) is exploratory and comes from the project, not the literature.

**Search scope.** About ten standard web queries, read at abstract or summary level. Only Gowda & Miller was read in more depth. The methods section of the Meta Nature 2025 paper could not be fetched, so this note makes no claim about its input features. A thorough search is still needed before claiming the piezo/FMG gap below.

## Each component maps to an established line of work

| Part of `rec1_borrowed` | Closest prior work | How close |
|---|---|---|
| Covariance matrices as gesture features, classified on the SPD manifold | Gowda & Miller, *Topology of surface electromyogram signals: hand gesture decoding on Riemannian manifolds* (J. Neural Eng. 2025; arXiv 2311.08548). 12–128 sEMG electrodes; minimum distance to mean, manifold SVM, k-medoids. | Same feature idea, applied to sEMG; it works on the manifold directly rather than in tangent space. Their unsupervised k-medoids parallels our label-free clustering experiment. |
| Tangent space at a reference mean; per-session re-centering | Riemannian alignment (RA-MDRM, Zanini et al., HAL 01923278) re-centers each session at the mean covariance of its rest trials; Euclidean alignment (He & Wu, arXiv 1808.05464) is a related variant. Gowda & Miller also suggest updating the tangent reference without labels to handle drift between sessions. | Very close to how we choose the tangent reference (mean of the 8-stroke prefix). In our data, re-centering on REST windows hurt (0.34). |
| Covariance of time-lagged copies of the channels (28×28) | Augmented covariance matrix (Carrara & Papadopoulo 2023, arXiv 2302.04508): a delay embedding that adds temporal structure to the spatial covariance; pyRiemann has an implementation. | Essentially the same construction. Lags 0–30 in steps of 5 are a delay embedding of order 7. |
| Model from earlier sessions, adapted with one calibration trial per class | Vidovic et al., *Improving the robustness of myoelectric pattern recognition for upper limb prostheses by covariate shift adaptation* (IEEE TNSRE 24(9), 2016). They shrink LDA means and covariance from earlier days toward a short calibration set, and report >92% accuracy over five days with one trial per class. | Closest match for "borrowed covariance + 1-shot means". The difference: they shrink toward the old parameters, while we keep only the covariance from other wears and take the means from the new wear. |
| Refining class means with soft assignments of unlabeled queries | Ren et al. 2018, semi-supervised prototypical networks (soft k-means refinement; arXiv 1803.00676), and later transductive few-shot work. A Waikato study applies few-shot prototypical networks to sEMG gestures. | Our two refinement passes are soft k-means in this style, but with a Mahalanobis distance and no learned embedding. |
| Robustness to band rotation | Array barrel-shifting augmentation for CNNs (UNB thesis); an orientation-correction algorithm for gesture armbands that needs only two calibration gestures (Sensors 2025, PMC11991082); energy-based orientation correction for EMG bracelets. | Cyclic channel shift and quick orientation calibration are already studied. This matters for the project's rotation-invariance framing. |
| Piezoelectric / mechanical wrist sensing | Coupled piezoelectric mechanomyography sensor (PMC8705252); piezoelectric finger-gesture wristbands (Calgary, Bath); force-myography benchmarks with donning and doffing (arXiv 2007.14918). | No covariance or Riemannian method on piezoelectric or FMG wristbands was found. This is the most plausible novelty angle, and it is not yet verified. |

## Positioning for the write-up

- Cite the components as established: augmented covariance in tangent space (BCI), re-centering (Zanini / RA-MDRM), shrinkage-style calibration (Vidovic 2016), and transductive prototypes (Ren 2018).
- The candidate contribution is empirical. It applies the method to four circumferential piezoelectric sensors and measures how much a short per-wear calibration recovers after rewearing at a different band angle. Calibration cost should be counted as every stroke the user performs before recognition starts, labeled or not. The classifier itself is not the contribution.
- The calibrated recipes ignore the band angle. Frame that result against the orientation-correction papers, which already show that two calibration gestures can fix armband rotation. The W02 result shows that per-wear calibration works. It does not show rotation invariance.
- Scope claims to what was tested: one participant, two days, presegmented strokes, development wears used repeatedly for tuning, and sealed wears W07/W10/W12 not opened.

## Open items

- Run a thorough search on Riemannian or covariance methods for piezoelectric, MMG and FMG wearables before claiming the gap.
- Read the Meta Nature 2025 methods for its input features and personalization procedure.
- Read Vidovic 2016 in full to state precisely how their shrinkage estimator differs from the borrowed metric.

## Sources

- [Gowda & Miller, Topology of sEMG signals on Riemannian manifolds](https://arxiv.org/html/2311.08548v2)
- [Decoding finger motion from EMG with Riemannian features and RNNs](https://arxiv.org/pdf/2604.22499)
- [Zanini et al., Riemannian alignment (HAL 01923278)](https://hal.archives-ouvertes.fr/hal-01923278)
- [He & Wu, Euclidean alignment for BCI transfer](https://arxiv.org/pdf/1808.05464)
- [Carrara & Papadopoulo, Augmented covariance matrix](https://arxiv.org/pdf/2302.04508)
- [pyRiemann augmented covariance example](https://pyriemann.readthedocs.io/en/v0.8/_downloads/556bea18047a62055fd6eaccae4921de/plot_augmented.py)
- [Vidovic et al. 2016, Covariate shift adaptation for myoelectric control](https://publications.goettingen-research-online.de/ID-76047)
- [Ren et al. 2018, Semi-supervised few-shot classification](https://arxiv.org/pdf/1803.00676)
- [EMG few-shot / transfer learning (Waikato)](https://researchcommons.waikato.ac.nz/entities/publication/235c3b2d-9bf7-4d91-be05-3f2069fbea19)
- [Orientation correction for gesture armbands (PMC11991082)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11991082)
- [Energy-based orientation correction for EMG bracelets](https://redi.cedia.edu.ec/document/19806)
- [Array barrel-shifting augmentation thesis (UNB)](https://unbscholar.dspace.lib.unb.ca/server/api/core/bitstreams/54407f52-28ec-4abf-95ef-33f55e98ff41/content)
- [Coupled piezoelectric MMG sensor](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8705252/)
- [Piezoelectric wristband for finger gestures (Calgary)](https://prism.ucalgary.ca/handle/11023/4230)
- [FMG benchmark data](https://arxiv.org/pdf/2007.14918)
- [Riemannian feature adaptation against contraction-force variation](https://repository.derby.ac.uk/item/q7478/a-robust-feature-adaptation-approach-against-variation-of-muscle-contraction-forces-for-myoelectric-pattern-recognition-based-gesture-characterization)
- [Meta, Generic non-invasive neuromotor interface (Nature 2025)](https://repository.cshl.edu/id/eprint/42049/)
