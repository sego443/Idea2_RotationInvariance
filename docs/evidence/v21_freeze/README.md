# v21 freeze manifests

| File | Stage | Training data | Status |
|---|---|---|---|
| `v21_freeze_S1.json` | S1 | C170 + W01 (`all_current`) | **Superseded** on 2026-10-12, before any W02 collection, when W01 was excluded from training (preliminary, non-rigorous feasibility data). Kept unchanged; never evaluate W02 with it. |
| `v21_freeze_S1_c170.json` | S1 | C170 only (`c170_all`) | Frozen 2026-10-14 by `scripts/launchers/launch_v21_stage1.py` (commit `9f7461f`). **Deviation:** frozen after W02 Part A was recorded, before any W02 stroke was scored; see the [2026-10-14 record](../../results/2026-10-14-w02-part-a-and-v21-freezes.md). |
| `v21_freeze_S2.json` | S2 | C170 + W02 Part A (`c170_plus_w02`) | Frozen 2026-10-14 by `scripts/launchers/launch_v21_stage2.py` (commit `00e17e7`), before any Part B recording; see the [2026-10-14 record](../../results/2026-10-14-w02-part-a-and-v21-freezes.md). |

`make_v21_freeze_manifest.py` and `launch_v21_stage2.py` refuse runs trained with W01.
