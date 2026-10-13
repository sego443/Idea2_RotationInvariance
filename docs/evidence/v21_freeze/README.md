# v21 freeze manifests

| File | Stage | Training data | Status |
|---|---|---|---|
| `v21_freeze_S1.json` | S1 | C170 + W01 (`all_current`) | **Superseded** on 2026-10-12, before any W02 collection, when W01 was excluded from training (preliminary, non-rigorous feasibility data). Kept unchanged; never evaluate W02 with it. |
| `v21_freeze_S1_c170.json` | S1 | C170 only (`c170_all`) | To be written by `scripts/launchers/launch_v21_stage1.py` and committed before W02 collection. |
| `v21_freeze_S2.json` | S2 | C170 + W02 Part A (`c170_plus_w02`) | To be written during the W02 midday break by `scripts/launchers/launch_v21_stage2.py`. |

`make_v21_freeze_manifest.py` and `launch_v21_stage2.py` refuse runs trained with W01.
