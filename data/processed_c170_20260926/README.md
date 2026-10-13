# Segmented wristband movements from 2026-09-26

Source: OpenBCISession_2026-09-26_20-05-53. Positions are 0,5,10,15,20,25,30 mm, then 40–160 mm in 10 mm steps, with UP/DOWN/LEFT/RIGHT at each position.

dataset.npz contains 1,600 windows. X_raw has shape `(1600,500,4)`: two seconds of raw EXG at 250 Hz. distance_mm, direction, block, and repetition are labels; source_repetition and is_duplicate identify copied movements. channel_map stores original EXG columns. Localization uses a 0.5–12 Hz bandpass and a 75-sample smoothed envelope; X_raw retains unfiltered windows.

segments.csv records source filenames, row boundaries, and centers. recordings.csv records detected counts, trimming/copy counts, active columns, and source SHA-256. excluded.csv lists 56 short unrecorded files. manifest.json records parameters and counts. Raw and .rejected files remain unchanged.

Forty-one accepted recordings use EXG 0,2,4,6; 39 from B058 onward use 0,1,2,3. B057/B058 both belong to 70 mm. The user confirmed that both orders correspond to S0,S1,S2,S3. The experimental split excludes all of 70 mm.

Keep the first 20 detected movements when more than 20 exist. B019 has 19; repetition 20 copies repetition 10 and is marked is_duplicate. Model helpers exclude this copy. Group windows by recording/position to avoid source/copy leakage. Frozen experimental counts are 879/320/320 train/validation/test.

Git includes metadata but excludes dataset.npz and raw recordings. Historical absolute paths are provenance, not portable runtime paths. Obtain the data bundle separately and pass --dataset explicitly if necessary.

## Movement and position reference (added 2026-10-12, participant report)

Each movement is an in-air swipe (UP/DOWN/LEFT/RIGHT) followed by a return to centre, so a window usually holds an outward and a return burst.

The band was worn on the right wrist (circumference 170 mm). A wrist tape marked 0–17 cm runs around the band line. The 1 cm mark lies on the dorsal midline, and readings increase toward the little finger, across the palm to the thumb and back. S0→S1→S2→S3 increase along the tape. `distance_mm` is the tape reading under the midpoint between S3 and S0. Assuming equally spaced sensors (42.5 mm apart), the angle of S0 from the dorsal midline in the tape direction is ((distance_mm + 11.25) mod 170) × 360 / 170. For example, 0 mm → 24°, 31.25 mm → 90° and 158.75 mm → 0°. This is the same frame as the W02 band angle. It is a geometric conversion from the reported setup, not an independent measurement.
