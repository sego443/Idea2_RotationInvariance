# Segmented wristband movements from 2026-09-26

Source: OpenBCISession_2026-09-26_20-05-53. Positions are 0,5,10,15,20,25,30 mm, then 40–160 mm in 10 mm steps, with UP/DOWN/LEFT/RIGHT at each position.

dataset.npz contains 1,600 windows. X_raw has shape `(1600,500,4)`: two seconds of raw EXG at 250 Hz. distance_mm, direction, block, and repetition are labels; source_repetition and is_duplicate identify copied movements. channel_map stores original EXG columns. Localization uses a 0.5–12 Hz bandpass and a 75-sample smoothed envelope; X_raw retains unfiltered windows.

segments.csv records source filenames, row boundaries, and centers. recordings.csv records detected counts, trimming/copy counts, active columns, and source SHA-256. excluded.csv lists 56 short unrecorded files. manifest.json records parameters and counts. Raw and .rejected files remain unchanged.

Forty-one accepted recordings use EXG 0,2,4,6; 39 from B058 onward use 0,1,2,3. B057/B058 both belong to 70 mm. The user confirmed that both orders correspond to S0,S1,S2,S3. The experimental split excludes all of 70 mm.

Keep the first 20 detected movements when more than 20 exist. B019 has 19; repetition 20 copies repetition 10 and is marked is_duplicate. Model helpers exclude this copy. Group windows by recording/position to avoid source/copy leakage. Frozen experimental counts are 879/320/320 train/validation/test.

Git includes metadata but excludes dataset.npz and raw recordings. Historical absolute paths are provenance, not portable runtime paths. Obtain the data bundle separately and pass --dataset explicitly if necessary.
