# C170 latent-angle CUDA launcher

`launch_c170_latent_angle_cuda.py` runs the approved train/validation experiment with one runner process per seed. Each experiment needs a new output directory. Probe output and formal output must use separate directories.

The launcher uses Python 3.10 with PyTorch 1.13.1+cu117, requires CUDA and `nvidia-smi`, and records GPU utilization and memory every two seconds. Each child process uses the configured CPU thread count (default 2). Runner stdout and stderr go to a separate `<variant>__seed-<seed>.stdout.log` file.

## Throughput probe

The probe suite always runs the same eight full-model seeds `17, 29, 43, 61, 89, 101, 113, 127`, for five epochs with patience five. Run each concurrency level in a separate output directory so the total work is comparable:

```powershell
& 'C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe' `
  '.\launch_c170_latent_angle_cuda.py' `
  --suite probe --concurrency 2 `
  --output-dir '.\outputs\c170_latent_angle_probe_w2'

& 'C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe' `
  '.\launch_c170_latent_angle_cuda.py' `
  --suite probe --concurrency 4 `
  --output-dir '.\outputs\c170_latent_angle_probe_w4'
```

Use `--concurrency 8` for the same eight-task probe only when the four-process measurements leave GPU, CPU, and memory capacity available. Probe checkpoints and validation results are for throughput and instrumentation checks; they do not enter model selection or formal evaluation.

## Formal matrix

The formal suite contains 25 fixed runs: five variants (`mlp`, `no_angle`, `full`, `no_rms`, and `shuffle`) for seeds `17, 29, 43, 61, 89`. Each run has a maximum of 160 epochs and patience 25. Formal concurrency can be set to two, four, or eight processes after the probe results and resource measurements have been reviewed. Start it only after the probe results have been reviewed and the concurrency level is chosen:

```powershell
& 'C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe' `
  '.\launch_c170_latent_angle_cuda.py' `
  --suite formal --concurrency 4 `
  --output-dir '.\outputs\c170_latent_angle_formal_w4'
```

The launcher passes the dataset path, device, CPU thread count, and physical channel-order confirmation to each runner. `--dataset PATH` can select another dataset file; its absolute path and SHA-256 become part of the run configuration.

## Outputs and resuming

Each experiment root contains `launcher_manifest.json`, `gpu_samples.csv`, one runner directory per seed, and one sibling stdout log per run. The launcher manifest records the exact command, return code, run status, source/data fingerprints, wall time, update throughput, startup/preprocessing/training time, and aggregate epoch timings. Each runner directory must contain a completed manifest, learning curve, validation predictions, checkpoint, and `RUN_COMPLETE.json`; the launcher verifies their hashes before marking a job complete.

To reuse already completed jobs, pass `--resume` with the same output root and identical suite, concurrency, dataset, runtime, and source files. Completed matching runs are verified and skipped. Existing incomplete run directories, changed configurations, missing artifacts, and hash mismatches stop the launcher without overwriting data; use a new output directory for a fresh run.

This launcher runs train/validation only. It does not compute final-test metrics.
