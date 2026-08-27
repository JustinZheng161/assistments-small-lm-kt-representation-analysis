# ASSISTments prompt-audit reproducibility code

This repository contains reusable code for auditing prompt interfaces and evaluating a conventional BKT baseline. It intentionally does **not** publish the manuscript, manuscript figures, raw or derived data, experiment outputs, manifests, model caches, or private analysis notes.

The scripts expect the user to obtain the relevant public dataset independently and to set a local data path before execution. No numerical result is asserted by this repository. Any local outputs should remain outside the repository or under ignored paths.

The repository is provided as a code-only companion. The manuscript and its core data/results are maintained separately in a private repository.

## Reproducible model-audit runtime

The two hidden-state and output-audit entry points now share `scripts/runtime_utils.py`. The helper centralizes device resolution, explicit precision selection, tensor movement, inference mode, and accelerator-cache cleanup. The default remains CPU-compatible float32 inference, so the numerical audit definition is unchanged. On a compatible accelerator, users may opt into a larger batch size or reduced precision and the run records the selected device, dtype, batch size, and maximum sequence length in its ignored JSON output.

For a default run, use:

```bash
python scripts/run_probe_controls_random_dev.py
python scripts/run_token_matched_output_audit_random.py
```

For a throughput-oriented local run on a CUDA device, use an explicitly selected precision and batch size, then compare the resulting metrics with the default float32 run:

```bash
python scripts/run_probe_controls_random_dev.py --device cuda --dtype bfloat16 --batch-size 64
python scripts/run_token_matched_output_audit_random.py --device cuda --dtype bfloat16 --batch-size 64
```

For resource-limited exploratory runs, `--permutations` controls the label-permutation count in the probe script and `--bootstrap-iterations` controls the bootstrap count in the output script. Both default to 1000, matching the documented audit settings; reduced counts are suitable only for smoke tests and must not replace the paper settings in a final rerun.

The optimization changes execution throughput and resource control only. It does not alter the sampling rule, labels, folds, classifier, prompt text, pooling definition, output score, bootstrap procedure, or reported interpretation boundaries. The repository does not publish model outputs or sensitive records.
