# Reproducing the completed prompt-interface diagnostics

This directory provides local, end-to-end entry points for the completed **prompt-interface audit** in the manuscript. It is designed for the corrected ASSISTments 2009-2010 Skill Builder CSV obtained independently from the official ASSISTments data page. The scripts reconstruct the paper’s seed-42 samples, Qwen inference inputs, mappings, five-fold accessibility metrics, lexical output-sensitivity values, and local figures. They do **not** implement a chronology-preserving future-response knowledge-tracing benchmark.

## Before running

Install the repository dependencies, download the corrected student-problem CSV from the official ASSISTments source, and keep that file outside version control. The CSV must include `user_id`, `problem_id`, `skill_id`, `correct`, and `order_id`. Local results default to `reproduce-output/`, which is ignored by Git.

```bash
pip install -r requirements.txt
export ASSISTMENTS_CSV=/absolute/path/to/skill-builder-data-corrected.csv
```

Run the deterministic mapping check before model inference:

```bash
python scripts/reproduce/reproduce-random-mappings.py --data "$ASSISTMENTS_CSV"
```

For the exact corrected file used in the paper, `appendix_g_exact_match` should be `true`. The script reports the two 11-row mappings in Appendix G, including both seeds and the complete without-replacement algorithms. A `false` result indicates a different source-file version or a changed frequency order; do not compare model results until this has been resolved.

## Figure 2: explicit-field accessibility

```bash
python scripts/reproduce/reproduce-fig2.py --data "$ASSISTMENTS_CSV" --mode full
```

This reconstructs the 264-record development sample by taking zero-based rows `600:624` after a PCG64 seed-42 within-skill permutation of each of the 11 most frequent skills. It runs the revised non-temporal prompt with mean/max/last-token pooling and five-fold `GroupKFold` by `user_id`. For the historical Figure 2 reference, it additionally runs the earlier `Previous response was` prompt on the fixed seed-42 `GroupShuffleSplit` and applies 1,000 skill-label permutations while retaining the hidden states and split. It writes both analyses to `fig2-results.json` and writes `fig2-probe-permutation.png` locally.

## Figure 4: lexical output sensitivity

```bash
python scripts/reproduce/reproduce-fig4.py --data "$ASSISTMENTS_CSV" --mode full
```

This reconstructs the 132-record audit sample by taking zero-based rows `624:636` after the same within-skill seed-42 permutation. It evaluates the revised prompt under `explicit`, `neutral`, `unknown`, `apple`, `table`, and `removed` conditions; retains the earlier template as a comparison output; produces EOS-only, skill-only, and `decl` operational baselines; runs conditionwise 5,000 label-permutation AUC nulls; and reports paired record-level bootstrap and student-cluster bootstrap intervals. The primary paired inference is the student-cluster interval because the audit set contains repeated students. It writes `fig4-results.json` and `fig4-output-sensitivity.png` locally.

## Smoke mode and computational boundary

Both figure runners accept `--mode smoke`. This reduces permutations/bootstrap iterations and is useful for installation, path, model-loading, and result-schema checks. **Smoke mode must not be used to replace the full 1,000/2,000/5,000 iteration values reported in the paper.**

The default inference configuration is CPU-compatible float32, batch size 32, and maximum sequence length 96. The default model revision is `main`, but each result records both the requested revision and the resolved Hugging Face commit hash. For numerical comparison, use the resolved commit hash printed in the result file rather than assuming that a mutable `main` revision will remain unchanged. Because the historical run recorded `main` but not an immutable commit, a new run reconstructs the procedure under its resolved revision and cannot promise bitwise equality to the historical manuscript values. Accelerator users may explicitly select, for example, `--device cuda --dtype bfloat16 --batch-size 64`. Results record these choices. They affect execution throughput and local logging, not the sampling rules, prompts, pooling definitions, classifier, score construction, or interpretation boundaries.

## What can and cannot be independently verified

The public code makes the completed data-to-diagnostic procedure inspectable from the official dataset. It does not publish manuscript files, source rows, raw user identifiers, private hash salts, model caches, weight files, or prior private result archives. The published de-identified manifests permit checks of counts, folds, conditions, predictions, and logit-difference summaries without revealing raw learner identifiers.

The manuscript’s results remain a bounded single-model case study using Qwen2.5-0.5B-Instruct, selected high-frequency skills, fixed seed-42 samples, and specified prompt templates. They should not be interpreted as semantic understanding, learner-state estimates, or future-response KT performance.
