# De-identified supplementary manifests for v18

This directory contains the de-identified row-level summaries used to verify the reported development-set folds and output-audit conditions for the Qwen2.5-0.5B-Instruct prompt-interface case study.

The files contain no raw `user_id`, private hash salt, model weights, tokenizer cache, or source data. `hashed_user_id` is an opaque 16-character identifier carried over from the private sampling manifest; it is not reversible from the public files. The files are released as reproducibility summaries, not as the ASSISTments dataset.

| File | Contents | Records represented |
|---|---|---:|
| `manifest-dev-v18.json` | `record_index`, `fold_id`, `hashed_user_id`, `true_skill`, `predicted_skill`, `pooling_type` | 264 records × 3 pooling rules |
| `manifest-audit-v18.json` | `record_index`, `hashed_user_id`, `true_label`, `skill`, `condition`, `logit_diff` | 132 records × 6 output conditions |
| `supplementary-summary-v18.json` | Schema and count checks | Summary only |

The development manifest supports recomputation of five-fold fold counts and probe accuracy. The audit manifest supports recomputation of condition-specific score summaries; it does not disclose the original record text or the private cross-set mapping. The public repository remains code-focused, and the core dataset and private evidence remain outside this repository.

To verify the files locally, run:

```bash
python scripts/verify-supplementary-v18.py
```
