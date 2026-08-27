#!/usr/bin/env python3
"""Reproduce the Figure 4 lexical output-sensitivity diagnostic locally.

Requires a corrected ASSISTments 2009-2010 Skill Builder CSV supplied by the user.
All local result files are written beneath an ignored output directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_utils import inference_runtime, move_batch, resolve_device, resolve_dtype

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
SAMPLE_SEED = 42
BOOTSTRAP_SEED = 141
AUC_NULL_SEED = 143
CLUSTER_BOOTSTRAP_SEED = 151


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Corrected ASSISTments CSV downloaded independently.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduce-output' / 'fig4')
    parser.add_argument('--mode', choices=['full', 'smoke'], default='full')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=96)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--model-revision', default='main', help='Hugging Face model revision; the resolved commit hash is recorded in output.')
    return parser.parse_args()


def sample_audit(data: Path) -> tuple[pd.DataFrame, list[str]]:
    required = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']
    frame = pd.read_csv(data, encoding='latin1', low_memory=False, usecols=required)
    frame = frame[frame.skill_id.notna()].copy()
    frame['skill_id'] = frame.skill_id.astype(str).str.strip()
    frame['user_id'] = frame.user_id.astype(str)
    top = frame.skill_id.value_counts().head(11).index.tolist()
    subset = frame[frame.skill_id.isin(top)].sort_values(['user_id', 'order_id'])
    rng = np.random.default_rng(SAMPLE_SEED)
    parts = []
    for skill in top:
        within_skill = subset[subset.skill_id == skill].reset_index(drop=True)
        if len(within_skill) < 636:
            raise ValueError(f'Skill {skill} has fewer than 636 rows; corrected input differs from the paper file.')
        parts.append(within_skill.iloc[rng.permutation(len(within_skill))[624:636]])
    return pd.concat(parts, ignore_index=True), top


def prompt(row, condition: str, template: str) -> str:
    if condition == 'eos-only':
        return '<|endoftext|>'
    if condition == 'skill-only':
        return f'Single interaction record. Skill {row.skill_id}.'
    if condition == 'explicit':
        phrase = 'correct' if int(row.correct) else 'wrong'
    else:
        phrase = condition
    if template == 'revised':
        return f'Single interaction record. Skill {row.skill_id}. The response to this skill question was {phrase}.'
    return f'Single interaction record. Skill {row.skill_id}. Previous response was {phrase}.'


def score_texts(texts, tokenizer, model, correct_id, wrong_id, device, batch_size, max_length):
    score = []
    with inference_runtime(device):
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            batch = move_batch(batch, device)
            hidden = model.model(**batch).last_hidden_state
            last_index = batch['attention_mask'].sum(1) - 1
            last_hidden = hidden[torch.arange(len(last_index), device=device), last_index]
            logits = model.lm_head(last_hidden)
            score.append((logits[:, correct_id] - logits[:, wrong_id]).float().cpu().numpy())
    return np.concatenate(score)


def auc_null(y, score, n_permutations, rng):
    observed = float(roc_auc_score(y, score))
    null = np.empty(n_permutations)
    for iteration in range(n_permutations):
        null[iteration] = roc_auc_score(rng.permutation(y), score)
    return {
        'auc': observed,
        'null_mean': float(null.mean()),
        'null_sd': float(null.std(ddof=1)),
        'two_sided_label_permutation_p_vs_0_5': float((1 + np.sum(np.abs(null - 0.5) >= abs(observed - 0.5))) / (n_permutations + 1)),
    }


def paired_record_bootstrap(y, explicit, replacement, n_bootstrap, rng):
    differences = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(y), len(y))
        differences.append(roc_auc_score(y[index], explicit[index]) - roc_auc_score(y[index], replacement[index]))
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])]


def paired_sign_flip(y, explicit, replacement, n_permutations, rng):
    observed = roc_auc_score(y, explicit) - roc_auc_score(y, replacement)
    null = np.empty(n_permutations)
    for iteration in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(y))
        null[iteration] = roc_auc_score(y, explicit * signs) - roc_auc_score(y, replacement * signs)
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_permutations + 1))


def student_cluster_bootstrap(y, users, explicit, replacement, n_bootstrap, rng):
    unique_users = np.unique(users)
    differences = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_users, size=len(unique_users), replace=True)
        index = np.concatenate([np.flatnonzero(users == user) for user in sampled])
        differences.append(roc_auc_score(y[index], explicit[index]) - roc_auc_score(y[index], replacement[index]))
    return [float(value) for value in np.percentile(differences, [2.5, 97.5])]


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f'Input data not found: {args.data}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit, top = sample_audit(args.data)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=args.model_revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision=args.model_revision, trust_remote_code=True, torch_dtype=dtype).to(device)
    resolved_revision = getattr(model.config, '_commit_hash', args.model_revision)
    model.eval()
    correct_id = tokenizer(' correct', add_special_tokens=False)['input_ids'][0]
    wrong_id = tokenizer(' wrong', add_special_tokens=False)['input_ids'][0]
    conditions = ['explicit', 'neutral', 'unknown', 'apple', 'table', 'removed']
    score = {
        condition: score_texts([prompt(row, condition, 'revised') for row in audit.itertuples()], tokenizer, model, correct_id, wrong_id, device, args.batch_size, args.max_length)
        for condition in conditions
    }
    baseline_conditions = ['eos-only', 'skill-only', 'decl']
    baselines = {
        condition: score_texts([prompt(row, condition, 'revised') for row in audit.itertuples()], tokenizer, model, correct_id, wrong_id, device, args.batch_size, args.max_length)
        for condition in baseline_conditions
    }
    old_score = {
        condition: score_texts([prompt(row, condition, 'old') for row in audit.itertuples()], tokenizer, model, correct_id, wrong_id, device, args.batch_size, args.max_length)
        for condition in conditions
    }
    y = audit.correct.to_numpy(dtype=int)
    users = audit.user_id.to_numpy()
    n_bootstrap = 2000 if args.mode == 'full' else 20
    n_permutations = 5000 if args.mode == 'full' else 50
    null_rng = np.random.default_rng(AUC_NULL_SEED)
    auc_tests = {condition: auc_null(y, score[condition], n_permutations, null_rng) for condition in conditions}
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    sign_rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    cluster_rng = np.random.default_rng(CLUSTER_BOOTSTRAP_SEED)
    paired = {}
    for condition in conditions[1:]:
        difference = float(roc_auc_score(y, score['explicit']) - roc_auc_score(y, score[condition]))
        sign_p = paired_sign_flip(y, score['explicit'], score[condition], n_permutations, sign_rng)
        paired[condition] = {
            'explicit_minus_replacement_auc': difference,
            'record_level_paired_bootstrap_ci95': paired_record_bootstrap(y, score['explicit'], score[condition], n_bootstrap, bootstrap_rng),
            'paired_sign_flip_p': sign_p,
            'bonferroni_p_5': float(min(1.0, 5 * sign_p)),
            'student_cluster_bootstrap_ci95': student_cluster_bootstrap(y, users, score['explicit'], score[condition], n_bootstrap, cluster_rng),
        }
    payload = {
        'model': MODEL_NAME,
        'sample': {'seed': SAMPLE_SEED, 'audit_rows': int(len(audit)), 'audit_students': int(audit.user_id.nunique()), 'selected_skills': top, 'sampling_rule': 'rows 625-636 after a within-skill PCG64 seed-42 permutation; rows use zero-based iloc 624:636'},
        'runtime': {'device': str(device), 'dtype': str(dtype).replace('torch.', ''), 'batch_size': args.batch_size, 'max_length': args.max_length, 'mode': args.mode, 'requested_model_revision': args.model_revision, 'resolved_model_revision': resolved_revision},
        'templates': {'revised': 'Single interaction record. Skill [skill]. The response to this skill question was [correct/wrong].', 'old': 'Single interaction record. Skill [skill]. Previous response was [correct/wrong].'},
        'conditionwise_auc_null': {'unit': 'record-level labels; fixed condition scores; two-sided deviation from AUC 0.5', 'seed': AUC_NULL_SEED, 'iterations': n_permutations, 'results': auc_tests},
        'paired_replacement_inference': {'primary': 'student-cluster bootstrap because audit records contain repeated students; record-level paired bootstrap is a sensitivity analysis', 'student_cluster_bootstrap_seed': CLUSTER_BOOTSTRAP_SEED, 'record_bootstrap_seed': BOOTSTRAP_SEED, 'sign_flip_seed': BOOTSTRAP_SEED + 1, 'iterations': n_bootstrap, 'sign_flip_iterations': n_permutations, 'comparisons': paired},
        'revised_template_auc': {condition: float(roc_auc_score(y, score[condition])) for condition in conditions},
        'old_template_auc': {condition: float(roc_auc_score(y, old_score[condition])) for condition in conditions},
        'operational_baseline_auc': {condition: float(roc_auc_score(y, baselines[condition])) for condition in baseline_conditions},
        'scope_note': 'The script reconstructs a prompt-interface output-sensitivity audit. The `removed` condition is a literal one-token replacement word, not phrase deletion; the separate skill-only baseline omits the response phrase. The EOS-only and decl conditions are reported as operational calibration baselines, not no-information or semantic controls. It is not a chronology-preserving next-response KT benchmark.',
    }
    (args.output_dir / 'fig4-results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plt.figure(figsize=(7.0, 4.2), dpi=300)
    values = [payload['revised_template_auc'][condition] for condition in conditions]
    labels = ['explicit', 'neutral', 'unknown', 'apple', 'table', 'removed']
    plt.bar(labels, values, color=['#185a9d'] + ['#75808c'] * 5)
    plt.axhline(0.5, color='black', linewidth=0.8, linestyle='--')
    plt.ylim(0.3, 1.05)
    plt.ylabel('ROC AUC')
    plt.title('Output sensitivity under the revised non-temporal prompt')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'fig4-output-sensitivity.png', dpi=300)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
