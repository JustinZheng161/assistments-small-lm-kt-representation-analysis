#!/usr/bin/env python3
"""Reproduce the Figure 2 explicit-field accessibility diagnostic locally.

The user must obtain the corrected ASSISTments 2009-2010 Skill Builder CSV independently.
Outputs are written only to an ignored local directory.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_utils import inference_runtime, move_batch, resolve_device, resolve_dtype

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
SAMPLE_SEED = 42
MAPPING_SEED_EQUAL = 42
MAPPING_SEED_FIVE_DIGIT = 137
EXPECTED_EQUAL = {
    '311': '978', '47': '72', '277': '166', '280': '304', '312': '430',
    '79': '15', '279': '119', '27': '69', '18': '70', '50': '91', '77': '29',
}
EXPECTED_FIVE_DIGIT = {
    '311': '85174', '47': '27084', '277': '63058', '280': '34752', '312': '54197',
    '79': '66277', '279': '52065', '27': '87228', '18': '41471', '50': '31139', '77': '48423',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Corrected ASSISTments CSV downloaded independently.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reproduce-output' / 'fig2')
    parser.add_argument('--mode', choices=['full', 'smoke'], default='full')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-length', type=int, default=96)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'float16', 'bfloat16'], default='auto')
    parser.add_argument('--model-revision', default='main', help='Hugging Face model revision; the resolved commit hash is recorded in output.')
    return parser.parse_args()


def sample_development(data: Path) -> tuple[pd.DataFrame, list[str]]:
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
        parts.append(within_skill.iloc[rng.permutation(len(within_skill))[600:624]])
    return pd.concat(parts, ignore_index=True), top


def equal_character_mapping(top: list[str]) -> dict[str, str]:
    rng = np.random.default_rng(MAPPING_SEED_EQUAL)
    mapping: dict[str, str] = {}
    for skill in top:
        length = len(skill)
        candidates = [f'{number:0{length}d}' for number in range(10 ** length) if f'{number:0{length}d}' != skill]
        rng.shuffle(candidates)
        for candidate in candidates:
            if candidate not in mapping.values():
                mapping[skill] = candidate
                break
    return mapping


def five_digit_mapping(top: list[str]) -> dict[str, str]:
    rng = np.random.default_rng(MAPPING_SEED_FIVE_DIGIT)
    values = rng.choice(np.arange(10000, 100000), size=len(top), replace=False).astype(str).tolist()
    return dict(zip(top, values))


def encode(texts: list[str], tokenizer, model, device: torch.device, batch_size: int, max_length: int) -> dict[str, np.ndarray]:
    means, maximums, lasts = [], [], []
    with inference_runtime(device):
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            batch = move_batch(batch, device)
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].bool()
            mask_float = mask.unsqueeze(-1).float()
            means.append(((hidden * mask_float).sum(1) / mask_float.sum(1).clamp_min(1)).float().cpu().numpy())
            maximums.append(hidden.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values.float().cpu().numpy())
            last_index = batch['attention_mask'].sum(1) - 1
            lasts.append(hidden[torch.arange(len(last_index), device=device), last_index].float().cpu().numpy())
    return {'mean': np.vstack(means), 'max': np.vstack(maximums), 'last-token': np.vstack(lasts)}


def five_fold_metrics(features: dict[str, np.ndarray], labels: np.ndarray, groups: np.ndarray, top: list[str]) -> dict:
    results = {}
    folds = list(GroupKFold(n_splits=5).split(features['mean'], labels, groups))
    for name, matrix in features.items():
        fold_metrics = []
        for fold, (train, test) in enumerate(folds, 1):
            classifier = LinearSVC(C=1.0, dual=False, max_iter=5000).fit(matrix[train], labels[train])
            prediction = classifier.predict(matrix[test])
            fold_metrics.append({
                'fold': fold,
                'test_rows': int(len(test)),
                'test_students': int(np.unique(groups[test]).size),
                'accuracy': float(accuracy_score(labels[test], prediction)),
                'macro_f1': float(f1_score(labels[test], prediction, labels=top, average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(labels[test], prediction, labels=top, average='weighted', zero_division=0)),
            })
        results[name] = {
            'folds': fold_metrics,
            'accuracy_mean': float(np.mean([fold['accuracy'] for fold in fold_metrics])),
            'accuracy_sd': float(np.std([fold['accuracy'] for fold in fold_metrics], ddof=1)),
        }
    return results


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f'Input data not found: {args.data}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development, top = sample_development(args.data)
    equal_map = equal_character_mapping(top)
    five_map = five_digit_mapping(top)
    expected_mapping_match = top == list(EXPECTED_EQUAL) and equal_map == EXPECTED_EQUAL
    if top == list(EXPECTED_FIVE_DIGIT):
        expected_mapping_match = expected_mapping_match and five_map == EXPECTED_FIVE_DIGIT

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=args.model_revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=args.model_revision, trust_remote_code=True, torch_dtype=dtype).to(device)
    resolved_revision = getattr(model.config, '_commit_hash', args.model_revision)
    model.eval()
    revised_texts = [
        f'Single interaction record. Skill {row.skill_id}. The response to this skill question was '
        f'{"correct" if int(row.correct) else "wrong"}.' for row in development.itertuples()
    ]
    old_texts = [
        f'Single interaction record. Skill {row.skill_id}. Previous response was '
        f'{"correct" if int(row.correct) else "wrong"}.' for row in development.itertuples()
    ]
    features = encode(revised_texts, tokenizer, model, device, args.batch_size, args.max_length)
    old_features = encode(old_texts, tokenizer, model, device, args.batch_size, args.max_length)
    labels = development.skill_id.to_numpy()
    groups = development.user_id.to_numpy()
    metrics = five_fold_metrics(features, labels, groups, top)

    permutation_count = 1000 if args.mode == 'full' else 10
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SAMPLE_SEED)
    train, test = next(splitter.split(development, labels, groups=groups))
    fixed_classifier = LinearSVC(C=1.0, dual=False, max_iter=1000, random_state=SAMPLE_SEED).fit(old_features['mean'][train], labels[train])
    observed = float(accuracy_score(labels[test], fixed_classifier.predict(old_features['mean'][test])))
    rng = np.random.default_rng(SAMPLE_SEED)
    null_accuracy = []
    for iteration in range(permutation_count):
        permuted = rng.permutation(labels)
        classifier = LinearSVC(C=1.0, dual=False, max_iter=1000, random_state=SAMPLE_SEED + iteration).fit(old_features['mean'][train], permuted[train])
        null_accuracy.append(float(accuracy_score(permuted[test], classifier.predict(old_features['mean'][test]))))
    permutation = {
        'template': 'Single interaction record. Skill [skill]. Previous response was [correct/wrong].',
        'split': 'one GroupShuffleSplit test fraction 0.2, seed 42, grouped by user_id',
        'unit': 'skill labels permuted across the 264 development records; old-template features and fixed student-level train/test assignment retained',
        'iterations': permutation_count,
        'observed_accuracy': observed,
        'null_mean': float(np.mean(null_accuracy)),
        'null_sd': float(np.std(null_accuracy, ddof=1)),
        'null_min': float(np.min(null_accuracy)),
        'null_max': float(np.max(null_accuracy)),
        'empirical_right_tail_p': float((1 + sum(value >= observed for value in null_accuracy)) / (permutation_count + 1)),
    }
    payload = {
        'model': MODEL_NAME,
        'prompt': 'Single interaction record. Skill [skill]. The response to this skill question was [correct/wrong].',
        'sample': {'seed': SAMPLE_SEED, 'development_rows': int(len(development)), 'selected_skills': top, 'sampling_rule': 'rows 601-624 after a within-skill PCG64 seed-42 permutation; rows use zero-based iloc 600:624'},
        'runtime': {'device': str(device), 'dtype': str(dtype).replace('torch.', ''), 'batch_size': args.batch_size, 'max_length': args.max_length, 'mode': args.mode, 'requested_model_revision': args.model_revision, 'resolved_model_revision': resolved_revision},
        'mappings': {'equal_character_seed': MAPPING_SEED_EQUAL, 'equal_character': equal_map, 'five_digit_seed': MAPPING_SEED_FIVE_DIGIT, 'five_digit': five_map, 'appendix_g_match_when_top_order_matches': expected_mapping_match},
        'five_fold_probe': metrics,
        'historical_fixed_split_label_permutation_null': permutation,
        'scope_note': 'This reconstructs the prompt-interface diagnostic; it is not a history-based next-response KT evaluation.',
    }
    (args.output_dir / 'fig2-results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plt.figure(figsize=(6.4, 4.0), dpi=300)
    plt.hist(null_accuracy, bins=min(30, max(5, permutation_count // 10)), color='#9aa5b1', edgecolor='white', label='label-permutation null')
    plt.axvline(observed, color='#185a9d', linewidth=2, label=f'observed fixed-split accuracy = {observed:.3f}')
    plt.xlabel('Fixed-split accuracy')
    plt.ylabel('Permutation count')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / 'fig2-probe-permutation.png', dpi=300)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
