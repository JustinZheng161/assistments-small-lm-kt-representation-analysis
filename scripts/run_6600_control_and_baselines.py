from pathlib import Path
import json
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'control_and_baselines_6600.json'
DEFAULT_SEED = 42
SEEDS = [17, 29, 41, 53, 67]
MODELS = {
    'Qwen2.5-0.5B-Instruct': 'Qwen/Qwen2.5-0.5B-Instruct',
    'TinyLlama-1.1B-Chat-v1.0': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}


def load_sample(seed=DEFAULT_SEED, sample_mode='random'):
    cols = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']
    df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=cols)
    df = df[df['skill_id'].notna()].copy()
    df['skill_id'] = df['skill_id'].astype(str).str.strip()
    top = df['skill_id'].value_counts().head(11).index.tolist()
    df = df[df['skill_id'].isin(top)].copy()
    parts = []
    for skill in top:
        skill_df = df[df['skill_id'] == skill]
        if sample_mode == 'random':
            parts.append(skill_df.sample(n=min(600, len(skill_df)), random_state=seed))
        else:
            parts.append(skill_df.sort_values(['user_id', 'order_id']).head(600))
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sort_values(['user_id', 'order_id']).reset_index(drop=True)
    return sample, top


def split_indices(sample, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(splitter.split(sample, groups=sample['user_id']))
    return tr, te


def fit_probe(X, y, groups, seed):
    tr, te = split_indices(pd.DataFrame({'user_id': groups}), seed)
    clf = LogisticRegression(max_iter=800, solver='lbfgs', multi_class='auto')
    clf.fit(X[tr], y[tr])
    return {
        'train_rows': int(len(tr)),
        'test_rows': int(len(te)),
        'train_students': int(np.unique(groups[tr]).size),
        'test_students': int(np.unique(groups[te]).size),
        'accuracy': float(accuracy_score(y[te], clf.predict(X[te]))),
    }


def compute_hidden_states(model_name, texts, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    states = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=96, return_tensors='pt')
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            states.append(pooled.cpu().numpy())
    X = np.vstack(states)
    del model, tokenizer, states
    gc.collect()
    return X


def handcrafted_baselines(sample):
    work = sample.sort_values(['user_id', 'order_id']).copy()
    work['prev_correct'] = work.groupby('user_id')['correct'].shift(1)
    work = work[work['prev_correct'].notna()].copy()
    skill_dummies = pd.get_dummies(work['skill_id'], prefix='skill', dtype=float)
    X = np.column_stack([skill_dummies.to_numpy(), work[['prev_correct']].to_numpy(dtype=float)])
    y = work['correct'].to_numpy(dtype=int)
    groups = work['user_id'].to_numpy()
    outputs = []
    for seed in SEEDS:
        tr, te = split_indices(work, seed)
        train_rate = float(y[tr].mean())
        majority = np.full(len(te), int(train_rate >= 0.5), dtype=int)
        previous = work.iloc[te]['prev_correct'].to_numpy(dtype=float)
        majority_prob = np.full(len(te), train_rate, dtype=float)
        logistic = LogisticRegression(max_iter=500, solver='lbfgs')
        logistic.fit(X[tr], y[tr])
        prob = logistic.predict_proba(X[te])[:, 1]
        outputs.append({
            'seed': seed,
            'train_rows': int(len(tr)),
            'test_rows': int(len(te)),
            'train_students': int(np.unique(groups[tr]).size),
            'test_students': int(np.unique(groups[te]).size),
            'majority_accuracy': float(accuracy_score(y[te], majority)),
            'majority_auc': float(roc_auc_score(y[te], majority_prob)),
            'majority_logloss': float(log_loss(y[te], majority_prob, labels=[0, 1])),
            'last_response_auc': float(roc_auc_score(y[te], previous)),
            'last_response_logloss': float(log_loss(y[te], previous, labels=[0, 1])),
            'skill_prev_logistic_auc': float(roc_auc_score(y[te], prob)),
            'skill_prev_logistic_logloss': float(log_loss(y[te], prob, labels=[0, 1])),
        })
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--sample-mode', choices=['random', 'deterministic'], default='random')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sample, top = load_sample(seed=args.seed, sample_mode=args.sample_mode)
    texts = [
        f"Single interaction record. Skill {row.skill_id}. Previous response was {'correct' if row.correct == 1 else 'wrong'}."
        for row in sample.itertuples()
    ]
    label_codes = {skill: i for i, skill in enumerate(top)}
    y_skill = sample['skill_id'].map(label_codes).to_numpy(dtype=int)
    groups = sample['user_id'].to_numpy()
    results = {'sample_rows': int(len(sample)), 'skills': top, 'models': {}, 'handcrafted_baselines': handcrafted_baselines(sample)}

    rng = np.random.default_rng(args.seed)
    permutation_maps = [rng.permutation(len(top)) for _ in range(100)]
    for label, model_name in MODELS.items():
        print(f'loading {label}', flush=True)
        X = compute_hidden_states(model_name, texts, batch_size=args.batch_size)
        model_results = {'skill_probe': [], 'randomized_control_probe': [], 'hidden_size': int(X.shape[1])}
        for seed in SEEDS:
            model_results['skill_probe'].append(fit_probe(X, y_skill, groups, seed))
        perm_acc = []
        for perm_idx, mapping in enumerate(permutation_maps):
            y_control = mapping[y_skill]
            control_fit = fit_probe(X, y_control, groups, DEFAULT_SEED + perm_idx)
            perm_acc.append(control_fit['accuracy'])
        model_results['randomized_control_probe'] = {
            'n_permutations': len(perm_acc),
            'accuracy_mean': float(np.mean(perm_acc)),
            'accuracy_sd': float(np.std(perm_acc, ddof=1)),
            'accuracy_min': float(np.min(perm_acc)),
            'accuracy_max': float(np.max(perm_acc)),
            'accuracies': [float(x) for x in perm_acc],
        }
        results['models'][label] = model_results
        print(model_results, flush=True)
    results['seed'] = args.seed
    results['sample_mode'] = args.sample_mode
    results['batch_size'] = args.batch_size
    OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
