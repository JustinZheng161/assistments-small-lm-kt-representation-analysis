import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModel, AutoTokenizer

from runtime_utils import inference_runtime, move_batch, resolve_device, resolve_dtype

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'probe_controls_random_development.json'
SEED = 42

parser = argparse.ArgumentParser(description='Run the random-development explicit-field accessibility audit.')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-length', type=int, default=96)
parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
parser.add_argument('--dtype', default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
parser.add_argument('--permutations', type=int, default=1000, help='Number of label permutations; default preserves the paper audit.')
args = parser.parse_args()
device = resolve_device(args.device)
dtype = resolve_dtype(args.dtype, device)
MODELS = {
    'Qwen2.5-0.5B-Instruct': 'Qwen/Qwen2.5-0.5B-Instruct',
}

cols = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']
df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=cols)
df = df[df.skill_id.notna()].copy()
df['skill_id'] = df.skill_id.astype(str).str.strip()
top = df.skill_id.value_counts().head(11).index.tolist()
sub = df[df.skill_id.isin(top)].copy().sort_values(['user_id', 'order_id'])
rng = np.random.default_rng(SEED)
rows = []
for skill in top:
    g = sub[sub.skill_id == skill]
    idx = rng.permutation(len(g))
    rows.append(g.iloc[idx[600:624]])
sample = pd.concat(rows, ignore_index=True)
label_codes = {s: i for i, s in enumerate(top)}
y = sample.skill_id.map(label_codes).to_numpy(dtype=int)
groups = sample.user_id.to_numpy()
texts = [f'Single interaction record. Skill {r.skill_id}. Previous response was {"correct" if r.correct == 1 else "wrong"}.' for r in sample.itertuples()]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(splitter.split(sample, y, groups=groups))

def fit_acc(X, labels, train=tr, test=te, seed=SEED):
    clf = LinearSVC(C=1.0, dual=False, max_iter=1000, random_state=seed)
    clf.fit(X[train], labels[train])
    return float(accuracy_score(labels[test], clf.predict(X[test])))

# Skill-ID one-hot is an explicit lexical null model, not a semantic baseline.
X_onehot = np.zeros((len(sample), len(top)), dtype=np.float32)
X_onehot[np.arange(len(sample)), y] = 1.0
result = {
    'seed': SEED,
    'sample_rule': 'random development set: rows 601-624 of a seed-42 random permutation within each top skill after reserving rows 1-600 for main and 625-636 for audit',
    'sample_rows': int(len(sample)),
    'student_count': int(sample.user_id.nunique()),
    'train_rows': int(len(tr)),
    'test_rows': int(len(te)),
    'train_students': int(np.unique(groups[tr]).size),
    'test_students': int(np.unique(groups[te]).size),
    'classifier': 'LinearSVC(C=1.0, dual=False)',
    'runtime': {'device': str(device), 'dtype': str(dtype).replace('torch.', ''), 'batch_size': args.batch_size, 'max_length': args.max_length, 'permutations': args.permutations},
    'one_hot_skill_token_accuracy': fit_acc(X_onehot, y),
    'models': {},
}

for label, name in MODELS.items():
    print('loading', label, flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.eval()
    states = []
    with inference_runtime(device):
        for start in range(0, len(texts), args.batch_size):
            b = tok(texts[start:start+args.batch_size], padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            b = move_batch(b, device)
            h = model(**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1)
            states.append(((h * m).sum(1) / m.sum(1).clamp_min(1)).float().cpu().numpy())
    X = np.vstack(states)
    true_acc = fit_acc(X, y)
    rng_perm = np.random.default_rng(SEED)
    accs = []
    for i in range(args.permutations):
        perm = rng_perm.permutation(y)
        accs.append(fit_acc(X, perm, seed=SEED+i))
    result['models'][label] = {
        'hidden_size': int(X.shape[1]),
        'true_label_accuracy': true_acc,
        'permutation_count': args.permutations,
        'permutation_accuracy_mean': float(np.mean(accs)),
        'permutation_accuracy_sd': float(np.std(accs, ddof=1)),
        'permutation_accuracy_min': float(np.min(accs)),
        'permutation_accuracy_max': float(np.max(accs)),
        'empirical_p_right_tail': float((1 + sum(a >= true_acc for a in accs)) / (len(accs) + 1)),
    }
    del model, tok, states, X
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
