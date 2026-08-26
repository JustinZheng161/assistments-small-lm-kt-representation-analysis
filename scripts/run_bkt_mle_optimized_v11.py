import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'bkt_mle_optimized_v11.json'
SEED = 42
COLS = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']

df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=COLS)
df = df[df.skill_id.notna()].copy()
df['skill_id'] = df.skill_id.astype(str).str.strip()
top = df.skill_id.value_counts().head(11).index.tolist()
df = df[df.skill_id.isin(top)].sort_values(['user_id', 'order_id']).reset_index(drop=True)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(splitter.split(df, groups=df.user_id))
train = df.iloc[tr_idx]
test = df.iloc[te_idx]

# Numerically stable BKT log likelihood with bounded parameters.
def nll(params, seqs):
    pi, learn, guess, slip = params
    total = 0.0
    for seq in seqs:
        mastery = pi
        for obs in seq:
            pred = mastery * (1.0 - slip) + (1.0 - mastery) * guess
            total += np.log(max(pred if obs else 1.0 - pred, 1e-12))
            if obs:
                post = mastery * (1.0 - slip) / max(pred, 1e-12)
            else:
                post = mastery * slip / max(1.0 - pred, 1e-12)
            mastery = post + (1.0 - post) * learn
    return float(-total)

starts = [[.2,.1,.2,.1],[.5,.2,.1,.1],[.8,.5,.2,.2],[.1,.7,.3,.1]]
bounds = [(0.001,0.999)] * 4

def fit(seqs):
    runs = []
    for start in starts:
        r = minimize(lambda p: nll(p, seqs), start, method='L-BFGS-B', bounds=bounds, options={'maxiter':500, 'ftol':1e-10, 'gtol':1e-7})
        runs.append({'start': start, 'params': r.x.tolist(), 'nll': float(r.fun), 'success': bool(r.success), 'iterations': int(r.nit), 'message': str(r.message)})
    return min(runs, key=lambda z: z['nll']), runs

params = {}
fit_runs = {}
for skill, g in train.groupby('skill_id'):
    seqs = [x.correct.astype(int).tolist() for _, x in g.groupby('user_id')]
    best, runs = fit(seqs)
    params[str(skill)] = best['params']
    fit_runs[str(skill)] = {'best': best, 'runs': runs}


def predict(param_map):
    y_true, y_score = [], []
    for _, g in test.groupby('user_id'):
        state = {s: param_map[s][0] for s in top}
        for r in g.sort_values('order_id').itertuples():
            pi, learn, guess, slip = param_map[str(r.skill_id)]
            m = state[str(r.skill_id)]
            pred = m * (1.0 - slip) + (1.0 - m) * guess
            y_true.append(int(r.correct)); y_score.append(float(pred))
            if r.correct:
                post = m * (1.0 - slip) / max(pred, 1e-12)
            else:
                post = m * slip / max(1.0 - pred, 1e-12)
            state[str(r.skill_id)] = post + (1.0 - post) * learn
    return float(roc_auc_score(y_true, y_score)), int(len(y_true)), float(np.mean(y_true))

mle_auc, n_test, prevalence = predict(params)
fixed_map = {s: [.2,.1,.2,.1] for s in top}
fixed_auc, _, _ = predict(fixed_map)
majority_auc = 0.5
result = {
    'seed': SEED,
    'dataset': DATA.name,
    'skills': top,
    'train_rows': int(len(train)), 'test_rows': int(len(test)),
    'train_students': int(train.user_id.nunique()), 'test_students': int(test.user_id.nunique()),
    'time_order': 'train/test split by student; test interactions sorted by order_id before state updates',
    'optimized_model': {'name': 'skill-wise BKT with bounded multi-start L-BFGS-B MLE', 'auc': mle_auc, 'test_prevalence': prevalence, 'n_test': n_test, 'params': params, 'fit_runs': fit_runs},
    'fixed_bkt_reference': {'params': [.2,.1,.2,.1], 'auc': fixed_auc},
    'majority_auc_reference': majority_auc,
    'improvement_over_fixed_auc': mle_auc - fixed_auc,
    'interpretation': 'Future-response benchmark on a student-level held-out split; separate from prompt-readability and lexical-output audits.'
}
OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'mle_auc': mle_auc, 'fixed_auc': fixed_auc, 'improvement': mle_auc-fixed_auc, 'train_rows': len(train), 'test_rows': len(test)}, indent=2))
