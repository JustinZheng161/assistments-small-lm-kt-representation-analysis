import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TOP_K = 11
MAIN_N = 600
DEV_N = 24
AUDIT_N = 12
COLS = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']


def row_key(df):
    return set(zip(df.user_id.astype(str), df.problem_id.astype(str), df.skill_id.astype(str), df.correct.astype(str), df.order_id.astype(str)))


def stats(df):
    per_student = df.groupby('user_id').size()
    return {
        'rows': int(len(df)),
        'students': int(df.user_id.nunique()),
        'problems': int(df.problem_id.nunique()),
        'correct_mean': float(df.correct.mean()),
        'correct_sd': float(df.correct.std(ddof=1)),
        'student_rows_mean': float(per_student.mean()),
        'student_rows_median': float(per_student.median()),
        'student_rows_max': int(per_student.max()),
        'order_id_min': float(df.order_id.min()),
        'order_id_median': float(df.order_id.median()),
        'order_id_max': float(df.order_id.max()),
        'skill_counts': {str(k): int(v) for k, v in df.skill_id.value_counts().items()},
    }


def hash_student(x):
    return hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest()[:16]


df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=COLS)
df = df[df.skill_id.notna()].copy()
df['skill_id'] = df['skill_id'].astype(str).str.strip()
top = df.skill_id.value_counts().head(TOP_K).index.tolist()
subset = df[df.skill_id.isin(top)].copy().sort_values(['user_id', 'order_id']).reset_index(drop=True)

# Historical deterministic slices.
deterministic = {n: pd.concat([subset[subset.skill_id == s].head(n) for s in top], ignore_index=True) for n in [DEV_N, MAIN_N, AUDIT_N]}

# One randomized draw per skill, partitioned into three mutually exclusive sets.
rng = np.random.default_rng(SEED)
parts = {'main_random_600': [], 'development_random_24': [], 'audit_random_12': []}
for s in top:
    g = subset[subset.skill_id == s]
    if len(g) < MAIN_N + DEV_N + AUDIT_N:
        raise ValueError(f'Not enough rows for skill {s}')
    idx = rng.permutation(len(g))
    parts['main_random_600'].append(g.iloc[idx[:MAIN_N]])
    parts['development_random_24'].append(g.iloc[idx[MAIN_N:MAIN_N + DEV_N]])
    parts['audit_random_12'].append(g.iloc[idx[MAIN_N + DEV_N:MAIN_N + DEV_N + AUDIT_N]])
random_sets = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}

# Public manifests use hashes for student identifiers and retain only reconstruction fields.
for name, data in random_sets.items():
    m = data[['skill_id', 'problem_id', 'correct', 'order_id', 'user_id']].copy()
    m['student_hash'] = m['user_id'].map(hash_student)
    m = m.drop(columns=['user_id'])
    m.insert(0, 'sample_row', np.arange(len(m)))
    m.to_csv(OUT / f'{name}_manifest.csv', index=False)

summary = {
    'seed': SEED,
    'data_file': DATA.name,
    'top_skills': top,
    'sampling_rule': 'For each of the 11 most frequent skills, draw 636 rows without replacement with PCG64 seed 42; assign 600 to main, 24 to development, and 12 to audit. The three sets are mutually exclusive.',
    'deterministic_slices': {str(n): stats(x) for n, x in deterministic.items()},
    'random_sets': {name: stats(x) for name, x in random_sets.items()},
    'overlap_random_sets': {},
    'overlap_random_vs_deterministic': {},
    'comparison': {},
}
for a_name, a in random_sets.items():
    for b_name, b in random_sets.items():
        if a_name < b_name:
            summary['overlap_random_sets'][f'{a_name}__{b_name}'] = len(row_key(a) & row_key(b))
for n, d in deterministic.items():
    summary['overlap_random_vs_deterministic'][str(n)] = {
        name: len(row_key(d) & row_key(r)) for name, r in random_sets.items()
    }
for n in [DEV_N, MAIN_N, AUDIT_N]:
    d = summary['deterministic_slices'][str(n)]
    rname = {DEV_N: 'development_random_24', MAIN_N: 'main_random_600', AUDIT_N: 'audit_random_12'}[n]
    r = summary['random_sets'][rname]
    summary['comparison'][str(n)] = {
        'random_set': rname,
        'correct_mean_difference_random_minus_deterministic': r['correct_mean'] - d['correct_mean'],
        'students_difference_random_minus_deterministic': r['students'] - d['students'],
        'problems_difference_random_minus_deterministic': r['problems'] - d['problems'],
        'order_id_median_difference_random_minus_deterministic': r['order_id_median'] - d['order_id_median'],
    }

(OUT / 'sampling_bias_and_disjoint_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
