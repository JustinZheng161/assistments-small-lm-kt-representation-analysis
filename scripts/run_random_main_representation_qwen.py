import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'random_main_representation_qwen.json'
SEED = 42
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
LABEL = 'Qwen2.5-0.5B-Instruct'
cols = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']
df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=cols)
df = df[df.skill_id.notna()].copy()
df['skill_id'] = df.skill_id.astype(str).str.strip()
top = df.skill_id.value_counts().head(11).tolist() if False else df.skill_id.value_counts().head(11).index.tolist()
sub = df[df.skill_id.isin(top)].copy().sort_values(['user_id', 'order_id'])
rng = np.random.default_rng(SEED)
parts = []
for skill in top:
    g = sub[sub.skill_id == skill]
    idx = rng.permutation(len(g))
    parts.append(g.iloc[idx[:600]])
sample = pd.concat(parts, ignore_index=True)
code = {s:i for i,s in enumerate(top)}
y = sample.skill_id.map(code).to_numpy(dtype=int)
groups = sample.user_id.to_numpy()
texts = [f'Single interaction record. Skill {r.skill_id}. Previous response was {"correct" if r.correct == 1 else "wrong"}.' for r in sample.itertuples()]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()
states = []
with torch.inference_mode():
    for start in range(0, len(texts), 32):
        b = tok(texts[start:start+32], padding=True, truncation=True, max_length=96, return_tensors='pt')
        h = model(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1)
        states.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu().numpy())
X = np.vstack(states)
sil = float(silhouette_score(X, y, metric='cosine'))
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(splitter.split(X, y, groups=groups))
clf = LinearSVC(C=1.0, dual=False, max_iter=1000, random_state=SEED)
clf.fit(X[tr], y[tr])
acc = float(accuracy_score(y[te], clf.predict(X[te])))
result = {'seed': SEED, 'model': LABEL, 'sample_rows': int(len(sample)), 'skills': top, 'students': int(sample.user_id.nunique()), 'problems': int(sample.problem_id.nunique()), 'correct_mean': float(sample.correct.mean()), 'train_rows': int(len(tr)), 'test_rows': int(len(te)), 'train_students': int(np.unique(groups[tr]).size), 'test_students': int(np.unique(groups[te]).size), 'silhouette_cosine': sil, 'probe_classifier': 'LinearSVC(C=1.0, dual=False)', 'heldout_probe_accuracy': acc, 'interpretation': 'Descriptive result for the seed-42 stratified-random main sample; not a population estimate.'}
OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
del model, tok, states, X
gc.collect()
