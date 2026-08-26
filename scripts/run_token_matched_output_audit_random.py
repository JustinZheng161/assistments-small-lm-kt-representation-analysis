import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'token_matched_output_audit_random.json'
SEED = 42
MODEL_LABEL = 'Qwen2.5-0.5B-Instruct'
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'

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
    rows.append(g.iloc[idx[624:636]])
sample = pd.concat(rows, ignore_index=True)
y = sample.correct.to_numpy(dtype=int)

# All tested words are one token in the native tokenizer; neutral retains this count.
def make_prompt(r, mode):
    if mode == 'explicit':
        phrase = 'correct' if r.correct == 1 else 'wrong'
    elif mode == 'neutral':
        phrase = 'neutral'
    elif mode == 'unknown':
        phrase = 'unknown'
    elif mode == 'removed':
        return f'Single interaction record. Skill {r.skill_id}.'
    return f'Single interaction record. Skill {r.skill_id}. Previous response was {phrase}.'


tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()
cor_id = tok(' correct', add_special_tokens=False)['input_ids'][0]
wrong_id = tok(' wrong', add_special_tokens=False)['input_ids'][0]
word_counts = {w: len(tok(w, add_special_tokens=False)['input_ids']) for w in ['correct', 'wrong', 'neutral', 'unknown']}

def score(texts):
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            b = tok(texts[start:start+32], padding=True, truncation=True, max_length=96, return_tensors='pt')
            hidden = model.model(**b).last_hidden_state
            idx = b['attention_mask'].sum(1) - 1
            last_hidden = hidden[torch.arange(len(idx)), idx]
            v = model.lm_head(last_hidden)
            out.append((v[:, cor_id] - v[:, wrong_id]).cpu().numpy())
    return np.concatenate(out)

results = {'seed': SEED, 'sample_rows': int(len(sample)), 'sample_rule': 'random audit set: rows 625-636 of a seed-42 within-skill permutation after reserving 600 main and 24 development records', 'model': MODEL_LABEL, 'token_counts': word_counts, 'modes': {}}
for mode in ['explicit', 'neutral', 'unknown', 'removed']:
    values = score([make_prompt(r, mode) for r in sample.itertuples()])
    auc = float(roc_auc_score(y, values))
    brng = np.random.default_rng(SEED)
    boot = []
    for _ in range(1000):
        idx = brng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            boot.append(float(roc_auc_score(y[idx], values[idx])))
    results['modes'][mode] = {'auc': auc, 'bootstrap_iterations': 1000, 'bootstrap_fixed_sample_percentile_range': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], 'score_mean': float(values.mean()), 'score_sd': float(values.std())}
OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
del model, tok
gc.collect()
