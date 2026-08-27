import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime_utils import inference_runtime, move_batch, resolve_device, resolve_dtype

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'token_matched_output_audit_random.json'
SEED = 42

parser = argparse.ArgumentParser(description='Run the random-audit token-matched output sensitivity audit.')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-length', type=int, default=96)
parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
parser.add_argument('--dtype', default='auto', choices=['auto', 'float32', 'float16', 'bfloat16'])
parser.add_argument('--bootstrap-iterations', type=int, default=1000, help='Bootstrap resamples; default preserves the paper audit.')
parser.add_argument('--template', default='revised', choices=['revised', 'old'], help='Prompt template; revised matches the manuscript primary audit.')
args = parser.parse_args()
device = resolve_device(args.device)
dtype = resolve_dtype(args.dtype, device)
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
    elif mode == 'apple':
        phrase = 'apple'
    elif mode == 'table':
        phrase = 'table'
    elif mode == 'removed':
        phrase = 'removed'
    if args.template == 'revised':
        return f'Single interaction record. Skill {r.skill_id}. The response to this skill question was {phrase}.'
    return f'Single interaction record. Skill {r.skill_id}. Previous response was {phrase}.'


tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=dtype).to(device)
model.eval()
cor_id = tok(' correct', add_special_tokens=False)['input_ids'][0]
wrong_id = tok(' wrong', add_special_tokens=False)['input_ids'][0]
word_counts = {w: len(tok(w, add_special_tokens=False)['input_ids']) for w in ['correct', 'wrong', 'neutral', 'unknown', 'apple', 'table']}

def score(texts):
    out = []
    with inference_runtime(device):
        for start in range(0, len(texts), args.batch_size):
            b = tok(texts[start:start+args.batch_size], padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
            b = move_batch(b, device)
            hidden = model.model(**b).last_hidden_state
            idx = b['attention_mask'].sum(1) - 1
            last_hidden = hidden[torch.arange(len(idx), device=device), idx]
            v = model.lm_head(last_hidden)
            out.append((v[:, cor_id] - v[:, wrong_id]).float().cpu().numpy())
    return np.concatenate(out)

results = {'seed': SEED, 'sample_rows': int(len(sample)), 'sample_rule': 'random audit set: rows 625-636 of a seed-42 within-skill permutation after reserving 600 main and 24 development records', 'model': MODEL_LABEL, 'runtime': {'device': str(device), 'dtype': str(dtype).replace('torch.', ''), 'batch_size': args.batch_size, 'max_length': args.max_length, 'bootstrap_iterations': args.bootstrap_iterations, 'template': args.template}, 'token_counts': word_counts, 'modes': {}, 'condition_note': 'The `removed` condition is a literal one-token replacement word. The separate skill-only baseline removes the response phrase.'}
for mode in ['explicit', 'neutral', 'unknown', 'apple', 'table', 'removed']:
    values = score([make_prompt(r, mode) for r in sample.itertuples()])
    auc = float(roc_auc_score(y, values))
    brng = np.random.default_rng(SEED)
    boot = []
    for _ in range(args.bootstrap_iterations):
        idx = brng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            boot.append(float(roc_auc_score(y[idx], values[idx])))
    results['modes'][mode] = {'auc': auc, 'bootstrap_iterations': args.bootstrap_iterations, 'bootstrap_fixed_sample_percentile_range': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], 'score_mean': float(values.mean()), 'score_sd': float(values.std())}
OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
del model, tok
gc.collect()
