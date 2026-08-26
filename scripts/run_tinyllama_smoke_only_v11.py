import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
text = 'Single interaction record. Skill 311. Previous response was correct.'
tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32)
model.eval()
b = tok([text], return_tensors='pt')
with torch.inference_mode():
    hidden = model.model(**b).last_hidden_state
    last = hidden[:, -1, :]
    logits = model.lm_head(last)
res = {'checkpoint': name, 'tokens': int(b['input_ids'].shape[1]), 'hidden_shape': list(hidden.shape), 'last_shape': list(last.shape), 'logits_shape': list(logits.shape), 'finite_hidden': bool(torch.isfinite(hidden).all()), 'finite_logits': bool(torch.isfinite(logits).all()), 'status': 'passed'}
Path('/home/ubuntu/paper_work/repo/results/tinyllama_smoke_v11.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
