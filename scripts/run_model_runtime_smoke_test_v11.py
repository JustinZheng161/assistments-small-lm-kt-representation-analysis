import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path('/home/ubuntu/paper_work/repo/results/model_runtime_smoke_test_v11.json')
MODELS = {
    'Qwen2.5-0.5B-Instruct': 'Qwen/Qwen2.5-0.5B-Instruct',
    'TinyLlama-1.1B-Chat-v1.0': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}
text = 'Single interaction record. Skill 311. Previous response was correct.'
results = {}
for label, name in MODELS.items():
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    batch = tok([text, text], padding=True, return_tensors='pt')
    with torch.inference_mode():
        hidden = model.model(**batch).last_hidden_state
        idx = batch['attention_mask'].sum(1) - 1
        last = hidden[torch.arange(len(idx)), idx]
        logits = model.lm_head(last)
    results[label] = {'checkpoint': name, 'token_count': int(batch['attention_mask'][0].sum()), 'hidden_shape': list(hidden.shape), 'last_hidden_shape': list(last.shape), 'lm_head_shape': list(logits.shape), 'finite_hidden': bool(torch.isfinite(hidden).all()), 'finite_logits': bool(torch.isfinite(logits).all()), 'status': 'passed'}
    del model, tok, batch, hidden, last, logits
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
