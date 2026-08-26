import json
import json
from pathlib import Path
from transformers import AutoTokenizer

MODELS = {
    'Qwen2.5-0.5B-Instruct': 'Qwen/Qwen2.5-0.5B-Instruct',
    'TinyLlama-1.1B-Chat-v1.0': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
}
SKILLS = ['311','47','277','280','312','79','279','27','18','50','77']
WORDS = ['correct','wrong','right','incorrect','true','false','unknown','neutral']
OUT = Path('/home/ubuntu/paper_work/revision_data')
OUT.mkdir(parents=True, exist_ok=True)
results = {}
for label, name in MODELS.items():
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    skill_rows=[]
    for s in SKILLS:
        text=f'Skill {s}'
        ids=tok(text, add_special_tokens=False)['input_ids']
        skill_rows.append({'skill_id':s,'text':text,'token_count':len(ids),'token_ids':ids,'tokens':tok.convert_ids_to_tokens(ids)})
    word_rows=[]
    for w in WORDS:
        ids=tok(w, add_special_tokens=False)['input_ids']
        word_rows.append({'word':w,'token_count':len(ids),'token_ids':ids,'tokens':tok.convert_ids_to_tokens(ids)})
    matched_skill_groups={}
    for row in skill_rows:
        matched_skill_groups.setdefault(row['token_count'],[]).append(row['skill_id'])
    matched_word_groups={}
    for row in word_rows:
        matched_word_groups.setdefault(row['token_count'],[]).append(row['word'])
    results[label]={'tokenizer':name,'skills':skill_rows,'words':word_rows,'matched_skill_groups':matched_skill_groups,'matched_word_groups':matched_word_groups}
Path(OUT/'tokenization_audit.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(results,indent=2,ensure_ascii=False))
