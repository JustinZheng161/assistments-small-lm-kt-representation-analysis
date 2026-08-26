import gc,json
from pathlib import Path
import numpy as np,pandas as pd,torch,json,gc
from transformers import AutoTokenizer,AutoModelForCausalLM
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv'
MODELS={'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct','TinyLlama-1.1B-Chat-v1.0':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id']);df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str);top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']);small=pd.concat([df[df.skill_id==s].head(12) for s in top]).reset_index(drop=True)
def make(r,mode):
 if mode=='explicit': prev='correct' if r.correct==1 else 'wrong'
 elif mode=='masked': prev='unknown'
 elif mode=='neutral': return f'Knowledge tracing record. Skill {r.skill_id}. Predict the next response.'
 return f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {prev}. Predict the next response.'
def scores(model,tok,texts):
 cor=tok(' correct',add_special_tokens=False)['input_ids'][0];wro=tok(' wrong',add_special_tokens=False)['input_ids'][0];out=[]
 for i in range(0,len(texts),16):
  b=tok(texts[i:i+16],padding=True,truncation=True,max_length=96,return_tensors='pt')
  with torch.no_grad(): z=model(**b).logits
  idx=b['attention_mask'].sum(1)-1;v=z[torch.arange(len(idx)),idx];out.append((v[:,cor]-v[:,wro]).cpu().numpy())
 return np.concatenate(out)
out={'n':len(small),'label_prevalence':float(small.correct.mean()),'bootstrap_iterations':1000,'bootstrap_unit':'records resampled with replacement within the fixed 132-record sample','modes':{}}
y=small.correct.to_numpy().astype(int)
for label,name in MODELS.items():
 tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True);model=AutoModelForCausalLM.from_pretrained(name,trust_remote_code=True,torch_dtype=torch.float32);model.eval();out['modes'][label]={}
 for mode in ['explicit','masked','neutral']:
  s=scores(model,tok,[make(r,mode) for r in small.itertuples()]); rng=np.random.default_rng(20240822); vals=[]
  for _ in range(1000):
   idx=rng.integers(0,len(y),len(y)); yy=y[idx]
   if len(np.unique(yy))>1: vals.append(roc_auc_score(yy,s[idx]))
  out['modes'][label][mode]={'auc':float(roc_auc_score(y,s)),'auc_bootstrap_ci':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'score_mean':float(s.mean()),'score_std':float(s.std()),'scores':s.tolist()}
 del model,tok;gc.collect()
Path('/home/ubuntu/auc_leakage_audit_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
