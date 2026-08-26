import json, gc
from pathlib import Path
import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

DATA='/home/ubuntu/assistments_corrected.csv'
MODELS={'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct','TinyLlama-1.1B-Chat-v1.0':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy(); df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist(); df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
parts=[df[df.skill_id==s].head(12) for s in top]; sample=pd.concat(parts).reset_index(drop=True)
next_skill={top[i]:top[(i+1)%len(top)] for i in range(len(top))}
texts=[f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {"correct" if r.correct==1 else "wrong"}. The next response is' for r in sample.itertuples()]
skillcf=[f'Knowledge tracing record. Skill {next_skill[r.skill_id]}. Previous response was {"correct" if r.correct==1 else "wrong"}. The next response is' for r in sample.itertuples()]
controlcf=[f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {"wrong" if r.correct==1 else "correct"}. The next response is' for r in sample.itertuples()]
labels=sample.correct.to_numpy().astype(int)
results=[]
for label,name in MODELS.items():
    tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(name,trust_remote_code=True,dtype=torch.float32)
    model.eval()
    def batch_logits(arr):
        outs=[]
        for i in range(0,len(arr),8):
            b=tok(arr[i:i+8],padding=True,truncation=True,max_length=96,return_tensors='pt')
            with torch.no_grad(): out=model(**b).logits
            idx=b['attention_mask'].sum(1)-1
            vals=out[torch.arange(len(idx)),idx]
            # first token of the strings, with fallbacks to tokenized first non-special token
            cor=tok(' correct',add_special_tokens=False)['input_ids'][0]
            wro=tok(' wrong',add_special_tokens=False)['input_ids'][0]
            score=(vals[:,cor]-vals[:,wro]).cpu().numpy()
            outs.append(score)
        return np.concatenate(outs)
    def hidden(arr):
        outs=[]
        for i in range(0,len(arr),8):
            b=tok(arr[i:i+8],padding=True,truncation=True,max_length=96,return_tensors='pt')
            with torch.no_grad(): h=model.model(**b).last_hidden_state
            idx=b['attention_mask'].sum(1)-1
            outs.append(h[torch.arange(len(idx)),idx].cpu().numpy())
        return np.vstack(outs)
    scores=batch_logits(texts); X=hidden(texts); Xs=hidden(skillcf); Xc=hidden(controlcf)
    auc=float(roc_auc_score(labels,scores))
    target=np.linalg.norm(X-Xs,axis=1); control=np.linalg.norm(X-Xc,axis=1)
    did=float(target.mean()-control.mean())
    results.append({'model':label,'n':len(sample),'auc_exploratory':auc,'target_skill_displacement_mean':float(target.mean()),'control_displacement_mean':float(control.mean()),'did_descriptive_displacement_contrast':did})
    del model,tok; gc.collect()
out={'top_skills':top,'rows':len(df),'sample_rows':len(sample),'results':results,'auc_protocol':'last-position logit difference for tokens correct vs wrong; exploratory only'}
Path('/home/ubuntu/did_auc_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
