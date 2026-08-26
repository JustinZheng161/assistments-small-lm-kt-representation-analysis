import os, json, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModel

DATA='/home/ubuntu/assistments_corrected.csv'
MODELS={'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct','TinyLlama-1.1B-Chat-v1.0':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy(); df['skill_id']=df.skill_id.astype(str)
counts=df.skill_id.value_counts(); top=counts.head(11).index.tolist(); df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
# one balanced sample per skill, capped for CPU reproducibility
parts=[]
for s in top:
    z=df[df.skill_id==s].head(24)
    parts.append(z)
sample=pd.concat(parts).reset_index(drop=True)
texts=[f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {"correct" if r.correct==1 else "wrong"}. Predict the next response.' for r in sample.itertuples()]
labels=sample.skill_id.to_numpy()
# counterfactual: replace skill with next skill in top list
next_skill={top[i]:top[(i+1)%len(top)] for i in range(len(top))}
cftexts=[f'Knowledge tracing record. Skill {next_skill[r.skill_id]}. Previous response was {"correct" if r.correct==1 else "wrong"}. Predict the next response.' for r in sample.itertuples()]
results=[]
for label,name in MODELS.items():
    print('loading',label,flush=True)
    tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    model=AutoModel.from_pretrained(name,trust_remote_code=True,torch_dtype=torch.float32)
    model.eval()
    def encode(arr):
        outs=[]
        for i in range(0,len(arr),8):
            b=tok(arr[i:i+8],padding=True,truncation=True,max_length=96,return_tensors='pt')
            with torch.no_grad(): h=model(**b).last_hidden_state
            mask=b['attention_mask'].unsqueeze(-1); pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            outs.append(pooled.numpy())
        return np.vstack(outs)
    X=encode(texts); Xcf=encode(cftexts)
    le=LabelEncoder(); y=le.fit_transform(labels)
    sil=float(silhouette_score(X,y,metric='cosine'))
    clf=LogisticRegression(max_iter=400).fit(X,y)
    probe=float(clf.score(X,y))
    disp=float(np.linalg.norm(X-Xcf,axis=1).mean())
    results.append({'model':label,'n':len(sample),'n_skills':len(top),'silhouette':sil,'linear_probe_accuracy':probe,'counterfactual_displacement':disp})
    del model,tok; import gc; gc.collect()
Path('/home/ubuntu/representation_pilot_results.json').write_text(json.dumps({'top_skills':top,'rows':len(df),'sample_rows':len(sample),'results':results},indent=2))
print(json.dumps({'top_skills':top,'rows':len(df),'sample_rows':len(sample),'results':results},indent=2))
