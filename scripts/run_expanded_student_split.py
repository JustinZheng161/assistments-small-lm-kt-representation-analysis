import json,gc
from pathlib import Path
import pandas as pd,numpy as np,torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer,AutoModel
DATA='/home/ubuntu/assistments_corrected.csv'
MODELS={'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct','TinyLlama-1.1B-Chat-v1.0':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
parts=[]
for s in top: parts.append(df[df.skill_id==s].head(600))
sample=pd.concat(parts).reset_index(drop=True)
texts=[f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {"correct" if r.correct==1 else "wrong"}. Predict the next response.' for r in sample.itertuples()]
y=LabelEncoder().fit_transform(sample.skill_id.astype(str)); groups=sample.user_id.to_numpy()
results=[]
for label,name in MODELS.items():
 print('loading',label,flush=True);tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True);model=AutoModel.from_pretrained(name,trust_remote_code=True,torch_dtype=torch.float32);model.eval()
 outs=[]
 for i in range(0,len(texts),16):
  b=tok(texts[i:i+16],padding=True,truncation=True,max_length=96,return_tensors='pt')
  with torch.no_grad(): h=model(**b).last_hidden_state
  m=b['attention_mask'].unsqueeze(-1);outs.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).numpy())
 X=np.vstack(outs)
 sil=float(silhouette_score(X,y,metric='cosine'))
 gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);tr,te=next(gss.split(X,y,groups))
 clf=LogisticRegression(max_iter=500).fit(X[tr],y[tr]);acc=float(clf.score(X[te],y[te]))
 results.append({'model':label,'n':len(sample),'n_skills':len(top),'silhouette':sil,'student_level_probe_accuracy':acc,'train_students':int(len(np.unique(groups[tr]))),'test_students':int(len(np.unique(groups[te]))),'train_n':len(tr),'test_n':len(te)})
 print(results[-1],flush=True);del model,tok;gc.collect()
out={'top_skills':top,'available_rows':len(df),'sample_rows':len(sample),'sampling':'600 per skill, deterministic first rows after user/order sort','results':results}
Path('/home/ubuntu/expanded_student_split_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
