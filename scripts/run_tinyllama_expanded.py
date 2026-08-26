import gc,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer,AutoModel

torch.set_num_threads(8)
DATA='/home/ubuntu/assistments_corrected.csv';SEED=42
name='TinyLlama/TinyLlama-1.1B-Chat-v1.0';label='TinyLlama-1.1B-Chat-v1.0'
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id']);df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str);top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
sample=pd.concat([df[df.skill_id==s].head(600) for s in top]).reset_index(drop=True)
texts=[f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {"correct" if r.correct==1 else "wrong"}. Predict the next response.' for r in sample.itertuples()]
y=LabelEncoder().fit_transform(sample.skill_id.astype(str));groups=sample.user_id.to_numpy();gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED);tr,te=next(gss.split(texts,y,groups))
print('loading',label,flush=True);tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True);model=AutoModel.from_pretrained(name,trust_remote_code=True,torch_dtype=torch.float32);model.eval();outs=[]
for i in range(0,len(texts),64):
 b=tok(texts[i:i+64],padding=True,truncation=True,max_length=96,return_tensors='pt')
 with torch.no_grad(): h=model(**b).last_hidden_state
 m=b['attention_mask'].unsqueeze(-1).to(h.dtype);outs.append(((h*m).sum(1)/m.sum(1).clamp_min(1)).numpy());
 if i%640==0: print('encoded',i,flush=True)
X=np.vstack(outs);clf=LogisticRegression(max_iter=500).fit(X[tr],y[tr]);out={'model':label,'seed':SEED,'n':len(sample),'n_skills':len(top),'silhouette':float(silhouette_score(X,y,metric='cosine')),'student_level_probe_accuracy':float(clf.score(X[te],y[te])),'student_level_probe_train_accuracy':float(clf.score(X[tr],y[tr])),'train_students':int(np.unique(groups[tr]).size),'test_students':int(np.unique(groups[te]).size),'train_n':int(len(tr)),'test_n':int(len(te)),'pooling':'masked mean over non-padding tokens','sampling':'600 records per skill, deterministic first records after user/order sorting'}
Path('/home/ubuntu/tinyllama_expanded_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));del model,tok;gc.collect()
