import gc, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

DATA='/home/ubuntu/assistments_corrected.csv'
MODELS={'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct','TinyLlama-1.1B-Chat-v1.0':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
SEED=42; rng=np.random.default_rng(SEED)

df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy(); df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist(); df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
parts=[df[df.skill_id==s].head(24) for s in top]; sample=pd.concat(parts).reset_index(drop=True)
small=pd.concat([df[df.skill_id==s].head(12) for s in top]).reset_index(drop=True)
cyclic={top[i]:top[(i+1)%len(top)] for i in range(len(top))}

def prompt(skill,correct,variant='normal'):
    resp='correct' if int(correct)==1 else 'wrong'
    if variant=='case': resp=resp.upper()
    elif variant=='synonym': resp='right' if int(correct)==1 else 'wrong'
    return f'Knowledge tracing record. Skill {skill}. Previous response was {resp}. Predict the next response.'

def bootstrap_mean(a,B=1000):
    a=np.asarray(a,float); idx=rng.integers(0,len(a),size=(B,len(a))); vals=a[idx].mean(1); return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def bootstrap_diff(a,b,B=1000):
    a=np.asarray(a,float);b=np.asarray(b,float); ia=rng.integers(0,len(a),size=(B,len(a))); ib=rng.integers(0,len(b),size=(B,len(b))); vals=a[ia].mean(1)-b[ib].mean(1); return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def bootstrap_silhouette(X,y,B=1000):
    vals=[]
    classes=np.unique(y)
    groups=[np.where(y==c)[0] for c in classes]
    for _ in range(B):
        idx=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        vals.append(silhouette_score(X[idx],y[idx],metric='cosine'))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]
def bootstrap_auc(y,s,B=1000):
    y=np.asarray(y);s=np.asarray(s); vals=[]
    for _ in range(B):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]
        if len(np.unique(yy))<2: continue
        vals.append(roc_auc_score(yy,s[idx]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],int(len(vals))
def token_stats(tok,arr):
    lens=[len(tok(x,add_special_tokens=True)['input_ids']) for x in arr]
    return {'min':int(min(lens)),'median':float(np.median(lens)),'max':int(max(lens)),'mean':float(np.mean(lens))}

def encode(model,tok,arr,batch=8):
    out=[]
    for i in range(0,len(arr),batch):
        b=tok(arr[i:i+batch],padding=True,truncation=True,max_length=96,return_tensors='pt')
        with torch.no_grad(): h=model(**b).last_hidden_state
        mask=b['attention_mask'].unsqueeze(-1).to(h.dtype); pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        out.append(pooled.cpu().numpy())
    return np.vstack(out)

def logits(model,tok,arr,batch=8):
    out=[]; cor=tok(' correct',add_special_tokens=False)['input_ids'][0]; wro=tok(' wrong',add_special_tokens=False)['input_ids'][0]
    for i in range(0,len(arr),batch):
        b=tok(arr[i:i+batch],padding=True,truncation=True,max_length=96,return_tensors='pt')
        with torch.no_grad(): z=model(**b).logits
        idx=b['attention_mask'].sum(1)-1; v=z[torch.arange(len(idx)),idx]; out.append((v[:,cor]-v[:,wro]).cpu().numpy())
    return np.concatenate(out)

results={'seed':SEED,'top_skills':top,'rows':int(len(df)),'sample_rows':int(len(sample)),'small_rows':int(len(small)),'models':{}}
for label,name in MODELS.items():
    print('loading',label,flush=True)
    tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    model=AutoModel.from_pretrained(name,trust_remote_code=True,torch_dtype=torch.float32); model.eval()
    base=[prompt(r.skill_id,r.correct) for r in sample.itertuples()]
    skill=[prompt(cyclic[r.skill_id],r.correct) for r in sample.itertuples()]
    case=[prompt(r.skill_id,r.correct,'case') for r in sample.itertuples()]
    synonym=[prompt(r.skill_id,r.correct,'synonym') for r in sample.itertuples()]
    X=encode(model,tok,base); Xskill=encode(model,tok,skill); Xcase=encode(model,tok,case); Xsyn=encode(model,tok,synonym)
    yskill=LabelEncoder().fit_transform(sample.skill_id.to_numpy())
    target=np.linalg.norm(X-Xskill,axis=1); case_disp=np.linalg.norm(X-Xcase,axis=1); syn_disp=np.linalg.norm(X-Xsyn,axis=1)
    random_disps=[]
    for j in range(5):
        rand_skill=[rng.choice([s for s in top if s!=r.skill_id]) for r in sample.itertuples()]
        rand_text=[prompt(s,r.correct) for s,r in zip(rand_skill,sample.itertuples())]
        random_disps.append(np.linalg.norm(X-encode(model,tok,rand_text),axis=1))
    random_all=np.vstack(random_disps)
    clf=LogisticRegression(max_iter=500).fit(X,yskill)
    # AUC is an output-level diagnostic; it remains on the original 132-record pilot and is separate from pooled hidden states.
    auc_text=[prompt(r.skill_id,r.correct) for r in small.itertuples()]
    causal_model=AutoModelForCausalLM.from_pretrained(name,trust_remote_code=True,torch_dtype=torch.float32); causal_model.eval()
    yauc=small.correct.to_numpy().astype(int); scores=logits(causal_model,tok,auc_text)
    results['models'][label]={'n':len(sample),'pooling':'masked mean over non-padding tokens','silhouette':float(silhouette_score(X,yskill,metric='cosine')),'silhouette_bootstrap_ci':bootstrap_silhouette(X,yskill),'probe_in_sample':float(clf.score(X,yskill)),'skill_cyclic_mean':float(target.mean()),'skill_cyclic_ci':bootstrap_mean(target),'correctness_case_mean':float(case_disp.mean()),'correctness_case_ci':bootstrap_mean(case_disp),'synonym_mean':float(syn_disp.mean()),'synonym_ci':bootstrap_mean(syn_disp),'random_skill_mean':float(random_all.mean()),'random_skill_ci':bootstrap_mean(random_all.ravel()),'cyclic_vs_case_diff':float(target.mean()-case_disp.mean()),'cyclic_vs_case_diff_ci':bootstrap_diff(target,case_disp),'auc_n':int(len(small)),'auc_exploratory':float(roc_auc_score(yauc,scores)),'auc_ci':bootstrap_auc(yauc,scores)[0],'auc_bootstrap_reps':bootstrap_auc(yauc,scores)[1],'token_lengths':{'base':token_stats(tok,base),'cyclic_skill':token_stats(tok,skill),'case_control':token_stats(tok,case),'synonym_control':token_stats(tok,synonym)}}
    del causal_model,model,tok;gc.collect()
Path('/home/ubuntu/robust_representation_audit_results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
