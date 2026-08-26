import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
DATA='/home/ubuntu/assistments_corrected.csv';SEED=42

df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id']);df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str);top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED);tri,tei=next(gss.split(df,groups=df.user_id));train=df.iloc[tri];test=df.iloc[tei]
def nll(p,seqs):
 pi,learn,guess,slip=p;ll=0.0
 for y in seqs:
  a=np.array([1-pi,pi])*np.array([guess if y[0] else 1-guess,1-slip if y[0] else slip]);z=max(a.sum(),1e-300);ll+=np.log(z);a/=z
  for obs in y[1:]:
   a=np.array([a[0]+a[1]*(1-learn),a[1]+a[0]*learn])*np.array([guess if obs else 1-guess,1-slip if obs else slip]);z=max(a.sum(),1e-300);ll+=np.log(z);a/=z
 return -ll
def fit(seqs):
 starts=[[.2,.1,.2,.1],[.5,.2,.1,.1],[.8,.5,.2,.2],[.1,.7,.3,.1]];runs=[]
 for s in starts:
  r=minimize(lambda p:nll(p,seqs),s,method='L-BFGS-B',bounds=[(.001,.999)]*4,options={'maxiter':300,'ftol':1e-10,'gtol':1e-7});runs.append({'start':s,'params':r.x.tolist(),'nll':float(r.fun),'success':bool(r.success),'message':r.message,'iterations':int(r.nit)})
 return sorted(runs,key=lambda x:x['nll'])[0],runs
def predict(params,skills):
 Y=[];P=[]
 for _,g in test.groupby('user_id'):
  state={s:params[s][0] for s in skills}
  for r in g.sort_values('order_id').itertuples():
   pi,learn,guess,slip=params[r.skill_id];m=state[r.skill_id];q=m*(1-slip)+(1-m)*guess;Y.append(int(r.correct));P.append(float(q));post=m*(1-slip)/max(q,1e-12) if r.correct else m*slip/max(1-q,1e-12);state[r.skill_id]=post+(1-post)*learn
 return float(roc_auc_score(Y,P)),len(Y)
fit_best={};all_runs={}
for s,g in train.groupby('skill_id'):
 seqs=[gg.correct.astype(int).tolist() for _,gg in g.groupby('user_id')];best,runs=fit(seqs);fit_best[s]=best['params'];all_runs[s]={'best':best,'all':runs}
auc,n=predict(fit_best,top)
shared_seqs=[g.correct.astype(int).tolist() for _,g in train.groupby('user_id')];shared_best,shared_runs=fit(shared_seqs);shared_auc,_=predict({s:shared_best['params'] for s in top},top)
fixed=[.2,.1,.2,.1];fixed_auc,_=predict({s:fixed for s in top},top)
out={'seed':SEED,'skills':top,'train_rows':len(train),'test_rows':len(test),'skillwise_mle':{'best_auc':auc,'best_params':fit_best,'runs':all_runs},'shared_mle':{'best_auc':shared_auc,'best':shared_best,'runs':shared_runs},'fixed_reference':{'params':fixed,'auc':fixed_auc}}
Path('/home/ubuntu/bkt_mle_crosscheck_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
