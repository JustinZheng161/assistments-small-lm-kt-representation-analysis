import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv';SEED=42

df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id']);df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str);top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED);tr_idx,te_idx=next(gss.split(df,groups=df.user_id));train=df.iloc[tr_idx];test=df.iloc[te_idx]

def forward_loglik(y,p):
 pi,learn,guess,slip=p; alpha=np.array([1-pi,pi])*np.array([1-guess if y[0]==0 else guess, slip if y[0]==0 else 1-slip]); ll=np.log(max(alpha.sum(),1e-300)); alpha/=max(alpha.sum(),1e-300)
 for obs in y[1:]:
  alpha=np.array([alpha[0]+alpha[1]*(1-learn),alpha[1]+alpha[0]*learn])*np.array([1-guess if obs==0 else guess, slip if obs==0 else 1-slip]); z=max(alpha.sum(),1e-300);ll+=np.log(z);alpha/=z
 return float(ll)

def em(seqs,init,iters=30,tol=1e-6):
 p=np.clip(np.array(init,float),.001,.999);prev=-np.inf;hist=[]
 for _ in range(iters):
  pi,learn,guess,slip=p;I0=I1=T00=T01=E00=E01=E10=E11=0.;ll=0.
  for y in seqs:
   y=np.asarray(y,int);T=len(y);emit=np.zeros((T,2));emit[:,0]=np.where(y==1,guess,1-guess);emit[:,1]=np.where(y==1,1-slip,slip)
   a=np.zeros((T,2));a[0]=np.array([1-pi,pi])*emit[0];z=max(a[0].sum(),1e-300);ll+=np.log(z);a[0]/=z
   for t in range(1,T):a[t,0]=(a[t-1,0]+a[t-1,1]*(1-learn))*emit[t,0];a[t,1]=(a[t-1,1]+a[t-1,0]*learn)*emit[t,1];z=max(a[t].sum(),1e-300);ll+=np.log(z);a[t]/=z
   b=np.ones((T,2))
   for t in range(T-2,-1,-1):b[t,0]=(1-learn)*emit[t+1,0]*b[t+1,0]+learn*emit[t+1,1]*b[t+1,1];b[t,1]=emit[t+1,0]*b[t+1,0]+emit[t+1,1]*b[t+1,1]
   gam=a*b;gam/=gam.sum(1,keepdims=True);I0+=gam[0,0];I1+=gam[0,1]
   for t in range(T-1):
    M=np.array([[1-learn,learn],[0,1]]);xi=a[t,:,None]*M*emit[t+1][None,:]*b[t+1];xi/=max(xi.sum(),1e-300);T00+=xi[0,0];T01+=xi[0,1]
   E00+=gam[:,0][y==0].sum();E01+=gam[:,0][y==1].sum();E10+=gam[:,1][y==0].sum();E11+=gam[:,1][y==1].sum()
  pnew=np.array([I1/max(I0+I1,1e-9),T01/max(T00+T01,1e-9),E01/max(E00+E01,1e-9),E10/max(E10+E11,1e-9)]);pnew=np.clip(pnew,.001,.999);hist.append({'loglik':ll,'params':pnew.tolist()})
  if np.max(np.abs(pnew-p))<tol: p=pnew;break
  p=pnew
 return p,hist

def predict(params,skills):
 Y=[];P=[]
 for _,g in test.groupby('user_id'):
  state={s:params.get(s,[.2,.1,.2,.1])[0] for s in skills}
  for r in g.sort_values('order_id').itertuples():
   pi,learn,guess,slip=params.get(r.skill_id,[.2,.1,.2,.1]);m=state.get(r.skill_id,pi);q=m*(1-slip)+(1-m)*guess;Y.append(int(r.correct));P.append(float(q));post=m*(1-slip)/max(q,1e-12) if r.correct else m*slip/max(1-q,1e-12);state[r.skill_id]=post+(1-post)*learn
 return float(roc_auc_score(Y,P)),len(Y)

# skill-wise multi-start EM
starts=[ [0.2,0.1,0.2,0.1],[0.5,0.2,0.1,0.1],[0.8,0.5,0.2,0.2] ]
fit_runs=[]
for si,start in enumerate(starts):
 params={};details={}
 for s,g in train.groupby('skill_id'):
  seqs=[gg.correct.astype(int).tolist() for _,gg in g.groupby('user_id')];p,h=em(seqs,start);params[s]=p.tolist();details[s]={'iterations':len(h),'final_loglik':h[-1]['loglik'],'monotone':bool(all(h[i]['loglik']>=h[i-1]['loglik']-1e-5 for i in range(1,len(h)))),'params':p.tolist()}
 auc,n=predict(params,top);fit_runs.append({'start':start,'auc':auc,'n_test':n,'details':details})
# shared-parameter EM using all training sequences
seqs=[g.correct.astype(int).tolist() for _,g in train.groupby('user_id')]
shared=[]
for start in starts:
 p,h=em(seqs,start);params={s:p.tolist() for s in top};auc,n=predict(params,top);shared.append({'start':start,'params':p.tolist(),'auc':auc,'iterations':len(h),'final_loglik':h[-1]['loglik'],'monotone':bool(all(h[i]['loglik']>=h[i-1]['loglik']-1e-5 for i in range(1,len(h))))})
fixed_params=[0.2,0.1,0.2,0.1];fixed={s:fixed_params for s in top};fixed_auc,n=predict(fixed,top)
out={'seed':SEED,'skills':top,'train_rows':len(train),'test_rows':len(test),'train_students':int(train.user_id.nunique()),'test_students':int(test.user_id.nunique()),'skillwise_multistart':fit_runs,'shared_multistart':shared,'fixed_parameter_reference':{'params':fixed_params,'auc':fixed_auc,'n_test':n}}
Path('/home/ubuntu/bkt_convergence_audit_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
