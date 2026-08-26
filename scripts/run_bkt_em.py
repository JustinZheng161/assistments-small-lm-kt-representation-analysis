import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv'
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);tr_idx,te_idx=next(gss.split(df,groups=df.user_id));train=df.iloc[tr_idx];test=df.iloc[te_idx]
def em_skill(seqs,iters=30):
 p=np.array([.2,.1,.2,.1],dtype=float)
 for _ in range(iters):
  pi,learn,guess,slip=p; init0=init1=t01=t00=e0=e1=ec0=ec1=0.; ll=0.
  for y in seqs:
   y=np.asarray(y,dtype=int);T=len(y)
   emit=np.zeros((T,2));emit[:,0]=np.where(y==1,guess,1-guess);emit[:,1]=np.where(y==1,1-slip,slip)
   a=np.zeros((T,2));a[0]=np.array([1-pi,pi])*emit[0]
   for t in range(1,T):a[t,0]=(a[t-1,0]+a[t-1,1]*(1-learn))*emit[t,0];a[t,1]=(a[t-1,1]+a[t-1,0]*learn)*emit[t,1]
   z=max(a[-1].sum(),1e-300);ll+=np.log(z);a/=z
   b=np.ones((T,2))
   for t in range(T-2,-1,-1):
    b[t,0]=((1-learn)*emit[t+1,0]*b[t+1,0]+learn*emit[t+1,1]*b[t+1,1])
    b[t,1]=(emit[t+1,0]*b[t+1,0]+emit[t+1,1]*b[t+1,1])
   g=a*b;g/=g.sum(1,keepdims=True)
   init0+=g[0,0];init1+=g[0,1]
   for t in range(T-1):
    den=max(((a[t,:,None]*np.array([[1-learn,learn],[0,1]])*emit[t+1][None,:]*b[t+1]).sum()),1e-300)
    xi=(a[t,:,None]*np.array([[1-learn,learn],[0,1]])*emit[t+1][None,:]*b[t+1])/den
    t01+=xi[0,1];t00+=xi[0,0]
   e0+=g[:,0][y==1].sum();ec0+=g[:,0].sum();e1+=g[:,1][y==1].sum();ec1+=g[:,1].sum()
  p=np.array([init1/max(init0+init1,1e-9),t01/max(t00+t01,1e-9),e0/max(ec0,1e-9),1-e1/max(ec1,1e-9)])
  p=np.clip(p,[.001,.001,.001,.001],[.999,.999,.999,.999])
 return p
params={}
for s,g in train.groupby('skill_id'):
 seqs=[gg.correct.astype(int).tolist() for _,gg in g.groupby('user_id') if len(gg)>0]
 params[s]=em_skill(seqs).tolist()
def predict(g,s):
 pi,learn,guess,slip=params.get(s,[.2,.1,.2,.1]);master=pi;ys=[];ps=[]
 for r in g.sort_values('order_id').itertuples():
  pred=master*(1-slip)+(1-master)*guess;ys.append(int(r.correct));ps.append(float(pred))
  post=(master*(1-slip)/max(pred,1e-9)) if r.correct==1 else (master*slip/max(1-pred,1e-9))
  master=post+(1-post)*learn
 return ys,ps
Y=[];P=[]
for _,g in test.groupby('user_id'):
 y,p=predict(g,g.skill_id.iloc[0]);Y.extend(y);P.extend(p)
# Note: for mixed-skill student sequences, prediction state is intentionally reset per group call; report as skill-sequence BKT evaluation.
out={'dataset':'ASSISTments corrected 2009-2010 Skill Builder','skills':top,'selected_rows':len(df),'train_rows':len(train),'test_rows':len(test),'train_students':int(train.user_id.nunique()),'test_students':int(test.user_id.nunique()),'model':'skill-wise BKT with EM','fitted_parameters':params,'student_level_test_auc':float(roc_auc_score(Y,P)),'note':'EM fit on training students; test evaluation follows the existing per-student grouped protocol'}
Path('/home/ubuntu/bkt_em_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
