import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv'
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
# Skill-level audit before splitting.
audit=[]
for s,g in df.groupby('skill_id'):
 audit.append({'skill_id':s,'interactions':int(len(g)),'students':int(g.user_id.nunique()),'items':int(g.problem_id.nunique()),'accuracy':float(g.correct.mean())})
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);tr_idx,te_idx=next(gss.split(df,groups=df.user_id));train=df.iloc[tr_idx];test=df.iloc[te_idx]
# Compact EM for a shared-parameter BKT across all skills; skill priors are initialized from train accuracy, transitions shared.
def em(seqs,iters=30):
 p=np.array([.2,.1,.2,.1],float)
 for _ in range(iters):
  pi,learn,guess,slip=p; I0=I1=T00=T01=E00=E01=E10=E11=0.
  for y in seqs:
   y=np.asarray(y,int); T=len(y); emit=np.zeros((T,2));emit[:,0]=np.where(y==1,guess,1-guess);emit[:,1]=np.where(y==1,1-slip,slip)
   a=np.zeros((T,2));a[0]=np.array([1-pi,pi])*emit[0]
   for t in range(1,T): a[t,0]=(a[t-1,0]+a[t-1,1]*(1-learn))*emit[t,0];a[t,1]=(a[t-1,1]+a[t-1,0]*learn)*emit[t,1]
   z=max(a[-1].sum(),1e-300);a/=z;b=np.ones((T,2))
   for t in range(T-2,-1,-1): b[t,0]=(1-learn)*emit[t+1,0]*b[t+1,0]+learn*emit[t+1,1]*b[t+1,1];b[t,1]=emit[t+1,0]*b[t+1,0]+emit[t+1,1]*b[t+1,1]
   gam=a*b;gam/=gam.sum(1,keepdims=True);I0+=gam[0,0];I1+=gam[0,1]
   for t in range(T-1):
    mat=np.array([[1-learn,learn],[0,1]])
    xi=a[t,:,None]*mat*emit[t+1][None,:]*b[t+1];xi/=max(xi.sum(),1e-300);T00+=xi[0,0];T01+=xi[0,1]
   E00+=gam[:,0][y==0].sum();E01+=gam[:,0][y==1].sum();E10+=gam[:,1][y==0].sum();E11+=gam[:,1][y==1].sum()
  p=np.array([I1/max(I0+I1,1e-9),T01/max(T00+T01,1e-9),E01/max(E00+E01,1e-9),E10/max(E10+E11,1e-9)])
  p=np.clip(p,.001,.999)
 return p
seqs=[g.correct.astype(int).tolist() for _,g in train.groupby(['user_id','skill_id'])]
params=em(seqs)
priors=float(train.correct.mean())
def pred(g):
 pi,learn,guess,slip=params; state=priors;Y=[];P=[]
 for r in g.sort_values('order_id').itertuples():
  q=state*(1-slip)+(1-state)*guess;Y.append(int(r.correct));P.append(float(q));post=state*(1-slip)/max(q,1e-9) if r.correct else state*slip/max(1-q,1e-9);state=post+(1-post)*learn
 return Y,P
Y=[];P=[]
for _,g in test.groupby('user_id'):
 y,p=pred(g);Y.extend(y);P.extend(p)
out={'dataset':'ASSISTments corrected 2009-2010 Skill Builder','selected_rows':len(df),'train_rows':len(train),'test_rows':len(test),'train_students':int(train.user_id.nunique()),'test_students':int(test.user_id.nunique()),'skills':top,'skill_audit':audit,'model':'shared-parameter BKT with EM','shared_fitted_parameters':{'p_init':float(params[0]),'p_learn':float(params[1]),'p_guess':float(params[2]),'p_slip':float(params[3])},'student_level_test_auc':float(roc_auc_score(Y,P))}
Path('/home/ubuntu/aggregate_bkt_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
