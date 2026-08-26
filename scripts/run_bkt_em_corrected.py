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
def em(seqs,iters=25):
 p=np.array([.2,.1,.2,.1],float)
 for _ in range(iters):
  pi,learn,guess,slip=p;I0=I1=T00=T01=E00=E01=E10=E11=0.
  for y in seqs:
   y=np.asarray(y,int);T=len(y);emit=np.zeros((T,2));emit[:,0]=np.where(y==1,guess,1-guess);emit[:,1]=np.where(y==1,1-slip,slip)
   a=np.zeros((T,2));a[0]=np.array([1-pi,pi])*emit[0]
   for t in range(1,T):a[t,0]=(a[t-1,0]+a[t-1,1]*(1-learn))*emit[t,0];a[t,1]=(a[t-1,1]+a[t-1,0]*learn)*emit[t,1]
   z=max(a[-1].sum(),1e-300);a/=z;b=np.ones((T,2))
   for t in range(T-2,-1,-1):b[t,0]=(1-learn)*emit[t+1,0]*b[t+1,0]+learn*emit[t+1,1]*b[t+1,1];b[t,1]=emit[t+1,0]*b[t+1,0]+emit[t+1,1]*b[t+1,1]
   gam=a*b;gam/=gam.sum(1,keepdims=True);I0+=gam[0,0];I1+=gam[0,1]
   for t in range(T-1):
    M=np.array([[1-learn,learn],[0,1]]);xi=a[t,:,None]*M*emit[t+1][None,:]*b[t+1];xi/=max(xi.sum(),1e-300);T00+=xi[0,0];T01+=xi[0,1]
   E00+=gam[:,0][y==0].sum();E01+=gam[:,0][y==1].sum();E10+=gam[:,1][y==0].sum();E11+=gam[:,1][y==1].sum()
  p=np.array([I1/max(I0+I1,1e-9),T01/max(T00+T01,1e-9),E01/max(E00+E01,1e-9),E10/max(E10+E11,1e-9)]);p=np.clip(p,.001,.999)
 return p
def evaluate(skills):
 tr=train[train.skill_id.isin(skills)];te=test[test.skill_id.isin(skills)]
 params={}
 for s,g in tr.groupby('skill_id'):
  seqs=[gg.correct.astype(int).tolist() for _,gg in g.groupby('user_id')];params[s]=em(seqs).tolist()
 Y=[];P=[]
 for _,g in te.groupby('user_id'):
  state={s:params.get(s,[.2,.1,.2,.1])[0] for s in skills}
  for r in g.sort_values('order_id').itertuples():
   pi,learn,guess,slip=params.get(r.skill_id,[.2,.1,.2,.1]);master=state.get(r.skill_id,pi);q=master*(1-slip)+(1-master)*guess;Y.append(int(r.correct));P.append(float(q));post=master*(1-slip)/max(q,1e-9) if r.correct else master*slip/max(1-q,1e-9);state[r.skill_id]=post+(1-post)*learn
 return {'skills':skills,'train_rows':len(tr),'test_rows':len(te),'train_students':int(tr.user_id.nunique()),'test_students':int(te.user_id.nunique()),'params':params,'auc':float(roc_auc_score(Y,P))}
# High-volume means >800 students in the selected subset; report exact count rather than assume five.
skill_counts=df.groupby('skill_id').user_id.nunique().to_dict();high=[s for s in top if skill_counts[s]>800]
out={'dataset':'ASSISTments corrected 2009-2010 Skill Builder','all11':evaluate(top),'high_volume_threshold':800,'high_volume_skills':high,'skill_student_counts':skill_counts,'high_volume':evaluate(high)}
Path('/home/ubuntu/bkt_em_corrected_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
