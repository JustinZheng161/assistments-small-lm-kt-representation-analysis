import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv';SEED=42

df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id']);df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str);top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED);tri,tei=next(gss.split(df,groups=df.user_id));train=df.iloc[tri];test=df.iloc[tei]
def em(seqs,init,iters=100,tol=1e-7):
 p=np.clip(np.array(init,float),.001,.999);hist=[]
 for it in range(iters):
  pi,learn,guess,slip=p;stats=np.zeros(8);total_ll=0.
  for y in seqs:
   y=np.asarray(y,int);T=len(y);e=np.zeros((T,2));e[:,0]=np.where(y==1,guess,1-guess);e[:,1]=np.where(y==1,1-slip,slip)
   a=np.zeros((T,2));c=np.zeros(T);a[0]=np.array([1-pi,pi])*e[0];c[0]=max(a[0].sum(),1e-300);a[0]/=c[0];total_ll+=np.log(c[0])
   for t in range(1,T):
    a[t,0]=(a[t-1,0]+a[t-1,1]*(1-learn))*e[t,0];a[t,1]=(a[t-1,0]*learn+a[t-1,1])*e[t,1];c[t]=max(a[t].sum(),1e-300);a[t]/=c[t];total_ll+=np.log(c[t])
   b=np.ones((T,2))
   for t in range(T-2,-1,-1):
    b[t,0]=((1-learn)*e[t+1,0]*b[t+1,0]+learn*e[t+1,1]*b[t+1,1])/c[t+1]
    b[t,1]=(e[t+1,0]*b[t+1,0]+e[t+1,1]*b[t+1,1])/c[t+1]
   g=a*b;g/=g.sum(1,keepdims=True)
   stats[0]+=g[0,0];stats[1]+=g[0,1]
   stats[4]+=g[:,0][y==0].sum();stats[5]+=g[:,0][y==1].sum();stats[6]+=g[:,1][y==0].sum();stats[7]+=g[:,1][y==1].sum()
   for t in range(T-1):
    x=np.array([[a[t,0]*(1-learn)*e[t+1,0]*b[t+1,0],a[t,0]*learn*e[t+1,1]*b[t+1,1]],[a[t,1]*0*e[t+1,0]*b[t+1,0],a[t,1]*e[t+1,1]*b[t+1,1]]]);x/=max(x.sum(),1e-300);stats[2]+=x[0,0];stats[3]+=x[0,1]
  pnew=np.array([stats[1]/max(stats[0]+stats[1],1e-300),stats[3]/max(stats[2]+stats[3],1e-300),stats[5]/max(stats[4]+stats[5],1e-300),stats[6]/max(stats[6]+stats[7],1e-300)]);pnew=np.clip(pnew,.001,.999);hist.append({'iteration':it+1,'loglik':float(total_ll),'params':pnew.tolist()})
  if np.max(np.abs(pnew-p))<tol: p=pnew;break
  p=pnew
 return p,hist
def predict(params,skills):
 yout=[];pout=[]
 for _,g in test.groupby('user_id'):
  state={s:params[s][0] for s in skills}
  for r in g.sort_values('order_id').itertuples():
   pi,learn,guess,slip=params[r.skill_id];m=state[r.skill_id];q=m*(1-slip)+(1-m)*guess;yout.append(int(r.correct));pout.append(q);post=m*(1-slip)/max(q,1e-12) if r.correct else m*slip/max(1-q,1e-12);state[r.skill_id]=post+(1-post)*learn
 return float(roc_auc_score(yout,pout))
starts=[[.2,.1,.2,.1],[.5,.2,.1,.1],[.8,.5,.2,.2]]
allruns=[]
for start in starts:
 params={};details={}
 for s,g in train.groupby('skill_id'):
  seqs=[gg.correct.astype(int).tolist() for _,gg in g.groupby('user_id')];p,h=em(seqs,start);params[s]=p.tolist();details[s]={'iterations':len(h),'loglik_history':h,'monotone':all(h[i]['loglik']>=h[i-1]['loglik']-1e-7 for i in range(1,len(h)))}
 allruns.append({'start':start,'auc':predict(params,top),'params':params,'details':details})
# choose highest total training log-likelihood across skill fits
best=max(allruns,key=lambda r:sum(r['details'][s]['loglik_history'][-1]['loglik'] for s in top))
out={'seed':SEED,'skills':top,'train_rows':len(train),'test_rows':len(test),'runs':allruns,'selected_start':best['start'],'selected_auc':best['auc']}
Path('/home/ubuntu/bkt_em_stable_results.json').write_text(json.dumps(out,indent=2));print(json.dumps({'selected_start':best['start'],'selected_auc':best['auc'],'runs':[(r['start'],r['auc'],all(r['details'][s]['monotone'] for s in top)) for r in allruns]},indent=2))
