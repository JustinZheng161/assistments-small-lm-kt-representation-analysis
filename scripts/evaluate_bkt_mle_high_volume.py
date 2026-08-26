import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv';SEED=42
mle=json.loads(Path('/home/ubuntu/bkt_mle_crosscheck_results.json').read_text())
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id']);df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str);top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED);tri,tei=next(gss.split(df,groups=df.user_id));test=df.iloc[tei]
counts=df.groupby('skill_id').user_id.nunique();high=[s for s in top if counts[s]>800]
def auc_for(skills):
 Y=[];P=[]
 for _,g in test[test.skill_id.isin(skills)].groupby('user_id'):
  state={s:mle['skillwise_mle']['best_params'][s][0] for s in skills}
  for r in g.sort_values('order_id').itertuples():
   pi,learn,guess,slip=mle['skillwise_mle']['best_params'][r.skill_id];m=state[r.skill_id];q=m*(1-slip)+(1-m)*guess;Y.append(int(r.correct));P.append(float(q));post=m*(1-slip)/max(q,1e-12) if r.correct else m*slip/max(1-q,1e-12);state[r.skill_id]=post+(1-post)*learn
 return float(roc_auc_score(Y,P)),len(Y),int(test[test.skill_id.isin(skills)].user_id.nunique())
out={'threshold_students':800,'high_volume_skills':high,'student_counts':counts.to_dict(),'high_volume_mle_skillwise_auc':auc_for(high),'all11_mle_skillwise_auc':auc_for(top)}
Path('/home/ubuntu/bkt_mle_high_volume_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
