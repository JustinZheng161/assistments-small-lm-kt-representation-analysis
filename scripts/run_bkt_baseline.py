import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
DATA='/home/ubuntu/assistments_corrected.csv'
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id']).reset_index(drop=True)
# Student-level split over all selected-skill interactions.
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42);tr_idx,te_idx=next(gss.split(df,groups=df.user_id));train=df.iloc[tr_idx];test=df.iloc[te_idx]
# Standard fixed-parameter BKT, fit using only training students. Parameters are explicit and reproducible.
p_init=.2;p_learn=.1;p_guess=.2;p_slip=.1
# Estimate skill priors from train while keeping transition parameters fixed.
priors={s:p_init for s in top}
for s,g in train.groupby('skill_id'):
 priors[s]=float(np.clip(g.correct.mean(),.05,.95))
# Initialize mastery per student and skill, update online only within each student's test sequence.
def predict_group(g):
 state={s:priors.get(s,p_init) for s in top};ys=[];ps=[]
 for r in g.sort_values('order_id').itertuples():
  s=r.skill_id; p=state.get(s,p_init); pred=p*(1-p_slip)+(1-p)*p_guess
  ys.append(int(r.correct));ps.append(float(pred))
  if r.correct==1: post=p*(1-p_slip)/max(pred,1e-9)
  else: post=p*p_slip/max(1-pred,1e-9)
  state[s]=post+(1-post)*p_learn
 return ys,ps
all_y=[];all_p=[]
for _,g in test.groupby('user_id'):
 y,p=predict_group(g);all_y.extend(y);all_p.extend(p)
auc=float(roc_auc_score(all_y,all_p))
out={'dataset':'ASSISTments corrected 2009-2010 Skill Builder','skills':top,'selected_rows':len(df),'train_rows':len(train),'test_rows':len(test),'train_students':int(train.user_id.nunique()),'test_students':int(test.user_id.nunique()),'model':'fixed-parameter BKT','parameters':{'p_init':p_init,'p_learn':p_learn,'p_guess':p_guess,'p_slip':p_slip},'student_level_test_auc':auc,'interpretation':'real student-level held-out baseline; parameters fixed except skill priors estimated on training students'}
Path('/home/ubuntu/bkt_baseline_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
