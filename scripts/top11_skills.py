import pandas as pd
p='/home/ubuntu/assistments_corrected.csv'
df=pd.read_csv(p,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct'])
# Use single-skill rows only for a clean skill-controlled experiment.
df=df[df.skill_id.notna()].copy()
counts=df.groupby('skill_id').size().sort_values(ascending=False)
print('rows_with_skill',len(df),'students',df.user_id.nunique(),'problems',df.problem_id.nunique(),'skills',df.skill_id.nunique())
print(counts.head(11).to_string())
top=counts.head(11).index.tolist()
sub=df[df.skill_id.isin(top)]
print('top11_rows',len(sub),'students',sub.user_id.nunique(),'problems',sub.problem_id.nunique(),'correct_mean',sub.correct.mean())
