import pandas as pd
from pathlib import Path
p=Path('/home/ubuntu/assistments_corrected.csv')
df=pd.read_csv(p, encoding='latin1')
print('rows',len(df))
print('columns',list(df.columns))
for c in ['studentId','problemId','skill','correct','skill_name']:
    if c in df: print(c,'nunique',df[c].nunique(),'missing',df[c].isna().sum())
if 'skill' in df:
    skills=df['skill'].dropna().astype(str).str.split('_').explode().str.strip()
    print('skill_tokens',skills.nunique(),'top',skills.value_counts().head(20).to_dict())
if 'correct' in df: print('correct_mean',df['correct'].mean())
print(df.head(3).to_string())
