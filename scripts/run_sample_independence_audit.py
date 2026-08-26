from pathlib import Path
import pandas as pd, json
DATA='/home/ubuntu/assistments_corrected.csv'
cols=['user_id','problem_id','skill_id','correct','order_id']
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=cols)
df=df[df.skill_id.notna()].copy();df['skill_id']=df.skill_id.astype(str)
top=df.skill_id.value_counts().head(11).index.tolist();df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
# Existing frozen protocol: both scripts use the first 24 records within each skill after sorting.
lora=pd.concat([df[df.skill_id==s].head(24) for s in top]).reset_index(drop=True)
repr_sample=pd.concat([df[df.skill_id==s].head(24) for s in top]).reset_index(drop=True)
keys=['user_id','problem_id','skill_id','correct','order_id']
def keyset(x): return set(map(tuple,x[keys].astype(str).itertuples(index=False,name=None)))
A=keyset(lora);B=keyset(repr_sample);overlap=A&B
out={'data_rows_after_skill_filter':int(len(df)),'top_skills':top,'lora_n':len(lora),'representation_n':len(repr_sample),'overlap_n':len(overlap),'overlap_fraction_of_representation':len(overlap)/len(B),'sampling_protocol':'deterministic first 24 records per skill after user_id/order_id sorting','representation_checkpoint':'untouched base AutoModel checkpoint; no LoRA adapter loaded','interpretation':'The record sets are identical under the frozen scripts, but the representation analysis uses untouched base checkpoints. Therefore the overlap is not a train/test evaluation of an adapted model; LoRA is a pipeline-feasibility pilot only.'}
Path('/home/ubuntu/sample_independence_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
