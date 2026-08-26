import json, gc, math
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

DATA='/home/ubuntu/assistments_corrected.csv'
MODELS={'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct','TinyLlama-1.1B-Chat-v1.0':'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
CFG={'r':8,'lora_alpha':16,'lora_dropout':0.05,'learning_rate':2e-4,'epochs':1,'batch_size':2,'max_length':64}
df=pd.read_csv(DATA,encoding='latin1',low_memory=False,usecols=['user_id','problem_id','skill_id','correct','order_id'])
df=df[df.skill_id.notna()].copy(); df['skill_id']=df.skill_id.astype(str); top=df.skill_id.value_counts().head(11).index.tolist(); df=df[df.skill_id.isin(top)].sort_values(['user_id','order_id'])
parts=[df[df.skill_id==s].head(24) for s in top]; data=pd.concat(parts).reset_index(drop=True)
texts=[f'Knowledge tracing record. Skill {r.skill_id}. Previous response was {"correct" if r.correct==1 else "wrong"}. The next response is {"correct" if r.correct==1 else "wrong"}.' for r in data.itertuples()]
class Txt(Dataset):
    def __init__(self,tok): self.items=[]; self.tok=tok
    def __len__(self): return len(texts)
    def __getitem__(self,i):
        z=self.tok(texts[i],padding='max_length',truncation=True,max_length=CFG['max_length'],return_tensors='pt')
        return {'input_ids':z['input_ids'][0],'attention_mask':z['attention_mask'][0],'labels':z['input_ids'][0].clone()}
out={}
for label,name in MODELS.items():
    print('loading',label,flush=True)
    tok=AutoTokenizer.from_pretrained(name,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(name,trust_remote_code=True,dtype=torch.float32)
    config=LoraConfig(r=CFG['r'],lora_alpha=CFG['lora_alpha'],lora_dropout=CFG['lora_dropout'],target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM)
    model=get_peft_model(model,config); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'])
    loader=DataLoader(Txt(tok),batch_size=CFG['batch_size'],shuffle=False)
    losses=[]
    for ep in range(CFG['epochs']):
        for step,b in enumerate(loader):
            opt.zero_grad(); o=model(**{k:v for k,v in b.items()}); loss=o.loss; loss.backward(); opt.step(); losses.append(float(loss.detach()));
            if step%20==0: print(label,ep,step,losses[-1],flush=True)
    out[label]={'config':CFG,'n_examples':len(texts),'loss_first':losses[0],'loss_last':losses[-1],'loss_min':min(losses),'loss_steps':losses}
    model.save_pretrained('/home/ubuntu/'+label.replace('.','_')+'_lora_adapter'); del model,tok; gc.collect()
Path('/home/ubuntu/lora_pilot_results.json').write_text(json.dumps({'top_skills':top,'results':out},indent=2))
print(json.dumps({'top_skills':top,'results':{k:{a:v for a,v in z.items() if a!='loss_steps'} for k,z in out.items()}},indent=2))
