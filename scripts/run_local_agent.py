import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
"""Local GPU inference and exact replay of recorded supervisory proposals."""
import argparse,json,time,csv,platform,hashlib
from pathlib import Path
import numpy as np
from saferefresh import episode

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='models/Qwen3-1.7B-4bit');ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--decisions',type=int,default=36);ap.add_argument('--out',default='results/reproduced');ap.add_argument('--replay',action='store_true');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 logs=[];rows=[];tapes=[]
 if not a.replay:
  import mlx.core as mx
  from mlx_lm import load,stream_generate
  from mlx_lm.sample_utils import make_sampler
  model,tok=load(a.model);mx.random.seed(2731)
  system='You supervise an eight-channel industrial IoT process. Move each channel toward its requested target using actuator pulses. At each turn choose one or two channels to adjust. A separate safety guard checks physical constraints and may refresh sensors or use a backup. Use only the cached observations provided. Return only JSON: {"channels":[0,1],"directions":[1,-1],"duration":6}. Channels are distinct integers 0 to 7. Directions must be +1 to increase or -1 to decrease. Duration must be 4,6,8,or 10 ticks. Choose useful progress; do not provide explanations.'
  def propose(domain,seed,t,m,r,target,amplitude,h):
   msg='Goal: bring the channels toward these targets: '+json.dumps(np.round(target,3).tolist())+'. Current cached estimates: '+json.dumps(np.round(m,3).tolist())+'. Sensor uncertainty half widths: '+json.dumps(np.round(r,3).tolist())+'. Choose the next actuator pulse.'
   prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':msg}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   start=time.perf_counter();parts=[];last=None
   for z in stream_generate(model,tok,prompt,max_tokens=96,sampler=make_sampler(temp=0)):
    parts.append(z.text);last=z
   elapsed=time.perf_counter()-start;raw=''.join(parts);valid=False;u=np.zeros(len(m));duration=h
   try:
    content=raw[raw.index('{'):raw.rindex('}')+1];v=json.loads(content)
    ids=v['channels'];directions=v['directions'];duration=v['duration']
    valid=(isinstance(ids,list) and 1<=len(ids)<=2 and len(ids)==len(set(ids)) and len(ids)==len(directions) and all(type(i)==int and 0<=i<8 for i in ids) and all(s in [-1,1] for s in directions) and duration in [4,6,8,10])
    if valid:
     for i,s in zip(ids,directions):u[i]=float(s)*amplitude
   except Exception:valid=False
   if not valid:u[:]=0;duration=h
   logs.append({'domain':domain,'seed':seed,'decision':t,'method':current_method,'prompt':msg,'response':raw,'valid':bool(valid),'seconds':elapsed,'tokens':getattr(last,'generation_tokens',None),'peak_memory_gb':getattr(last,'peak_memory',None),'u':u.tolist(),'h':int(duration)})
   return u,int(duration)
  for domain in ['water','thermal']:
   for seed in range(51000,51000+a.seeds):
    for method in ['trigger_all','greedy','refresh']:
     current_method=method
     result=episode(domain,method,seed,decisions=a.decisions,proposal_fn=propose,return_trace=True)
     tapes.append({'domain':domain,'method':method,'seed':seed,'decisions':a.decisions,'proposals':[{'u':z['u'],'h':z['h']} for z in result.pop('proposals')],'trace':result.pop('trace')})
     result.pop('timeline',None)
     rows.append(result)
     (out/'local_agent_inference.json').write_text(json.dumps(logs,indent=2));(out/'local_agent_tapes.json').write_text(json.dumps(tapes,indent=2))
     print(domain,seed,method,'calls',len(logs),'accepted',result['accepted'],'cost',result['cost'],flush=True)
  metadata={'model':a.model.split('/')[-1],'mlx_version':mx.__version__,'platform':platform.platform(),'seeds_per_regime':a.seeds,'decisions':a.decisions,'temperature':0,'max_tokens':96,'thinking':False,'system_prompt':system,'parameters':'1.7 billion, 4-bit weights'}
  (out/'local_agent_run.json').write_text(json.dumps(metadata,indent=2))
 else:
  for tape in json.loads((out/'local_agent_tapes.json').read_text()):
   rows.append(episode(tape['domain'],tape['method'],tape['seed'],decisions=tape['decisions'],proposal_tape=tape['proposals']))
 name='local_agent_replay.csv' if a.replay else 'local_agent.csv'
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
