import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import os
for x in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[x]='1'
import argparse,json,time,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from saferefresh import episode,METHODS

def job(t):return episode(**t)
def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=128);p.add_argument('--workers',type=int,default=10);p.add_argument('--decisions',type=int,default=60);p.add_argument('--suite',default='main');p.add_argument('--out',default='results');a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 tasks=[]
 if a.suite=='main':
  tasks=[dict(domain=d,method=m,seed=10000+s,decisions=a.decisions) for d in ['water','thermal'] for m in METHODS for s in range(a.seeds)]
 elif a.suite=='deadline':
  tasks=[dict(domain=d,method=m,seed=20000+s,decisions=a.decisions,deadline=b) for d in ['water','thermal'] for m in ['always','trigger_all','age','greedy','refresh','no_wait'] for b in [1,2,3,4,6] for s in range(a.seeds)]
 elif a.suite=='recheck':
  tasks=[dict(domain=d,method=m,seed=70000+s,decisions=a.decisions,noise_mode=mode,query_loss=loss) for d in ['water','thermal'] for m in ['refresh','trust_forecast','point'] for mode,loss in [('uniform',0),('persistent',0),('persistent',.3)] for s in range(a.seeds)]
 elif a.suite=='persistent':
  tasks=[dict(domain=d,method=m,seed=40000+s,decisions=a.decisions,noise_mode='persistent') for d in ['water','thermal'] for m in METHODS for s in range(a.seeds)]
 elif a.suite=='stress':
  tasks=[dict(domain=d,method=m,seed=30000+s,decisions=a.decisions,noise_scale=w,query_loss=l) for d in ['water','thermal'] for m in ['trigger_all','refresh','point'] for w,l in [(1,0),(1,.1),(1,.3),(2,0),(4,0)] for s in range(a.seeds)]
 start=time.perf_counter();rows=[]
 with ProcessPoolExecutor(a.workers) as ex:
  for i,r in enumerate(ex.map(job,tasks,chunksize=2)):
   rows.append(r)
   if (i+1)%100==0:print(a.suite,i+1,'/',len(tasks),round(time.perf_counter()-start,1),flush=True)
 with (out/(a.suite+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/(a.suite+'_run.json')).write_text(json.dumps({'suite':a.suite,'seeds':a.seeds,'decisions':a.decisions,'workers':a.workers,'seconds':time.perf_counter()-start,'tasks':len(tasks)},indent=2))
 print('complete',len(tasks),round(time.perf_counter()-start,2),flush=True)
if __name__=='__main__':main()
