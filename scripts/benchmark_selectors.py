import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import time,json,csv,argparse
import numpy as np
from saferefresh import Plant,choose_exact,choose_milp,choose_prefix

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/reproduced');ap.add_argument('--cases',type=int,default=100);a=ap.parse_args();from pathlib import Path
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True);rng=np.random.default_rng(912715);rows=[]
 for n in [4,8,12,16,24,32]:
  p=Plant('thermal',n)
  for case in range(a.cases):
   m=rng.uniform(-.7,.7,n);r=rng.uniform(.04,.3,n);u=np.zeros(n);ids=rng.choice(n,2,replace=False);u[ids]=rng.choice([-1,1],2)*.045;h=10;c=rng.integers(1,5,n).astype(float)
   # Put one action-relevant coordinate near its boundary to exercise selection.
   i=ids[0];m[i]=np.sign(u[i])*rng.uniform(.55,.72)
   if p.safe(m,r,u,h):continue
   for name in (['enumerate','milp','greedy'] if n<=16 else ['milp','greedy']):
    start=time.perf_counter()
    S=choose_exact(p,m,r,u,h,c,6) if name=='enumerate' else choose_milp(p,m,r,u,h,c,6) if name=='milp' else choose_prefix(p,m,r,u,h,c,6,np.ones(n)*10,'greedy')
    seconds=time.perf_counter()-start
    rows.append({'n':n,'case':case,'solver':name,'seconds':seconds,'feasible':int(S is not None),'cost':float(c[S].sum()) if S is not None else None})
   if n<=16:
    a1,a2=rows[-3:-1]
    assert a1['feasible']==a2['feasible'] and a1['cost']==a2['cost'],(a1,a2)
  print('n',n,'rows',len(rows),flush=True)
 with (out/'scaling.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
