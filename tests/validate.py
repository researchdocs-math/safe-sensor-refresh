import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
"""Numerical checks of invariance, exact support functions and selector optimality."""
import itertools,json
import numpy as np
from saferefresh import Plant,choose_exact,predicted_violation,masks,latency,episode
rng=np.random.default_rng(281828)
checks={'backup_boundary_points':0,'tube_corner_extrema':0,'selector_cases':0,'trajectory_checks':0}
for dom in ['water','thermal']:
 p=Plant(dom,4)
 # All vertices of the coordinate box that also satisfy the aggregate bounds.
 for x in itertools.product([-1.,1.],repeat=4):
  x=np.array(x)
  if p.violation(x)>1e-10:continue
  for w in itertools.product([-p.w,p.w],repeat=4):
   assert p.violation(p.A@x+np.array(w))<1e-10
   checks['backup_boundary_points']+=1
 # Independent worst-case direct rollout for each linear constraint: initial and
 # disturbance corners are chosen using the signs of their exact coefficients.
 for z in range(100):
  m=rng.uniform(-.8,.8,4);r=rng.uniform(.01,.25,4);u=rng.uniform(-.08,.08,4);h=int(rng.integers(1,11))
  got=p.constraints(m,r,u,h).reshape(h+1,-1)
  for j in range(h+1):
   for k,C in enumerate(p.C):
    x=m+np.sign(C@np.linalg.matrix_power(p.A,j))*r
    for t in range(j):
     w=np.sign(C@np.linalg.matrix_power(p.A,j-1-t))*p.w
     x=p.A@x+u+w
    assert abs(C@x-p.b[k]-got[j,k])<1e-10
    checks['tube_corner_extrema']+=1
  c=rng.integers(1,5,4).astype(float);rem=int(rng.integers(1,7))
  chosen=choose_exact(p,m,r,u,h,c,rem)
  feasible=[]
  for bits in itertools.product([False,True],repeat=4):
   S=np.array(bits)
   if S.any() and latency(c[S].sum())<=rem and predicted_violation(p,m,r,u,h,S,c)<=1e-10:feasible.append(c[S].sum())
  assert (chosen is None)==(len(feasible)==0)
  if feasible:assert c[chosen].sum()==min(feasible)
  checks['selector_cases']+=1
 for seed in range(20):
  q=episode(dom,'refresh',60000+seed,decisions=72)
  assert q['containment_failures']==0 and q['violating_steps']==0
  checks['trajectory_checks']+=1
print(json.dumps({'passed':True,'checks':checks},indent=2))
