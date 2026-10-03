import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
"""Aggregate episode-level results and paired bootstrap intervals."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import beta

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',default='results/reproduced');a=ap.parse_args();p=Path(a.results);rng=np.random.default_rng(48103);out={}
 for suite in ['main','deadline','persistent','stress','recheck','local_agent']:
  f=p/(suite+'.csv')
  if not f.exists():continue
  d=pd.read_csv(f);groups=['domain','method']+(['deadline'] if suite=='deadline' else ['noise_scale','query_loss'] if suite=='stress' else ['noise_mode','query_loss'] if suite=='recheck' else [])
  d['accept_pct']=100*d.accepted/d.decisions;d['cost_per_proposal']=d.cost/d.decisions;d['wait_per_proposal']=d.wait_ticks/d.decisions
  cols=['accept_pct','cost','cost_per_proposal','queries','tracking_mae','violation_episode','violating_steps','wait_per_proposal','forecast_failures','containment_failures']
  summary=d.groupby(groups)[cols].mean().reset_index();summary.to_csv(p/(suite+'_summary.csv'),index=False);out[suite+'_episodes']=len(d)
  if suite=='main':
   comparisons=[]
   for domain in ['water','thermal']:
    for comparator in ['trigger_all','greedy','age','no_refresh']:
     x=d[(d.domain==domain)&(d.method=='refresh')].sort_values('seed');y=d[(d.domain==domain)&(d.method==comparator)].sort_values('seed');ids=rng.integers(0,len(x),(10000,len(x)))
     costx=x.cost.to_numpy();costy=y.cost.to_numpy();accdiff=x.accept_pct.to_numpy()-y.accept_pct.to_numpy()
     row={'domain':domain,'comparator':comparator,'accept_difference_pp':accdiff.mean(),'accept_ci':np.quantile(accdiff[ids].mean(1),[.025,.975]).tolist()}
     if costy.mean()>0:
      saving=100*(1-costx[ids].mean(1)/costy[ids].mean(1));row.update(cost_reduction_pct=100*(1-costx.mean()/costy.mean()),cost_reduction_ci=np.quantile(saving,[.025,.975]).tolist())
     comparisons.append(row)
   out['comparisons']=comparisons
  if suite=='persistent':
   for domain in ['water','thermal']:
    x=d[(d.domain==domain)&(d.method=='point')];k=int(x.violation_episode.sum());n=len(x)
    out[domain+'_persistent_point_failure']={'k':k,'n':n,'ci':[float(beta.ppf(.025,k,n-k+1)),float(beta.ppf(.975,k+1,n-k))]}
 out['zero_failures_upper95_main_regime']=float(1-.05**(1/512))
 if (p/'local_agent_inference.json').exists():
  z=pd.read_json(p/'local_agent_inference.json');out['local_inference']={'calls':len(z),'valid':int(z.valid.sum()),'seconds_median':float(z.seconds.median()),'seconds_p95':float(z.seconds.quantile(.95)),'peak_memory_gb_max':float(z.peak_memory_gb.max()),'total_seconds':float(z.seconds.sum())}
 if (p/'scaling.csv').exists():
  z=pd.read_csv(p/'scaling.csv');summ=z.groupby(['n','solver']).seconds.agg(['median',lambda x:x.quantile(.95),'max']);summ.columns=['median_s','p95_s','max_s'];summ.to_csv(p/'scaling_summary.csv')
 (p/'statistics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
