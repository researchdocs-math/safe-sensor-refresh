"""Reproduce simulation results and replay the recorded local-agent proposals."""
import argparse,json,os,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(*args):
 subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=10);ap.add_argument('--benchmark',action='store_true');a=ap.parse_args()
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
 cfg=json.loads((ROOT/'config/study.json').read_text());out=ROOT/'results/reproduced';out.mkdir(exist_ok=True)
 run('tests/validate.py')
 for suite in ['main','deadline','persistent','stress','recheck']:
  c=cfg[suite];run('scripts/run_experiments.py','--suite',suite,'--seeds',c['seeds'],'--decisions',c['decisions'],'--workers',a.workers,'--out',out)
 for n in ['local_agent_tapes.json','local_agent_inference.json']:shutil.copy2(ROOT/'data'/n,out/n)
 run('scripts/run_local_agent.py','--replay','--out',out)
 shutil.copy2(out/'local_agent_replay.csv',out/'local_agent.csv')
 if a.benchmark:run('scripts/benchmark_selectors.py','--cases',cfg['scaling_cases'],'--out',out)
 run('scripts/analyze.py','--results',out)
 run('scripts/verify_results.py','--results',out)
if __name__=='__main__':main()
