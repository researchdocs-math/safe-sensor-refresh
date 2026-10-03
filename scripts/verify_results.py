import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--results',default='results/reproduced');a=ap.parse_args();out=Path(a.results);checks=[]
for n in ['main','deadline','persistent','stress','recheck','local_agent']:
 expected=pd.read_csv(ROOT/'results/reference'/(n+'.csv'));actual=pd.read_csv(out/(n+'.csv'))
 pd.testing.assert_frame_equal(expected,actual,check_exact=False,rtol=1e-12,atol=1e-12)
 checks.append({'suite':n,'rows':len(actual),'match':True})
(out/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
