"""Deterministic robust certificates and deadline-constrained sensor refreshing."""
from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

@dataclass
class Plant:
    domain: str
    n: int = 8
    def __post_init__(self):
        self.rho,self.coupling,self.w,self.e,self.amplitude = ((.980,.006,.006,.012,.065) if self.domain=='water' else (.992,.003,.0035,.010,.045))
        self.A=(self.rho-2*self.coupling)*np.eye(self.n)
        for i in range(self.n):
            self.A[i,(i-1)%self.n]+=self.coupling
            self.A[i,(i+1)%self.n]+=self.coupling
        self.C=np.vstack([np.eye(self.n),-np.eye(self.n),np.ones((1,self.n))/self.n,-np.ones((1,self.n))/self.n])
        self.b=np.r_[np.ones(2*self.n),.70,.70]
        self.tubes={}
        self.powers=[np.linalg.matrix_power(self.A,j) for j in range(65)]
        self.wait_r=[sum((self.powers[k]@np.full(self.n,self.w) for k in range(j)),start=np.zeros(self.n)) for j in range(65)]
    def tube(self,h):
        if h not in self.tubes:
            H=[];U=[];W=[]
            for j in range(h+1):
                H.append(self.C@self.powers[j])
                U.append(self.C@sum(self.powers[:j],start=np.zeros((self.n,self.n))))
                W.append(sum((np.abs(self.C@self.powers[k])@np.full(self.n,self.w) for k in range(j)),start=np.zeros(len(self.b))))
            self.tubes[h]=(np.vstack(H),np.vstack(U),np.concatenate(W),np.tile(self.b,h+1))
        return self.tubes[h]
    def constraints(self,m,r,u,h):
        H,U,W,b=self.tube(h)
        return H@m+np.abs(H)@r+U@u+W-b
    def safe(self,m,r,u,h):
        return np.max(self.constraints(m,r,u,h))<=1e-10
    def wait(self,m,r,d):
        return self.powers[d]@m,self.powers[d]@r+self.wait_r[d]
    def violation(self,x):return max(0.,float(np.max(self.C@x-self.b)))

@lru_cache(maxsize=16)
def masks(n):return ((np.arange(1<<n)[:,None]>>np.arange(n))&1).astype(bool)

def latency(cost):return max(1,int(math.ceil(float(cost)/6)))

def predicted_violation(p,m,r,u,h,S,c,ignore_wait=False):
    d=0 if ignore_wait else latency(np.sum(c[S]))
    m1,r1=p.wait(m,r,d)
    r1=np.where(S,np.minimum(r1,p.e),r1)
    return float(np.max(p.constraints(m1,r1,u,h)))

def choose_exact(p,m,r,u,h,c,remaining,available=None,ignore_wait=False):
    M=masks(p.n)
    keep=np.any(M,axis=1)
    if available is not None:keep &= ~np.any(M&~available,axis=1)
    M=M[keep];cost=M@c;delay=np.maximum(1,np.ceil(cost/6).astype(int));keep=delay<=remaining
    M,cost,delay=M[keep],cost[keep],delay[keep]
    H,U,W,b=p.tube(h);AH=np.abs(H);best=None
    for d in np.unique(delay):
        ids=np.flatnonzero(delay==d)
        m1,r1=p.wait(m,r,0 if ignore_wait else int(d))
        R=np.where(M[ids],np.minimum(r1,p.e),r1)
        scores=np.max(R@AH.T+(H@m1+U@u+W-b)[None,:],axis=1)
        good=ids[scores<=1e-10]
        if len(good):
            j=good[np.argmin(cost[good])]
            key=(float(cost[j]),int(delay[j]),int(np.flatnonzero(keep)[j]))
            if best is None or key<best[0]:best=(key,M[j].copy())
    return None if best is None else best[1]

def choose_prefix(p,m,r,u,h,c,remaining,ages,kind,available=None):
    available=np.ones(p.n,dtype=bool) if available is None else available
    if kind=='age':score=ages/c
    elif kind=='width':score=r/c
    else:
        v=p.constraints(m,r,u,h);H,_,_,_=p.tube(h)
        active=np.maximum(v,0)
        score=(np.abs(H).T@active)*(np.maximum(r-p.e,0))/c
    order=np.argsort(-score,kind='stable');S=np.zeros(p.n,dtype=bool)
    for i in order:
        if not available[i]:continue
        S[i]=True
        if latency(c[S].sum())>remaining:
            S[i]=False;continue
        if predicted_violation(p,m,r,u,h,S,c)<=1e-10:return S
    return None

def propose(m,target,amplitude,h,style='feedback'):
    err=target-m+.015*h*m
    ids=np.argsort(-np.abs(err),kind='stable')[:2]
    u=np.zeros(len(m))
    # A bounded, goal-directed supervisory proposal; safety is delegated to the guard.
    u[ids]=np.clip(err[ids]/max(2,h),-amplitude,amplitude)
    return u

METHODS=['always','trigger_all','periodic','age','width','greedy','refresh','no_refresh','point']

def episode(domain,method,seed,n=8,decisions=60,deadline=6,noise_scale=1.,query_loss=0.,return_trace=False,proposal_tape=None,noise_mode="uniform",proposal_fn=None):
    p=Plant(domain,n);rng=np.random.default_rng(seed)
    costs=rng.permutation(np.resize(np.array([1.,2.,3.,4.]),n))
    x=rng.uniform(-.55,.55,n);measurement=x+rng.uniform(-p.e,p.e,n);m=measurement.copy();r=np.full(n,p.e)
    ages=np.zeros(n,dtype=int);clock=0
    disturbances=rng.uniform(-p.w,p.w,(12000,n))*noise_scale
    if noise_mode=='persistent':
        signs=rng.choice([-1.,1.],n)
        disturbances[:]=signs*p.w*noise_scale
    measurement_errors=rng.uniform(-p.e,p.e,(12000,n))
    loss_draws=rng.uniform(size=(12000,n))
    # Persistent operating goals: interior, near-boundary and infeasible requests.
    # Hold each goal for 12 proposals; balanced signs avoid a trivial mean violation.
    targets=[]
    for block in range((decisions+11)//12):
        signs=rng.permutation(np.r_[np.ones(n//2),-np.ones(n-n//2)])
        magnitude=rng.choice([.75,.95,1.08],n,p=[.25,.60,.15])
        goal=signs*magnitude
        targets.extend([goal.copy() for _ in range(12)])
    targets=np.asarray(targets[:decisions]);horizons=rng.choice([4,6,8,10],decisions)
    # Add heterogeneous initial ages while maintaining interval consistency.
    initial_periods=rng.integers(2,13,n)
    queries=cost=accepted=violations=cert_fail=contain_fail=rounds=0
    inconsistent=forecast_fail=0
    maxviolation=0.;tracking=0.;wait_ticks=0;act_ticks=0;falseblocks=0;truthsafe=0
    traces=[];proposals=[];timeline=[]
    def step(u):
        nonlocal x,m,r,clock,violations,maxviolation,contain_fail
        x=p.A@x+u+disturbances[clock]
        m=p.A@m+u;r=p.A@r+p.w;clock+=1;ages[:]+=1
        v=p.violation(x);violations+=int(v>1e-9);maxviolation=max(maxviolation,v)
        contain_fail+=int(np.any(np.abs(x-m)>r+1e-8))
        if return_trace:timeline.append({'clock':clock,'x':x.tolist(),'m':m.tolist(),'r':r.tolist(),'u':u.tolist()})
    for _ in range(16):
        step(np.zeros(n))
        idx=(clock%initial_periods)==0
        y=x+measurement_errors[clock]
        lo=np.maximum(m-r,y-p.e);hi=np.minimum(m+r,y+p.e)
        m[idx]=(lo[idx]+hi[idx])/2;r[idx]=np.maximum(0,(hi[idx]-lo[idx])/2);ages[idx]=0
    violations=contain_fail=0
    for t in range(decisions):
        target=targets[t];h=int(horizons[t]);u=propose(m,target,p.amplitude,h)
        if proposal_fn is not None:u,h=proposal_fn(domain,seed,t,m.copy(),r.copy(),target.copy(),p.amplitude,h)
        if proposal_tape is not None:u=np.asarray(proposal_tape[t]['u'],dtype=float);h=int(proposal_tape[t]['h'])
        proposals.append({'x':x.tolist(),'clock':clock,'ages':ages.tolist(),'costs':costs.tolist(),'m':m.tolist(),'r':r.tolist(),'target':target.tolist(),'u':u.tolist(),'h':h})
        trusted_forecast=False
        remaining=deadline;available=np.ones(n,dtype=bool);q0=queries;c0=cost;age0=ages.copy();x0=x.copy()
        def acquire(S):
            nonlocal m,r,queries,cost,remaining,rounds,wait_ticks,inconsistent
            cc=float(costs[S].sum());d=latency(cc)
            for _ in range(d):step(np.zeros(n))
            remaining-=d;wait_ticks+=d;queries+=int(S.sum());cost+=cc;rounds+=1
            delivered=S&(loss_draws[clock]>=query_loss)
            y=x+measurement_errors[clock]
            lo=np.maximum(m-r,y-p.e);hi=np.minimum(m+r,y+p.e)
            # Empty intersections are recorded under out-of-model stress; resetting the
            # measured coordinate does not restore the disturbance-bound guarantee.
            bad=delivered&(lo>hi)
            if np.any(bad):
                inconsistent+=int(bad.sum());lo[bad]=y[bad]-p.e;hi[bad]=y[bad]+p.e
            m[delivered]=(lo[delivered]+hi[delivered])/2;r[delivered]=np.maximum(0,(hi[delivered]-lo[delivered])/2)
            ages[delivered]=0;available[S]=False
        allS=np.ones(n,dtype=bool)
        if method=='always' or (method=='periodic' and t%4==0) or (method=='trigger_all' and not p.safe(m,r,u,h)):
            if latency(costs.sum())<=remaining:acquire(allS)
        elif method in ['refresh','age','width','greedy','no_wait','trust_forecast']:
            for _ in range(3):
                if p.safe(m,r,u,h) or remaining<=0:break
                if method in ['refresh','no_wait','trust_forecast']:S=choose_exact(p,m,r,u,h,costs,remaining,available,ignore_wait=method=='no_wait')
                else:S=choose_prefix(p,m,r,u,h,costs,remaining,ages,method,available)
                if S is None:break
                acquire(S)
                forecast_fail+=int(not p.safe(m,r,u,h))
                if method=='trust_forecast':trusted_forecast=True;break
        ok=trusted_forecast or p.safe(m,np.zeros(n) if method=='point' else r,u,h)
        # True-state oracle used exclusively as an offline diagnostic.
        osafe=p.safe(x,np.zeros(n),u,h);truthsafe+=int(osafe);falseblocks+=int(osafe and not ok)
        accepted+=int(ok)
        if ok and not p.safe(m,r,u,h):cert_fail+=1
        applied=u if ok else np.zeros(n)
        max_before=maxviolation
        for _ in range(h):step(applied);tracking+=float(np.mean(np.abs(x-target)));act_ticks+=1
        if return_trace:
            traces.append({'decision':t,'clock':clock,'x':x.tolist(),'start_x':x0.tolist(),'m':m.tolist(),'r':r.tolist(),'ages':ages.tolist(),'start_ages':age0.tolist(),'target':target.tolist(),'u':u.tolist(),'h':h,'accepted':bool(ok),'queries':queries-q0,'cost':cost-c0,'max_violation':maxviolation})
    out={'domain':domain,'method':method,'seed':seed,'n':n,'decisions':decisions,'deadline':deadline,'noise_scale':noise_scale,'query_loss':query_loss,'noise_mode':noise_mode,'inconsistent_intervals':inconsistent,'forecast_failures':forecast_fail,'queries':queries,'cost':cost,'accepted':accepted,'violating_steps':violations,'violation_episode':int(violations>0),'max_violation':maxviolation,'tracking_mae':tracking/max(1,act_ticks),'query_rounds':rounds,'wait_ticks':wait_ticks,'act_ticks':act_ticks,'certificate_failures':cert_fail,'containment_failures':contain_fail,'oracle_safe':truthsafe,'false_blocks':falseblocks}
    if return_trace:out.update(trace=traces,proposals=proposals,timeline=timeline)
    return out

def choose_milp(p,m,r,u,h,c,remaining,available=None):
    """Exact forecast selector via one binary covering program per delay class."""
    from scipy.optimize import milp,Bounds,LinearConstraint
    H,U,W,b=p.tube(h);AH=np.abs(H)
    available=np.ones(p.n,dtype=bool) if available is None else available
    best=None
    for d in range(1,remaining+1):
        md,rd=p.wait(m,r,d)
        deficit=H@md+AH@rd+U@u+W-b
        gain=AH*np.maximum(rd-p.e,0)[None,:]
        # Integer communication units make c(S) >= 6(d-1)+1 exact.
        mat=np.vstack([gain,c[None,:]])
        lo=np.r_[deficit,6*(d-1)+1];hi=np.r_[np.full(len(deficit),np.inf),6*d]
        sol=milp(c,integrality=np.ones(p.n),bounds=Bounds(np.zeros(p.n),available.astype(float)),constraints=LinearConstraint(mat,lo,hi),options={'mip_rel_gap':0.,'presolve':True})
        if sol.success:
            S=sol.x>.5
            # Numerical solver tolerances never constitute a safety certificate.
            if predicted_violation(p,m,r,u,h,S,c)>1e-10:continue
            cost=float(c[S].sum())
            if best is None or cost<best[0]:best=(cost,S)
    return None if best is None else best[1]
