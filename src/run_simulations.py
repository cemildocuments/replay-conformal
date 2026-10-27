"""Known-CDF validation, assumption stress tests, and quantile timing."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[v]='1'
from pathlib import Path
import json,time,argparse
import numpy as np
import pandas as pd
from scipy.stats import norm,t as student
from calibration import rank_floor,replay_rank,conformal_rank
ROOT=Path(__file__).resolve().parents[1]

def cdf(name,x):
    if name=='halfnormal':return np.maximum(0,2*norm.cdf(x)-1)
    if name=='exponential':return np.where(x>=0,-np.expm1(-x),0)
    if name=='half_t3':return np.maximum(0,2*student.cdf(x,3)-1)
    if name=='rounded_halfnormal':return np.where(x>=0,2*norm.cdf(x+.25)-1,0)
    raise ValueError(name)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='results');args=parser.parse_args()
    outdir=(ROOT/args.output_dir).resolve();outdir.mkdir(parents=True,exist_ok=True)
    R=10000;records=[]
    for di,name in enumerate(['halfnormal','exponential','half_t3','rounded_halfnormal']):
        for n in [32,128,512]:
            rng=np.random.default_rng(781000+di*1000+n)
            if name=='exponential':s=rng.exponential(size=(R,n))
            elif name=='half_t3':s=np.abs(rng.standard_t(3,size=(R,n)))
            else:s=np.abs(rng.normal(size=(R,n)))
            if name=='rounded_halfnormal':s=np.round(2*s)/2
            s.sort(axis=1)
            for cap in [2,4,8]:
                for total in sorted(set(int(round(x*n)) for x in np.linspace(1,cap,29))):
                    for alpha in [.05,.1,.2]:
                        k=conformal_rank(total,alpha)
                        for method,kk in [('row',k),('replay',replay_rank(n,total,cap,alpha)),('count',total-n+conformal_rank(n,alpha))]:
                            if kk>total:q=np.full(R,np.inf);r=n+1
                            else:
                                # Low-score-first replay attains the sharp rank envelope simultaneously for every k.
                                r=rank_floor(kk,n,total,cap);q=s[:,r-1]
                            coverage=cdf(name,q)
                            records.append(dict(distribution=name,n=n,cap=cap,N=total,ratio=total/n,alpha=alpha,method=method,
                                                rank=r,exact_floor=min(1,r/(n+1)),coverage=float(coverage.mean()),
                                                mc_se=float(coverage.std(ddof=1)/np.sqrt(R)),mean_width=float((2*q).mean()),replicates=R))
    pd.DataFrame(records).to_csv(outdir/'simulation.csv',index=False,float_format='%.12g')
    # Downward score perturbations and incorrect metadata: intentionally outside unadjusted exact-replay assumptions.
    n=128;total=512;cap=8;alpha=.1
    rng=np.random.default_rng(110093);s=np.sort(np.abs(rng.normal(size=(R,n))),axis=1)
    r=rank_floor(replay_rank(n,total,cap,alpha),n,total,cap);q=s[:,r-1]
    stress=[]
    for delta in [0,.05,.1,.2,.4,.8]:
        changed=np.maximum(0,q-delta)
        for method,qq in [('unadjusted',changed),('delta_adjusted',changed+delta)]:
            v=cdf('halfnormal',qq)
            stress.append(dict(experiment='score_perturbation',value=delta,method=method,coverage=v.mean(),mc_se=v.std(ddof=1)/np.sqrt(R)))
    for claimed in [4,5,6,7,8,10,16,32]:
        k=replay_rank(n,total,claimed,alpha);rank=rank_floor(k,n,total,cap);v=cdf('halfnormal',s[:,rank-1])
        stress.append(dict(experiment='claimed_cap',value=claimed,method='replay',coverage=v.mean(),mc_se=v.std(ddof=1)/np.sqrt(R)))
    pd.DataFrame(stress).to_csv(outdir/'stress.csv',index=False,float_format='%.12g')
    timings=[];rng=np.random.default_rng(40772)
    for N in [10000,100000,1000000,4000000]:
        rows=rng.exponential(size=N);k=int(.975*N)
        for method in ['partition','sort']:
            elapsed=[]
            for rep in range(6):
                start=time.perf_counter()
                q=np.partition(rows,k-1)[k-1] if method=='partition' else np.sort(rows)[k-1]
                dt=time.perf_counter()-start
                if rep:elapsed.append(dt)
            timings.append(dict(rows=N,method=method,median_seconds=np.median(elapsed),score_megabytes=rows.nbytes/1e6))
    pd.DataFrame(timings).to_csv(outdir/'timings.csv',index=False,float_format='%.12g')
    (outdir/'simulation_metadata.json').write_text(json.dumps({'replicates_per_cell':R,'score_distributions':4,'source_sizes':[32,128,512],'simulated_rows':len(records),'primary_seed':781000,'stress_seed':110093,'timing_seed':40772},indent=2)+'\n')
    print('Simulation cells:',len(records));print(pd.DataFrame(stress).to_string(index=False));print(pd.DataFrame(timings).to_string(index=False))
if __name__=='__main__':main()
