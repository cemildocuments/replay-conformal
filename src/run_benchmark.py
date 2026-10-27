"""Deterministic parallel training and paired replay-calibration experiments."""
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[v]='1'
from pathlib import Path
import argparse, json, time, platform, hashlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import sklearn, scipy, psutil
from calibration import thresholds, replay_counts, rank_floor, conformal_rank
ROOT=Path(__file__).resolve().parents[1]


def fit_one(dataset,model_name,seed,dsidx,config):
    outdir=Path(config['_output_dir'])
    cache=outdir/'scores'/f'{dataset}__{model_name}__{seed:03d}.npz'
    dest=outdir/'runs'/f'{dataset}__{model_name}__{seed:03d}.csv'
    if dest.exists():return str(dest)
    seedvalue=config['seed_offset']+1000*dsidx+seed
    raw=np.load(ROOT/'data/processed'/f'{dataset}.npz');X,y=raw['X'],raw['y']
    if cache.exists():
        c=np.load(cache);cal=c['cal_scores'];test=c['test_scores'];fit_seconds=float(c['fit_seconds']);rmse=float(c['rmse'])
    else:
        rng=np.random.default_rng(seedvalue);ix=rng.permutation(len(y));nt=len(y)//2;nte=len(y)//4
        itrain=ix[:nt];itest=ix[nt:nt+nte];ical=ix[nt+nte:]
        if model_name=='ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        elif model_name=='forest':model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features=1.0,n_jobs=1,random_state=seedvalue)
        elif model_name=='histboost':model=HistGradientBoostingRegressor(max_iter=200,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0,early_stopping=False,random_state=seedvalue)
        else:raise ValueError(model_name)
        t=time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(X[itrain],y[itrain]);pc=model.predict(X[ical]);pt=model.predict(X[itest])
        fit_seconds=time.perf_counter()-t;cal=np.abs(y[ical]-pc);test=np.abs(y[itest]-pt)
        rmse=float(np.sqrt(mean_squared_error(y[itest],pt)))
        np.savez_compressed(cache,train_index=itrain,calibration_index=ical,test_index=itest,
                            cal_scores=cal,test_scores=test,cal_prediction=pc,test_prediction=pt,
                            cal_target=y[ical],test_target=y[itest],fit_seconds=fit_seconds,rmse=rmse)
    records=[]
    for n in config['calibration_sizes']:
        assert len(cal)>=n
        scores=cal[:n]
        for scid,sc in enumerate(config['scenarios']):
            rng=np.random.default_rng(np.random.SeedSequence([seedvalue,n,scid,9187]))
            total=int(n*sc['ratio']);counts=replay_counts(scores,total,sc['cap'],sc['mechanism'],rng)
            for alpha in config['alphas']:
                out=thresholds(scores,counts,alpha,sc['cap'])
                for method,q in out.items():
                    covered=int(np.sum(test<=q))
                    records.append(dict(dataset=dataset,model=model_name,seed=seed,calibration_n=n,
                        alpha=alpha,cap=sc['cap'],ratio=sc['ratio'],mechanism=sc['mechanism'],method=method,
                        N=total,n_test=len(test),covered=covered,coverage=covered/len(test),width=2*q,
                        width_ratio=q/out['dedup'],threshold=q,rmse=rmse,fit_seconds=fit_seconds,
                        effective_n=total**2/np.square(counts).sum(),
                        row_floor=rank_floor(conformal_rank(total,alpha),n,total,sc['cap'])/(n+1)))
    pd.DataFrame(records).to_csv(dest,index=False,float_format='%.12g')
    return str(dest)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=10);parser.add_argument('--seeds',type=int)
    parser.add_argument('--output-dir',default='results')
    args=parser.parse_args();config=json.loads((ROOT/'config/main.json').read_text())
    if args.seeds is not None:config['seeds']=args.seeds
    outdir=(ROOT/args.output_dir).resolve();outdir.mkdir(parents=True,exist_ok=True)
    config['_output_dir']=str(outdir)
    for folder in ['scores','runs']: (outdir/folder).mkdir(exist_ok=True)
    t=time.perf_counter()
    jobs=[(d,m,s,di,config) for di,d in enumerate(config['datasets']) for m in config['models'] for s in range(config['seeds'])]
    paths=Parallel(n_jobs=args.workers,verbose=10)(delayed(fit_one)(*j) for j in jobs)
    frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    frame.to_csv(outdir/'benchmark.csv',index=False,float_format='%.12g')
    meta={'config':config,'config_sha256':hashlib.sha256((ROOT/'config/main.json').read_bytes()).hexdigest(),
          'wall_seconds':time.perf_counter()-t,'model_fits':len(jobs),'result_rows':len(frame),'workers':args.workers,
          'python':platform.python_version(),'os':platform.system(),'architecture':platform.machine(),
          'logical_cpus':psutil.cpu_count(),'memory_gib':psutil.virtual_memory().total/2**30,
          'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}
    (outdir/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
