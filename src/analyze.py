"""Paired summaries with seed-block uncertainty conditional on benchmark datasets."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]

def mean_ci(values):
    v=np.asarray(values,dtype=float);m=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)))
    e=float(t.ppf(.975,len(v)-1)*se)
    return {'mean':m,'lo':m-e,'hi':m+e,'se':se,'blocks':len(v)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-dir',default='results');args=parser.parse_args()
    outdir=(ROOT/args.results_dir).resolve()
    d=pd.read_csv(outdir/'benchmark.csv')
    primary=d.query('alpha==.1 and calibration_n==128 and cap==8 and ratio==4 and mechanism=="low"')
    cells=primary.groupby(['dataset','model','method'],as_index=False)[['coverage','width_ratio','width']].mean()
    cells.to_csv(outdir/'primary_cells.csv',index=False,float_format='%.12g')
    rows=[]
    group=['calibration_n','alpha','cap','ratio','mechanism','method']
    seedmeans=d.groupby(group+['seed'])[['coverage','width_ratio']].mean().reset_index()
    for values,g in seedmeans.groupby(group):
        rec=dict(zip(group,values))
        for metric in ['coverage','width_ratio']:
            for k,v in mean_ci(g[metric]).items():rec[metric+'_'+k]=v
        rows.append(rec)
    summary=pd.DataFrame(rows);summary.to_csv(outdir/'aggregate.csv',index=False,float_format='%.12g')
    base=primary.groupby(['seed','method'])[['coverage','width_ratio']].mean().reset_index()
    cp=base.pivot(index='seed',columns='method',values='coverage')
    outputs={'primary':{},'replay_minus_row':mean_ci(cp.replay-cp.row),'replay_minus_ess':mean_ci(cp.replay-cp.ess),
             'all_primary_thresholds_equal_dedup':bool(np.allclose(primary.query('method=="replay"').width_ratio,1)),
             'dataset_model_cells':len(cells.query('method=="replay"'))}
    for method,g in base.groupby('method'):
        outputs['primary'][method]={k:mean_ci(g[k]) for k in ['coverage','width_ratio']}
    # Cap information value at uniform full replication: identical source pool for every method.
    sat=d.query('alpha==.1 and calibration_n==128 and cap==8 and ratio==8 and mechanism=="balanced"')
    piv=sat.pivot(index=['dataset','model','seed'],columns='method',values='width')
    savings=(1-piv.replay/piv['count']).groupby('seed').mean()
    outputs['saturated_width_saving']=mean_ci(savings)
    outputs['corrected_minimum_cell_mean']=float(cells.query('method=="replay"').coverage.min())
    outputs['corrected_maximum_cell_mean']=float(cells.query('method=="replay"').coverage.max())
    outputs['main_rows']=len(d)
    (outdir/'key_results.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print(json.dumps(outputs,indent=2))
if __name__=='__main__':main()
