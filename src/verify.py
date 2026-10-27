"""Verify data provenance, independent primary calculations, or a fresh reproduction."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
KEYS=['dataset','model','seed','calibration_n','alpha','cap','ratio','mechanism','method']

def verify_inputs():
    manifests=json.loads((ROOT/'data/source_manifest.json').read_text())
    for key,m in manifests.items():
        path=ROOT/'data/raw'/f'{key}.zip'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256'],str(path)
    info=pd.read_csv(ROOT/'data/dataset_summary.csv')
    for _,row in info.iterrows():
        csv=ROOT/'data/processed'/f'{row.dataset}.csv'
        assert hashlib.sha256(csv.read_bytes()).hexdigest()==row.csv_sha256
        a=np.load(ROOT/'data/processed'/f'{row.dataset}.npz')
        v=pd.read_csv(csv).to_numpy(float)
        np.testing.assert_allclose(v[:,:-1],a['X'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(v[:,-1],a['y'],rtol=1e-12,atol=1e-12)
        assert len(v)==row.unique_rows and len(np.unique(v,axis=0))==len(v)
    return len(manifests),len(info)

def independent_primary(results):
    d=pd.read_csv(results/'benchmark.csv')
    p=d.query('alpha==.1 and calibration_n==128 and cap==8 and ratio==4 and mechanism=="low"')
    checked=0
    for (dataset,model,seed),g in p.groupby(['dataset','model','seed']):
        f=results/'scores'/f'{dataset}__{model}__{seed:03d}.npz';z=np.load(f)
        for i,j in [('train_index','calibration_index'),('train_index','test_index'),('calibration_index','test_index')]:
            assert not np.intersect1d(z[i],z[j]).size
        src=np.sort(z['cal_scores'][:128]);test=z['test_scores']
        # Independently evaluated sharp source ranks for this fixed primary scenario.
        expected={'row':src[77], 'replay':src[116], 'dedup':src[116], 'count':src[116]}
        for method,q in expected.items():
            r=g[g.method==method].iloc[0]
            np.testing.assert_allclose(r.threshold,q,rtol=2e-11,atol=2e-11)
            np.testing.assert_allclose(r.coverage,np.mean(test<=q),rtol=2e-11,atol=2e-11)
        checked+=1
    return checked

def compare(reference,candidate):
    checked={}
    for filename,keys,ignore in [
        ('benchmark.csv',KEYS,['fit_seconds']),
        ('aggregate.csv',['calibration_n','alpha','cap','ratio','mechanism','method'],[]),
        ('primary_cells.csv',['dataset','model','method'],[]),
        ('simulation.csv',['distribution','n','cap','N','alpha','method'],[]),
        ('stress.csv',['experiment','value','method'],[])]:
        a=pd.read_csv(reference/filename).sort_values(keys).reset_index(drop=True)
        b=pd.read_csv(candidate/filename).sort_values(keys).reset_index(drop=True)
        assert a.shape==b.shape,(filename,a.shape,b.shape)
        for c in a.columns:
            if c in ignore:continue
            if pd.api.types.is_numeric_dtype(a[c]):
                np.testing.assert_allclose(a[c],b[c],rtol=2e-10,atol=2e-10,err_msg=f'{filename}:{c}')
            else:assert a[c].equals(b[c]),(filename,c)
        checked[filename]=len(a)
    return checked

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',default='results');ap.add_argument('--candidate');args=ap.parse_args()
    ref=(ROOT/args.reference).resolve();report={'input_archives_and_tasks':verify_inputs(),'independent_primary_models':independent_primary(ref)}
    if args.candidate:report['reproduced_rows']=compare(ref,(ROOT/args.candidate).resolve())
    report['status']='passed'
    (ref/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
