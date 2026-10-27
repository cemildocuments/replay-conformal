"""Fetch versioned UCI archives; build exact-record-deduplicated regression tasks."""
from pathlib import Path
import hashlib, io, json, zipfile
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'wine': (186, 'wine+quality', '10.24432/C56S3T'),
    'abalone': (1, 'abalone', '10.24432/C55C7W'),
    'concrete': (165, 'concrete+compressive+strength', '10.24432/C5PK67'),
    'energy': (242, 'energy+efficiency', '10.24432/C51307'),
    'airfoil': (291, 'airfoil+self+noise', '10.24432/C5VW2C'),
}


def main():
    manifest, frames = {}, {}
    for key, (ident, slug, doi) in SOURCES.items():
        url = f'https://archive.ics.uci.edu/static/public/{ident}/{slug}.zip'
        path = ROOT / 'data/raw' / f'{key}.zip'
        if not path.exists():
            r = requests.get(url, timeout=90); r.raise_for_status();path.write_bytes(r.content)
        content = path.read_bytes()
        z = zipfile.ZipFile(io.BytesIO(content))
        manifest[key] = {'url': url, 'doi': doi, 'sha256':hashlib.sha256(content).hexdigest(),
                         'license':'CC BY 4.0 (UCI dataset record)', 'archive_files':z.namelist()}
        if key == 'wine':
            for color in ['red','white']:
                filename = next(n for n in z.namelist() if n.endswith(f'winequality-{color}.csv'))
                frames[f'wine_{color}'] = pd.read_csv(z.open(filename), sep=';')
        elif key == 'abalone':
            filename=next(n for n in z.namelist() if n.endswith('abalone.data'))
            df=pd.read_csv(z.open(filename),header=None)
            df.columns=['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']
            frames[key]=df
        elif key == 'airfoil':
            filename=next(n for n in z.namelist() if n.endswith('.dat'))
            frames[key]=pd.read_csv(z.open(filename),sep=r'\s+',header=None)
        elif key == 'concrete':
            filename=next(n for n in z.namelist() if n.lower().endswith('.xls'))
            frames[key]=pd.read_excel(io.BytesIO(z.read(filename)))
        elif key == 'energy':
            filename=next(n for n in z.namelist() if n.lower().endswith('.xlsx'))
            frames['energy_heating']=pd.read_excel(io.BytesIO(z.read(filename))).iloc[:,:9].dropna()
    info=[]
    for key,df in frames.items():
        original=len(df)
        df=df.drop_duplicates().reset_index(drop=True)
        y=df.iloc[:,-1].to_numpy(dtype=float)
        X=df.iloc[:,:-1]
        if key=='abalone':
            X=pd.get_dummies(X,columns=['sex'],dtype=float)
        X=X.to_numpy(dtype=float)
        assert np.isfinite(X).all() and np.isfinite(y).all()
        np.savez_compressed(ROOT/'data/processed'/f'{key}.npz', X=X,y=y)
        csvpath=ROOT/'data/processed'/f'{key}.csv'
        pd.DataFrame(np.c_[X,y],columns=[f'x{i}' for i in range(X.shape[1])]+['target']).to_csv(csvpath,index=False,float_format='%.17g')
        info.append({'dataset':key,'raw_rows':original,'unique_rows':len(y),'features':X.shape[1],
                     'duplicate_rows_removed':original-len(y),'csv_sha256':hashlib.sha256(csvpath.read_bytes()).hexdigest()})
    (ROOT/'data/source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(info).to_csv(ROOT/'data/dataset_summary.csv',index=False)
    print(pd.DataFrame(info).to_string(index=False))

if __name__=='__main__': main()
