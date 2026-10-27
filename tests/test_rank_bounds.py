"""Exhaustively test the sharp rank envelope, corrected rank, and perturbation bound."""
from pathlib import Path
import sys, itertools, json
from fractions import Fraction
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from calibration import conformal_rank, rank_floor, replay_rank, order_statistic

def main():
    precision=0
    for n in range(1,10001):
        for alpha in ["0.01","0.05","0.1","0.2","0.5","0.9"]:
            z=(n+1)*(1-Fraction(alpha))
            exact=(z.numerator+z.denominator-1)//z.denominator
            assert conformal_rank(n,float(alpha))==exact,(n,alpha)
            precision+=1
    checked=0; sharp=0
    for n in range(1,9):
        for cap in range(1,5):
            minima={}
            for v in itertools.product(range(1,cap+1),repeat=n):
                rows=np.repeat(np.arange(1,n+1),v)
                total=len(rows)
                for k,observed in enumerate(rows,1):
                    lower=rank_floor(k,n,total,cap)
                    assert observed>=lower,(n,cap,v,k,observed,lower)
                    minima[(total,k)]=min(minima.get((total,k),n+1),int(observed))
                    checked+=1
            for (total,k),val in minima.items():
                assert val==rank_floor(k,n,total,cap)
                sharp+=1
            for total in range(n,n*cap+1):
                for alpha in [.01,.05,.1,.2,.5,.9]:
                    r=conformal_rank(n,alpha)
                    k=replay_rank(n,total,cap,alpha)
                    if r>n: assert k==total+1
                    else:
                        assert rank_floor(k,n,total,cap)>=r
                        if k>1: assert rank_floor(k-1,n,total,cap)<r
    rng=np.random.default_rng(90017)
    for _ in range(1000):
        n=128;cap=8;s=rng.integers(0,30,n)/10;v=rng.integers(1,cap+1,n)
        rows=np.repeat(s,v);delta=.2
        noisy=rows+rng.uniform(-delta,delta,len(rows))
        k=replay_rank(n,len(rows),cap,.1)
        assert order_statistic(noisy,k)+delta+1e-12>=order_statistic(s,conformal_rank(n,.1))
    result={'exact_decimal_rank_checks':precision,'rank_inequalities_checked':checked,'sharp_cases_checked':sharp,'perturbation_cases':1000,'status':'passed'}
    out=Path(__file__).resolve().parents[1]/'results/theorem_verification.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
