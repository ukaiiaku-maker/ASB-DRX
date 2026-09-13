#!/usr/bin/env python3
from __future__ import annotations
import csv, math, re, sys
from pathlib import Path

def f(row, key, default=float('nan')):
    try:
        v=row.get(key,'')
        if v is None or v=='': return default
        return float(v)
    except Exception:
        return default

def read_rows(p):
    q=p/'drx_v25_restart_asb_diagnostics.csv'
    if not q.exists(): return []
    with q.open(newline='') as fh: return list(csv.DictReader(fh))

def infer(p):
    m=re.search(r'(?:(asb_only|drx_isothermal|coupled)_)?rate_([^_/]+).*?_seed(\d+)',p.name)
    return (m.group(1) or 'coupled', m.group(2).replace('p','.'), m.group(3)) if m else ('','', '')

def summarize_case(p):
    rows=read_rows(p); branch,rate,seed=infer(p)
    out=dict(case=str(p),branch=branch,rate=rate,seed=seed,n_diag=len(rows))
    if not rows:
        out['status']='missing_diag'; return out
    r0,r1=rows[0],rows[-1]
    max_haz=max(f(r,'grain_hazard_births',0) for r in rows)
    max_haz_step=max(f(r,'grain_hazard_step_births',0) for r in rows)
    max_top=max(f(r,'grain_topology_births',0) for r in rows)
    max_cand=max(f(r,'nuc_candidate_active',0) for r in rows)
    max_cand_age=max(f(r,'nuc_candidate_age_max',0) for r in rows)
    max_Tstd=max(f(r,'asb_T_std',f(r,'T_std',0)) for r in rows)
    min_rhc=min(f(r,'asb_rho_hot_over_cold',1) for r in rows)
    min_Trho=min(f(r,'asb_corr_T_logrho',f(r,'asb_corr_T_rho',1)) for r in rows)
    max_T=max(f(r,'T_max',0) for r in rows)
    max_stress=max(f(r,'sigma_MPa',0) for r in rows)
    eps=f(r1,'eps_pct',f(r1,'eps',float('nan')))
    drx=max_haz>0.5 or max_haz_step>0.5
    cand=max_cand>0.5
    asb=(max_Tstd>=5.0) or (min_rhc<0.9) or (min_Trho<-0.15)
    if drx and asb: cls='persistent_DRX+ASB'
    elif drx: cls='persistent_DRX'
    elif cand and asb: cls='candidate_DRX+ASB'
    elif cand: cls='candidate_DRX_only'
    elif asb: cls='ASB_only'
    else: cls='neither'
    out.update(status='ok',classification=cls,final_step=f(r1,'step'),eps_pct_final=eps,
               n_grains_final=f(r1,'n_grains'),grain_hazard_births_max=max_haz,
               grain_topology_births_max=max_top,nuc_candidate_active_max=max_cand,
               nuc_candidate_age_max=max_cand_age,T_max=max_T,asb_T_std_max=max_Tstd,
               asb_rho_hot_over_cold_min=min_rhc,asb_corr_T_logrho_min=min_Trho,
               stress_peak_MPa=max_stress)
    return out

def main():
    if len(sys.argv)<2:
        print('usage: summarize_v34_drx_asb_rate_sweep.py <root>'); sys.exit(2)
    root=Path(sys.argv[1])
    cases=sorted(p for p in root.iterdir() if p.is_dir() and 'rate_' in p.name)
    rows=[summarize_case(p) for p in cases]
    outcsv=root/'v34_drx_asb_summary.csv'
    if rows:
        fields=sorted({k for r in rows for k in r})
        with outcsv.open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
    counts={}
    for r in rows: counts[r.get('classification',r.get('status','unknown'))]=counts.get(r.get('classification',r.get('status','unknown')),0)+1
    txt=root/'v34_drx_asb_summary.txt'
    txt.write_text(f'root: {root}\nn_cases: {len(rows)}\nclassification_counts: {counts}\ncsv: {outcsv}\n')
    print(txt.read_text())
if __name__=='__main__': main()
