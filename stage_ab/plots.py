"""Plot and CSV export for completed Stage A/B runs (no invented data)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def export(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=Path(directory)
    p=json.loads((directory/'results.json').read_text())
    files=[]
    def save(fig,name):
        path=directory/name
        fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig);files.append(str(path))
    if p['stage']=='A':
        records=[]
        for split,policies in p['results'].items():
            fig,ax=plt.subplots(figsize=(7,5))
            for policy,rows in policies.items():
                runtime=np.mean([r['runtime_sec'] for r in rows])*1000
                error=np.mean([r['error'] for r in rows])
                ax.scatter(runtime,error,label=policy)
                records.append({'split':split,'policy':policy,'mean_error':error,
                                'mean_runtime_ms':runtime,'acceptance':np.mean([r['acceptance'] for r in rows]),
                                'speedup':rows[0]['cohort_speedup']})
            ax.set(xlabel='Runtime per trajectory [ms]',ylabel='Final discrete L1 error',title=f'Stage A: {split.upper()}')
            ax.set_xscale('log');ax.legend(fontsize=8)
            save(fig,f'accuracy_runtime_{split}.png')
        fig,ax=plt.subplots(figsize=(7,4))
        for split,rows in p['certified_threshold_sweep'].items():
            ax.plot([r['step_tolerance'] for r in rows],[r['acceptance'] for r in rows],marker='o',label=split)
        ax.set(xlabel='Certified local-error tolerance',ylabel='Neural acceptance fraction',title='Stage A: certified gate')
        ax.set_xscale('log');ax.legend();save(fig,'certified_acceptance.png')
    else:
        records=[]
        for case,data in p['results'].items():
            for policy,row in data['policies'].items():
                records.append({'case':case,'role':data['role'],'policy':policy,
                                'runtime_ms':1000*row['runtime_sec'],
                                'speedup':row['speedup_vs_fastest_classical'],
                                'acceptance':row['acceptance'],
                                'max_scaled_error':row['max_scaled_state_error'],
                                'ignition_delay_s':row['ignition']['temperature_threshold_delay_s'],
                                'ignition_delay_error_s':row['ignition_delay_absolute_error_s'],
                                'ignition_classification_correct':row['ignition_classification_correct']})
        representative=next(k for k in p['results'] if p['results'][k]['role']=='test_id')
        data=p['results'][representative];t=np.array(data['times_s'])*1e6
        names=p['mechanism']['species']
        for variable,ylabel in [('T','Temperature [K]'),('OH','OH mass fraction'),('p','Pressure [Pa]'),('q','Enthalpy heat release [W/m3]')]:
            fig,ax=plt.subplots(figsize=(7,4))
            for policy in ['always_solver','always_neural','residual','consistency']:
                row=data['policies'][policy]
                if variable=='T':y=np.array(row['trajectory'])[:,0]
                elif variable=='OH':y=np.array(row['trajectory'])[:,1+names.index('OH')]
                elif variable=='p':y=row['pressure_Pa']
                else:y=[v['enthalpy_heat_release_W_m3'] for v in row['heat_release']]
                ax.plot(t,y,label=policy)
            ax.set(xlabel='Time [microseconds]',ylabel=ylabel,title=f'Stage B: {representative}')
            ax.legend(fontsize=8);save(fig,f'trajectory_{variable}.png')
        fig,ax=plt.subplots(figsize=(7,4))
        for policy in ['residual','consistency','uncertainty']:
            rows=data['policies'][policy]['rows']
            ax.step([r['time']*1e6 for r in rows],[int(not r['accept']) for r in rows],where='post',label=policy)
        ax.set(xlabel='Time [microseconds]',ylabel='Fallback (1) / neural (0)',title='Stage B: fallback decisions')
        ax.legend();save(fig,'fallback_history.png')
        fig,ax=plt.subplots(figsize=(7,5))
        for policy in ['always_neural','physical','residual','consistency','uncertainty']:
            rows=[r for r in records if r['policy']==policy]
            ax.scatter([r['speedup'] for r in rows],[r['max_scaled_error'] for r in rows],label=policy)
        ax.set(xlabel='Speedup vs fastest classical baseline',ylabel='Max scaled state error',title='Stage B: accuracy / runtime')
        ax.set_xscale('log');ax.legend(fontsize=8);save(fig,'accuracy_runtime.png')
    with (directory/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    args=parser.parse_args();print(json.dumps(export(args.directory),indent=2))


if __name__=='__main__':main()
