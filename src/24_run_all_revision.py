"""Run revision stages 20 -> 21 -> five Stage22 folds -> 23 sequentially.

"""
from __future__ import annotations
import argparse, subprocess, sys, time, traceback, json
from pathlib import Path
from datetime import datetime


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--skip-threshold',action='store_true',help='Only run Stage20 and Stage21; skip Stage22 and final full audit')
    ap.add_argument('--save-models',action='store_true',help='Optional large joblib RF model saves in Stage22')
    args=ap.parse_args()
    root=args.root.resolve();src=root/'src';logdir=root/'outputs'/'revision_run_logs';logdir.mkdir(parents=True,exist_ok=True)
    logfile=logdir/('revision_pipeline_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
    tasks=[('Stage 20 CSV statistical audit',['20_revision_existing_outputs.py'])]
    tasks.append(('Stage 21 features and exception ROIs',['21_revision_features_exceptions.py']))
    if not args.skip_threshold:
        for fold in range(1,6):
            cmd=['22_revision_threshold_sensitivity.py','--fold',str(fold)]
            if args.save_models:cmd+=['--save-model']
            tasks.append((f'Stage 22 threshold fold {fold}/5',cmd))
        tasks.append(('Stage 23 final scientific audit',['23_revision_final_audit.py']))
    print('PROJECT ROOT:',root)
    print('LOG FILE:',logfile)
    print('TASKS:',', '.join(n for n,_ in tasks))
    with logfile.open('a',encoding='utf-8',buffering=1) as log:
        for i,(name,command) in enumerate(tasks,1):
            args_=[sys.executable,'-u',str(src/command[0]),*command[1:],'--root',str(root)]
            banner=f'\n[{datetime.now().isoformat()}] TASK {i}/{len(tasks)} {name}\nCOMMAND {args_}\n'
            print(banner,flush=True);log.write(banner)
            start=time.monotonic()
            with subprocess.Popen(args_,cwd=str(root),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,encoding='utf-8',errors='replace') as child:
                for line in child.stdout:
                    print(line,end='',flush=True);log.write(line)
                rc=child.wait()
            if rc!=0:
                msg=f'PIPELINE STOPPED at {name}, exit={rc}, elapsed={time.monotonic()-start:.1f}s. Original data remain untouched. Check log: {logfile}'
                print(msg,flush=True);log.write(msg+'\n');return rc
            print(f'TASK PASS: {name} ({time.monotonic()-start:.1f}s)',flush=True)
        print('COMPUTATIONAL STAGES PASS. Outputs: outputs/revision_stage20..23',flush=True)
        log.write('ALL COMPUTATIONAL STAGES PASS\n')
    return 0

if __name__=='__main__':sys.exit(main())
