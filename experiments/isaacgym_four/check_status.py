import argparse,json,os,time
from pathlib import Path
from datetime import datetime,timezone
parser=argparse.ArgumentParser(description='Read live status and the first five-minute rewards.')
parser.add_argument('manifest',type=Path)
a=parser.parse_args()
b=a.manifest.resolve().parent
results=[]
for m in json.loads(a.manifest.read_text()):
 p=Path(m['run_dir']);r={'key':m['key'],'gpu':m['gpu']}
 if (p/'worker.pid').exists():
  pid=int((p/'worker.pid').read_text());r['pid']=pid;r['alive']=Path('/proc/'+str(pid)).exists()
 for name in ['exit_status.txt','supervisor_exit_status.txt']:
  if (p/name).exists():r[name]=(p/name).read_text().strip()
 if (p/'training_started.json').exists():
  start=json.loads((p/'training_started.json').read_text())['unix_time'];r['training_wall_seconds']=round(time.time()-start,1)
  for line in (p/'worker.log').read_text().splitlines():
   if '[campaign] PROGRESS ' not in line:continue
   metrics=json.loads(line.split('[campaign] PROGRESS ',1)[1])
   elapsed=datetime.fromisoformat(metrics['updated_utc']).timestamp()-start
   if elapsed>=300:
    results.append(dict(key=m['key'],gpu=m['gpu'],wandb_url=m['wandb_url'],wall_seconds=elapsed,**metrics));break
 else:r['stage']='initializing'
 if (p/'progress.json').exists():
  progress=json.loads((p/'progress.json').read_text());r.update({k:progress.get(k) for k in ['train/epoch','train/env_steps','train/episode_reward','train/success_tolerance']})
  r['progress_age_seconds']=round(time.time()-datetime.fromisoformat(progress['updated_utc']).timestamp(),1)
 print(json.dumps(r),flush=True)
if results:
 (b/'five_minute_rewards.json').write_text(json.dumps(results,indent=2)+'\n')
 print('FIVE_MINUTE_SNAPSHOTS',len(results),flush=True)
