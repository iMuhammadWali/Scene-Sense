"""Usage: python scripts/fetch_charades.py <test.parquet> <videos_dir>   (downloads only the needed videos; resumable)"""
import sys, os, time, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from remotezip import RemoteZip
ann, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
vids = sorted(pd.read_parquet(ann)['video'].unique())
URL='https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip'
done=0; t=time.time()
with RemoteZip(URL) as z:
    for v in vids:
        dst=os.path.join(out,v)
        if os.path.exists(dst) and os.path.getsize(dst)>0: done+=1; continue
        for attempt in range(3):
            try:
                data=z.read('Charades_v1_480/'+v)
                open(dst+'.part','wb').write(data); os.replace(dst+'.part',dst); break
            except Exception as e:
                print('retry',v,e,flush=True); time.sleep(3)
        done+=1
        if done%50==0: print(f'{done}/{len(vids)} {time.time()-t:.0f}s',flush=True)
print('DONE',done,flush=True)
