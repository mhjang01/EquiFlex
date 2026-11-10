"""Replay the training callback's loss evaluation without suppressing exceptions.

This never fits, updates, or saves model parameters. The checkpoint is loaded
strictly, EMA parameters applied as in RMSDEvaluationCallback, and each system
is evaluated under torch.no_grad(). Autocast defaults off because the callback
itself does not establish an autocast context. Each exception includes the full
trace and is optionally retried with fewer timestep copies to distinguish
local VRAM limits from data/model errors.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
from equiflex.model.trainer import FlexTrainer
from equiflex.model.loader import FlexDataset, collate_single_system


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,default=Path('data/posebusters/featurized'))
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--retry-batch-size',type=int,default=1)
    p.add_argument('--autocast',action='store_true')
    p.add_argument('--systems-file',type=Path)
    p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=990131)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required to reproduce GPU evaluation failures.')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('high')
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=ckpt['hyper_parameters']['config']
    module=FlexTrainer(cfg)
    module.load_state_dict(ckpt['state_dict'],strict=True)
    shadow=ckpt.get('ema_state_dict',{}).get('shadow',{})
    with torch.no_grad():
        for name,param in module.model.named_parameters():
            if name in shadow:param.copy_(shadow[name])
    metadata={'checkpoint':str(a.checkpoint.resolve()),'sha256':digest,
              'global_step':ckpt.get('global_step'),'epoch':ckpt.get('epoch'),
              'ema_applied':bool(shadow),'torch':torch.__version__,
              'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
              'vram_gib':torch.cuda.get_device_properties(0).total_memory/1024**3,
              'batch_size':a.batch_size,'autocast':a.autocast,
              'seed':a.seed,'data_dir':str(a.data_dir.resolve())}
    del ckpt,shadow;gc.collect()
    module.eval().to('cuda')
    ids=None
    if a.systems_file:ids=a.systems_file.read_text().split()
    ds=FlexDataset(str(a.data_dir),system_ids=ids)
    metadata['systems']=len(ds) if a.limit is None else min(a.limit,len(ds))
    (a.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)
    # Record the last major region entered in addition to the Python traceback.
    stage={'name':'loading'}
    def mark(name,fn):
        def wrapped(*args,**kwargs):
            stage['name']=name
            return fn(*args,**kwargs)
        return wrapped
    module.model.forward_chemistry=mark('forward_chemistry',module.model.forward_chemistry)
    module.model.forward_flow=mark('forward_flow',module.model.forward_flow)
    module.loss_fn.register_forward_pre_hook(lambda *args:stage.update(name='loss_fn'))

    def attempt(sample,bsize):
        batch=None;losses=None
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        try:
            stage['name']='collation'
            batch=collate_single_system([sample]*bsize)
            batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            stage['name']='_shared_step'
            context=torch.autocast('cuda',dtype=torch.bfloat16) if a.autocast else nullcontext()
            with torch.no_grad(),context:losses=module._shared_step(batch)
            torch.cuda.synchronize()
            values={k:float(v.detach().cpu()) for k,v in losses.items() if isinstance(v,torch.Tensor) and v.numel()==1}
            return {'status':'ok','batch_size':bsize,'losses':values,
                    'seconds':time.monotonic()-started,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
        except Exception as e:
            result={'status':'error','batch_size':bsize,'stage':stage['name'],
                    'type':type(e).__name__,'message':str(e),
                    'traceback':traceback.format_exc(),'seconds':time.monotonic()-started,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
            return result
        finally:
            del batch,losses
            gc.collect();torch.cuda.empty_cache()

    results=[]
    with (a.output_dir/'systems.jsonl').open('w') as out:
        for i in range(metadata['systems']):
            sid=ds.system_dirs[i].name
            sample=None
            try:
                stage['name']='data_loading';sample=ds[i]
                result={'system':sid,'n_rec':sample['rec_atom_feats'].shape[0],
                        'n_lig':sample['lig_atom_feats'].shape[0],**attempt(sample,a.batch_size)}
                if result['status']=='error' and 0<a.retry_batch_size<a.batch_size:
                    result['smaller_batch_retry']=attempt(sample,a.retry_batch_size)
            except Exception as e:
                result={'system':sid,'status':'error','stage':stage['name'],
                        'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
            results.append(result);out.write(json.dumps(result)+'\n');out.flush()
            if result['status']=='error':
                print(f"ERROR {sid}: {result['type']}: {result['message'][:220]}",flush=True)
                print(result['traceback'],flush=True)
                retry=result.get('smaller_batch_retry')
                if retry:print('SMALLER_BATCH',sid,retry['status'],retry.get('message','')[:200],flush=True)
            elif (i+1)%20==0:print(f"PROGRESS {i+1}/{metadata['systems']} last={sid}",flush=True)
            del sample;gc.collect();torch.cuda.empty_cache()
    errors=[r for r in results if r['status']=='error']
    summary={**metadata,'completed':len(results),'errors':len(errors),
             'error_types':dict(Counter(r['type'] for r in errors)),
             'error_systems':[r['system'] for r in errors],
             'smaller_batch_still_fails':[r['system'] for r in errors if r.get('smaller_batch_retry',{}).get('status')=='error']}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
