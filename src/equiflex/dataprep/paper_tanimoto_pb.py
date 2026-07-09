import argparse, glob, json, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = Path('/mnt/drive/plinder/2024-06/v2')
PB = Path('/home/mhjang/workspace/flex/prep/posebusters_pdb_ccd_ids.txt')
ANNOT = '/mnt/drive/new_plinder/v2/index/annotation_table.parquet'

def pb_systems():
    pairs={(x.split('_',1)[0].lower(),x.split('_',1)[1].upper()) for x in PB.read_text().splitlines() if x}
    a=pq.read_table(ANNOT,columns=['system_id','entry_pdb_id','ligand_ccd_code']).to_pandas(); a['p']=a.entry_pdb_id.str.lower(); a['c']=a.ligand_ccd_code.astype(str).str.upper()
    return set(a[[(p,c) in pairs for p,c in zip(a.p,a.c)]].system_id)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--split-csv'); ap.add_argument('--output',required=True); ap.add_argument('--label',required=True); args=ap.parse_args()
    split=pd.read_csv(args.split_csv) if args.split_csv else pq.read_table('/mnt/drive/plinder/2024-06/v0/splits/plinder-pl50.parquet',columns=['system_id','split']).to_pandas()
    train=set(split[split.split=='train'].system_id); pb=pb_systems()
    l=pq.read_table(ROOT/'fingerprints/ligands_per_system.parquet',columns=['system_id','number_id_by_inchikeys']).to_pandas().dropna(); l.number_id_by_inchikeys=l.number_id_by_inchikeys.astype('int64')
    tl=set(l[l.system_id.isin(train)].number_id_by_inchikeys); pl=set(l[l.system_id.isin(pb)].number_id_by_inchikeys)
    con=duckdb.connect(); con.execute('PRAGMA threads=16'); con.execute("PRAGMA memory_limit='24GB'"); con.execute('create temp table tl(id bigint)'); con.executemany('insert into tl values (?)',[(int(x),) for x in tl]); con.execute('create temp table pl(id bigint)'); con.executemany('insert into pl values (?)',[(int(x),) for x in pl])
    parts=[]; shards=sorted(glob.glob(str(ROOT/'ligand_scores/*.parquet')))
    for direction in ('forward','reverse'):
      for i,s in enumerate(shards,1):
        st=time.monotonic(); print(f'[{args.label} {direction} {i}/{len(shards)}] {Path(s).name}',flush=True)
        if direction=='forward': q='target_ligand_id as pb_id'; w='query_ligand_id in (select id from tl) and target_ligand_id in (select id from pl)'
        else: q='query_ligand_id as pb_id'; w='query_ligand_id in (select id from pl) and target_ligand_id in (select id from tl)'
        d=con.execute(f"select {q}, max(tanimoto_similarity_max) sim from read_parquet('{s}') where {w} group by pb_id").fetchdf(); parts.append(d); print(f'  {len(d)} PB ligands in {time.monotonic()-st:.1f}s',flush=True)
    m=pd.concat(parts).groupby('pb_id',as_index=False).sim.max(); ligsys=l[l.system_id.isin(pb)].groupby('system_id').number_id_by_inchikeys.apply(set)
    mp=dict(zip(m.pb_id,m.sim)); v=np.array([max((mp.get(x,0) for x in ids),default=0) for ids in ligsys],float)
    out={'label':args.label,'n_train':len(train),'n_pb_systems':len(pb),'n_pb_ligands':len(pl),'fraction_ge_30':float((v>=30).mean()),'count_ge_30':int((v>=30).sum()),'mean':float(v.mean()),'max':float(v.max())}; Path(args.output).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
