# prepare_real_dataset.py
# Organises the raw .npy semiconductor dataset into data/real/ layout.
# Usage: python scripts/prepare_real_dataset.py [--dry-run]
from __future__ import annotations
import argparse, random, shutil, sys
from pathlib import Path

DEFAULT_TRAIN_GT    = r'C:/Users/admin/Downloads/train/train/GT'
DEFAULT_TRAIN_NOISY = r'C:/Users/admin/Downloads/train/train/NoisyLR'
DEFAULT_TEST_NOISY  = r'C:/Users/admin/Downloads/Test_NoisyLR/NoisyLR'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT  = PROJECT_ROOT / 'data' / 'real'

def _is_junk(n): return n.startswith('._') or n in ('.DS_Store',)
def _npy(d): return sorted(p for p in Path(d).iterdir() if p.is_file() and p.suffix.lower()=='.npy' and not _is_junk(p.name))
def _cp(s,d): Path(d).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d)

def prepare(train_gt, train_noisy, test_noisy, out_root, val_split, seed, dry_run):
    train_gt=Path(train_gt); train_noisy=Path(train_noisy); test_noisy=Path(test_noisy); out_root=Path(out_root)
    print(f'[prepare] GT:    {train_gt}')
    print(f'[prepare] Noisy: {train_noisy}')
    print(f'[prepare] Test:  {test_noisy}')
    print(f'[prepare] Out:   {out_root}  val={val_split:.0%}  seed={seed}  dry={dry_run}')
    gts=_npy(train_gt); ns=_npy(train_noisy)
    gm={p.name:p for p in gts}; nm={p.name:p for p in ns}
    paired=sorted(set(gm)&set(nm))
    if not paired: sys.exit('[prepare] ERROR: No paired files found.')
    rng=random.Random(seed); sh=list(paired); rng.shuffle(sh)
    nv=max(1,round(len(sh)*val_split)); vset=set(sh[:nv])
    tr=[n for n in paired if n not in vset]; vl=sorted(vset)
    print(f'[prepare] Total={len(paired)} Train={len(tr)} Val={len(vl)}')
    c={'train':0,'val':0,'test':0}; sk=0
    for n in tr:
        dc=out_root/'train'/'clean'/n; dd=out_root/'train'/'degraded'/n
        if dc.exists() and dd.exists(): sk+=1; continue
        if not dry_run: _cp(gm[n],dc); _cp(nm[n],dd)
        c['train']+=1
    for n in vl:
        dc=out_root/'val'/'clean'/n; dd=out_root/'val'/'degraded'/n
        if dc.exists() and dd.exists(): sk+=1; continue
        if not dry_run: _cp(gm[n],dc); _cp(nm[n],dd)
        c['val']+=1
    tf=_npy(test_noisy); print(f'[prepare] Test={len(tf)} (no GT)')
    for src in tf:
        dd=out_root/'test'/'degraded'/src.name
        if dd.exists(): sk+=1; continue
        if not dry_run: _cp(src,dd)
        c['test']+=1
    tag='[DRY RUN] ' if dry_run else ''
    print(f'{tag}DONE train={c[chr(116)+chr(114)+chr(97)+chr(105)+chr(110)]} val={c[chr(118)+chr(97)+chr(108)]} test={c[chr(116)+chr(101)+chr(115)+chr(116)]} skipped={sk}')
    if not dry_run:
        for sp in ('train','val','test'):
            sd=out_root/sp
            if sd.exists():
                for sub in sorted(d.name for d in sd.iterdir() if d.is_dir()):
                    print(f'  {sp}/{sub}/ -> {len(list((sd/sub).glob(chr(42)+chr(46)+chr(110)+chr(112)+chr(121))))} files')

def main():
    p=argparse.ArgumentParser(description='Organise real semiconductor .npy dataset into data/real/')
    p.add_argument('--train-gt',    default=DEFAULT_TRAIN_GT)
    p.add_argument('--train-noisy', default=DEFAULT_TRAIN_NOISY)
    p.add_argument('--test-noisy',  default=DEFAULT_TEST_NOISY)
    p.add_argument('--out',         default=str(DEFAULT_OUT))
    p.add_argument('--val-split',   type=float, default=0.10)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--dry-run',     action='store_true')
    a=p.parse_args()
    prepare(a.train_gt, a.train_noisy, a.test_noisy, a.out, a.val_split, a.seed, a.dry_run)

if __name__=='__main__': main()
