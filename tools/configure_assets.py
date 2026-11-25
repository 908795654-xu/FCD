"""Plan or create directory links to existing assets; never copy or overwrite."""
from pathlib import Path
import argparse
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--model-cache',type=Path)
    ap.add_argument('--apply',action='store_true')
    args=ap.parse_args()
    data=args.data_root.expanduser().resolve()
    if not data.is_dir():raise FileNotFoundError(data)
    for name in ('ETTh1.csv','weather.csv','electricity.csv'):
        if not (data/name).is_file():raise FileNotFoundError(data/name)
    projects=[ROOT/'experiments/frozen'/name for name in ('SACQ_TSFM',)]
    links=[(ROOT/'dataset',data)]+[(p/'datasets',data) for p in projects]
    if args.model_cache:
        cache=args.model_cache.expanduser().resolve()
        if not cache.is_dir():raise FileNotFoundError(cache)
        for name in ('ttm-512-96-ft-r2.1','ttm-512-720-r2','chronos-bolt-mini','timer-base-84m'):
            if not (cache/name).is_dir():raise FileNotFoundError(cache/name)
        links += [(p/'models_cache',cache) for p in projects]
    for dest,target in links:
        if dest.exists() or dest.is_symlink():
            if dest.resolve()!=target:raise FileExistsError(f'Will not replace {dest}')
    print(json.dumps([{'link':str(d),'target':str(t),'already_present':d.exists()} for d,t in links],indent=2))
    if args.apply:
        for dest,target in links:
            if not dest.exists():dest.symlink_to(target,target_is_directory=True)
        print('ASSET_LINKS_CREATED: existing targets untouched')
    else:print('DRY_RUN_ONLY: add --apply to create these links')

if __name__=='__main__':main()
