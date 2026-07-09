'''
Run all diagnostic stats against v2_plinder_5A:
  1. duplicate_stats          -> duplicate_stats.json
  2. com_distance_stats       -> com_distance_stats.json
  3. pocket_cutoff_violations -> pocket_cutoff_violations.json
  4. check_features           -> feature_check.json
  5. check_atom_one_hots      -> atom_one_hot_check.json (5k train sample)

Each underlying module imports its own STORAGE_ROOT/NEW_ROOT. We monkey-patch
those to point at v2_plinder_5A and invoke main().
'''
import sys
sys.path.insert(0, '/home/mhjang/workspace/flex')

from pathlib import Path

ROOT = Path('/home/mhjang/workspace/flex/v2_plinder_5A')


def run(label, fn, argv):
    print('\n' + '=' * 70)
    print(f' {label}')
    print('=' * 70)
    saved = sys.argv
    sys.argv = argv
    try:
        fn()
    finally:
        sys.argv = saved


def main():
    import equiflex.dataprep.duplicate_stats as M1
    M1.STORAGE_ROOT = ROOT
    run('duplicate_stats', M1.main, ['duplicate_stats'])

    import equiflex.dataprep.com_distance_stats as M2
    M2.STORAGE_ROOT = ROOT
    run('com_distance_stats', M2.main, ['com_distance_stats'])

    import equiflex.dataprep.pocket_cutoff_violations as M3
    M3.STORAGE_ROOT = ROOT
    run('pocket_cutoff_violations', M3.main, ['pocket_cutoff_violations'])

    import equiflex.dataprep.check_features as M4
    M4.NEW_ROOT = ROOT
    run('check_features', M4.main, ['check_features'])

    import equiflex.dataprep.check_atom_one_hots as M5
    M5.NEW_ROOT = ROOT
    run('check_atom_one_hots', M5.main,
        ['check_atom_one_hots', '--sample', '5000'])

    print('\n=== ALL STATS DONE ===')
    for fname in ['duplicate_stats.json', 'com_distance_stats.json',
                  'pocket_cutoff_violations.json', 'feature_check.json',
                  'atom_one_hot_check.json']:
        fp = ROOT / fname
        print(f'  {fp}  {"OK" if fp.exists() else "MISSING"}')


if __name__ == '__main__':
    main()
