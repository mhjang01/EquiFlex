#!/usr/bin/env bash
# Apply the Uni-Mol v2 padding-invariance patch to the unimol_tools install of
# the active Python (see scripts/unimol_padding.patch for what and why).
#
#   scripts/patch_unimol.sh            apply (no-op if already applied)
#   scripts/patch_unimol.sh --revert   restore stock unimol_tools
#   scripts/patch_unimol.sh --verify [cpu]
#                                      apply, then embed a padded batch with the
#                                      1.1B model (on cpu if asked) and compare
#                                      with standalone
#
# PYTHON=/path/to/python selects another interpreter (default: python on PATH).

here=$(cd "$(dirname "$0")" && pwd)
patch_file="$here/unimol_padding.patch"
python=${PYTHON:-python}
mode=${1:-apply}

pkg=$("$python" -c 'import pathlib, unimol_tools; print(pathlib.Path(unimol_tools.__file__).parent)')
version=$("$python" -c 'import importlib.metadata as m; print(m.version("unimol_tools"))')
marker="PADDING_INVARIANT = True"
echo "unimol_tools $version at $pkg"

is_patched() { grep -q "^$marker" "$pkg/models/transformersv2.py"; }

case "$mode" in
  --revert)
    if ! is_patched; then echo "not patched, nothing to revert"; exit 0; fi
    patch -d "$pkg" -p1 -R --no-backup-if-mismatch < "$patch_file"
    find "$pkg/models" -name __pycache__ -type d -exec rm -rf {} +
    echo "reverted to stock unimol_tools"
    exit 0 ;;
  apply|--verify) ;;
  *) echo "usage: $0 [--revert|--verify]" >&2; exit 2 ;;
esac

if is_patched; then
  echo "already patched"
else
  if [[ "$version" != "0.1.5" ]]; then
    echo "warning: patch was written against unimol_tools 0.1.5; trying anyway" >&2
  fi
  # dry run first so a partial application never leaves a half-patched install
  patch -d "$pkg" -p1 --dry-run --no-backup-if-mismatch < "$patch_file" > /dev/null
  patch -d "$pkg" -p1 --no-backup-if-mismatch < "$patch_file"
  find "$pkg/models" -name __pycache__ -type d -exec rm -rf {} +
  echo "patched"
fi
"$python" -c 'from unimol_tools.models import transformersv2; assert transformersv2.PADDING_INVARIANT'

if [[ "$mode" == "--verify" ]]; then
  VERIFY_DEVICE=${2:-cuda} "$python" - <<'PY'
import os
import numpy as np
from equiflex.prep.featurize import UniMolExtractor
ext = UniMolExtractor(model_size='1.1B', use_cuda=os.environ['VERIFY_DEVICE'] != 'cpu')
smi, big = 'N=C1N/C(=N\\Nc2ccccc2)c2ccccc21', 'O=C(N[C@@H]1C[C@H]1c1ccccc1)c1cc(Cl)cc(COc2cnc3[nH]ccc3c2)c1'
def rep(data):
    return [np.asarray(a) for a in ext.model.get_repr(data=data, return_atomic_reprs=True)['atomic_reprs']]
alone = rep([smi])[0]
padded = rep([smi, big])[0]
diff = float(np.abs(alone - padded).max())
print(f'padded vs standalone max|diff| = {diff:.2e} (mean|x| = {np.abs(alone).mean():.2f}); stock gives ~1e2')
assert diff < 1e-2, 'padding still leaks'
print('verified')
PY
fi
