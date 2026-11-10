#!/usr/bin/env bash
# Build the smina binary used by scripts/vina_rescore.py --scoring vinardo.
#
# The smina sources target Open Babel 2.x; scripts/smina_ob3_compat.h is
# force-included so they compile against Open Babel 3 and a current Boost. The
# build runs in a scratch copy, so the source tree only gains the binary.
#
#   scripts/build_smina.sh [SMINA_SRC] [OUT]
#
# SMINA_SRC defaults to ~/references/dock/smina, OUT to
# SMINA_SRC/build/linux/release/smina. OPENBABEL_PREFIX (default /usr/local)
# must hold include/openbabel3 and lib/libopenbabel.so.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
src=$(realpath "${1:-$HOME/references/dock/smina}")
out=${2:-$src/build/linux/release/smina}
ob=${OPENBABEL_PREFIX:-/usr/local}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$src" "$work/smina"
rm -f "$work/smina/build/linux/release/"*.o

make -C "$work/smina/build/linux/release" -j"$(nproc)" smina \
  CC="g++ -std=c++14 -fpermissive -O3 -DNDEBUG -w -include $here/smina_ob3_compat.h -I$ob/include/openbabel3" \
  LDFLAGS="-L$ob/lib -Wl,-rpath,$ob/lib"

install -m 755 "$work/smina/build/linux/release/smina" "$out"
echo "installed $out"
"$out" --version
