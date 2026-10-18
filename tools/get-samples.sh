#!/usr/bin/env bash
# Fetches and prepares the real-instrument samples for fleece's generated music (ska and
# mariachi). fleece uses its synthesiser for anything missing, so all of this is optional.
#
#   tools/get-samples.sh [DEST]        DEST defaults to ~/.local/share/fleece/samples
#
# It fetches:
#   - MuseScore General 0.2, a General MIDI SoundFont (MuseScore_General.sf2 and its License,
#     Readme and Sample_Sources.csv) from ftp.osuosl.org: MIT license (FluidR3 by Frank Wen and
#     others; its notices must travel with it). Used for guitars, organs, bass, piano and drums.
#   - Folders of Versilian Studios' VS Chamber Orchestra 2: Community Edition
#     (github.com/sgossner/VSCO-2-CE) and Versilian Community Sample Library
#     (github.com/sgossner/VCSL), both CC0 1.0 (public domain), as sparse, shallow, blobless git
#     clones pinned to a commit: violins, violin, trumpet, trombone, contrabass, harp, tenor sax,
#     guiro, shakers, claps and cajon.
# and writes into DEST: the SoundFont files; multi/<instrument>/<articulation>/*.wav (48 kHz,
# 16-bit, trimmed); manifest.tsv (what the sampler reads); sources.tsv (each file's origin);
# SOURCES.md; and licenses/ (each source's license and readme, verbatim).
#
# Disk space: about 515 MB in DEST (216 MB SoundFont, 299 MB multi/), plus about 0.9 GB
# of git clones in the cache (FLEECE_SAMPLES_SRC, default ~/.cache/fleece/samples-src), which
# is only needed to rebuild multi/ and can be deleted afterwards.
#
# Needs bash, git (2.27 or newer), curl, ffmpeg with ffprobe, and python3 (3.8 or newer,
# standard library only). Idempotent: what is present and valid is skipped. Set
# FLEECE_SAMPLES_FORCE=1 to rebuild multi/ anyway.
set -euo pipefail

DEST=${1:-$HOME/.local/share/fleece/samples}
SRC=${FLEECE_SAMPLES_SRC:-$HOME/.cache/fleece/samples-src}
HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/samples_manifest.py"
SF_URL=https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General
# MuseScore General 0.2 (files dated 2020-07-10): name, size, sha256
SF_FILES=(
  "MuseScore_General.sf2 215614036 ee51d2c4b1525e70f19a45909c4fd7a2e26d91d115fa89dbf5a6bc413d8b9bf3"
  "MuseScore_General_License.md 3419 5ad8d737e13c7f01f5b9674872a82a92b4ba253603e8ed14b9db12293550b4b9"
  "MuseScore_General_Readme.md 5161 e4ee85d097cda49a7926ea73e59ea8b7bb90d6ad715f82d20c6bfddf5dcfeb4c"
  "MuseScore_General_Sample_Sources.csv 12817 cbec757614fa47d2ba71a2f1276bf010918261d94987cfb678352e9755e9bdd4"
)

say() { printf '%s\n' "$*"; }
die() { printf 'get-samples: %s\n' "$*" >&2; exit 1; }
helper() { python3 -I "$HELPER" "$@"; }

for t in git curl ffmpeg ffprobe python3 sha256sum; do
  command -v "$t" >/dev/null || die "needs $t"
done
python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' || die "needs python3 3.8 or newer"
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
say "samples directory: $DEST"

# --- MuseScore General ------------------------------------------------------------------------
sf_valid() { # file size sha256
  [ -f "$1" ] && [ "$(stat -c %s "$1")" = "$2" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$3" ]
}
progress=(-sS)
[ -t 2 ] && progress=(--progress-bar)
for entry in "${SF_FILES[@]}"; do
  read -r name size sum <<<"$entry"
  if sf_valid "$DEST/$name" "$size" "$sum"; then
    say "present: $name"
    continue
  fi
  say "downloading $name ($size bytes) from $SF_URL"
  curl -fL --retry 3 "${progress[@]}" -o "$DEST/$name.part" "$SF_URL/$name"
  sf_valid "$DEST/$name.part" "$size" "$sum" || { rm -f "$DEST/$name.part"; die "$name: wrong size or sha256"; }
  mv "$DEST/$name.part" "$DEST/$name"
done
[ "$(head -c 4 "$DEST/MuseScore_General.sf2")" = RIFF ] \
  && [ "$(dd if="$DEST/MuseScore_General.sf2" bs=1 skip=8 count=4 2>/dev/null)" = sfbk ] \
  || die "MuseScore_General.sf2 is not a RIFF sfbk file"

# --- the multi-samples ------------------------------------------------------------------------
if [ "${FLEECE_SAMPLES_FORCE:-0}" != 1 ] && helper uptodate "$DEST"; then
  say "present: multi/ and manifest.tsv (up to date with tools/samples_manifest.py)"
else
  mkdir -p "$SRC"
  while IFS=$'\t' read -r name url commit <&3; do
    repo="$SRC/$name"
    mapfile -t folders < <(helper folders "$name")
    if [ -d "$repo/.git" ]; then
      [ "$(git -C "$repo" config --get remote.origin.url)" = "$url" ] \
        || die "$repo is not a clone of $url; move it away"
    elif [ -e "$repo" ]; then
      die "$repo exists and is not a git clone; move it away"
    else
      say "cloning $url (blobless, shallow, sparse) into $repo"
      git clone -q --filter=blob:none --sparse --depth 1 --no-checkout "$url" "$repo"
    fi
    if [ "$(git -C "$repo" rev-parse -q --verify HEAD 2>/dev/null)" != "$commit" ] \
       || [ "$(git -C "$repo" sparse-checkout list | sort)" != "$(printf '%s\n' "${folders[@]}" | sort)" ] \
       || [ -n "$(git -C "$repo" status --porcelain)" ]; then
      say "checking out ${#folders[@]} folders of $name at ${commit:0:12}"
      git -C "$repo" sparse-checkout set --cone "${folders[@]}"
      git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null \
        || git -C "$repo" fetch -q --depth 1 --filter=blob:none origin "$commit"
      git -C "$repo" -c advice.detachedHead=false checkout -q --force --detach "$commit"
    else
      say "present: $name at ${commit:0:12}"
    fi
  done 3< <(helper repos)
  # (the helper checks that every file is audio and not a Git LFS pointer: neither repository
  # uses LFS today)
  helper build "$SRC" "$DEST"
fi

helper summary "$DEST"
say "multi/: $(du -sh "$DEST/multi" | cut -f1); $DEST: $(du -sh "$DEST" | cut -f1)"
