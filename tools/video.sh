#!/bin/sh
# Renders an Electric Sheep style video with fleece (headless: no window).
#   tools/video.sh [-o OUT.mp4] [-s SECONDS] [-l LINES] [-q QUALITY] [-r FPS] [-S SEED] [-k KEYS]
# Defaults: Electric Sheep's 800x592 frame and 2000 samples per pixel; -l 1080
# renders 1920x1080 instead. KEYS are frame:key presses, e.g. "300: ,600:m"
# (space = next sheep, m = a mutation). The sheep's credit is burned in.
set -e
OUT="$HOME/Videos/fleece-demo.mp4"; SECS=60; LINES=""; Q=2000; FPS=30; SEED=""; KEYS=""
while getopts o:s:l:q:r:S:k: opt; do
  case $opt in
    o) OUT=$OPTARG;; s) SECS=$OPTARG;; l) LINES=$OPTARG;; q) Q=$OPTARG;; r) FPS=$OPTARG;;
    S) SEED=$OPTARG;; k) KEYS=$OPTARG;; *) exit 2;;
  esac
done
SIZE=""
[ -n "$LINES" ] && SIZE="$(( (LINES * 16 / 9 + 1) / 2 * 2 ))x$LINES"
BIN="$(dirname "$0")/../target/release/fleece"
[ -x "$BIN" ] || (cd "$(dirname "$0")/.." && cargo build --release)
"$BIN" --video "$OUT" --seconds "$SECS" --fps "$FPS" --quality "$Q" --credit $SIZE \
  ${SEED:+--seed "$SEED"} ${KEYS:+--keys "$KEYS"}
