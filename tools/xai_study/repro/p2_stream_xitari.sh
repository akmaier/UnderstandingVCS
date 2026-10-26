#!/usr/bin/env bash
# p2_stream_xitari.sh — xitari side of the Paper-2 stream validation.
#
# Reference build (once; upstream google-deepmind/xitari @ 4945fc6 + three edits):
#   git clone https://github.com/google-deepmind/xitari && cd xitari
#   common/Defaults.cpp : random_seed "time" -> "0"   (both places; Superchip RAM, bug_fix_log #122)
#   emucore/Random.cxx  : bool Random::ourSeeded = true;
#   common/Defaults.cpp : use_starting_actions = (getenv("XI_NO_STARTING_ACTIONS") == nullptr)
#   cmake . -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration -DHAVE_UNISTD_H"
#   make -j8                                       # -> libxitari.a
#   c++ -std=c++11 -O2 -I. -Iemucore -Iemucore/m6502/src -Iemucore/m6502/src/bspf/src \
#       -Icommon -Icontrollers -Ienvironment -Igames -Ios_dependent \
#       -o trace_min <repo>/tools/xai_study/repro/trace_min.cpp libxitari.a -lz
#
# Run (DIR holds p2_stream_dump.jl output):
#   tools/xai_study/repro/p2_stream_xitari.sh /path/to/trace_min DIR
#   python3 tools/xai_study/repro/p2_stream_check.py DIR --md tools/xai_study/repro/p2_stream_check.md
set -euo pipefail
TM=$1; DIR=$2
for meta in "$DIR"/*.meta.json; do
  g=$(basename "$meta" .meta.json)
  rom=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['rom'])" "$meta")
  # P2 boots each game with its real RomSettings (common/game_sets.jl::xai_settings_for),
  # i.e. WITH ALE's starting actions, as xitari does by default. (XI_NO_STARTING_ACTIONS=1
  # reproduces the pre-2026-10-26 generic boot, for validating the old records.)
  "$TM" "$rom" "$DIR/$g.actions.txt" > "$DIR/$g.xitari.jsonl"
  echo "xitari $g"
done
