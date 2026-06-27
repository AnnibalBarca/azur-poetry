#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI AZUR.

Usage :
  python3 azur_cli.py "Laissez la nuit couver l'œuf bleu de l'angoisse"
  python3 azur_cli.py -f poeme.txt
  python3 azur_cli.py -f poeme.txt --judge      # ajoute le juge LLM (API key)
  echo "..." | python3 azur_cli.py
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from azur.scorer import score, render
from azur.judge import judge


def main():
    ap = argparse.ArgumentParser(description="AZUR — note de perfection poétique")
    ap.add_argument("text", nargs="?", help="vers ou poème (sinon -f / stdin)")
    ap.add_argument("-f", "--file", help="fichier texte à noter")
    ap.add_argument("--judge", action="store_true",
                    help="active le juge LLM (ANTHROPIC_API_KEY requis)")
    ap.add_argument("--json", action="store_true", help="sortie JSON")
    args = ap.parse_args()

    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    rep = score(text)
    if args.json:
        import json
        out = {k: getattr(rep, k) for k in
               ("score", "F", "N", "PHI", "O", "D", "T_mean", "C",
                "delta_mean", "lyric_mean", "instrumental", "cliches",
                "echoes", "notes")}
        out["scansion"] = [{"vers": s.text, "syllabes": s.total,
                            "cesure": s.cesure_kind,
                            "hiatus": s.hiatus} for s in rep.scansion]
        out["mots"] = [{"mot": w.word, "T": round(w.T, 3),
                        "delta": round(w.delta, 3),
                        "lyrique": round(w.lyric, 3), "regime": w.regime}
                       for w in rep.words]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(render(rep))
    if args.judge:
        j = judge(text)
        print("\n[juge LLM]", j if j else "indisponible (clé API ou réseau)")


if __name__ == "__main__":
    main()
