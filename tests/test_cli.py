# -*- coding: utf-8 -*-
"""Tests du CLI AZUR (via subprocess pour couvrir l'entry point réel)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLI = os.path.join(ROOT, "azur_cli.py")
PY = sys.executable

V1 = "Laissez la nuit couver l'œuf bleu de l'angoisse"


def _run(args, stdin=None, env=None):
    return subprocess.run(
        [PY, CLI, *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
    )


def test_cli_texte_arg():
    r = _run([V1])
    assert r.returncode == 0
    assert "AZUR" in r.stdout
    assert "66.7" in r.stdout           # score reproductible du vers-test


def test_cli_json_valide():
    r = _run([V1, "--json"])
    assert r.returncode == 0
    out = json.loads(r.stdout)          # doit parser sans erreur
    assert "score" in out
    assert out["score"] == 66.7
    for k in ("F", "N", "PHI", "O", "D", "T_mean", "C", "scansion", "mots"):
        assert k in out
    assert out["scansion"][0]["syllabes"] == 11


def test_cli_stdin():
    r = _run([], stdin="Le navire glissant sur les gouffres amers\n")
    assert r.returncode == 0
    assert "AZUR" in r.stdout


def test_cli_judge_sans_cle_indisponible():
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)  # garantir l'absence de clé
    r = _run([V1, "--judge"], env=env)
    assert r.returncode == 0
    assert "--judge inactif" in r.stderr   # avertissement sur stderr
    assert "indisponible" not in r.stdout  # stdout laissé propre


def test_cli_fichier_introuvable_exit_1():
    r = _run(["-f", "/chemin/inexistant/azur.txt"])
    assert r.returncode == 1
    assert "impossible de lire" in r.stderr


def test_cli_aucun_argument_lit_stdin_vide():
    # pas d'arg, pas de -f : stdin lu (vide ici) -> ne crash pas
    r = _run([], stdin="")
    assert r.returncode == 0
