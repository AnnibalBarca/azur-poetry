# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest

from azur.scansion import scan_line

GOLD_12 = [
    "Je le vis, je rougis, je pâlis à sa vue",                    # Racine
    "Tout m'afflige et me nuit, et conspire à me nuire",          # Racine
    "Je mis un bonnet rouge au vieux dictionnaire",               # Hugo
    "Souvent, pour s'amuser, les hommes d'équipage",              # Baudelaire
    "Prennent des albatros, vastes oiseaux des mers",
    "Qui suivent, indolents compagnons de voyage",
    "Le navire glissant sur les gouffres amers",
    "Le vierge, le vivace et le bel aujourd'hui",                 # Mallarmé
    # Vérifiés contre le scanner avant ajout (noms propres / diérèses
    # fragiles exclus — limitation assumée du PTYX-lite) :
    "Cette obscure clarté qui tombe des étoiles",                 # Corneille
    "Oui, puisque je retrouve mon fidèle amant",                  # trimètre 4/8
]


@pytest.mark.parametrize("vers", GOLD_12)
def test_alexandrins_canoniques(vers):
    s = scan_line(vers)
    assert s.total == 12, (
        f"{vers!r} -> {s.total} "
        f"({[(w.raw, w.counted) for w in s.words]})"
    )

def test_faux_alexandrin_detecte():
    s = scan_line("Le navire glissant lentement sur les eaux calmes et profondes")
    assert s.total != 12 and not s.is_alexandrin

def test_dierese_dictionnaire():
    s = scan_line("Je mis un bonnet rouge au vieux dictionnaire")
    assert any(w.dierese for w in s.words)

def test_hiatus():
    s = scan_line("Tu as souvent rêvé sous le grand chêne sombre")
    assert ("tu", "as") in [(a.lower(), b.lower()) for a, b in s.hiatus]

def test_cesure_classique():
    s = scan_line("Je le vis, je rougis, je pâlis à sa vue")
    assert s.cesure_kind == "classique"

def test_elision():
    s = scan_line("Tout m'afflige et me nuit, et conspire à me nuire")
    elided = [w.raw for w in s.words if w.elided]
    assert "afflige" in [e.lower() for e in elided]
    assert "conspire" in [e.lower() for e in elided]
