# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from azur import rhyme as R

BAUDELAIRE_DERNIERS = ["équipage", "mers", "voyage", "amers"]


def test_genre_masculin_feminin():
    assert R.is_feminine("amère") is True
    assert R.is_feminine("équipage") is True
    assert R.is_feminine("voyage") is True
    assert R.is_feminine("amers") is False
    assert R.is_feminine("cher") is False
    # -és / -ées / -ès : masculin phonétique même si graphie en e
    assert R.is_feminine("étés") is False


def test_genre_insensible_accent_et_ponctuation():
    assert R.is_feminine("Amère,") is True
    assert R.is_feminine("AMERS") is False


def test_phonetic_tail_stable_par_forme():
    # la partie pertinente pour la rime = suffixe [-2:] (corps tronqué)
    assert R.phonetic_tail("amers", 3)[-2:] == R.phonetic_tail("mers", 3)[-2:]
    assert R.phonetic_tail("équipage", 3)[-2:] == R.phonetic_tail("voyage", 3)[-2:]


def test_richesse_croissante():
    # rime riche (>= 3) bat suffisante (2) bat pauvre (1)
    riche = R.richness("amers", "mers")
    pauvre = R.richness("amers", "navire")
    assert riche > pauvre
    assert R.richness("amers", "amers") >= 3


def test_richesse_positif_pour_rime():
    assert R.richness("amers", "chers") >= 1


def test_scheme_baudelaire_ABAB():
    sch = R.rhyme_scheme(BAUDELAIRE_DERNIERS)
    assert len(sch) == 4
    assert sch[0] == sch[2]      # équipage / voyage -> A
    assert sch[1] == sch[3]      # mers / amers -> B
    assert sch[0] != sch[1]
    assert sch == "ABAB"


def test_scheme_rimes_identiques_meme_label():
    assert R.rhyme_scheme(["amers", "mers"]) == "AA"


def test_scheme_pas_de_rime_labels_distincts():
    assert R.rhyme_scheme(["amers", "parking"]) == "AB"
