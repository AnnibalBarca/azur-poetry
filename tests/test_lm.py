# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from azur import lm as LM
from azur import semantics as S


def test_rarity_ultra_frequent_faible():
    assert LM.rarity("le") < 0.1
    assert LM.rarity("et") < 0.1


def test_rarity_rare_eleve():
    assert LM.rarity("albatros") > 0.5


def test_rarity_borne_01():
    for w in ("le", "maison", "albatros", "chosquilbertz"):
        r = LM.rarity(w)
        assert 0.0 <= r <= 1.0


def test_tension_idiome_plafonne():
    seg = S.Segment(["sent", "la", "poisse"], regime="cliché")
    assert LM.tension(seg, []) == 0.12


def test_tension_stopword_nulle():
    seg = S.Segment(["la"])
    assert LM.tension(seg, []) == 0.0


def test_tension_distance_locale():
    # « œuf » seul : tension faible ; « œuf » loin de « table » : plus tendu
    seg_proche = S.Segment(["œuf"])
    t_proche = LM.tension(seg_proche, ["couver"])   # même champ (gestation)
    t_loin = LM.tension(seg_proche, ["parking"])    # aucun lien
    assert t_loin > t_proche


def test_tension_borne_01():
    for w in ("nuit", "soupe", "albatros", "bleu"):
        seg = S.Segment([w])
        assert 0.0 <= LM.tension(seg, ["nuit"]) <= 1.0


def test_declic_nul_sans_relief():
    # mot sans aucune connexion au reste du poème -> Δ = 0
    assert LM.declic("parking", 0.9, ["nuit", "œuf", "bleu"]) == 0.0


def test_declic_positif_avec_relief():
    d = LM.declic("couver", 0.8, ["nuit", "œuf", "angoisse"])
    assert 0.0 < d <= 0.8          # = T × relief, relief <= 1


def test_wordfreq_backend_interface():
    b = LM.WordfreqBackend()
    assert b.name == "wordfreq-unigram"
    assert 0.0 <= b.surprisal("nuit", []) <= 1.0


def test_ngram_backend_interface():
    corpus = [
        "le chat mange la soupe",
        "la nuit tombe sur la mer",
        "le navire glisse sur les flots",
    ]
    b = LM.NGramBackend(corpus)
    assert b.name == "ngram-prose"
    # surprisal dans [0,1] qu'il y ait bigramme vu ou non
    s_vu = b.surprisal("soupe", ["mange"])
    s_inconnu = b.surprisal("albatros", ["vastes"])
    assert 0.0 <= s_vu <= 1.0
    assert 0.0 <= s_inconnu <= 1.0
