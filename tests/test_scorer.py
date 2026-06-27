# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from azur.scorer import score

V1 = "Laissez la nuit couver l'œuf bleu de l'angoisse"
V2 = "Et mange ta soupe qui sent la poisse"

def test_v1_domine_v2():
    r1, r2 = score(V1), score(V2)
    assert r1.score > 2 * r2.score

def test_v2_instrumental_et_cliche():
    r2 = score(V2)
    assert r2.instrumental
    assert any("poisse" in c for c in r2.cliches)
    assert r2.T_mean < 0.2

def test_v1_profil_dents_de_scie():
    """Pic prospectif sur 'couver', retombée sur 'œuf' (récupération locale)."""
    r1 = score(V1)
    t = {w.word: w.T for w in r1.words}
    assert t["couver"] > 0.6
    assert t["œuf"] < t["couver"] * 0.5
    assert r1.delta_mean > 0.3          # le déclic : tension qui se résout

def test_pas_de_faux_echo_nuit_etoilee():
    r1 = score(V1)
    assert all("étoilée" not in e for e in r1.echoes)

def test_paraphrase_perd():
    a = score("Le navire glissant sur les gouffres amers")
    b = score("Le bateau qui avance sur l'eau très profonde")
    assert a.score > b.score

def test_bruit_aleatoire_coule():
    """Écart maximal mais récupérabilité nulle -> le produit T×C punit (L2)."""
    bruit = "Pylône cactus stéthoscope vermillon tracteur ozone"
    r = score(bruit)
    v1 = score(V1)
    assert r.N < v1.N
