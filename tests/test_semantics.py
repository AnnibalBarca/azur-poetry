# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from azur import semantics as S


def test_segment_idiome_gele():
    toks = ["et", "mange", "ta", "soupe", "qui", "sent", "la", "poisse"]
    segs = S.segment(toks)
    geles = [s for s in segs if s.frozen]
    assert len(geles) == 1
    assert geles[0].text == "sent la poisse"
    assert geles[0].regime == "cliché"


def test_segment_sans_idiome_tout_libre():
    segs = S.segment(["la", "nuit", "sombre"])
    assert all(not s.frozen for s in segs)
    assert [s.text for s in segs] == ["la", "nuit", "sombre"]


def test_segment_idiome_poetique():
    segs = S.segment(["un", "clair", "de", "lune"])
    geles = [s for s in segs if s.frozen]
    assert len(geles) == 1
    assert geles[0].regime == "cliché_poétique"


def test_lyric_axis_poles():
    # mots explicitement listés -> saturation
    assert S.lyric_axis("azur", 3.0) == 1.0
    assert S.lyric_axis("spleen", 3.0) == 1.0
    assert S.lyric_axis("soupe", 3.0) == -1.0
    assert S.lyric_axis("parking", 3.0) == -1.0


def test_lyric_axis_courbe_en_u_inversee():
    # zone optimale autour de zipf 2-3 ; l'ultra-fréquent pénalisé
    milieu = S.lyric_axis("table", 2.8)
    trop_bas = S.lyric_axis("inconnu", 0.0)
    assert milieu > trop_bas
    assert trop_bas == -0.2          # fallback_zipf<=0


def test_relatedness_denotative_vs_connotative():
    # nuit / ombre : même champ dénotatif (obscurité) -> relatedness 1.0
    assert S.relatedness("nuit", "ombre") == 1.0
    assert S.denotative_relatedness("nuit", "ombre") == 1.0
    # nuit / bleu : pas de champ dénotatif commun, mais connotatif -> 0.55 global
    assert S.relatedness("nuit", "bleu") == 0.55
    # ... et la distance dénotative seule reste 0 (séparation écart/résolution)
    assert S.denotative_relatedness("nuit", "bleu") == 0.0


def test_relatedness_identite():
    assert S.relatedness("nuit", "nuit") == 1.0
    assert S.relatedness("Nuit,", "nuit") == 1.0   # insensible casse/ponctuation


def test_connotative_relief_ordre():
    # relief plus fort pour connexion dénotative (1.0) que connotative seule (0.8)
    r_denot = S.connotative_relief("nuit", ["ombre"])     # obscurité
    r_connot = S.connotative_relief("nuit", ["bleu"])     # mélancolie bleue
    assert r_denot == 1.0
    assert r_connot == 0.8


def test_idiom_echo_garde_fou_nuit_etoilee():
    # « nuit » seul ne doit PAS réactiver « nuit étoilée » (mot présent
    # trop fréquent vs partenaire absent) -> garde-fou du §.
    echoes = S.idiom_echo(["la", "nuit", "couver", "l", "œuf", "bleu"])
    assert all("étoilée" not in e for e in echoes)


def test_idiom_echo_detecte_substitution():
    # « brisé » présent, « cœur » absent -> « cœur brisé » désautomatisé
    # (brisé est le terme distinctif : zipf plus faible que cœur).
    echoes = S.idiom_echo(["un", "rêve", "brisé", "substitué"])
    assert any("brisé" in e for e in echoes)
