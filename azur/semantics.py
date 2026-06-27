# -*- coding: utf-8 -*-
"""
azur.semantics — Ressources sémantiques et segmentation collocationnelle.

Couche-jouet AUDITABLE qui tient lieu d'espace d'embeddings :
  - champs dénotatifs  -> distance locale (tension T)
  - champs connotatifs -> récupérabilité (cohérence C, déclic Δ)
  - idiomes figés      -> tokenisation en expressions (§2.3) : une collocation
    figée est UN item lexical ; « poisse » dans « sentir la poisse » ne reçoit
    aucun crédit de rareté.

Interface stable : remplacer `relatedness()` par une similarité cosinus
CamemBERT ne change rien au reste du pipeline.
"""
from __future__ import annotations

import json
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

with open(os.path.join(DATA, "champs.json"), encoding="utf-8") as f:
    _R = json.load(f)

DENOTATIF: dict[str, set[str]] = {k: set(v) for k, v in _R["denotatif"].items()}
CONNOTATIF: dict[str, set[str]] = {k: set(v) for k, v in _R["connotatif"].items()}
POLE_LYRIQUE = set(_R["axe_lyrique"]["pole_lyrique"])
POLE_PROSAIQUE = set(_R["axe_lyrique"]["pole_prosaique"])
VERBES_COURANTS = set(_R["verbes_courants_dampening"])
IMPERATIFS = set(_R["imperatifs_instrumentaux"])

STOPWORDS = set("""le la les un une des de du d l au aux et ou mais donc or ni
car que qu qui quoi dont où à a en dans par pour sur sous avec sans ne pas
plus moins très se sa son ses ta ton tes ma mon mes votre vos notre nos leur
leurs ce cet cette ces il elle ils elles on nous vous je tu me te lui y est
sont était fut être avoir ai as ont si comme quand""".split())

IDIOMS: list[tuple[str, str]] = []
with open(os.path.join(DATA, "idiomes.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            expr, regime = [x.strip() for x in line.split("|")]
            IDIOMS.append((expr, regime))


def norm(w: str) -> str:
    w = w.lower().strip(".,;:!?…«»\"'’()")
    return w


# Vocabulaire connu (union des champs) — défini avant lemma_lite qui le
# consulte (lisibilité, et pour ne pas dépendre d'une résolution différée).
_ALL_WORDS = set().union(*DENOTATIF.values(), *CONNOTATIF.values(),
                         POLE_LYRIQUE, POLE_PROSAIQUE)


def lemma_lite(w: str) -> str:
    """Dé-flexion grossière pour la consultation des champs."""
    w = norm(w)
    for suf in ("s",):
        if len(w) > 3 and w.endswith(suf) and w[:-1] in _ALL_WORDS:
            return w[:-1]
    return w


def fields_of(w: str, table: dict[str, set[str]]) -> set[str]:
    w = norm(w)
    out = {k for k, v in table.items() if w in v}
    if not out:
        l = lemma_lite(w)
        out = {k for k, v in table.items() if l in v}
    return out


def relatedness(w1: str, w2: str) -> float:
    """0..1 — proximité sémantique globale (dénotatif fort, connotatif moyen).
    Sert au régime instrumental et aux usages génériques."""
    if norm(w1) == norm(w2):
        return 1.0
    if fields_of(w1, DENOTATIF) & fields_of(w2, DENOTATIF):
        return 1.0
    if fields_of(w1, CONNOTATIF) & fields_of(w2, CONNOTATIF):
        return 0.55
    return 0.0


def denotative_relatedness(w1: str, w2: str) -> float:
    """Proximité DÉNOTATIVE seule -> distance locale de la tension T (§2.1).
    Les isotopies connotatives ne doivent PAS amortir la tension : elles
    servent à la récupérabilité (C, Δ). C'est la séparation théorique
    écart-dénotatif / résolution-connotative (Cohen vs Riffaterre)."""
    if norm(w1) == norm(w2):
        return 1.0
    if fields_of(w1, DENOTATIF) & fields_of(w2, DENOTATIF):
        return 1.0
    return 0.0


def connotative_relief(w: str, others: list[str]) -> float:
    """Récupérabilité rétrospective : meilleure connexion (dénot. OU connot.)
    de w au reste du poème — sert à Δ et à C."""
    best = 0.0
    cw = fields_of(w, CONNOTATIF) | fields_of(w, DENOTATIF)
    for o in others:
        if norm(o) == norm(w):
            continue
        co = fields_of(o, CONNOTATIF) | fields_of(o, DENOTATIF)
        if cw & co:
            d = fields_of(w, DENOTATIF) & fields_of(o, DENOTATIF)
            best = max(best, 1.0 if d else 0.8)
    return best


# ---------------------------------------------------------------- idiomes
class Segment:
    """Unité lexicale après segmentation : mot simple ou idiome figé."""
    def __init__(self, words: list[str], regime: str = "libre"):
        self.words = words
        self.regime = regime          # libre | cliché | cliché_poétique

    @property
    def text(self) -> str:
        return " ".join(self.words)

    @property
    def frozen(self) -> bool:
        return self.regime != "libre"

    def __repr__(self):
        return f"<{self.text}|{self.regime}>"


def segment(tokens: list[str]) -> list[Segment]:
    """Tokenisation en expressions : repère les idiomes (matching de surface
    sur formes normalisées) et les gèle en un seul Segment."""
    toks = [norm(t) for t in tokens if norm(t)]
    out: list[Segment] = []
    i = 0
    while i < len(toks):
        hit = None
        for expr, regime in IDIOMS:
            ew = expr.split()
            if toks[i:i + len(ew)] == ew:
                hit = (ew, regime)
                break
        if hit:
            out.append(Segment(hit[0], hit[1]))
            i += len(hit[0])
        else:
            out.append(Segment([toks[i]]))
            i += 1
    return out


def idiom_echo(tokens: list[str]) -> list[str]:
    """Détecte un cliché DÉSAUTOMATISÉ : exactement un mot plein d'un idiome
    présent, son partenaire substitué (l'« œuf bleu » réactivant « peur
    bleue »). Garde-fou : le mot présent doit être le terme DISTINCTIF de
    l'idiome (pas plus fréquent que le partenaire absent), sinon « nuit »
    seul réactiverait faussement « nuit étoilée »."""
    try:
        from wordfreq import zipf_frequency as _z
    except ImportError:  # pragma: no cover
        _z = lambda w, l: 3.0
    toks = {lemma_lite(t) for t in tokens if norm(t) not in STOPWORDS}
    echoes = []
    for expr, _ in IDIOMS:
        content = [w for w in expr.split() if w not in STOPWORDS]
        present = [w for w in content if lemma_lite(w) in toks]
        absent = [w for w in content if lemma_lite(w) not in toks]
        if len(present) == 1 and absent and len(content) >= 2:
            if _z(present[0], "fr") <= min(_z(a, "fr") for a in absent) + 0.2:
                echoes.append(expr)
    return echoes


def lyric_axis(word: str, zipf: float) -> float:
    """Score lexical -1 (prosaïque) .. +1 (lyrique).
    Liste explicite > courbe en U sur la fréquence (L5 lexicale)."""
    w = lemma_lite(word)
    if w in POLE_LYRIQUE:
        return 1.0
    if w in POLE_PROSAIQUE:
        return -1.0
    # U inversée : le lyrique vit autour de zipf 2.0–3.5 ;
    # l'ultra-fréquent et l'introuvable sont pénalisés.
    if zipf <= 0:
        return -0.2
    peak, width = 2.8, 1.6
    return max(-0.6, 1.0 - ((zipf - peak) / width) ** 2) * 0.6
