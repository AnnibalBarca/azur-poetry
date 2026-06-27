# -*- coding: utf-8 -*-
"""
azur.scorer — Agrégation des cinq axes et rapport.

AZUR = 100 · F^.20 · N^.35 · Φ^.15 · O^.20 · D^.10   (moyenne géométrique :
un axe effondré ne se rachète pas ailleurs).

  F : forme (scansion PTYX-lite : isométrie, hiatus, césures, rimes)
  N : noyau Tension × Récupérabilité + indice de déclic Δ (§2)
  Φ : surdétermination phonique (allitérations/assonances > hasard) (L4)
  O : originalité de 2ᵉ ordre (clichés subis vs désautomatisés, rareté) (L7)
  D : densité sémantique par syllabe sous contrainte (L6)

Le régime « proposition logique optimale » (mange ta soupe) est détecté
clause par clause : impératif quotidien + AUCUNE paire à forte distance
sémantique interne => tension plafonnée (§2.3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

from . import scansion as SC
from . import semantics as S
from . import rhyme as R
from . import lm as LM
from .lm import zipf_frequency


@dataclass
class WordReport:
    word: str
    T: float = 0.0
    delta: float = 0.0
    lyric: float = 0.0
    regime: str = "libre"


@dataclass
class AzurReport:
    text: str
    F: float = 0.0
    N: float = 0.0
    PHI: float = 0.0
    O: float = 0.0
    D: float = 0.0
    score: float = 0.0
    T_mean: float = 0.0
    C: float = 0.0
    delta_mean: float = 0.0
    lyric_mean: float = 0.0
    instrumental: bool = False
    cliches: list = field(default_factory=list)
    echoes: list = field(default_factory=list)
    words: list = field(default_factory=list)
    scansion: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _content_segments(segs):
    return [s for s in segs
            if s.frozen or S.norm(s.words[0]) not in S.STOPWORDS]


def _phonics(tokens: list[str]) -> float:
    """Densité d'échos phoniques au-delà de l'attendu (proxy L4)."""
    import collections
    onsets = collections.Counter()
    nuclei = collections.Counter()
    for t in tokens:
        w = S.norm(t)
        if not w or w in S.STOPWORDS:
            continue
        tail = R.phonetic_tail(w, 4)
        if w[0] not in "aàâeéèêiîoôuùûyœh":
            onsets[w[0]] += 1
        for ph in ("â", "ô", "ê", "wa", "u", "2", "o", "E", "i", "e", "y", "8i"):
            if ph in tail:
                nuclei[ph] += 1
    rep = sum(c - 1 for c in onsets.values() if c > 1) \
        + sum(c - 1 for c in nuclei.values() if c > 1)
    n = max(1, len([t for t in tokens if S.norm(t) not in S.STOPWORDS]))
    return min(1.0, 0.25 + rep / (1.6 * n))


def _form(scans: list[SC.LineScan]) -> tuple[float, list[str]]:
    notes = []
    if not scans:
        return 0.0, notes
    counts = [s.total for s in scans]
    f = 1.0
    nh = sum(len(s.hiatus) for s in scans)
    if nh:
        f -= 0.18 * nh
        notes.append(f"hiatus x{nh}")
    if len(scans) > 1:
        iso = counts.count(max(set(counts), key=counts.count)) / len(counts)
        f *= 0.45 + 0.55 * iso
        if iso < 1:
            notes.append(f"isométrie {iso:.0%} ({counts})")
        if all(c == 12 for c in counts):
            ces = sum(1 for s in scans if s.cesure_kind == "classique")
            f *= 0.75 + 0.25 * (ces / len(scans))
            last = [SC.tokenize(s.text)[-1] for s in scans]
            scheme = R.rhyme_scheme(last)
            notes.append(f"alexandrins, césures classiques {ces}/{len(scans)},"
                         f" schéma {scheme}")
    return max(0.05, min(1.0, f)), notes


def score(text: str, backend=None) -> AzurReport:
    backend = backend or LM.WordfreqBackend()
    rep = AzurReport(text=text)
    rep.scansion = SC.scan_poem(text)

    tokens = [t for s in rep.scansion for t in SC.tokenize(s.text)]
    plain = [S.norm(t).rstrip("'") for t in tokens]
    segs = S.segment(plain)
    content = _content_segments(segs)
    poem_words = [w for s in content for w in s.words]

    # ---- régime instrumental (la « proposition logique optimale ») ----
    has_imperative = any(S.lemma_lite(w) in S.IMPERATIFS for w in plain)
    pair_dists = []
    free_words = [s.words[0] for s in content if not s.frozen]
    for i, w in enumerate(free_words):
        for v in free_words[i + 1:]:
            pair_dists.append(1.0 - S.relatedness(w, v))
    max_dist = max(pair_dists, default=0.0)
    rep.instrumental = has_imperative and max_dist < 0.5
    if rep.instrumental:
        rep.notes.append("régime instrumental : impératif quotidien sans"
                         " écart sémantique interne -> tension plafonnée")

    # ---- T, Δ, axe lyrique, mot par mot ----
    # Tension PROSPECTIVE : contexte gauche seul (le lecteur ne connaît pas
    # la suite). « couver » est tendu après « nuit » ; « œuf » retombe après
    # « couver » — c'est le profil en dents de scie du §3 de la spec.
    T_vals, D_vals, L_vals = [], [], []
    for i, seg in enumerate(content):
        neigh = [s.words[0] for s in content[max(0, i - 3):i]
                 if not s.frozen]
        t = LM.tension(seg, neigh, backend)
        if rep.instrumental:
            t = min(t, 0.18)
        w0 = seg.words[-1] if seg.frozen else seg.words[0]
        d = LM.declic(w0, t, poem_words)
        z = zipf_frequency(S.norm(w0), "fr")
        ly = S.lyric_axis(w0, z) if not seg.frozen else -0.5
        T_vals.append(t)
        D_vals.append(d)
        L_vals.append(ly)
        rep.words.append(WordReport(seg.text, t, d, ly, seg.regime))
        if seg.frozen:
            rep.cliches.append(seg.text)

    rep.T_mean = sum(T_vals) / len(T_vals) if T_vals else 0.0
    rep.delta_mean = sum(sorted(D_vals, reverse=True)[:4]) / max(1, min(4, len(D_vals)))
    rep.lyric_mean = sum(L_vals) / len(L_vals) if L_vals else 0.0

    # ---- C : récupérabilité globale (isotopie) ----
    if poem_words:
        connected = [w for w in poem_words
                     if S.connotative_relief(w, poem_words) > 0]
        rep.C = 0.25 + 0.75 * len(connected) / len(poem_words)
    cliche_words = {S.lemma_lite(w) for c in rep.cliches for w in c.split()}
    rep.echoes = [e for e in S.idiom_echo(plain)
                  if not ({S.lemma_lite(w) for w in e.split()} & cliche_words)]

    # ---- axes ----
    rep.F, fnotes = _form(rep.scansion)
    rep.notes.extend(fnotes)
    # zone optimale (L5) : tension hors-bande amortie en U inversée
    wundt = 1.0 - 0.9 * max(0.0, rep.T_mean - 0.82) ** 2 * 10
    noyau = (0.5 * rep.T_mean + 0.5 * rep.delta_mean) * (rep.C ** 0.7)
    rep.N = max(0.02, min(1.0, noyau * max(0.4, wundt) * 1.45))
    rep.PHI = _phonics(plain)
    cliche_pen = 0.30 * (len(rep.cliches) / max(1, len(content)))
    echo_bonus = 0.18 * min(2, len(rep.echoes))
    rar = sum(LM.rarity(w) for w in poem_words) / max(1, len(poem_words))
    rep.O = max(0.05, min(1.0, 0.45 + 0.45 * rar - cliche_pen + echo_bonus
                          + 0.15 * max(0.0, rep.lyric_mean)))
    syll = sum(s.total for s in rep.scansion) or 1
    rep.D = max(0.05, min(1.0, sum(T_vals) / (0.32 * syll)))

    w = dict(F=0.20, N=0.35, PHI=0.15, O=0.20, D=0.10)
    logsum = sum(w[k] * math.log(max(getattr(rep, k), 0.02))
                 for k in ("F", "N", "PHI", "O", "D"))
    rep.score = round(100 * math.exp(logsum), 1)
    return rep


def render(rep: AzurReport) -> str:
    bar = lambda x: "█" * int(round(x * 20)) + "░" * (20 - int(round(x * 20)))
    L = [f"\n=== AZUR {rep.score}/100 ===  «{rep.text.strip().splitlines()[0][:60]}…»"
         if len(rep.text.strip()) > 60 else
         f"\n=== AZUR {rep.score}/100 ===  «{rep.text.strip()}»"]
    for k, lab in [("F", "Forme        "), ("N", "Noyau T×C    "),
                   ("PHI", "Phonique     "), ("O", "Originalité  "),
                   ("D", "Densité      ")]:
        v = getattr(rep, k)
        L.append(f"  {lab} {bar(v)} {v:.2f}")
    L.append(f"  -- T̄={rep.T_mean:.2f}  C={rep.C:.2f}  Δ̄={rep.delta_mean:.2f}"
             f"  axe-lyrique={rep.lyric_mean:+.2f}"
             f"{'  [INSTRUMENTAL]' if rep.instrumental else ''}")
    for s in rep.scansion:
        L.append(f"  {s.total:>2} syll | césure {s.cesure_kind:<11}| {s.text.strip()}")
    if rep.cliches:
        L.append(f"  clichés subis : {', '.join(rep.cliches)}")
    if rep.echoes:
        L.append(f"  clichés DÉSAUTOMATISÉS (prime) : {', '.join(rep.echoes)}")
    hot = sorted(rep.words, key=lambda w: -(w.T + w.delta))[:4]
    L.append("  points chauds : " + " ; ".join(
        f"{w.word}(T={w.T:.2f},Δ={w.delta:.2f})" for w in hot))
    for n in rep.notes:
        L.append(f"  note : {n}")
    return "\n".join(L)
