# -*- coding: utf-8 -*-
"""
azur.rhyme — Phonétisation approchée des finales : genre, richesse, schéma.

G2P de queue de mot par substitutions ordonnées (suffisant pour la rime ;
un déploiement sérieux branche Lexique.org ou espeak-ng).
"""
from __future__ import annotations

import re

_SILENT_FINALS = re.compile(r"(?<=[a-zéèêàâîôûœ])[sxztdp]$")

_TAIL_RULES = [
    (r"eaux?$", "o"), (r"aux?$", "o"), (r"ault$", "o"), (r"ots?$", "o"),
    (r"ez$", "e"), (r"ers?$", "e"), (r"és?$|ée s?$|ées?$", "e"),
    (r"ais$|ait$|aient$|aix$|ès$|êts?$", "E"), (r"ets?$", "E"),
    (r"oins?$", "wê"), (r"ions?$", "jô"),
    (r"ans?$|ams?$|ens?$|ents?$", "â"), (r"onds?$|onts?$|ons?$|oms?$", "ô"),
    (r"ins?$|ains?$|eins?$|yns?$|uns?$|ums?$", "ê"),
    (r"oux?$|ous$", "u"), (r"eux?$|œux?$", "2"),
    (r"oi[sxt]?$|oie s?$|oies?$", "wa"),
    (r"ui[ts]?$", "8i"), (r"i[sxt]?$|ies?$|ys?$", "i"),
    (r"u[st]?$|ues?$", "y"), (r"a[st]?$", "a"), (r"o$", "o"),
]
_BODY = [
    ("ch", "S"), ("ph", "f"), ("gn", "N"), ("qu", "k"), ("gu", "g"),
    ("ille", "ij"), ("ill", "ij"), ("eau", "o"), ("au", "o"), ("oi", "wa"),
    ("ou", "u"), ("eu", "2"), ("œu", "2"), ("ai", "E"), ("ei", "E"),
    ("an", "â"), ("en", "â"), ("on", "ô"), ("in", "ê"), ("é", "e"),
    ("è", "E"), ("ê", "E"), ("ç", "s"), ("œ", "2"), ("ss", "s"),
]


def phonetic_tail(word: str, depth: int = 5) -> str:
    """Phonèmes approchés de la finale (suffixe de longueur ~depth)."""
    w = word.lower().strip(".,;:!?…«»\"'’")
    if w.endswith("e") and len(w) > 1 and w[-2] != "é":
        w = w[:-1]
    elif w.endswith("es") and len(w) > 2:
        w = w[:-2]
    tail = None
    for pat, ph in _TAIL_RULES:
        m = re.search(pat, w)
        if m:
            tail, w = ph, w[: m.start()]
            break
    if tail is None:
        w = _SILENT_FINALS.sub("", w)
        tail = ""
    body = w[-depth:]
    for a, b in _BODY:
        body = body.replace(a, b)
    return (body + tail)[-depth:]


def is_feminine(word: str) -> bool:
    w = word.lower().strip(".,;:!?…«»\"'’")
    return bool(re.search(r"(e|es)$", w)) and not re.search(r"(és?|ées?|ès)$", w)


def richness(w1: str, w2: str) -> int:
    """Nombre de phonèmes communs en fin de mot (1=pauvre, 2=suffisante, 3+=riche)."""
    a, b = phonetic_tail(w1), phonetic_tail(w2)
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def rhyme_scheme(last_words: list[str]) -> str:
    """Schéma de rimes (ABAB…) par regroupement des finales phonétiques."""
    labels, seen = [], {}
    for w in last_words:
        key = phonetic_tail(w, 3)[-2:]
        if key not in seen:
            seen[key] = chr(ord("A") + len(seen))
        labels.append(seen[key])
    return "".join(labels)
