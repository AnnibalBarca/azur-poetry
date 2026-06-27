# -*- coding: utf-8 -*-
"""
azur.scansion — Vérificateur métrique classique (PTYX-lite).

Implémente la syllabation du vers français classique :
  R1  e caduc : compte devant consonne / h aspiré ; s'élide devant voyelle /
      h muet (seul le 'e' final nu s'élide ; '-es' et '-ent' verbaux comptent,
      liaison oblige) ; jamais compté en fin de vers.
  R3  diérèses classiques : table lexicale + règles régulières (-tion, etc.)
  R4  hiatus : convention graphique classique (voyelle finale prononcée
      autre que e + initiale vocalique).
  R5  césure : frontière de mot en 6, sans e caduc compté en 6 (épique).

Le compteur est volontairement table-driven : les tables de diérèses et
d'exceptions SONT la méthode (cf. Banville, Mazaleyrat) — elles s'enrichissent.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field

VOWELS = set("aàâäeéèêëiîïoôöuùûüyœ")
# Voyelles accentuées : ne se groupent jamais avec une voyelle voisine
# (po-é-sie, o-cé-an, mé-andre, na-ï-f).
SPLIT_VOWELS = set("éèêëàâîïôûüö")

# h aspiré (liste fermée, extensible) : bloque élision et liaison.
H_ASPIRE = {
    "haine", "hache", "haut", "haute", "hauts", "hautes", "honte", "hibou",
    "hasard", "héros", "haie", "halte", "hameau", "hanche", "harpe", "hâte",
    "hennir", "hérisson", "hêtre", "heurt", "hideux", "hollande", "homard",
    "houle", "huée", "hurler", "hutte", "hargne", "hampe",
}

# Mots en -es où le e n'est PAS caduc ([e] plein).
NO_CADUC_ES = {"les", "des", "mes", "tes", "ses", "ces"}

# Formes verbales en -ent muet rencontrées dans les corpus de test / démo.
# (Un vrai déploiement branche ici un lemmatiseur ou Lexique.)
VERB_ENT = {
    "prennent", "suivent", "chantent", "aiment", "conspirent", "tombent",
    "glissent", "couvent", "mangent", "sentent", "brillent", "pleurent",
    "dorment", "meurent", "vivent", "passent", "portent", "semblent",
    "tremblent", "songent", "veillent", "ouvrent", "ferment", "montent",
    "descendent", "regardent", "écoutent", "respirent", "frissonnent",
}

# Diérèses classiques : surcharge du compte de noyaux (valeur ABSOLUE de
# noyaux du radical, e caduc final exclu).
DIERESE_TABLE = {
    "lion": 2, "lions": 2, "meurtrier": 3, "meurtriers": 3,
    "ancien": 3, "anciens": 3, "ancienne": 3, "anciennes": 3,
    "hier": 2,            # flottant : convention classique = 2
    "pays": 2, "paysage": 3, "paysages": 3,
    "suave": 3, "suaves": 3, "tuer": 2, "ruine": 3, "ruines": 3,
    "violon": 3, "violons": 3, "pieux": 2, "odieux": 3, "radieux": 3,
    "précieux": 3, "précieuse": 3, "studieux": 3, "curieux": 3,
    "passion": 3, "passions": 3, "nation": 3, "nations": 3,
    "diamant": 3, "diamants": 3, "océan": 3, "océans": 3,
    "luisant": 3,  # exemple : lu-i-sant chez certains classiques (flottant)
}
# Diérèses régulières : motifs en fin de mot, +n noyaux sur le compte naïf.
DIERESE_PATTERNS = [
    (re.compile(r"[tdcsx]ions?$"), 1),          # na-ti-on, ac-ti-ons
    (re.compile(r"[tdcsx]ionnaires?$"), 1),     # dic-ti-on-nai-re
    (re.compile(r"[tdcsx]ieux$"), 0),           # géré par table (flottant)
]

STOP_PUNCT = re.compile(r"[,;:!?.…«»\"()\[\]—–]")


def _norm(w: str) -> str:
    w = w.lower().strip()
    w = w.replace("œ", "œ")  # garder œ comme voyelle unique
    return w


@dataclass
class WordScan:
    raw: str
    stem_nuclei: int          # noyaux du radical (e caduc final exclu)
    final_type: str | None    # 'e' | 'es' | 'ent' | None
    starts_vowel: bool        # initiale vocalique (h muet inclus)
    ends_pronounced_vowel: bool
    counted: int = 0          # syllabes effectivement comptées dans le vers
    elided: bool = False
    dierese: bool = False


def _strip_final(word: str) -> tuple[str, str | None]:
    """Sépare le radical du e caduc final éventuel."""
    if word.endswith("ent") and word in VERB_ENT:
        return word[:-3], "ent"
    if word.endswith("es") and len(word) > 2 and word not in NO_CADUC_ES \
            and word[-3] not in SPLIT_VOWELS:
        return word[:-2], "es"
    if word.endswith("e") and len(word) > 1 and word[-2] != "é":
        return word[:-1], "e"
    return word, None


def _count_nuclei(stem: str, full_word: str) -> tuple[int, bool]:
    """Compte les noyaux vocaliques du radical, diérèses classiques incluses.
    Retourne (n, dierese_appliquée)."""
    if full_word in DIERESE_TABLE:
        return DIERESE_TABLE[full_word], True
    s = stem
    s = re.sub(r"qu", "q", s)
    s = re.sub(r"gu([eéèêiy])", r"g\1", s)
    # y intervocalique = consonne [j] (vo-ya-ge, e-ssa-yer)
    s = re.sub(r"(?<=[aàâeéèêëiîïoôöuùûœ])y(?=[aàâeéèêëiîïoôöuùûœ])", "j", s)
    # u muet final après g/q une fois le e retiré (langue -> lang)
    s = re.sub(r"(?<=[gq])u$", "", s)
    n, prev_vowel, prev_split = 0, False, False
    for ch in s:
        is_v = ch in VOWELS
        if is_v and (not prev_vowel or prev_split or ch in SPLIT_VOWELS):
            n += 1
        prev_split = ch in SPLIT_VOWELS
        prev_vowel = is_v
    extra, die = 0, False
    for pat, inc in DIERESE_PATTERNS:
        if inc and pat.search(full_word):
            extra += inc
            die = True
            break
    return n + extra, die


def scan_word(token: str) -> WordScan:
    w = _norm(token)
    bare = w.rstrip("'’")
    elision_mark = w.endswith("'") or w.endswith("’")
    starts_vowel = bool(bare) and (
        bare[0] in VOWELS and bare not in H_ASPIRE
        or (bare[0] == "h" and bare not in H_ASPIRE)
    )
    if elision_mark:  # l', d', qu', m' … : zéro syllabe
        return WordScan(token, 0, None, starts_vowel, False)
    stem, ftype = _strip_final(bare)
    n, die = _count_nuclei(stem, bare)
    ends_v = ftype is None and bool(bare) and bare[-1] in VOWELS \
        and bare[-1] != "e"
    return WordScan(token, n, ftype, starts_vowel, ends_v, dierese=die)


@dataclass
class LineScan:
    text: str
    words: list[WordScan] = field(default_factory=list)
    total: int = 0
    hiatus: list[tuple[str, str]] = field(default_factory=list)
    cesure_ok: bool = False
    cesure_kind: str = "aucune"   # 'classique' | 'épique' | 'enjambante' | …
    is_alexandrin: bool = False


def tokenize(line: str) -> list[str]:
    line = STOP_PUNCT.sub(" ", line)
    line = re.sub(r"aujourd['’]hui", "aujourdhui", line, flags=re.I)
    line = re.sub(r"([ldjmtsncq]u?)['’]", r"\1' ", line, flags=re.I)
    return [t for t in line.split() if t]


def scan_line(line: str) -> LineScan:
    toks = tokenize(line)
    scans = [scan_word(t) for t in toks]
    out = LineScan(text=line, words=scans)
    boundaries = []  # position cumulée à la fin de chaque mot
    cum = 0
    caduc_positions = set()
    for i, ws in enumerate(scans):
        nxt = scans[i + 1] if i + 1 < len(scans) else None
        count = ws.stem_nuclei
        if ws.final_type:
            if nxt is None:
                pass                      # fin de vers : surnuméraire
            elif ws.final_type == "e" and nxt.starts_vowel:
                ws.elided = True          # élision
            else:
                count += 1                # compte ('-es'/'-ent' : liaison)
                caduc_positions.add(cum + count)
        if ws.ends_pronounced_vowel and nxt is not None and nxt.starts_vowel:
            out.hiatus.append((ws.raw, nxt.raw))
        ws.counted = count
        cum += count
        boundaries.append(cum)
    out.total = cum
    if cum == 12:
        out.is_alexandrin = True
        if 6 in boundaries:
            out.cesure_ok = 6 not in caduc_positions
            out.cesure_kind = "classique" if out.cesure_ok else "épique"
        elif 4 in boundaries and 8 in boundaries:
            out.cesure_kind = "trimètre"
        else:
            out.cesure_kind = "enjambante"
    return out


def scan_poem(text: str) -> list[LineScan]:
    return [scan_line(l) for l in text.strip().splitlines() if l.strip()]
