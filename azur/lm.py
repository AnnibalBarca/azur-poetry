# -*- coding: utf-8 -*-
"""
azur.lm — Tension prospective T et indice de déclic Δ.

Trois backends derrière une même interface :
  WordfreqBackend     : rareté unigramme (wordfreq, dispo partout) ;
  NGramBackend        : bigrammes Kneser-Ney-lite sur un corpus de PROSE
                        fourni par l'utilisateur (la norme dont on mesure
                        l'écart, loi L1 de Cohen) ;
  TransformersBackend : LM causal HuggingFace (Mistral/GPT-fr) — surprisal
                        réel en 2 passes pour Δ (prospectif vs rétrospectif).
                        Code complet, à exécuter sur machine avec GPU/poids.

Dans le prototype, T combine rareté (backend) et DISTANCE SÉMANTIQUE LOCALE
(semantics.relatedness) ; Δ = T × récupérabilité rétrospective. C'est la
version dégradée-mais-honnête de la définition du §2 : l'improbable
prospectif qui devient nécessaire rétrospectivement.
"""
from __future__ import annotations

import math
from collections import Counter

from . import semantics as S

try:
    from wordfreq import zipf_frequency
    _WF = True
except ImportError:  # pragma: no cover
    _WF = False
    def zipf_frequency(w, lang):
        return 3.0


def rarity(word: str) -> float:
    """0..1 — composante de rareté (zipf inversé, saturé)."""
    z = zipf_frequency(S.norm(word), "fr")
    if z == 0:
        z = 1.5  # inconnu de wordfreq : rare mais pas infini
    return max(0.0, min(1.0, (4.6 - z) / 3.4))


class WordfreqBackend:
    name = "wordfreq-unigram"

    def surprisal(self, word: str, context: list[str]) -> float:
        return rarity(word)


class NGramBackend:
    """Bigrammes avec repli unigramme wordfreq. corpus = itérable de phrases."""
    name = "ngram-prose"

    def __init__(self, sentences):
        self.bi = Counter()
        self.uni = Counter()
        for s in sentences:
            toks = ["<s>"] + [S.norm(t) for t in s.split()] + ["</s>"]
            self.uni.update(toks)
            self.bi.update(zip(toks, toks[1:]))
        self.N = sum(self.uni.values()) or 1

    def surprisal(self, word: str, context: list[str]) -> float:
        w = S.norm(word)
        prev = S.norm(context[-1]) if context else "<s>"
        c_bi, c_prev = self.bi[(prev, w)], self.uni[prev]
        p_bi = c_bi / c_prev if c_prev else 0.0
        p_uni = 10 ** (zipf_frequency(w, "fr") - 9)  # zipf -> proba approx.
        p = 0.7 * p_bi + 0.3 * max(p_uni, 1e-9)
        s = -math.log2(max(p, 1e-9))
        return max(0.0, min(1.0, (s - 4.0) / 22.0))


class TransformersBackend:  # pragma: no cover - nécessite GPU + poids HF
    """Surprisal réel sous LM causal ; Δ par double passe (cf. AZUR §2.2)."""
    name = "transformers"

    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.3"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto")
        self.torch = torch

    def token_surprisals(self, text: str, prefix: str = "") -> list[float]:
        t = self.torch
        ids = self.tok(prefix + text, return_tensors="pt").input_ids
        with t.no_grad():
            logits = self.model(ids).logits
        logp = t.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        s = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        n_prefix = len(self.tok(prefix).input_ids) - 1 if prefix else 0
        return s[n_prefix:].tolist()

    def declic(self, poem: str, line: str) -> float:
        """Δ = surprisal(vers | contexte gauche) - surprisal(vers | poème)."""
        prosp = sum(self.token_surprisals(line))
        retro = sum(self.token_surprisals(line, prefix=poem + "\n"))
        return max(0.0, prosp - retro)


# ------------------------------------------------------------ T et Δ jouets
def tension(seg: "S.Segment", neighbors: list[str],
            backend=None) -> float:
    """T d'un segment : rareté × distance sémantique locale, régimes §2.3.
    - idiome figé        -> plafonné (cliché subi)
    - verbe ultra-courant-> distance amortie
    """
    backend = backend or WordfreqBackend()
    if seg.frozen:
        return 0.12  # régime cliché : un seul item, tension plafonnée
    w = seg.words[0]
    if S.norm(w) in S.STOPWORDS:
        return 0.0
    r = backend.surprisal(w, neighbors)
    if neighbors:
        dist = 1.0 - max(S.denotative_relatedness(w, n) for n in neighbors)
    else:
        dist = 0.3
    if S.lemma_lite(w) in S.VERBES_COURANTS:
        dist *= 0.35
    return max(0.0, min(1.0, 0.45 * r + 0.55 * dist))


def declic(word: str, t_word: float, poem_words: list[str]) -> float:
    """Δ jouet : la tension qui SE RÉSOUT rétrospectivement.
    Δ = T × relief connotatif (la surprise chute quand le poème entier
    éclaire le mot)."""
    return t_word * S.connotative_relief(word, poem_words)
