# -*- coding: utf-8 -*-
"""
azur.judge — Juge LLM optionnel (couche 6 de l'architecture).

Nécessite ANTHROPIC_API_KEY dans l'environnement et `pip install anthropic`.
Le juge ne voit JAMAIS les scores des couches 1-5 (indépendance) ; il couvre
ce que les features ratent : ironie, intertexte, tenue d'ensemble, et le test
du résidu de paraphrase (L3) impossible hors-ligne.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("azur.judge")

# Modèle surchargeable via env (AZUR_JUDGE_MODEL) sans toucher au code.
DEFAULT_MODEL = os.environ.get("AZUR_JUDGE_MODEL", "claude-sonnet-4-5")

RUBRIC = """Tu es un juge de poésie française outillé par une rubrique stricte.
Note le texte ci-dessous sur 4 critères, chacun entre 0.0 et 1.0 :
- ecart (L1) : la langue dévie-t-elle de la prose probable ?
- recuperabilite (L2) : la déviation se résout-elle en isotopie cohérente
  (≠ charabia) ?
- residu_de_paraphrase (L3) : paraphrase mentalement le texte en prose
  neutre ; qu'est-ce qui se perd ? (0 = rien ne se perd, proposition
  logique close ; 1 = perte massive, hérésie de la paraphrase)
- tenue (ensemble) : unité de ton, progression, clausule.
Réponds UNIQUEMENT en JSON : {"ecart":x,"recuperabilite":x,
"residu_de_paraphrase":x,"tenue":x,"justification":"<=40 mots"}."""


def judge(text: str, model: str | None = None) -> dict | None:
    """Note le texte via un LLM. Retourne None (et log) si indisponible
    ou en cas d'erreur API : le pipeline 1-5 reste autonome."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(api_key=key)
    try:
        msg = client.messages.create(
            model=model or DEFAULT_MODEL, max_tokens=400,
            messages=[{"role": "user",
                       "content": f"{RUBRIC}\n\n<texte>\n{text}\n</texte>"}],
        )
    except Exception as exc:  # APIError, timeout, quota, réseau…
        log.warning("juge LLM indisponible : %s", exc)
        return None
    raw = "".join(b.text for b in msg.content if b.type == "text")
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("réponse du juge non JSON : %.80s", raw)
        return None
