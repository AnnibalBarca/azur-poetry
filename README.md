# AZUR — prototype exécutable

Implémentation testable de la spec AZUR (notation de perfection poétique)
et du vérificateur PTYX-lite (scansion classique).

## Installation & test (60 secondes)

```bash
pip install wordfreq pytest
python3 -m pytest tests/ -q        # 12 tests : scansion or + invariants scorer
python3 demo.py                    # étude de cas §3 + paire minimale + adversarial
python3 azur_cli.py "Laissez la nuit couver l'œuf bleu de l'angoisse"
python3 azur_cli.py -f poeme.txt --json
ANTHROPIC_API_KEY=... python3 azur_cli.py -f poeme.txt --judge
```

## Correspondance spec -> code

| Spec AZUR | Module | État |
|---|---|---|
| PTYX-lite : e caduc, élision, -es/-ent, diérèses, hiatus, césure | `azur/scansion.py` | réel, testé sur Racine/Hugo/Baudelaire/Mallarmé |
| §2.3 tokenisation en expressions (idiomes figés = 1 item) | `azur/semantics.py::segment` | réel |
| L1 écart / tension prospective T (contexte gauche) | `azur/lm.py::tension` | réel (rareté wordfreq × distance dénotative) |
| L2 récupérabilité C (isotopies connotatives) | `semantics.py` + `scorer.py` | réel sur ressource-jouet |
| §2.2 indice de déclic Δ | `lm.py::declic` (jouet) ; `TransformersBackend.declic` (réel, 2 passes LM) | jouet ici / réel sur machine GPU |
| L4 phonique, L5 U inversée, L6 densité, L7 clichés & désautomatisation | `scorer.py`, `rhyme.py` | réel |
| Régime instrumental (« mange ta soupe ») | `scorer.py` | réel : impératif quotidien + écart interne < seuil |
| Juge LLM (couche 6) + test de paraphrase L3 | `azur/judge.py` | réel, optionnel (clé API) |
| Agrégation géométrique F/N/Φ/O/D | `scorer.py::score` | réel |

## Résultats de la démo (reproductibles)

```
V1 « Laissez la nuit couver l'œuf bleu de l'angoisse »   AZUR 66.7
   T̄=0.46  C=0.88  Δ̄=0.55 — pic sur « couver » (T=0.81), retombée sur « œuf »
V2 « Et mange ta soupe qui sent la poisse »              AZUR 26.5
   [INSTRUMENTAL] T̄=0.12, idiome figé « sent la poisse », axe lexical -0.83
Baudelaire (Albatros, strophe 1)                          AZUR 73.3
Sa paraphrase prosaïque (paire minimale P4)               AZUR 59.5  (résidu -13.8)
Faux alexandrin « de Baudelaire »                         17 syllabes -> prémisse rejetée
```

## Limites assumées (et points de branchement)

1. **`data/champs.json` est un stub d'embeddings.** Les isotopies sont
   codées à la main pour le vocabulaire de la démo : le vers de Mallarmé
   obtient C=0.25 faute de champs adéquats — c'est la démonstration en acte
   que la récupérabilité exige une vraie sémantique distributionnelle.
   Brancher CamemBERT : remplacer `relatedness`/`denotative_relatedness`
   par des cosinus (interface inchangée).
2. **Δ jouet vs Δ réel.** Ici Δ = T × relief connotatif. La vraie définition
   (surprisal prospectif − rétrospectif, 2 passes LM) est codée dans
   `TransformersBackend.declic` — à exécuter avec Mistral-7B/GPT-fr local.
3. **G2P de rime approché.** Suffisant pour genre/schéma/richesse en démo ;
   brancher Lexique.org (colonne `phon`) ou espeak-ng pour du sérieux.
4. **`VERB_ENT` et `DIERESE_TABLE` sont des tables ouvertes** — comme chez
   Banville : elles s'enrichissent, elles ne se devinent pas.
5. **O sans RAG.** L'originalité de 2ᵉ ordre complète (voisins Gallica,
   n-grammes) attend l'index FAISS de la phase 2.

## Arborescence

```
azur/
  azur/scansion.py    scansion classique (PTYX-lite)
  azur/rhyme.py       finales phonétiques, genre, richesse, schéma
  azur/semantics.py   champs, segmentation idiomatique, axe lyrique, échos
  azur/lm.py          backends surprisal (wordfreq / n-gramme / transformers), T, Δ
  azur/scorer.py      axes F·N·Φ·O·D, régime instrumental, rapport
  azur/judge.py       juge LLM optionnel (rubrique L1-L7, JSON)
  data/champs.json    ressource sémantique (remplaçable par embeddings)
  data/idiomes.txt    collocations figées
  tests/              12 tests (scansion or + invariants)
  demo.py             étude de cas §3 exécutable
  azur_cli.py         CLI texte/fichier/stdin, sortie humaine ou JSON
```
