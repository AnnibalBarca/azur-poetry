# AZUR — prototype exécutable

Implémentation testable de la spec AZUR (notation de perfection poétique)
et du vérificateur PTYX-lite (scansion classique). Le pipeline note un texte
poétique français sur 5 axes (Forme, Noyau T×C, Phonique, Originalité,
Densité) agrégés en un score géométrique AZUR/100, complétés optionnellement
par un juge LLM (couche 6).

> Statut : prototype auditable, scores reproductibles. Les ressources
> sémantiques sont des stubs remplaçables par de vrais embeddings (CamemBERT).

## Prérequis

- **Python ≥ 3.9** (testé sur 3.9, 3.11, 3.12).
- Dépendances : voir `requirements.txt` (`wordfreq`, `pytest`).
- Optionnel : `anthropic` pour le juge LLM ; `torch` + `transformers` pour le
  backend surprisal réel (machine GPU).

## Installation & test (60 secondes)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -q        # 56 tests : scansion, rhyme, semantics, lm, scorer, cli
python3 demo.py                    # étude de cas §3 + paire minimale + adversarial
python3 azur_cli.py "Laissez la nuit couver l'œuf bleu de l'angoisse"
python3 azur_cli.py -f poeme.txt --json
ANTHROPIC_API_KEY=... python3 azur_cli.py -f poeme.txt --judge      # juge LLM (optionnel)
AZUR_JUDGE_MODEL=claude-sonnet-4-5 python3 azur_cli.py -f poeme.txt --judge   # modèle au choix
```

Le CLI lit aussi l'entrée standard : `echo "..." | python3 azur_cli.py`.
Codes de sortie : `0` succès, `1` erreur (ex. fichier introuvable). En mode
`--json`, le verdict du juge part sur stderr afin de garder stdout strictement
JSON.

## Correspondance spec → code

| Spec AZUR | Module | État |
|---|---|---|
| PTYX-lite : e caduc, élision, -es/-ent, diérèses, hiatus, césure | `azur/scansion.py` | réel, testé sur Racine/Hugo/Baudelaire/Mallarmé/Corneille |
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
   T̄=0.45  C=0.88  Δ̄=0.55 — pic sur « couver » (T=0.81), retombée sur « œuf »
V2 « Et mange ta soupe qui sent la poisse »              AZUR 26.5
   [INSTRUMENTAL] T̄=0.12, idiome figé « sent la poisse », axe lexical -0.83
Baudelaire (Albatros, strophe 1)                          AZUR 73.3
Sa paraphrase prosaïque (paire minimale P4)               AZUR 59.5  (résidu -13.8)
Faux alexandrin « de Baudelaire »                         17 syllabes -> prémisse rejetée
```

## Au-delà du prototype (backends)

`azur/lm.py` expose trois backends derrière la même interface `surprisal()` :

- **`WordfreqBackend`** (défaut, partout) : rareté unigramme.
- **`NGramBackend`** : bigrammes Kneser-Ney-lite sur un **corpus de prose**
  fourni (la norme dont on mesure l'écart, loi L1) :
  ```python
  from azur.lm import NGramBackend
  backend = NGramBackend(open("corpus_prose.txt").read().splitlines())
  print(score("Le navire glissant sur les gouffres amers", backend=backend).score)
  ```
- **`TransformersBackend`** : LM causal HuggingFace (Mistral/GPT-fr), surprisal
  réel en 2 passes pour Δ. Code complet, à exécuter sur machine avec GPU/poids.

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
4. **Les tables ouvertes vivent maintenant dans `data/`.** `VERB_ENT` et
   `DIERESE_TABLE` (scansion) sont dans `data/scansion_tables.json`, les
   idiomes dans `data/idiomes.txt` : comme chez Banville, elles s'enrichissent
   à la main, elles ne se devinent pas.
5. **`lemma_lite` est volontairement minimal.** Elle lève le pluriel `-s`
   mais pas le féminin `-e` (« bleue » vs « bleu ») : la désautomatisation
   « œuf bleu / peur bleue » n'est donc pas encore détectée. Un vrai
   lemmatiseur (ou lemmatisation par dictionnaire) la remplacerait.
6. **O sans RAG.** L'originalité de 2ᵉ ordre complète (voisins Gallica,
   n-grammes) attend l'index FAISS de la phase 2.

## Arborescence

```
azur/
  azur/scansion.py        scansion classique (PTYX-lite)
  azur/rhyme.py           finales phonétiques, genre, richesse, schéma
  azur/semantics.py       champs, segmentation idiomatique, axe lyrique, échos
  azur/lm.py              backends surprisal (wordfreq / n-gramme / transformers), T, Δ
  azur/scorer.py          axes F·N·Φ·O·D, régime instrumental, rapport
  azur/judge.py           juge LLM optionnel (rubrique L1-L7, JSON)
  data/champs.json        ressource sémantique (remplaçable par embeddings)
  data/idiomes.txt        collocations figées
  data/scansion_tables.json   tables -ent muet / diérèses classiques
  tests/                  56 tests (scansion or, rhyme, semantics, lm, scorer, cli)
  demo.py                 étude de cas §3 exécutable
  azur_cli.py             CLI texte/fichier/stdin, sortie humaine ou JSON
  pyproject.toml          config pytest + ruff
  .github/workflows/ci.yml   ruff + pytest (3.9/3.11/3.12) + demo
```

## Licence

MIT — voir [LICENSE](LICENSE).
