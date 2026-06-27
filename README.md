# AZUR — a computational engine for scoring French poetry

AZUR asks a question that sounds impossible: **can you measure how good a line of poetry is?**

Not whether it *rhymes* — that part is mechanical — but whether it *works*: whether a word lands with surprise and then resolves, whether an image earns its place, whether the line is doing something or just filling a metrical slot. AZUR is a runnable attempt to turn that judgment into numbers you can audit, defend, and reproduce.

It scores a French poetic text along five axes — **Form, Core (tension × recovery), Phonics, Originality, Density** — and aggregates them into a single geometric **AZUR/100** score. An optional sixth layer hands the text to an LLM judge against an explicit rubric.

The point isn't to replace a reader. It's to make explicit the rules a careful reader applies implicitly — and then to test whether a machine can apply them too. That second question is what the companion benchmark, **PTYX**, is built to answer.

---

## Why this exists

I came to code from philosophy and logic, and AZUR is where those two halves meet. Formalizing "the laws of poetic beauty" is, underneath, an exercise in evaluation design: take something subtle and contested, decompose it into decidable and non-decidable parts, score the decidable parts rigorously, and stay honest about the rest. That is the same problem you face when you try to evaluate the output of *any* AI system on a task where "correct" is a matter of degree.

So AZUR is two things at once. On the surface it's a poetry scorer. Underneath it's a worked example of how to build a rigorous evaluation pipeline for qualitative output — which is the unsolved problem sitting at the center of agentic AI.

---

## What it does, concretely

Give it a line and it returns a structured analysis: syllable count with every *e caduc* and *diérèse* accounted for, caesura position, rhyme scheme and gender, phonemic richness, and per-axis scores with a written rationale.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 -m pytest tests/ -q     # 56 tests: scansion, rhyme, semantics, lm, scorer, cli
python3 demo.py                 # case study + minimal pair + adversarial example

python3 azur_cli.py "Laissez la nuit couver l'œuf bleu de l'angoisse"
python3 azur_cli.py -f poem.txt --json
ANTHROPIC_API_KEY=... python3 azur_cli.py -f poem.txt --judge
```

The CLI also reads from stdin (`echo "..." | python3 azur_cli.py`). Exit codes: `0` success, `1` error. In `--json` mode the judge's verdict goes to stderr so stdout stays strictly JSON.

A few reproducible results from the demo:

```
V1  "Laissez la nuit couver l'œuf bleu de l'angoisse"   AZUR 66.7
    peak tension on "couver" (T=0.81), falling off on "œuf"
V2  "Et mange ta soupe qui sent la poisse"              AZUR 26.5
    [INSTRUMENTAL] frozen idiom "sent la poisse", lexical axis -0.83
Baudelaire (L'Albatros, stanza 1)                        AZUR 73.3
its prose paraphrase (minimal pair)                      AZUR 59.5  (residual -13.8)
fake alexandrine "by Baudelaire"                         17 syllables → premise rejected
```

That last line matters: the engine refuses to analyze a 17-syllable line presented as a classical alexandrine, instead of inventing a caesura to please you. Resistance to a false premise is a property, not an accident.

---

## How the score is built

The scansion layer (`PTYX-lite`) does classical French metrics — *e caduc*, elision, mute `-es`/`-ent`, diérèses, hiatus, caesura — and is tested against real lines from Racine, Hugo, Baudelaire, Mallarmé and Corneille. On top of it sit the semantic and language-model layers that estimate **tension** (how far a word departs from what the left context predicts, via unigram rarity and denotative distance) and **recovery** (whether connotative isotopies let the reader re-integrate that departure). The scorer combines everything geometrically across the five axes and flags an *instrumental* regime — an everyday imperative like "eat your soup" with no internal departure scores low on purpose.

The language-model component sits behind one `surprisal()` interface with three interchangeable backends, so you can run AZUR anywhere and scale up when you have the hardware:

- **WordfreqBackend** (default, runs everywhere): unigram rarity.
- **NGramBackend**: Kneser-Ney-lite bigrams over a prose corpus you supply — the norm whose departure the engine measures.
- **TransformersBackend**: a causal HuggingFace LM (Mistral / GPT-fr), real two-pass surprisal for the *déclic* index, meant for a GPU machine.

```python
from azur.lm import NGramBackend
backend = NGramBackend(open("corpus_prose.txt").read().splitlines())
print(score("Le navire glissant sur les gouffres amers", backend=backend).score)
```

The semantic resources (`data/champs.json`), idiom lists, scansion tables and rhyme tables all live in `data/` as plain files you can extend by hand — they grow the way Banville's tables grew, by curation, not guessing. Swapping the toy semantic resource for real CamemBERT embeddings is a drop-in: replace the relatedness functions with cosine similarities, interface unchanged.

---

## PTYX — the benchmark behind AZUR

**PTYX** (*Poésie Technique : Yardstick eXigeant*) is the evaluation suite AZUR grew out of. It's named after Mallarmé's *Sonnet en -yx*, built on a rhyme word the poet had to *invent* — exactly the kind of constraint the benchmark imposes. Where AZUR scores a given poem, PTYX measures whether a language model can read, generate, and reason about classical French versification at all.

Classical versification turns out to be an almost ideal testbed for an LLM, for four reasons:

1. **Partial formal verifiability.** Syllable count, caesura, rhyme scheme and alternation are objectively decidable (given a diérèse lexicon). Roughly 60% of the score has no subjectivity in it.
2. **A tokenization/phonology conflict.** Models see tokens, not phonemes. The *e caduc*, the diérèse, the liaison are sub-token phenomena — so the benchmark measures whether a model can reconstruct a phonological layer it doesn't natively perceive. It's the literary cousin of counting the letters in "strawberry," but far deeper.
3. **Constraints in conflict.** Meaning, syntax, meter, rhyme and register pull in opposite directions. Generating under several simultaneous hard constraints is a proxy for long-horizon planning.
4. **Measurable anti-regurgitation.** The canon (Rimbaud, Mallarmé) is in the training data, so the benchmark has to *penalize* recall and reward fresh production under novel constraint — imposed rhyme words, imposed lexicon.

Guiding principle: **one wrong line invalidates the item.** A 13-syllable alexandrine isn't "almost right," it's wrong. The gates are hard.

### Structure

120 items across five parts, each weighted and verified differently:

| Part | Skill | Items | Weight | Verification |
|---|---|---|---|---|
| A | Scansion & analysis | 40 | 25% | Fully automatable |
| B | Constrained generation | 30 | 35% | ~70% automatable |
| C | Stylometric pastiche | 20 | 20% | Human rubric + LLM judge |
| D | Minimal correction | 15 | 10% | Mixed |
| E | Adversarial & meta | 15 | 10% | Fully automatable |

**Part B** is the discriminating core: difficulty rises strictly from tier to tier, each tier adding a constraint without removing the previous ones. By the upper tiers a model is producing a regular French sonnet on *imposed rhyme words, in imposed order* (impossible to recite from memory), then a lipogram in alexandrines, then an acrostic-plus-bouts-rimés triple constraint, and finally "the tomb" — a sonnet that re-attempts Mallarmé's feat without reusing a single one of his rhyme words, forcing the model either to find its own lexical stock or to honestly admit the lexicon is exhausted.

**Part E** is the part I care most about for AI evaluation generally. It measures *technical sycophancy*: false premises ("here is an alexandrine by Racine" — followed by a forged 13-syllable line), false attributions ("an unpublished Mallarmé quatrain found in 2024"), planted definition traps, and self-verification (the model is handed back its *own* failed output and asked to check it). That last one is, empirically, the item most correlated with a model's general reliability in agentic settings — which is exactly why a poetry benchmark turns out to say something about agents.

### Reading the scores

```
PTYX = 0.25·A + 0.35·B + 0.20·C + 0.10·D + 0.10·E      (out of 100)
```

Three derived indices are published separately because they tell different stories: **Φ** (pure phonological capacity), **Σ** (planning under crossed constraints), and **Ε** (resistance to sycophancy). A model can be brilliant at Σ and useless at Ε — it versifies beautifully but swallows every false premise — and the aggregate alone would hide that.

The most instructive gap the benchmark looks for is **high A, low B**: a model that *knows* the rules but can't *generate under* them. That declarative/procedural split is the same thing code benchmarks measure. Closing it is a real signal about planning.

### Anti-gaming

Generated lines are checked (8-gram match) against a ~50,000-poem reference corpus — any match zeroes the item and flags it. 40% of items regenerate from parametric templates each release, so the suite can't be memorized. A handful of "canary" items carry an instruction invisible to a hurried human, to catch harnesses that cheat via human post-processing.

The full specification — the normative rule set (R1–R12), every tier, the automatic verifier architecture, and ready-to-run example items — lives in [`docs/PTYX.md`](docs/PTYX.md).

---

## Project layout

```
azur/
  scansion.py      classical scansion (PTYX-lite)
  rhyme.py         phonetic endings, gender, richness, scheme
  semantics.py     fields, idiomatic segmentation, lyric axis, echoes
  lm.py            surprisal backends (wordfreq / n-gram / transformers), T, Δ
  scorer.py        F·N·Φ·O·D axes, instrumental regime, report
  judge.py         optional LLM judge (L1–L7 rubric, JSON)
data/
  champs.json      semantic resource (swappable for embeddings)
  idiomes.txt      frozen collocations
  scansion_tables.json   mute -ent / classical diérèse tables
tests/             56 tests
demo.py            runnable case study
azur_cli.py        text / file / stdin CLI, human or JSON output
docs/PTYX.md       full benchmark specification
```

## Requirements

Python ≥ 3.9 (tested on 3.9, 3.11, 3.12). Dependencies in `requirements.txt` (`wordfreq`, `pytest`). Optional: `anthropic` for the LLM judge; `torch` + `transformers` for the real surprisal backend.

## License

MIT — see [LICENSE](LICENSE).