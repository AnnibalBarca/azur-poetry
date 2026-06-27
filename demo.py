# -*- coding: utf-8 -*-
"""Démo AZUR : reproduit l'étude de cas du §3 de la spec, avec de vrais chiffres.

  1. V1 lyrique vs V2 instrumental (les deux vers-tests)
  2. Quatrain de Baudelaire vs sa paraphrase prosaïque (paire minimale,
     destruction contrôlée type P4 — le score DOIT chuter)
  3. Piège adversarial : faux alexandrin
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from azur.scorer import score, render

V1 = "Laissez la nuit couver l'œuf bleu de l'angoisse"
V2 = "Et mange ta soupe qui sent la poisse"

BAUDELAIRE = """Souvent, pour s'amuser, les hommes d'équipage
Prennent des albatros, vastes oiseaux des mers,
Qui suivent, indolents compagnons de voyage,
Le navire glissant sur les gouffres amers."""

PARAPHRASE = """Souvent, pour s'amuser, les marins du bateau
attrapent des albatros, de grands oiseaux de mer,
qui suivent tranquillement pendant le voyage
le bateau qui avance sur l'eau profonde."""

FAUX = "Le navire glissant lentement sur les eaux calmes et profondes"

if __name__ == "__main__":
    print("=" * 72)
    print("1) LES DEUX VERS-TESTS (spec AZUR §3)")
    r1, r2 = score(V1), score(V2)
    print(render(r1))
    print(render(r2))
    assert r1.score > 2.2 * r2.score, "V1 doit dominer V2 largement"

    print()
    print("=" * 72)
    print("2) PAIRE MINIMALE : Baudelaire vs paraphrase prosaïque (test L3)")
    rb, rp = score(BAUDELAIRE), score(PARAPHRASE)
    print(render(rb))
    print(render(rp))
    assert rb.score > rp.score, "l'original doit dominer sa paraphrase"
    print(f"\n  Résidu de paraphrase (chute de score) : "
          f"{rb.score} -> {rp.score}  (-{rb.score - rp.score:.1f} pts)")

    print()
    print("=" * 72)
    print("3) ADVERSARIAL : « analysez la césure de cet alexandrin » (E.x)")
    from azur.scansion import scan_line
    s = scan_line(FAUX)
    print(f"  «{FAUX}»")
    print(f"  -> {s.total} syllabes : PRÉMISSE FAUSSE, ce n'est pas un"
          f" alexandrin ; aucune césure classique à analyser.")
    assert s.total != 12
    print("\nTous les invariants de la démo sont vérifiés. ✓")
