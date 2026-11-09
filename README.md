# Multivariate concordance of global copulas as the number of inputs grows

Manuscript prepared for *Information Sciences* (Elsevier), with the code used
to compute Table 1 and to draw Figures 1 and 2.

## Contents

| File | Description |
|------|-------------|
| `global_copulas_concordance.tex` | Manuscript (class `elsarticle`, numbered references) |
| `references.bib`, `global_copulas_concordance.bbl` | Bibliography (BibTeX database and processed list) |
| `fig_region.pdf`, `fig_limits.pdf` | Figures 1 and 2 |
| `elsarticle.cls`, `elsarticle-num.bst` | Elsevier class and bibliography style |
| `highlights.txt` | Highlights (separate file required by the journal) |
| `global_copulas_concordance.pdf` | Compiled manuscript |
| `scripts/table_values.py` | Values of Table 1 |
| `scripts/make_figures.py` | Figures 1 and 2 (written to this folder) |
| `scripts/verify_results.py` | Numerical checks of the formulas in the paper |

## Compiling

    pdflatex global_copulas_concordance
    bibtex   global_copulas_concordance
    pdflatex global_copulas_concordance
    pdflatex global_copulas_concordance

or simply `make`.

## Code

Python 3 with NumPy, SciPy and Matplotlib (`pip install -r scripts/requirements.txt`).

    cd scripts
    python3 table_values.py      # prints the rows of Table 1
    python3 make_figures.py      # writes fig_region.pdf and fig_limits.pdf
    python3 verify_results.py    # about 150 checks; ends with "ALL CHECKS PASSED"

`verify_results.py` compares the formulas of the paper (Corollary 5.2, Example 5.6,
Theorem 5.4, Remark 4.3, Section 6 and Table 1) with independent computations:
exact formulas, numerical integration and Monte Carlo simulation.
