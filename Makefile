MAIN = global_copulas_concordance

pdf:
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

figures:
	cd scripts && python3 make_figures.py

table:
	cd scripts && python3 table_values.py

verify:
	cd scripts && python3 verify_results.py

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).blg $(MAIN).out $(MAIN).spl

.PHONY: pdf figures table verify clean
