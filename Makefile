# Makefile — reprodução do projeto Breast Cancer Dataset
# Usa o Python do venv local (.venv) por padrão.
PY := .venv/bin/python
JUPYTER := .venv/bin/jupyter

.PHONY: help venv pipeline notebook report slides all clean

help:
	@echo "Alvos disponíveis:"
	@echo "  make venv      - cria o virtualenv e instala as dependências"
	@echo "  make pipeline  - roda o pipeline e gera as figuras em figures/"
	@echo "  make notebook  - constrói e executa o notebook (.ipynb + .html)"
	@echo "  make report    - gera o relatório em PDF (report/relatorio.pdf)"
	@echo "  make slides    - gera os slides em PDF (slides/slides.pdf)"
	@echo "  make all       - executa pipeline + notebook + report + slides"
	@echo "  make clean      - remove artefatos gerados"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

pipeline:
	$(PY) -m src.pipeline

notebook:
	$(PY) build/build_notebook.py
	$(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=300 notebooks/breast_cancer_analise.ipynb
	$(JUPYTER) nbconvert --to html notebooks/breast_cancer_analise.ipynb

report:
	$(PY) build/md_to_pdf.py report/relatorio.md report/relatorio.pdf "Relatório - Breast Cancer"

slides:
	npx -y @marp-team/marp-cli@latest slides/slides.md -o slides/slides.pdf --pdf --allow-local-files

all: pipeline notebook report slides

clean:
	rm -rf figures/*.png figures/*.csv
	rm -f notebooks/*.html report/*.pdf slides/*.pdf
	find . -type d -name __pycache__ -exec rm -rf {} +
