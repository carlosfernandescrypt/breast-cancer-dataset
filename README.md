# Breast Cancer Dataset — Machine Learning

Trabalho prático da disciplina de **Inteligência Artificial** — pipeline completo de
ciência de dados sobre o **Breast Cancer Wisconsin (Diagnostic) Dataset** (Scikit-Learn),
um problema de **classificação binária** (tumor **maligno** × **benigno**).

**Grupo 6** · Integrantes: _[Nomes dos integrantes]_

## Visão geral

- **569 amostras**, **30 atributos** numéricos contínuos, alvo binário (0 = maligno, 1 = benigno).
- Pipeline em 8 etapas: carregamento → EDA → estatísticas → visualizações →
  pré-processamento → split treino/teste → modelagem → avaliação.
- **4 algoritmos** comparados: Regressão Logística, KNN, Random Forest e SVM (RBF).

### Principais resultados (conjunto de teste, 114 amostras)

| Modelo | Acurácia | F1 | Recall maligno | ROC-AUC |
|---|---|---|---|---|
| **Regressão Logística** | **0,9825** | **0,9861** | 0,9762 | **0,9954** |
| **SVM (RBF)** | **0,9825** | **0,9861** | 0,9762 | 0,9950 |
| KNN | 0,9561 | 0,9655 | 0,9286 | 0,9788 |
| Random Forest | 0,9474 | 0,9583 | 0,9286 | 0,9937 |

## Estrutura do repositório

```
breast-cancer/
├── notebooks/
│   ├── breast_cancer_analise.ipynb   # Notebook executado (entregável principal)
│   └── breast_cancer_analise.html    # Versão HTML para leitura
├── src/                              # Código-fonte modular (entregável .py)
│   ├── data_loading.py               # Etapa 1 — carregamento
│   ├── eda.py                        # Etapas 2-4 — EDA, estatísticas, gráficos
│   ├── preprocessing.py              # Etapas 5-6 — padronização e split
│   ├── modeling.py                   # Etapa 7 — algoritmos e validação cruzada
│   ├── evaluation.py                 # Etapa 8 — métricas, ROC, matriz de confusão
│   └── pipeline.py                   # Orquestra todas as etapas
├── figures/                          # Gráficos e CSVs de resultados gerados
├── report/
│   ├── relatorio.md                  # Relatório acadêmico (fonte)
│   └── relatorio.pdf                 # Relatório acadêmico (PDF)
├── slides/
│   ├── slides.md                     # Slides Marp (fonte)
│   └── slides.pdf                    # Slides da apresentação (PDF)
├── build/                            # Scripts de geração (notebook e PDF)
├── requirements.txt
└── Makefile
```

## Como reproduzir

Requer Python 3.10+. Para os PDFs, é necessário **Chromium/Chrome** (relatório) e
**Node.js/npx** (slides via Marp).

```bash
# 1. Ambiente e dependências
make venv          # cria .venv e instala requirements.txt

# 2. Gerar tudo de uma vez (figuras, notebook, relatório e slides)
make all

# Ou individualmente:
make pipeline      # roda o pipeline e salva as figuras em figures/
make notebook      # constrói e executa o notebook (.ipynb + .html)
make report        # gera report/relatorio.pdf
make slides        # gera slides/slides.pdf
```

Sem `make`, com o venv ativado:

```bash
python -m src.pipeline
python build/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/breast_cancer_analise.ipynb
```

## Entregáveis

1. **Notebook Jupyter** (`notebooks/breast_cancer_analise.ipynb`) — código executado com outputs.
2. **Relatório acadêmico** (`report/relatorio.pdf`) — artigo estruturado.
3. **Slides** (`slides/slides.pdf`) — apresentação oral.
4. **Código-fonte** (`src/`) — scripts `.py` organizados por etapa.

## Referências

- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. *JMLR*, v. 12, 2011.
- WOLBERG, W. H.; MANGASARIAN, O. L. Multisurface method of pattern separation for medical
  diagnosis applied to breast cytology. *PNAS*, v. 87, n. 23, 1990.
- Scikit-Learn — Toy Datasets: https://scikit-learn.org/stable/datasets/toy_dataset.html
