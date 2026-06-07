"""Gera o notebook breast_cancer_analise.ipynb (auto-contido).

O notebook é construído programaticamente com nbformat e depois executado
com nbconvert (ver Makefile / README). O código é inline e independente do
pacote src/, para que o notebook possa ser lido e executado de ponta a ponta.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "breast_cancer_analise.ipynb"

cells = []


def md(texto: str) -> None:
    cells.append(new_markdown_cell(texto.strip("\n")))


def code(texto: str) -> None:
    cells.append(new_code_cell(texto.strip("\n")))


# --------------------------------------------------------------------------- #
md(r"""
# Explorando Toy Datasets do Scikit-Learn — **Breast Cancer Dataset**
## Da Análise à Modelagem em Machine Learning

**Disciplina:** Inteligência Artificial · **Tipo de problema:** Classificação binária

**Grupo 6 — Integrantes:** _[Nomes dos integrantes]_

---

Este notebook conduz o pipeline completo de um projeto de ciência de dados sobre o
**Breast Cancer Wisconsin (Diagnostic) Dataset** (569 amostras, 30 atributos),
cujo objetivo é classificar tumores em **maligno** ou **benigno** a partir de
características extraídas de imagens digitalizadas de aspirados por agulha fina (FNA).

**Etapas:** (1) Carregamento · (2) EDA · (3) Estatísticas · (4) Visualizações ·
(5) Pré-processamento · (6) Split treino/teste · (7) Modelagem · (8) Avaliação.
""")

md("## Configuração do ambiente")
code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42
PALETA = {"maligno": "#d62728", "benigno": "#2ca02c"}
pd.set_option("display.max_columns", 40)
print("Bibliotecas carregadas com sucesso.")
""")

# ---- Etapa 1 ---- #
md(r"""
## Etapa 1 — Carregamento dos Dados

Carregamos o dataset via `load_breast_cancer` e exibimos dimensões, nomes dos
atributos, classes-alvo e os primeiros registros.
""")
code(r"""
dados = load_breast_cancer(as_frame=True)
X = dados.data.copy()
y = dados.target.copy()
y.name = "diagnostico"

print(f"Dimensões (amostras x atributos): {X.shape[0]} x {X.shape[1]}")
print(f"Classes-alvo (target_names): {list(dados.target_names)}  -> 0=maligno, 1=benigno")
print(f"\nNomes dos 30 atributos:\n{list(dados.feature_names)}")
""")
code(r"""
# DataFrame completo com rótulo textual para facilitar a leitura dos gráficos.
df = X.copy()
df["diagnostico"] = y.values
df["diagnostico_rotulo"] = df["diagnostico"].map({0: "maligno", 1: "benigno"})
df.head()
""")

# ---- Etapa 2 ---- #
md(r"""
## Etapa 2 — Análise Exploratória de Dados (EDA)

### 2.1 Balanceamento das classes
""")
code(r"""
contagem = df["diagnostico_rotulo"].value_counts()
perc = (contagem / len(df) * 100).round(2)
print(pd.DataFrame({"contagem": contagem, "percentual_%": perc}))

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.countplot(data=df, x="diagnostico_rotulo", order=["maligno", "benigno"],
              hue="diagnostico_rotulo", palette=PALETA, legend=False, ax=ax)
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{int(h)}\n({h/len(df)*100:.1f}%)",
                (p.get_x()+p.get_width()/2, h), ha="center", va="bottom")
ax.set_title("Balanceamento das classes (diagnóstico)")
ax.set_xlabel("Diagnóstico"); ax.set_ylabel("Número de amostras")
ax.set_ylim(0, contagem.max()*1.15)
plt.tight_layout(); plt.show()
""")
md(r"""
O dataset apresenta **desbalanceamento moderado**: ~37% maligno e ~63% benigno.
Não é severo, mas justifica o uso de **estratificação** no split e a observação de
métricas além da acurácia (precision, recall, F1).
""")

md("### 2.2 Correlação entre as variáveis")
code(r"""
corr = X.corr()
plt.figure(figsize=(15, 12))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
            linewidths=0.3, cbar_kws={"shrink": 0.7}, vmin=-1, vmax=1)
plt.title("Mapa de calor — correlação entre os 30 atributos")
plt.tight_layout(); plt.show()
""")
code(r"""
# Atributos mais correlacionados com o diagnóstico (em módulo).
corr_alvo = df.drop(columns=["diagnostico_rotulo"]).corr()["diagnostico"].drop("diagnostico")
top = corr_alvo.reindex(corr_alvo.abs().sort_values(ascending=False).index).head(10)
print("Top 10 correlações com o diagnóstico (0=maligno, 1=benigno):")
print(top)
""")
md(r"""
Atributos ligados a **tamanho e forma do núcleo celular** — `worst concave points`,
`worst perimeter`, `mean concave points`, `worst radius` — são os mais associados ao
diagnóstico (correlação negativa: valores maiores → mais propensão a **maligno**).
Há ainda forte **multicolinearidade** entre os grupos `radius`/`perimeter`/`area`
(esperado, pois descrevem geometricamente a mesma estrutura).
""")

# ---- Etapa 3 ---- #
md(r"""
## Etapa 3 — Estatísticas Descritivas
""")
code(r"""
desc = X.describe().T
desc["mediana"] = X.median()
desc = desc.rename(columns={"mean": "media", "std": "desvio_padrao",
                            "min": "minimo", "max": "maximo"})
desc[["media", "mediana", "desvio_padrao", "minimo", "maximo"]].round(3)
""")
code(r"""
# Detecção de outliers pela regra do IQR (1.5 * IQR).
q1, q3 = X.quantile(0.25), X.quantile(0.75)
iqr = q3 - q1
mascara = (X < (q1 - 1.5*iqr)) | (X > (q3 + 1.5*iqr))
outliers = pd.DataFrame({"n_outliers": mascara.sum(),
                         "percentual_%": (mascara.sum()/len(X)*100).round(2)})
print(outliers.sort_values("n_outliers", ascending=False).head(8))
""")
md(r"""
As escalas são **muito heterogêneas** (ex.: `mean area` na casa das centenas vs.
`mean smoothness` ~ 0,1), o que **exige padronização** antes de algoritmos sensíveis
à escala (KNN, SVM, Regressão Logística). Há outliers (notadamente em `area error`),
porém **clinicamente plausíveis** — representam tumores atípicos e não erros de
medição, portanto não serão removidos.
""")

# ---- Etapa 4 ---- #
md(r"""
## Etapa 4 — Visualizações Gráficas

### 4.1 Histogramas de distribuição
""")
code(r"""
n = X.shape[1]; ncols = 5; nrows = int(np.ceil(n/ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*2.6))
axes = axes.ravel()
for i, col in enumerate(X.columns):
    sns.histplot(X[col], kde=True, ax=axes[i], color="#4c72b0", bins=30)
    axes[i].set_title(col, fontsize=8); axes[i].set_xlabel(""); axes[i].set_ylabel("")
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("Histogramas dos 30 atributos", fontsize=15, y=1.001)
plt.tight_layout(); plt.show()
""")
md("### 4.2 Boxplots (atributos padronizados) — detecção de outliers")
code(r"""
X_z = (X - X.mean()) / X.std()
plt.figure(figsize=(18, 8))
sns.boxplot(data=X_z, orient="h", fliersize=2, color="#8da0cb")
plt.title("Boxplots dos atributos (z-score)")
plt.xlabel("Valor padronizado (z-score)")
plt.tight_layout(); plt.show()
""")
md("### 4.3 Dispersão entre pares de variáveis relevantes")
code(r"""
pares = [("mean radius", "mean texture"), ("mean perimeter", "mean area"),
         ("mean concavity", "mean concave points"), ("worst radius", "worst concave points")]
fig, axes = plt.subplots(2, 2, figsize=(13, 10)); axes = axes.ravel()
for ax, (xv, yv) in zip(axes, pares):
    sns.scatterplot(data=df, x=xv, y=yv, hue="diagnostico_rotulo",
                    palette=PALETA, alpha=0.7, ax=ax, s=30)
    ax.set_title(f"{xv} x {yv}"); ax.legend(title="diagnóstico")
fig.suptitle("Dispersão entre pares de variáveis relevantes", fontsize=14)
plt.tight_layout(); plt.show()
""")
md("### 4.4 Pairplot colorido por classe")
code(r"""
cols = ["mean radius", "mean texture", "mean concavity", "mean concave points", "diagnostico_rotulo"]
sns.pairplot(df[cols], hue="diagnostico_rotulo", palette=PALETA,
             diag_kind="kde", plot_kws={"alpha": 0.6, "s": 25})
plt.suptitle("Pairplot dos principais atributos por classe", y=1.02)
plt.show()
""")
md(r"""
Os gráficos mostram **boa separabilidade** entre as classes: tumores malignos
concentram-se em valores maiores de raio, perímetro, concavidade e pontos côncavos.
Há sobreposição parcial, o que indica que modelos lineares já devem ir bem, mas
sem fronteira perfeitamente separável.
""")

# ---- Etapa 5 ---- #
md(r"""
## Etapa 5 — Pré-processamento

- **Valores faltantes:** verificados (o dataset não possui).
- **Variáveis categóricas:** não há — todos os 30 atributos são numéricos contínuos.
- **Padronização:** aplicamos `StandardScaler` (média 0, desvio 1). É a escolha
  adequada porque as escalas variam em ordens de grandeza e algoritmos baseados em
  distância (KNN, SVM) e otimização (Regressão Logística) são sensíveis a isso.
  O scaler é ajustado **apenas no treino** para evitar *data leakage*.
""")
code(r"""
print("Total de valores faltantes:", int(X.isna().sum().sum()))
""")

# ---- Etapa 6 ---- #
md(r"""
## Etapa 6 — Separação Treino e Teste (80/20, estratificada)
""")
code(r"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
prop = pd.DataFrame({
    "treino_%": (y_train.value_counts(normalize=True).sort_index()*100).round(2),
    "teste_%": (y_test.value_counts(normalize=True).sort_index()*100).round(2),
}).rename(index={0: "maligno", 1: "benigno"})
print(prop)
""")
md("A estratificação preservou a proporção das classes (~37% / ~63%) em ambos os conjuntos.")

# ---- Etapa 7 ---- #
md(r"""
## Etapa 7 — Aplicação dos Algoritmos de Machine Learning

Comparamos **quatro** classificadores (acima do mínimo de dois):
Regressão Logística, KNN, Random Forest e SVM (kernel RBF).
""")
code(r"""
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}
for nome, modelo in modelos.items():
    modelo.fit(X_train_s, y_train)
print("Modelos treinados:", list(modelos))
""")
md("### 7.1 Validação cruzada (cv=5) no conjunto de treino")
code(r"""
linhas = []
for nome, modelo in modelos.items():
    acc = cross_val_score(modelo, X_train_s, y_train, cv=5, scoring="accuracy")
    f1 = cross_val_score(modelo, X_train_s, y_train, cv=5, scoring="f1")
    linhas.append({"modelo": nome,
                   "cv_acuracia": f"{acc.mean():.4f} ± {acc.std():.4f}",
                   "cv_f1": f"{f1.mean():.4f} ± {f1.std():.4f}"})
pd.DataFrame(linhas).set_index("modelo")
""")

# ---- Etapa 8 ---- #
md(r"""
## Etapa 8 — Avaliação dos Modelos

No contexto de **diagnóstico médico**, o erro mais grave é o **falso negativo**
(classificar um tumor maligno como benigno). Por isso, além das métricas globais,
destacamos o **recall da classe maligna** (sensibilidade).
""")
code(r"""
def scores_prob(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)

linhas = []
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test_s)
    linhas.append({
        "modelo": nome,
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall_maligno": recall_score(y_test, y_pred, pos_label=0),
        "roc_auc": roc_auc_score(y_test, scores_prob(modelo, X_test_s)),
    })
resultados = pd.DataFrame(linhas).set_index("modelo").round(4).sort_values("f1", ascending=False)
resultados
""")
md("### 8.1 Comparação gráfica das métricas")
code(r"""
metricas = ["acuracia", "precisao", "recall", "f1"]
ax = resultados[metricas].plot(kind="bar", figsize=(10, 6), width=0.8, colormap="viridis")
ax.set_title("Comparação de métricas entre os modelos (teste)")
ax.set_ylabel("Valor"); ax.set_xlabel(""); ax.set_ylim(0.8, 1.01)
ax.legend(title="Métrica", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.show()
""")
md("### 8.2 Matrizes de confusão")
code(r"""
fig, axes = plt.subplots(2, 2, figsize=(11, 9)); axes = axes.ravel()
for ax, (nome, modelo) in zip(axes, modelos.items()):
    cm = confusion_matrix(y_test, modelo.predict(X_test_s))
    ConfusionMatrixDisplay(cm, display_labels=["maligno", "benigno"]).plot(
        cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(nome); ax.set_xlabel("Predito"); ax.set_ylabel("Real")
fig.suptitle("Matrizes de confusão — comparação dos modelos", fontsize=14)
plt.tight_layout(); plt.show()
""")
md("### 8.3 Curvas ROC")
code(r"""
plt.figure(figsize=(7, 6))
for nome, modelo in modelos.items():
    fpr, tpr, _ = roc_curve(y_test, scores_prob(modelo, X_test_s))
    auc = roc_auc_score(y_test, scores_prob(modelo, X_test_s))
    plt.plot(fpr, tpr, lw=2, label=f"{nome} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
plt.xlabel("Taxa de falsos positivos"); plt.ylabel("Taxa de verdadeiros positivos (recall)")
plt.title("Curvas ROC"); plt.legend(loc="lower right")
plt.tight_layout(); plt.show()
""")
md("### 8.4 Relatório de classificação do melhor modelo e importância de atributos")
code(r"""
melhor = resultados.index[0]
print(f"Melhor modelo por F1: {melhor}\n")
print(classification_report(y_test, modelos[melhor].predict(X_test_s),
                            target_names=["maligno", "benigno"]))
""")
code(r"""
imp = pd.Series(modelos["Random Forest"].feature_importances_, index=X.columns)
imp = imp.sort_values(ascending=False).head(15)
plt.figure(figsize=(9, 6))
sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="rocket", legend=False)
plt.title("Top 15 atributos mais importantes (Random Forest)")
plt.xlabel("Importância"); plt.tight_layout(); plt.show()
""")

# ---- Conclusão ---- #
md(r"""
## Discussão e Conclusão

- Todos os modelos atingiram desempenho **alto** (acurácia ≥ 0,94; ROC-AUC ≥ 0,97),
  confirmando a boa separabilidade observada na EDA.
- A **Regressão Logística** e o **SVM (RBF)** lideraram (acurácia ≈ **0,98**,
  ROC-AUC ≈ **0,995**), com apenas **2 erros** em 114 amostras de teste e **recall
  da classe maligna de 0,976** (1 falso negativo).
- A **padronização** foi decisiva para o bom desempenho dos modelos baseados em
  distância/otimização.
- **Aplicação real:** um classificador como este pode atuar como **sistema de apoio
  à decisão** para o radiologista/patologista, priorizando casos suspeitos — sempre
  como *triagem*, nunca substituindo o laudo médico.
- **Limitações:** dataset pequeno e de uma única fonte; hiperparâmetros não foram
  exaustivamente otimizados (espaço para `GridSearchCV`); avaliar custo assimétrico
  dos erros (falso negativo ≫ falso positivo) com ajuste de limiar de decisão.
""")

# --------------------------------------------------------------------------- #
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3", "language": "python", "name": "python3",
}
nb.metadata["language_info"] = {"name": "python"}

DESTINO.parent.mkdir(exist_ok=True)
with open(DESTINO, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook gerado: {DESTINO} ({len(cells)} células)")
