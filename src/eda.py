"""Etapas 2, 3 e 4 — Análise Exploratória, Estatísticas e Visualizações.

Funções que produzem estatísticas descritivas e salvam as figuras da
análise exploratória no diretório ``figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")

# Paleta consistente para as duas classes em todos os gráficos.
PALETA = {"maligno": "#d62728", "benigno": "#2ca02c"}


# --------------------------------------------------------------------------- #
# Etapa 2 — Balanceamento e correlação
# --------------------------------------------------------------------------- #
def balanceamento_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna a contagem e o percentual de cada classe."""
    contagem = df["diagnostico_rotulo"].value_counts()
    perc = (contagem / len(df) * 100).round(2)
    return pd.DataFrame({"contagem": contagem, "percentual_%": perc})


def grafico_balanceamento(df: pd.DataFrame, caminho: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ordem = ["maligno", "benigno"]
    sns.countplot(data=df, x="diagnostico_rotulo", order=ordem,
                  hue="diagnostico_rotulo", palette=PALETA, legend=False, ax=ax)
    total = len(df)
    for p in ax.patches:
        altura = p.get_height()
        ax.annotate(f"{int(altura)}\n({altura / total * 100:.1f}%)",
                    (p.get_x() + p.get_width() / 2, altura),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title("Balanceamento das classes (diagnóstico)")
    ax.set_xlabel("Diagnóstico")
    ax.set_ylabel("Número de amostras")
    ax.set_ylim(0, df["diagnostico_rotulo"].value_counts().max() * 1.15)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def grafico_heatmap_correlacao(X: pd.DataFrame, caminho: Path) -> Path:
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax,
                annot=False, vmin=-1, vmax=1)
    ax.set_title("Mapa de calor — correlação entre os 30 atributos", fontsize=14)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def top_correlacoes_com_alvo(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """Atributos mais correlacionados com o diagnóstico (em módulo)."""
    corr = df.drop(columns=["diagnostico_rotulo"]).corr()["diagnostico"]
    corr = corr.drop("diagnostico")
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(n)


# --------------------------------------------------------------------------- #
# Etapa 3 — Estatísticas descritivas
# --------------------------------------------------------------------------- #
def estatisticas_descritivas(X: pd.DataFrame) -> pd.DataFrame:
    """Resumo estatístico completo (.describe + mediana)."""
    desc = X.describe().T
    desc["mediana"] = X.median()
    desc = desc.rename(columns={
        "mean": "media", "std": "desvio_padrao",
        "min": "minimo", "max": "maximo",
    })
    colunas = ["count", "media", "mediana", "desvio_padrao",
               "minimo", "25%", "50%", "75%", "maximo"]
    return desc[colunas].round(3)


def deteccao_outliers_iqr(X: pd.DataFrame) -> pd.DataFrame:
    """Conta outliers por atributo usando a regra do IQR (1.5*IQR)."""
    q1 = X.quantile(0.25)
    q3 = X.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    mascara = (X < lim_inf) | (X > lim_sup)
    contagem = mascara.sum()
    perc = (contagem / len(X) * 100).round(2)
    out = pd.DataFrame({"n_outliers": contagem, "percentual_%": perc})
    return out.sort_values("n_outliers", ascending=False)


# --------------------------------------------------------------------------- #
# Etapa 4 — Visualizações gráficas
# --------------------------------------------------------------------------- #
def grafico_histogramas(X: pd.DataFrame, caminho: Path) -> Path:
    n = X.shape[1]
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 2.6))
    axes = axes.ravel()
    for i, col in enumerate(X.columns):
        sns.histplot(X[col], kde=True, ax=axes[i], color="#4c72b0", bins=30)
        axes[i].set_title(col, fontsize=8)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Histogramas de distribuição dos 30 atributos", fontsize=15, y=1.001)
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_boxplots(X: pd.DataFrame, caminho: Path) -> Path:
    """Boxplots dos atributos padronizados (z-score) para comparação visual."""
    X_z = (X - X.mean()) / X.std()
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.boxplot(data=X_z, orient="h", ax=ax, fliersize=2, color="#8da0cb")
    ax.set_title("Boxplots dos atributos (padronizados em z-score) — detecção de outliers")
    ax.set_xlabel("Valor padronizado (z-score)")
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def grafico_scatter_pares(df: pd.DataFrame, caminho: Path) -> Path:
    """Scatter plots de pares de variáveis relevantes, coloridos por classe."""
    pares = [
        ("mean radius", "mean texture"),
        ("mean perimeter", "mean area"),
        ("mean concavity", "mean concave points"),
        ("worst radius", "worst concave points"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    for ax, (xv, yv) in zip(axes, pares):
        sns.scatterplot(data=df, x=xv, y=yv, hue="diagnostico_rotulo",
                        palette=PALETA, alpha=0.7, ax=ax, s=30)
        ax.set_title(f"{xv} x {yv}")
        ax.legend(title="diagnóstico")
    fig.suptitle("Dispersão entre pares de variáveis relevantes", fontsize=14)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def grafico_pairplot(df: pd.DataFrame, caminho: Path) -> Path:
    """Pairplot colorido por classe para um subconjunto de atributos 'mean'."""
    cols = ["mean radius", "mean texture", "mean concavity",
            "mean concave points", "diagnostico_rotulo"]
    g = sns.pairplot(df[cols], hue="diagnostico_rotulo", palette=PALETA,
                     diag_kind="kde", plot_kws={"alpha": 0.6, "s": 25})
    g.figure.suptitle("Pairplot dos principais atributos por classe", y=1.02)
    g.savefig(caminho, dpi=120)
    plt.close(g.figure)
    return caminho


if __name__ == "__main__":
    from data_loading import carregar_dataset, montar_dataframe_completo

    X, y, info = carregar_dataset()
    df = montar_dataframe_completo(X, y)
    print(balanceamento_classes(df))
    print("\nTop correlações com o alvo:")
    print(top_correlacoes_com_alvo(df))
