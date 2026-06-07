"""Etapa 1 — Carregamento dos Dados.

Carrega o Breast Cancer Wisconsin (Diagnostic) Dataset diretamente do
``sklearn.datasets`` e o disponibiliza como ``pandas.DataFrame`` para as
demais etapas do pipeline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer


def carregar_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Carrega o Breast Cancer Dataset.

    Returns:
        X: DataFrame (569 x 30) com os atributos numéricos.
        y: Series com o alvo (0 = maligno, 1 = benigno).
        info: dicionário com metadados (nomes das classes, descrição etc.).
    """
    dados = load_breast_cancer(as_frame=True)

    X = dados.data.copy()
    y = dados.target.copy()
    y.name = "diagnostico"

    info = {
        "feature_names": list(dados.feature_names),
        "target_names": list(dados.target_names),  # ['malignant', 'benign']
        "n_amostras": X.shape[0],
        "n_atributos": X.shape[1],
        "descricao": dados.DESCR,
    }
    return X, y, info


def montar_dataframe_completo(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Junta atributos e alvo em um único DataFrame (útil para EDA)."""
    df = X.copy()
    df["diagnostico"] = y.values
    # Rótulo textual ajuda na leitura de gráficos e tabelas.
    df["diagnostico_rotulo"] = df["diagnostico"].map({0: "maligno", 1: "benigno"})
    return df


def resumo_carregamento(X: pd.DataFrame, y: pd.Series, info: dict) -> str:
    """Gera um texto-resumo das dimensões e estrutura do dataset."""
    contagem = y.value_counts().sort_index()
    linhas = [
        "=== Etapa 1 — Carregamento dos Dados ===",
        f"Dimensões (amostras x atributos): {info['n_amostras']} x {info['n_atributos']}",
        f"Classes-alvo: {info['target_names']} (0=maligno, 1=benigno)",
        f"Distribuição do alvo: maligno={contagem.get(0, 0)} | benigno={contagem.get(1, 0)}",
        f"Primeiros atributos: {info['feature_names'][:5]} ...",
    ]
    return "\n".join(linhas)


if __name__ == "__main__":
    X, y, info = carregar_dataset()
    print(resumo_carregamento(X, y, info))
    print("\nPrimeiros registros:")
    print(montar_dataframe_completo(X, y).head())
