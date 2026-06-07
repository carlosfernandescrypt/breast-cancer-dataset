"""Etapas 5 e 6 — Pré-processamento e separação treino/teste.

O Breast Cancer Dataset não possui valores faltantes nem variáveis
categóricas, portanto o pré-processamento se concentra na padronização
(StandardScaler), justificada pela grande diferença de escala entre os
atributos (ex.: 'mean area' ~ centenas vs. 'mean smoothness' ~ 0,1).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def checar_valores_faltantes(X: pd.DataFrame) -> int:
    """Retorna o total de valores faltantes (esperado: 0)."""
    return int(X.isna().sum().sum())


def separar_treino_teste(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Divide em treino/teste (80/20) com estratificação pelas classes."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,  # mantém a proporção maligno/benigno nos dois conjuntos
    )
    return X_train, X_test, y_train, y_test


def padronizar(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Ajusta o StandardScaler no treino e aplica em treino e teste.

    Importante: o scaler é ajustado APENAS no treino para evitar
    vazamento de informação (data leakage) do conjunto de teste.
    """
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index,
    )
    return X_train_s, X_test_s, scaler


def proporcao_classes(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Compara a proporção das classes após a divisão."""
    def perc(s):
        return (s.value_counts(normalize=True).sort_index() * 100).round(2)

    return pd.DataFrame({
        "treino_%": perc(y_train),
        "teste_%": perc(y_test),
    }).rename(index={0: "maligno", 1: "benigno"})


if __name__ == "__main__":
    from data_loading import carregar_dataset

    X, y, _ = carregar_dataset()
    print("Valores faltantes:", checar_valores_faltantes(X))
    X_tr, X_te, y_tr, y_te = separar_treino_teste(X, y)
    X_tr_s, X_te_s, _ = padronizar(X_tr, X_te)
    print(f"Treino: {X_tr.shape} | Teste: {X_te.shape}")
    print(proporcao_classes(y_tr, y_te))
