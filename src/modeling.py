"""Etapa 7 — Aplicação dos Algoritmos de Machine Learning.

Implementa e treina quatro classificadores (mais que o mínimo de dois
exigido) para comparação: Regressão Logística, KNN, Random Forest e SVM.
Também executa validação cruzada (cv=5) para uma estimativa robusta.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

RANDOM_STATE = 42


def construir_modelos() -> dict:
    """Retorna o dicionário de modelos a serem comparados.

    Observação: SVM e Regressão Logística recebem ``probability``/padrão
    adequados para permitir o cálculo de probabilidades (curva ROC).
    """
    return {
        "Regressão Logística": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def treinar_modelos(modelos: dict, X_train, y_train) -> dict:
    """Treina cada modelo no conjunto de treino padronizado."""
    treinados = {}
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        treinados[nome] = modelo
    return treinados


def validacao_cruzada(modelos: dict, X_train, y_train, cv: int = 5) -> pd.DataFrame:
    """Calcula acurácia e F1 médios via validação cruzada (cv=5)."""
    linhas = []
    for nome, modelo in modelos.items():
        acc = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy")
        f1 = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="f1")
        linhas.append({
            "modelo": nome,
            "cv_acuracia_media": acc.mean().round(4),
            "cv_acuracia_desvio": acc.std().round(4),
            "cv_f1_media": f1.mean().round(4),
            "cv_f1_desvio": f1.std().round(4),
        })
    return pd.DataFrame(linhas).set_index("modelo")


if __name__ == "__main__":
    from data_loading import carregar_dataset
    from preprocessing import padronizar, separar_treino_teste

    X, y, _ = carregar_dataset()
    X_tr, X_te, y_tr, y_te = separar_treino_teste(X, y)
    X_tr_s, X_te_s, _ = padronizar(X_tr, X_te)
    modelos = treinar_modelos(construir_modelos(), X_tr_s, y_tr)
    print(validacao_cruzada(construir_modelos(), X_tr_s, y_tr))
