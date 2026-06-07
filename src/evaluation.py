"""Etapa 8 — Avaliação dos Modelos.

Calcula as métricas de classificação (accuracy, precision, recall, F1,
ROC-AUC), gera matrizes de confusão e curvas ROC, e produz gráficos
comparativos entre os algoritmos.

No contexto de diagnóstico médico, dá-se atenção especial ao RECALL da
classe maligna (sensibilidade): um falso negativo (câncer classificado
como benigno) é o erro mais grave.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROTULOS = ["maligno", "benigno"]


def _probabilidades(modelo, X_test):
    """Obtém scores de probabilidade da classe positiva (benigno=1)."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X_test)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X_test)
    return None


def avaliar_modelos(modelos: dict, X_test, y_test) -> pd.DataFrame:
    """Tabela comparativa com as principais métricas no conjunto de teste."""
    linhas = []
    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_score = _probabilidades(modelo, X_test)
        linha = {
            "modelo": nome,
            "acuracia": accuracy_score(y_test, y_pred),
            "precisao": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            # Recall da classe maligna (0) = sensibilidade ao câncer.
            "recall_maligno": recall_score(y_test, y_pred, pos_label=0),
        }
        if y_score is not None:
            linha["roc_auc"] = roc_auc_score(y_test, y_score)
        linhas.append(linha)
    df = pd.DataFrame(linhas).set_index("modelo")
    return df.round(4).sort_values("f1", ascending=False)


def relatorio_classificacao(modelo, X_test, y_test) -> str:
    """classification_report textual para um modelo."""
    y_pred = modelo.predict(X_test)
    return classification_report(y_test, y_pred, target_names=ROTULOS)


def grafico_matriz_confusao(modelo, nome, X_test, y_test, caminho: Path) -> Path:
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ROTULOS)
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Matriz de confusão — {nome}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def grafico_matrizes_confusao_grade(modelos, X_test, y_test, caminho: Path) -> Path:
    """Matrizes de confusão de todos os modelos em uma grade 2x2."""
    n = len(modelos)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, (nome, modelo) in zip(axes, modelos.items()):
        cm = confusion_matrix(y_test, modelo.predict(X_test))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ROTULOS)
        disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
        ax.set_title(nome)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Matrizes de confusão — comparação dos modelos", fontsize=14)
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)
    return caminho


def grafico_curvas_roc(modelos, X_test, y_test, caminho: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome, modelo in modelos.items():
        y_score = _probabilidades(modelo, X_test)
        if y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, lw=2, label=f"{nome} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curvas ROC — comparação dos modelos")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def grafico_comparacao_metricas(resultados: pd.DataFrame, caminho: Path) -> Path:
    metricas = ["acuracia", "precisao", "recall", "f1"]
    dados = resultados[metricas]
    fig, ax = plt.subplots(figsize=(10, 6))
    dados.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    ax.set_title("Comparação de métricas entre os modelos (conjunto de teste)")
    ax.set_ylabel("Valor da métrica")
    ax.set_xlabel("")
    ax.set_ylim(0.8, 1.01)
    ax.legend(title="Métrica", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(dados.index, rotation=20, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def importancia_atributos(modelo_rf, feature_names, caminho: Path, top: int = 15) -> Path:
    """Importância dos atributos segundo o Random Forest."""
    imp = pd.Series(modelo_rf.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="rocket",
                legend=False, ax=ax)
    ax.set_title(f"Top {top} atributos mais importantes (Random Forest)")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho
