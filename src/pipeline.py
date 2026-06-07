"""Pipeline completo — executa as Etapas 1 a 8 de ponta a ponta.

Roda todo o fluxo de Machine Learning, salva as figuras em ``figures/``
e grava as tabelas de resultados em ``figures/`` (CSV). É a forma
"script" do projeto (entregável de código-fonte organizado), espelhando
o conteúdo do notebook.

Uso:
    python -m src.pipeline
    # ou, de dentro de src/:  python pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Permite executar tanto como módulo (`python -m src.pipeline`) quanto
# diretamente (`python src/pipeline.py`).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data_loading as dl
    import eda
    import evaluation as ev
    import modeling as md
    import preprocessing as pp
else:
    from . import data_loading as dl
    from . import eda
    from . import evaluation as ev
    from . import modeling as md
    from . import preprocessing as pp

RAIZ = Path(__file__).resolve().parent.parent
FIG = RAIZ / "figures"


def main() -> None:
    FIG.mkdir(exist_ok=True)

    # ---- Etapa 1 — Carregamento --------------------------------------- #
    X, y, info = dl.carregar_dataset()
    df = dl.montar_dataframe_completo(X, y)
    print(dl.resumo_carregamento(X, y, info))

    # ---- Etapa 2 — EDA ------------------------------------------------ #
    print("\n=== Etapa 2 — Balanceamento das classes ===")
    print(eda.balanceamento_classes(df))
    eda.grafico_balanceamento(df, FIG / "01_balanceamento_classes.png")
    eda.grafico_heatmap_correlacao(X, FIG / "04_heatmap_correlacao.png")
    print("\nTop 10 correlações com o diagnóstico:")
    print(eda.top_correlacoes_com_alvo(df))

    # ---- Etapa 3 — Estatísticas descritivas --------------------------- #
    desc = eda.estatisticas_descritivas(X)
    desc.to_csv(FIG / "estatisticas_descritivas.csv")
    outliers = eda.deteccao_outliers_iqr(X)
    print("\n=== Etapa 3 — Outliers (top 5 atributos por IQR) ===")
    print(outliers.head())

    # ---- Etapa 4 — Visualizações -------------------------------------- #
    eda.grafico_histogramas(X, FIG / "02_histogramas.png")
    eda.grafico_boxplots(X, FIG / "03_boxplots.png")
    eda.grafico_scatter_pares(df, FIG / "05_scatter_pares.png")
    eda.grafico_pairplot(df, FIG / "06_pairplot.png")

    # ---- Etapa 5 e 6 — Pré-processamento e split ---------------------- #
    print("\n=== Etapa 5/6 — Pré-processamento e split ===")
    print("Valores faltantes:", pp.checar_valores_faltantes(X))
    X_tr, X_te, y_tr, y_te = pp.separar_treino_teste(X, y)
    X_tr_s, X_te_s, _ = pp.padronizar(X_tr, X_te)
    print(f"Treino: {X_tr.shape} | Teste: {X_te.shape}")
    print(pp.proporcao_classes(y_tr, y_te))

    # ---- Etapa 7 — Modelagem ------------------------------------------ #
    modelos = md.treinar_modelos(md.construir_modelos(), X_tr_s, y_tr)
    cv = md.validacao_cruzada(md.construir_modelos(), X_tr_s, y_tr)
    cv.to_csv(FIG / "resultados_validacao_cruzada.csv")
    print("\n=== Etapa 7 — Validação cruzada (cv=5) ===")
    print(cv)

    # ---- Etapa 8 — Avaliação ------------------------------------------ #
    resultados = ev.avaliar_modelos(modelos, X_te_s, y_te)
    resultados.to_csv(FIG / "resultados_teste.csv")
    print("\n=== Etapa 8 — Métricas no conjunto de teste ===")
    print(resultados)

    ev.grafico_matrizes_confusao_grade(modelos, X_te_s, y_te,
                                       FIG / "07_matrizes_confusao.png")
    ev.grafico_curvas_roc(modelos, X_te_s, y_te, FIG / "08_curvas_roc.png")
    ev.grafico_comparacao_metricas(resultados, FIG / "09_comparacao_metricas.png")
    ev.importancia_atributos(modelos["Random Forest"], X.columns,
                             FIG / "10_importancia_atributos.png")

    melhor = resultados.index[0]
    print(f"\n>>> Melhor modelo por F1: {melhor}")
    print(ev.relatorio_classificacao(modelos[melhor], X_te_s, y_te))
    print(f"Figuras e CSVs salvos em: {FIG}")


if __name__ == "__main__":
    main()
