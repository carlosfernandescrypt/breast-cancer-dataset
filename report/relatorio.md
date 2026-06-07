# Classificação de Tumores Mamários com Machine Learning: Análise do Breast Cancer Wisconsin (Diagnostic) Dataset

**Disciplina de Inteligência Artificial — Trabalho Prático em Grupo**
**Grupo 6 — Breast Cancer Dataset (Classificação Binária)**

**Integrantes:** Carlos, Pedro, Luiz, Victor, Vinicius e Wesley

**Data:** Junho de 2026

---

## Resumo

Este trabalho apresenta um pipeline completo de ciência de dados aplicado ao *Breast Cancer Wisconsin (Diagnostic) Dataset*, disponibilizado pela biblioteca Scikit-Learn. O objetivo é classificar tumores mamários como **malignos** ou **benignos** a partir de 30 atributos extraídos de imagens digitalizadas de aspirados por agulha fina (FNA). Foram conduzidas as etapas de análise exploratória, pré-processamento e modelagem, comparando-se quatro algoritmos de classificação. Os modelos de **Regressão Logística** e **SVM (RBF)** alcançaram acurácia de **98,25%** e ROC-AUC de **0,995** no conjunto de teste, com apenas um falso negativo. Discute-se a relevância da métrica de *recall* da classe maligna no contexto de diagnóstico médico.

**Palavras-chave:** Machine Learning; Classificação Binária; Diagnóstico de Câncer de Mama; Scikit-Learn.

---

## 1. Introdução

O câncer de mama é uma das principais causas de mortalidade entre mulheres no mundo, e o diagnóstico precoce é determinante para o prognóstico. A análise citológica de aspirados por agulha fina (*Fine Needle Aspirate* — FNA) permite caracterizar quantitativamente os núcleos celulares presentes em uma amostra tumoral. O *Breast Cancer Wisconsin (Diagnostic) Dataset*, originado dos trabalhos de Wolberg e Mangasarian (1990), reúne essas medições e tornou-se um conjunto de referência para problemas de classificação supervisionada.

O dataset contém **569 amostras** e **30 atributos numéricos** contínuos, derivados de dez características de núcleo celular (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), reportadas em três agregações: média (*mean*), erro-padrão (*error*) e pior caso (*worst*). O alvo é binário: **maligno** (0) ou **benigno** (1).

Este trabalho tem como objetivo desenvolver e avaliar criticamente modelos de Machine Learning capazes de auxiliar nesse diagnóstico, percorrendo todo o ciclo de um projeto de ciência de dados — do carregamento e exploração dos dados à modelagem, avaliação e interpretação dos resultados.

## 2. Metodologia

O pipeline foi implementado em Python com as bibliotecas `pandas`, `numpy`, `matplotlib`, `seaborn` e `scikit-learn`, organizado nas oito etapas descritas a seguir.

### 2.1 Carregamento e exploração

Os dados foram carregados via `sklearn.datasets.load_breast_cancer`. A distribuição das classes (Figura 1) revela **desbalanceamento moderado**: 212 amostras malignas (37,3%) e 357 benignas (62,7%). Esse desbalanceamento, embora não severo, motivou o uso de **estratificação** na divisão treino/teste e a análise de métricas além da acurácia.

![Figura 1 — Balanceamento das classes](figures/01_balanceamento_classes.png)
<p class="legenda">Figura 1 — Distribuição das classes do diagnóstico.</p>

A análise de correlação (Figura 2) evidenciou forte associação entre o diagnóstico e atributos de **tamanho e forma do núcleo**. Os dez atributos mais correlacionados com o alvo foram, em ordem decrescente de magnitude: `worst concave points` (−0,79), `worst perimeter` (−0,78), `mean concave points` (−0,78), `worst radius` (−0,78), `mean perimeter` (−0,74), `worst area` (−0,73), `mean radius` (−0,73), `mean area` (−0,71), `mean concavity` (−0,70) e `worst concavity` (−0,66). A correlação negativa indica que valores maiores desses atributos estão associados a tumores **malignos**. Observou-se ainda forte **multicolinearidade** entre os grupos `radius`/`perimeter`/`area`, esperada por descreverem geometricamente a mesma estrutura.

![Figura 2 — Mapa de calor de correlação](figures/04_heatmap_correlacao.png)
<p class="legenda">Figura 2 — Mapa de calor da correlação entre os 30 atributos.</p>

### 2.2 Estatísticas descritivas

As estatísticas descritivas confirmaram a **heterogeneidade de escalas** entre os atributos — por exemplo, `mean area` na ordem de centenas frente a `mean smoothness` na ordem de 0,1. A detecção de outliers pela regra do IQR (1,5 × IQR) apontou `area error` (11,4%), `radius error` (6,7%) e `perimeter error` (6,7%) como os atributos com mais valores extremos. Tais outliers foram considerados **clinicamente plausíveis** (tumores atípicos) e não erros de medição, sendo, portanto, mantidos. As distribuições e os outliers são visualizados nas Figuras 3 e 4.

![Figura 3 — Histogramas](figures/02_histogramas.png)
<p class="legenda">Figura 3 — Histogramas de distribuição dos 30 atributos.</p>

![Figura 4 — Boxplots](figures/03_boxplots.png)
<p class="legenda">Figura 4 — Boxplots dos atributos padronizados (z-score).</p>

A separabilidade entre classes é evidenciada nos gráficos de dispersão e no pairplot (Figuras 5 e 6): tumores malignos concentram-se em valores mais altos de raio, perímetro, concavidade e pontos côncavos, com sobreposição apenas parcial.

![Figura 5 — Dispersão entre pares](figures/05_scatter_pares.png)
<p class="legenda">Figura 5 — Dispersão entre pares de variáveis relevantes, por classe.</p>

![Figura 6 — Pairplot](figures/06_pairplot.png)
<p class="legenda">Figura 6 — Pairplot dos principais atributos, colorido por classe.</p>

### 2.3 Pré-processamento e divisão dos dados

O dataset **não possui valores faltantes nem variáveis categóricas**. Aplicou-se a **padronização** com `StandardScaler` (média 0, desvio 1), justificada pela disparidade de escalas e pela sensibilidade dos algoritmos baseados em distância (KNN, SVM) e em otimização (Regressão Logística). O *scaler* foi ajustado **exclusivamente no conjunto de treino**, evitando vazamento de informação (*data leakage*).

Os dados foram divididos em **80% treino (455 amostras) e 20% teste (114 amostras)**, com `random_state=42` e **estratificação** pelas classes, o que preservou a proporção de ~37%/63% em ambos os conjuntos.

### 2.4 Modelagem e avaliação

Foram implementados e comparados **quatro classificadores** (acima do mínimo de dois exigido): **Regressão Logística**, **K-Nearest Neighbors** (k=5), **Random Forest** (300 árvores) e **SVM** com kernel RBF. Utilizou-se **validação cruzada (cv=5)** no treino para uma estimativa robusta, e a avaliação final no conjunto de teste empregou as métricas de **acurácia, precisão, recall, F1-score, ROC-AUC e matriz de confusão**.

## 3. Resultados

A Tabela 1 apresenta o desempenho na validação cruzada (cv=5) sobre o conjunto de treino.

**Tabela 1 — Validação cruzada (cv=5), conjunto de treino**

| Modelo | Acurácia (média ± desvio) | F1 (média ± desvio) |
|---|---|---|
| Regressão Logística | 0,9802 ± 0,0128 | 0,9843 ± 0,0101 |
| SVM (RBF) | 0,9714 ± 0,0179 | 0,9773 ± 0,0140 |
| KNN | 0,9670 ± 0,0209 | 0,9741 ± 0,0164 |
| Random Forest | 0,9582 ± 0,0224 | 0,9668 ± 0,0174 |

A Tabela 2 consolida as métricas no **conjunto de teste**. A coluna *Recall maligno* corresponde à sensibilidade para a classe de maior gravidade clínica.

**Tabela 2 — Métricas no conjunto de teste (114 amostras)**

| Modelo | Acurácia | Precisão | Recall | F1 | Recall maligno | ROC-AUC |
|---|---|---|---|---|---|---|
| Regressão Logística | **0,9825** | 0,9861 | 0,9861 | **0,9861** | 0,9762 | **0,9954** |
| SVM (RBF) | **0,9825** | 0,9861 | 0,9861 | **0,9861** | 0,9762 | 0,9950 |
| KNN | 0,9561 | 0,9589 | 0,9722 | 0,9655 | 0,9286 | 0,9788 |
| Random Forest | 0,9474 | 0,9583 | 0,9583 | 0,9583 | 0,9286 | 0,9937 |

As Figuras 7, 8 e 9 ilustram, respectivamente, a comparação das métricas, as matrizes de confusão e as curvas ROC.

![Figura 7 — Comparação de métricas](figures/09_comparacao_metricas.png)
<p class="legenda">Figura 7 — Comparação das métricas entre os modelos no conjunto de teste.</p>

![Figura 8 — Matrizes de confusão](figures/07_matrizes_confusao.png)
<p class="legenda">Figura 8 — Matrizes de confusão dos quatro modelos.</p>

![Figura 9 — Curvas ROC](figures/08_curvas_roc.png)
<p class="legenda">Figura 9 — Curvas ROC e respectivos valores de AUC.</p>

A importância de atributos segundo o Random Forest (Figura 10) reforça os achados da EDA: `worst concave points`, `worst perimeter`, `worst area` e `mean concave points` figuram entre os mais determinantes.

![Figura 10 — Importância de atributos](figures/10_importancia_atributos.png)
<p class="legenda">Figura 10 — Top 15 atributos mais importantes (Random Forest).</p>

## 4. Discussão

Todos os modelos atingiram desempenho elevado (acurácia ≥ 0,94; ROC-AUC ≥ 0,97), confirmando a boa separabilidade observada na análise exploratória. A **Regressão Logística** e o **SVM (RBF)** lideraram, com acurácia de **98,25%** e apenas **dois erros** em 114 amostras de teste. O fato de um modelo linear simples igualar o SVM sugere que as classes são, em grande medida, **linearmente separáveis** no espaço padronizado — resultado coerente com a estrutura dos dados.

No contexto de **diagnóstico médico**, a métrica mais crítica é o **recall da classe maligna** (sensibilidade), pois um **falso negativo** — classificar como benigno um tumor maligno — pode atrasar o tratamento e ter consequências graves. Os melhores modelos atingiram recall maligno de **0,976**, equivalente a **um único falso negativo** no conjunto de teste. A padronização mostrou-se decisiva para o desempenho dos modelos baseados em distância e otimização.

## 5. Conclusão

O trabalho demonstrou, de ponta a ponta, a aplicação de um pipeline de Machine Learning a um problema real de diagnóstico médico, com resultados consistentes e elevada acurácia. Um classificador como o desenvolvido pode atuar, na prática profissional, como **sistema de apoio à decisão** — auxiliando radiologistas e patologistas na **triagem** e priorização de casos suspeitos —, sempre como ferramenta complementar e **nunca em substituição ao laudo médico**.

Como **limitações**, destacam-se: o tamanho reduzido e a origem única do conjunto de dados, que limitam a generalização; a ausência de otimização exaustiva de hiperparâmetros (oportunidade para `GridSearchCV`); e a não consideração explícita do **custo assimétrico dos erros** (falso negativo ≫ falso positivo), que poderia ser tratada com ajuste do limiar de decisão. Como trabalhos futuros, sugere-se a validação em bases externas, a calibração de probabilidades e a análise de interpretabilidade (ex.: SHAP) para uso clínico responsável.

## 6. Referências

GÉRON, A. **Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras e TensorFlow**. 2. ed. Rio de Janeiro: Alta Books, 2021.

McKINNEY, W. **Python para Análise de Dados**. 2. ed. São Paulo: Novatec, 2018.

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825–2830, 2011.

RASCHKA, S.; MIRJALILI, V. **Python Machine Learning**. 3. ed. Birmingham: Packt Publishing, 2019.

SCIKIT-LEARN DEVELOPERS. **Toy Datasets — Scikit-Learn Documentation**. Disponível em: https://scikit-learn.org/stable/datasets/toy_dataset.html. Acesso em: jun. 2026.

WOLBERG, W. H.; MANGASARIAN, O. L. Multisurface method of pattern separation for medical diagnosis applied to breast cytology. **Proceedings of the National Academy of Sciences**, v. 87, n. 23, p. 9193–9196, 1990.
