# Redes Neurais Artificias - 2020/2 

Este repositório contém os códigos desenvolvidos ao longo da disciplina. Foram desenvolvidos os seguintes trabalhos:

## Exercícios computacionais
Exercícios contendo análises e comparações de diferentes estruturas de redes neurais artificiais: Perceptron, Multi-Layer Perceptron, ELM, Redes RBF e SVM. 

https://github.com/vicrsp/rna-ppgee/tree/main/exercicios

Cada pasta contém um notebook do jupyter com a implementação e análise dos resultados. 

## Survey
Este trabalho tem como objetivo apresentar uma revisão da literatura de redes neurais artificiais, com enfoque nas evoluções desenvolvidas a partir dos principais trabalhos lássicos que estabeleceram os fundamentos desta enorme área de pesquisa.

https://github.com/vicrsp/rna-ppgee/blob/main/survey/survey.pdf

## Artigo Computacional - Comparação de modelos lineares de redes neurais artificiais
Este trabalho tem como objetivo avaliar o desempenho de diferentes modelos de redes neurais artificiais estudados durante a disciplina sobre bases de dados de benchmark presentes na literatura. Serão considerados o Perceptron, Adaline, Redes RBF, ELM e ELM com aprendizado Hebbiano. Para três problemas de regressão e classificação binária escolhidos, um experimento foi desenhado seguindo as recomendações da literatura. Os resultados de cada modelos são comparados por meio de testes estatíscticos para as métricas AUC (classficação) e erro quadrático médio (regressão)

### Relatório
https://github.com/vicrsp/rna-ppgee/blob/main/artigo2/report/artigo2.pdf

### Códigos
https://github.com/vicrsp/rna-ppgee/tree/main/artigo2

Os resultados do artigo podem ser replicados a partir do arquivo principal: main.py

## Artigo Metodológico - Avaliação de Classificadores utilizando Técnicas de Estimativa de Densidades
A modelagem de dados não-lineares com redes neurais artificiais depende da qualidade projeção aplicada sobre as entradas, geralmente feita através de funções de kernel. A otimização de seus parâmetros é uma etapa importante e pode ser feita via técnicas de estimativa de densidade. Além de fornecer uma forma automática de seleção dos parâmetros ótimos, estas técnicas possuem a tendência de gerar projeções ortogonais das entradas no espaço de projetado. A partir desta observação, este trabalho tem como objetivo avaliar o desempenho de classificadores lineares sobre esta projeção, considerando problemas de benchmark presentes na literatura. Considerando 15 bases de dados de benchmark, os modelos ELM, Hebbiano e Perceptron sobre a projeção do espaço de verossimilhanças e SVM/RBF com otimização de largura foram comparados estatisticamente.

### Relatório
https://github.com/vicrsp/rna-ppgee/blob/main/artigo3/report/artigo3.pdf

### Códigos
https://github.com/vicrsp/rna-ppgee/tree/main/artigo3/src

Antes de iniciar, instalar o pacote com o modelo das projeções: https://github.com/vicrsp/rna-ppgee/blob/main/artigo3/kerneloptimizer/kerneloptimizer-0.0.1-py3-none-any.whl:

`pip install kerneloptimizer-0.0.1-py3-none-any.whl`

Em seguida, os resultados do artigo podem ser replicados executando a partir do arquivo principal: main.py
