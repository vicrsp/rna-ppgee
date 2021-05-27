- Para instalar o pacote em python, deve-se executar:
    pip3 install kerneloptimizer-0.0.1-py3-none-any.whl

- O objeto KernelOptimizer contém basicamente duas funções:
    - fit() aprende os parâmetros do kernel.
    - get_likelihood_space() obtém o espaço de verossimilhanças com os parâmetros aprendidos.

- O optimizador foi implementado em Python, mas pode ser executado em R com o pacote "reticulate". Um exemplo em R está disponível em test.R.
- Ao mesmo tempo, test.py contém um exemplo em Python.
