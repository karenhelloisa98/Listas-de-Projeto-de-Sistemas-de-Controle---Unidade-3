# Análise de Sistemas de Controle no Espaço de Estados

Script interativo em Python (`numpy`, `sympy` e `fpdf2`) para análise estrutural e projeto por alocação de polos via fórmula de Ackermann, com exibição passo a passo das matrizes e exportação do relatório completo em PDF.

## Funcionalidades

- **Análise inicial:**
  - Matriz $(sI - A)$ e polinômio característico expandido $p(s) = \det(sI - A)$.
  - Polos de malha aberta e classificação de estabilidade.
  - Matriz de controlabilidade $U$ (com colunas individuais) e verificação de controlabilidade pelo posto.
  - Matriz de observabilidade $V$ (com linhas individuais) e verificação de observabilidade pelo posto.
- **Realimentação de estados ($K$):** cálculo via fórmula de Ackermann para alocação dos polos desejados.
- **Observador de estados ($L$):** cálculo via fórmula de Ackermann para os polos desejados do observador.
- **Modo interativo e exportação em PDF:** permite escolher uma ou múltiplas opções por execução e salva o resultado em um arquivo `.pdf` na mesma pasta do script.

## Instalação

```bash
pip install numpy sympy fpdf2
```

## Como usar

Execute o script pelo terminal (evite a aba *Output* de leitura):

```bash
python Lista3.1-Controle.py
```

Durante a execução:
1. Informe a ordem $n$ do sistema e os elementos das matrizes $A$, $B$ e $C$.
2. Escolha o nome do arquivo PDF (ou pressione `Enter` para usar o padrão).
3. Selecione uma ou mais opções (`1`, `2`, `3` ou `4` para tudo) e informe os polos desejados quando aplicável.