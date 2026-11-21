# Análise de Sistemas de Controle no Espaço de Estados

Script interativo em Python (`numpy`, `sympy` e `fpdf2`) para análise estrutural e projeto por alocação de polos via fórmula de Ackermann, com opção de entrada direta das matrizes ou via função de transferência $G(s)$, exibição passo a passo das matrizes e exportação do relatório completo em PDF.

## Funcionalidades

- **Modos de entrada do sistema:**
  - Digitar diretamente as matrizes $A$, $B$ e $C$ de ordem $n$.
  - Informar uma função de transferência $G(s)$ (coeficientes do numerador e denominador) e gerar uma realização na **Forma Canônica Controlável** ou **Forma Canônica Observável**.
- **Análise inicial (`[1]`):**
  - Matriz $(sI - A)$ e polinômio característico expandido $p(s) = \det(sI - A)$.
  - Polos de malha aberta e classificação de estabilidade.
  - Matriz de controlabilidade $U$ (com colunas individuais) e verificação de controlabilidade pelo posto.
  - Matriz de observabilidade $V$ (com linhas individuais) e verificação de observabilidade pelo posto.
- **Realimentação de estados ($K$) (`[2]`):** cálculo via fórmula de Ackermann para alocação dos polos desejados.
- **Observador de estados ($L$) (`[3]`):** cálculo via fórmula de Ackermann para os polos desejados do observador.
- **Múltiplas opções e PDF (`[4]` ou combinação):** permite selecionar uma ou várias opções por execução (ex.: `1, 2` ou `4` para tudo) e salva o relatório completo em `.pdf` na mesma pasta do script.

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
1. Escolha o modo de entrada: `[1]` para digitar $A$, $B$ e $C$ diretamente ou `[2]` para informar $G(s)$ e escolher a realização canônica.
2. Informe o nome do arquivo PDF (ou pressione `Enter` para usar `lista_3_1_controle.pdf`).
3. Escolha uma ou mais opções (`1`, `2`, `3` ou `4` para tudo) e informe os polos desejados quando necessário.