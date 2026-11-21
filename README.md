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
- **Realimentação de estados ($K$) (`[2]`):** cálculo via fórmula de Ackermann para alocação dos polos desejados, com verificação dos autovalores de $A + BK$.
- **Observador de estados ($L$) (`[3]`):** cálculo via fórmula de Ackermann para os polos desejados do observador, com verificação dos autovalores de $A - LC$.
- **Seguidor de referência para entrada degrau (`[4]`):** projeto com ação integral (sistema aumentado com estado $[e;\ x]$ e $\dot{e} = y - r$) via Ackermann, com $n + 1$ polos desejados. Exibe o ganho no formato $K = [k_1 \mid K_2]$ e as equações de estado em malha fechada, com verificação dos autovalores. Se o sistema aumentado não for controlável, o cálculo é interrompido com aviso.
- **Múltiplas opções e PDF (`[5]` ou combinação):** permite selecionar uma ou várias opções por execução (ex.: `1, 2` ou `2, 4`; `5` para tudo) e salva o relatório completo em `.pdf` na mesma pasta do script. A fonte do PDF é reduzida automaticamente para que as matrizes largas não sejam cortadas.
- **Validação dos polos desejados:** os polos aceitam reais, decimais, frações e complexos com `j` ou `i` (ex.: `-1+0.1j -1-0.1j -5 -12`). Polos complexos devem vir em pares conjugados, caso contrário o programa pede a entrada novamente.

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
3. Escolha uma ou mais opções (`1`, `2`, `3`, `4` ou `5` para tudo) e informe os polos desejados quando necessário: $n$ polos para $K$ e para $L$, e $n + 1$ polos para o seguidor de referência.
4. Ao final, escolha se deseja calcular outra combinação para o mesmo sistema. Todos os resultados vão para o mesmo PDF.

## Exemplo: seguidor de referência

Para $G(s) = \dfrac{3s^2 + 4s - 2}{s^3 + 3s^2 + 7s + 5}$ na forma canônica controlável, com polos do seguidor em $-1 \pm 0{,}1j$, $-5$ e $-12$:

| Campo | Valor informado |
|---|---|
| Modo de entrada | `2` |
| Numerador | `3 4 -2` |
| Denominador | `1 3 7 5` |
| Realização | `1` (controlável) |
| Opção | `4` |
| Polos | `-1+0.1j -1-0.1j -5 -12` |

Resultado esperado: $\Delta(s) = s^4 + 19s^3 + 95{,}01s^2 + 137{,}17s + 60{,}6$ e $K = [\,30{,}3 \mid -253{,}37\ \ -178{,}91\ \ -16\,]$.

## Convenção do seguidor de referência

O código usa $\dot{e} = y - r$ e a lei de controle $u = k_1 \int e\,d\tau + K_2\,x$, com malha fechada $A_a + B_a K$. Se o seu material usa $\dot{e} = r - y$, o ganho $k_1$ aparece com o sinal oposto, e $K_2$ não muda.