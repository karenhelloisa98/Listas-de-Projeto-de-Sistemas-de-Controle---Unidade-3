# Controle no Espaço de Estados (contínuo e discreto)

Scripts interativos em Python (`numpy`, `sympy` e `fpdf2`) para análise e projeto por alocação de polos (fórmula de Ackermann), com resolução passo a passo e relatório em PDF.

| Script | Sistema |
|---|---|
| `Listas de Controle U3 - Continuo.py` | $\dot{x} = Ax + Bu$, $y = Cx$ |
| `Listas de Controle U3 - Discreto.py` | $x(k+1) = Gx(k) + Hu(k)$, $y(k) = Cx(k)$ |

## O que calculam

- Polos e estabilidade, controlabilidade e observabilidade.
- Realimentação de estados $K$ e observador de estados $L$.
- Seguidor de referência para entrada degrau.
- **Só no discreto:** representação discreta (ZOH) de um sistema contínuo, com $T$ informado pelo usuário, mostrando a equação de estado $x(k+1) = Gx(k) + Hu(k)$.
- Relatório completo em PDF, com cabeçalho (universidade, disciplina e aluno).

## Instalação

```bash
pip install numpy sympy fpdf2
```

## Como usar

Os nomes têm espaços, então use aspas:

```bash
python "Listas de Controle U3 - Continuo.py"
python "Listas de Controle U3 - Discreto.py"
```

**Entrada do sistema**

| Contínuo | Discreto |
|---|---|
| `[1]` matrizes $A$, $B$, $C$ | `[1]` matrizes $G$, $H$, $C$ |
| `[2]` $G(s)$ + forma canônica | `[2]` sistema contínuo $A$, $B$, $C$ + $T$ |
| | `[3]` $G(s)$ + forma canônica + $T$ |

**Opções de cálculo** (uma ou várias, ex.: `1, 3`)

| Contínuo | Discreto |
|---|---|
| `[1]` Análise inicial | `[1]` Representação discreta |
| `[2]` Realimentação $K$ | `[2]` Análise inicial |
| `[3]` Observador $L$ | `[3]` Realimentação $K$ |
| `[4]` Seguidor de referência | `[4]` Observador $L$ |
| `[5]` Todas | `[5]` Seguidor de referência |
| | `[6]` Todas |

**Polos desejados:** $n$ para $K$ e $L$, e $n + 1$ para o seguidor. Aceitam reais, frações e complexos com `j` ou `i` (ex.: `-1+0.1j -1-0.1j -5`), e os complexos precisam vir em pares conjugados. No discreto, use $|z| < 1$ para estabilidade.

Ao final, o programa pergunta se quer outra combinação. Tudo vai para o mesmo PDF, salvo na pasta do script.

## Convenções

- **Seguidor contínuo:** estado aumentado $[e;\ x]$, $\dot{e} = y - r$, $K = [k_1 \mid K_2]$.
- **Seguidor discreto:** método do professor, $u(k) = -K_2 x(k) + k_1 v(k)$, com $[k_2 \mid k_1]$.

## Cabeçalho do PDF

Para editar, altere `PDF_UNIVERSIDADE`, `PDF_DISCIPLINA` e `PDF_ALUNO` no topo de cada script.