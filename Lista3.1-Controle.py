from contextlib import redirect_stdout
import io
import os
from fpdf import FPDF
import numpy as np
import sympy as sp


def to_caret(expr):
    return str(expr).replace("**", "^")


def format_poles(poles, tol=1e-7):
    formatted = []
    for p in poles:
        re = 0.0 if abs(p.real) < tol else round(float(p.real), 6)
        im = 0.0 if abs(p.imag) < tol else round(float(p.imag), 6)
        if im == 0.0:
            formatted.append(f"{re}")
        else:
            sinal = "+" if im > 0 else "-"
            formatted.append(f"{re} {sinal} {abs(im)}j")
    return formatted


def classify_stability_from_poles(poles, tol=1e-7):
    """Classifica a estabilidade com base na parte real dos polos."""
    real_parts = np.real(poles)
    if np.any(real_parts > tol):
        return (
            "INSTAVEL: existe pelo menos um polo com parte real positiva "
            "(fora do semiplano esquerdo aberto)."
        )
    elif np.any(np.abs(real_parts) <= tol):
        return (
            "MARGINALMENTE ESTAVEL / CRITICO: ha polo(s) com parte real nula "
            "(sobre o eixo imaginario)."
        )
    else:
        return (
            "ASSINTOTICAMENTE ESTAVEL: todos os polos tem parte real estritamente negativa."
        )


def ackermann_gain_explicit(
    A_sym,
    B_sym,
    C_sym,
    I_sym,
    U_sym,
    V_sym,
    n,
    desired_poles,
    mode="controller",
    use_plus_bk=True,
    use_minus_lc=True,
):
    """
    Executa e detalha passo a passo a formula de Ackermann:
      1. Montagem do polinomio desejado
      2. Calculo de A^1, A^2, ..., A^n
      3. Matriz Q(A)
      4. Inversa da matriz de controlabilidade/observabilidade
      5. Matriz de ganho final (K ou L) e verificacao dos polos
    """
    s = sp.Symbol("s")
    fatores = [(s - sp.nsimplify(p)) for p in desired_poles]
    produto_fatores = sp.Mul(*fatores, evaluate=False)
    delta_poly = sp.expand(sp.Mul(*fatores))
    poly_obj = sp.Poly(delta_poly, s)
    coeffs_desc = [sp.simplify(c) for c in poly_obj.all_coeffs()]

    label_poly = "Q_c" if mode == "controller" else "Q_l"

    print(f"\n--- 1. Polinomio desejado {label_poly}(s) ---")
    print("Produto dos fatores (s - p_i):")
    print(" ", to_caret(produto_fatores))
    print(f"\nPolinomio expandido {label_poly}(s):")
    print(" ", to_caret(delta_poly))

    print("\n--- 2. Calculo explicito das potencias de A ---")
    potencias_A = {}
    for k in range(1, n + 1):
        A_pow = sp.simplify(A_sym ** k)
        potencias_A[k] = A_pow
        print(f"A^{k}:")
        sp.pprint(A_pow)
        print()

    Q_mat = sp.zeros(n, n)
    for idx, c in enumerate(coeffs_desc):
        grau = n - idx
        c_simp = sp.simplify(c)
        if grau >= 1:
            termo_sym = sp.simplify(c_simp * potencias_A[grau])
        else:
            termo_sym = sp.simplify(c_simp * I_sym)
        Q_mat = sp.simplify(Q_mat + termo_sym)

    print(f"--- 3. Matriz {label_poly}(A) ---")
    sp.pprint(Q_mat)

    if mode == "controller":
        U_inv = sp.simplify(U_sym.inv())
        print("\n--- 4. Inversa da matriz de controlabilidade U^(-1) ---")
        sp.pprint(U_inv)

        e_last_row = sp.Matrix([[0] * (n - 1) + [1]])
        sinal_mult = -1 if use_plus_bk else 1
        malha_str = "A + BK" if use_plus_bk else "A - BK"
        K_sym = sp.simplify(sinal_mult * (e_last_row * U_inv * Q_mat))

        print("\n--- 5. Matriz de ganhos K ---")
        sp.pprint(K_sym)

        A_np = np.array(A_sym.tolist(), dtype=float)
        B_np = np.array(B_sym.tolist(), dtype=float)
        K_np = np.array(K_sym.tolist(), dtype=float).reshape(1, -1)
        A_mf = (A_np + B_np @ K_np) if use_plus_bk else (A_np - B_np @ K_np)
        polos_mf = np.linalg.eigvals(A_mf)

        print(f"\nVerificacao - autovalores obtidos de ({malha_str}):")
        for idx, p_str in enumerate(format_poles(polos_mf), start=1):
            print(f"  lambda_c_{idx} = {p_str}")
        return K_sym

    else:
        V_inv = sp.simplify(V_sym.inv())
        print("\n--- 4. Inversa da matriz de observabilidade V^(-1) ---")
        sp.pprint(V_inv)

        e_last_col = sp.Matrix([[0] * (n - 1) + [1]]).T
        sinal_mult = 1 if use_minus_lc else -1
        erro_str = "A - LC" if use_minus_lc else "A + LC"
        L_sym = sp.simplify(sinal_mult * (Q_mat * V_inv * e_last_col))

        print("\n--- 5. Matriz de ganhos L ---")
        sp.pprint(L_sym)

        A_np = np.array(A_sym.tolist(), dtype=float)
        C_np = np.array(C_sym.tolist(), dtype=float)
        L_np = np.array(L_sym.tolist(), dtype=float).reshape(-1, 1)
        A_obs = (A_np - L_np @ C_np) if use_minus_lc else (A_np + L_np @ C_np)
        polos_obs = np.linalg.eigvals(A_obs)

        print(f"\nVerificacao - autovalores obtidos de ({erro_str}):")
        for idx, p_str in enumerate(format_poles(polos_obs), start=1):
            print(f"  lambda_l_{idx} = {p_str}")
        return L_sym


def full_control_analysis_explicit(
    A,
    B,
    C_out,
    poles_k=None,
    poles_l=None,
    calc_initial=True,
    calc_k=True,
    calc_l=True,
    use_plus_bk=True,
    use_minus_lc=True,
    show_header=True,
):
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float).reshape(-1, 1)
    C_np = np.array(C_out, dtype=float).reshape(1, -1)
    n = A_np.shape[0]

    s = sp.Symbol("s")
    A_sym = sp.Matrix(A_np)
    B_sym = sp.Matrix(B_np)
    C_sym = sp.Matrix(C_np)
    I_sym = sp.eye(n)

    if show_header:
        print("=" * 75)
        print("DADOS DO SISTEMA FORNECIDOS NA ENTRADA")
        print("=" * 75)
        print("Matriz A:")
        sp.pprint(A_sym)
        print("\nMatriz B:")
        sp.pprint(B_sym)
        print("\nMatriz C:")
        sp.pprint(C_sym)
        if calc_k and poles_k is not None:
            print("\nPolos desejados para o Projeto de Realimentacao de Estados (Realimentacao K):")
            for idx, p in enumerate(poles_k, start=1):
                print(f"  p_c_{idx} = {p}")
        if calc_l and poles_l is not None:
            print("\nPolos desejados para Projeto de Observador de Estados (Ganho L):")
            for idx, p in enumerate(poles_l, start=1):
                print(f"  p_l_{idx} = {p}")
        print("=" * 75)

    ctrl_cols = []
    for k in range(n):
        if k == 0:
            col = B_sym
        elif k == 1:
            col = A_sym * B_sym
        else:
            col = (A_sym ** k) * B_sym
        ctrl_cols.append(col)
    U_ctrl = sp.Matrix.hstack(*ctrl_cols)
    rank_c = int(U_ctrl.rank())
    is_ctrl = (rank_c == n)

    obs_rows = []
    for k in range(n):
        if k == 0:
            row = C_sym
        elif k == 1:
            row = C_sym * A_sym
        else:
            row = C_sym * (A_sym ** k)
        obs_rows.append(row)
    V_obs = sp.Matrix.vstack(*obs_rows)
    rank_o = int(V_obs.rank())
    is_obs = (rank_o == n)

    if calc_initial:
        sI_minus_A = s * I_sym - A_sym
        det_expr = sp.det(sI_minus_A)
        char_poly = sp.expand(det_expr)

        poly_obj = sp.Poly(char_poly, s)
        coeffs_desc = [complex(c) for c in poly_obj.all_coeffs()]
        poles = np.roots(coeffs_desc)

        print("\n1. POLINOMIO CARACTERISTICO, POLOS E ESTABILIDADE")
        print("-" * 75)
        print("Matriz (sI - A):")
        sp.pprint(sI_minus_A)
        print("\nPolinomio caracteristico expandido p(s) = det(sI - A):")
        print(" ", to_caret(char_poly))
        print("\nPolos de malha aberta (raizes de p(s) = 0):")
        for idx, p_str in enumerate(format_poles(poles), start=1):
            print(f"  p_{idx} = {p_str}")
        print("\nConclusao sobre a estabilidade:")
        print(" ", classify_stability_from_poles(poles))

        print("\n2. MATRIZ DE CONTROLABILIDADE")
        print("-" * 75)
        for k, col in enumerate(ctrl_cols):
            label = "B" if k == 0 else ("AB" if k == 1 else f"A^{k}B")
            print(f"Coluna {label}:")
            sp.pprint(col)
            print()

        print("Matriz de controlabilidade U = [B  AB  A^2B ...]:")
        sp.pprint(U_ctrl)
        print(f"\nOrdem do sistema: n = {n}")
        print(f"Posto da matriz de controlabilidade: posto(U) = {rank_c}")
        print("Conclusao:", "O sistema eh controlavel." if is_ctrl else "O sistema nao eh controlavel.")

        print("\n3. MATRIZ DE OBSERVABILIDADE")
        print("-" * 75)
        for k, row in enumerate(obs_rows):
            label = "C" if k == 0 else ("CA" if k == 1 else f"CA^{k}")
            print(f"Linha {label}:")
            sp.pprint(row)
            print()

        print("Matriz de observabilidade V = [C; CA; CA^2; ...]:")
        sp.pprint(V_obs)
        print(f"\nOrdem do sistema: n = {n}")
        print(f"Posto da matriz de observabilidade: posto(V) = {rank_o}")
        print("Conclusao:", "O sistema eh observavel." if is_obs else "O sistema nao eh observavel.")

    if calc_k:
        print("\n4. REALIMENTACAO DE ESTADOS (FORMULA DE ACKERMANN)")
        print("-" * 75)
        if is_ctrl:
            ackermann_gain_explicit(
                A_sym=A_sym,
                B_sym=B_sym,
                C_sym=C_sym,
                I_sym=I_sym,
                U_sym=U_ctrl,
                V_sym=V_obs,
                n=n,
                desired_poles=poles_k,
                mode="controller",
                use_plus_bk=use_plus_bk,
                use_minus_lc=use_minus_lc,
            )
        else:
            print("Calculo de K cancelado: o sistema nao eh controlavel.")

    if calc_l:
        print("\n5. OBSERVADOR DE ESTADOS (FORMULA DE ACKERMANN)")
        print("-" * 75)
        if is_obs:
            ackermann_gain_explicit(
                A_sym=A_sym,
                B_sym=B_sym,
                C_sym=C_sym,
                I_sym=I_sym,
                U_sym=U_ctrl,
                V_sym=V_obs,
                n=n,
                desired_poles=poles_l,
                mode="observer",
                use_plus_bk=use_plus_bk,
                use_minus_lc=use_minus_lc,
            )
        else:
            print("Calculo de L cancelado: o sistema nao eh observavel.")


def sanitize_for_pdf(text):
    """Substitui caracteres especiais de desenho de caixa por equivalentes ASCII."""
    replacements = {
        "\u2500": "-",
        "\u2502": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
        "\u251c": "+",
        "\u2524": "+",
        "\u252c": "+",
        "\u2534": "+",
        "\u253c": "+",
        "\u23a1": "[",
        "\u23a2": "|",
        "\u23a3": "[",
        "\u23a4": "]",
        "\u23a5": "|",
        "\u23a6": "]",
        "\u239b": "(",
        "\u239c": "|",
        "\u239d": "(",
        "\u239e": ")",
        "\u239f": "|",
        "\u23a0": ")",
        "\u00b7": "*",
        "\u2212": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_text_to_pdf(text, filename="lista_3_1_controle.pdf"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filename)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Courier", size=8.5)

    clean_text = sanitize_for_pdf(text)
    line_height = 4.2

    for line in clean_text.splitlines():
        pdf.cell(0, line_height, text=line, new_x="LMARGIN", new_y="NEXT")

    pdf.output(full_path)
    return full_path


def parse_number_list(raw_text, expected_len=None):
    """Converte string com numeros (separados por espaco ou virgula) em lista."""
    tokens = [tok.strip() for tok in raw_text.replace(",", " ").split() if tok.strip()]
    if expected_len is not None and len(tokens) != expected_len:
        raise ValueError(f"Esperados {expected_len} valores, mas recebi {len(tokens)}.")
    values = []
    for tok in tokens:
        expr_str = tok.replace("i", "j")
        val = complex(sp.N(sp.sympify(expr_str)))
        if abs(val.imag) < 1e-12:
            values.append(float(val.real))
        else:
            values.append(complex(val))
    return values


def prompt_matrix_A(n):
    print(f"\nDigite a matriz A ({n}x{n}), uma linha por vez (elementos separados por espaco ou virgula):")
    A = []
    for i in range(n):
        row = parse_number_list(input(f"  Linha {i + 1}: "), expected_len=n)
        A.append(row)
    return A


def prompt_vector_B(n):
    print(f"\nDigite a matriz/vetor B ({n} linhas, 1 coluna):")
    B = []
    for i in range(n):
        val = parse_number_list(input(f"  B[{i + 1}]: "), expected_len=1)[0]
        B.append(val)
    return B


def prompt_vector_C(n):
    raw = input(f"\nDigite a matriz/vetor C (1 linha, {n} colunas separados por espaco ou virgula): ")
    return parse_number_list(raw, expected_len=n)


def prompt_poles(n, label):
    raw = input(f"Digite os {n} polos desejados para {label} (ex.: -2+1j -2-1j -10): ")
    return parse_number_list(raw, expected_len=n)


def parse_options(raw_opts):
    """Interpreta uma ou mais opcoes como '1, 2', '1 2', '2, 3' ou '4'."""
    tokens = [t.strip() for t in raw_opts.replace(",", " ").split() if t.strip()]
    if not tokens:
        return {"1", "2", "3"}
    if "4" in tokens:
        return {"1", "2", "3"}
    valid = {t for t in tokens if t in {"1", "2", "3"}}
    return valid if valid else {"1", "2", "3"}


def run_interactive():
    print("=== ANALISE DE SISTEMAS DE CONTROLE (ESPACO DE ESTADOS) ===")
    n = int(input("Ordem do sistema n (ex.: 2, 3): ").strip())
    if n < 1:
        raise ValueError("A ordem do sistema deve ser >= 1.")

    A = prompt_matrix_A(n)
    B = prompt_vector_B(n)
    C = prompt_vector_C(n)

    pdf_name = input("\nNome do arquivo PDF (Enter para 'lista_3_1_controle.pdf'): ").strip()
    if not pdf_name:
        pdf_name = "lista_3_1_controle.pdf"
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    buffer = io.StringIO()
    primeira_execucao = True

    while True:
        print("\nO que voce deseja calcular? (Voce pode digitar UMA ou VARIAS opcoes, ex.: 1, 2)")
        print("  [1] Analise inicial (polinomio, estabilidade, controlabilidade e observabilidade)")
        print("  [2] Realimentacao de estados (K)")
        print("  [3] Observador de estados (L)")
        print("  [4] Todas as anteriores (1 + 2 + 3)")
        raw_opts = input("Digite a(s) opcao(oes) desejada(s) [ex.: 1, 2 ou 2, 3 ou 4]: ").strip()
        escolhas = parse_options(raw_opts)

        calc_initial = "1" in escolhas
        calc_k = "2" in escolhas
        calc_l = "3" in escolhas

        poles_k = prompt_poles(n, "o controlador (K)") if calc_k else None
        poles_l = prompt_poles(n, "o observador (L)") if calc_l else None

        with redirect_stdout(buffer):
            full_control_analysis_explicit(
                A=A,
                B=B,
                C_out=C,
                poles_k=poles_k,
                poles_l=poles_l,
                calc_initial=calc_initial,
                calc_k=calc_k,
                calc_l=calc_l,
                use_plus_bk=True,
                use_minus_lc=True,
                show_header=primeira_execucao,
            )

        primeira_execucao = False

        mais = input("\nDeseja calcular outra combinacao para este mesmo sistema? [s/N]: ").strip().lower()
        if mais not in ("s", "sim", "y", "yes"):
            break

    saida_completa = buffer.getvalue()
    print(saida_completa)

    caminho_pdf = export_text_to_pdf(saida_completa, filename=pdf_name)
    print(f"\nPDF gerado com sucesso em: {caminho_pdf}")


if __name__ == "__main__":
    run_interactive()