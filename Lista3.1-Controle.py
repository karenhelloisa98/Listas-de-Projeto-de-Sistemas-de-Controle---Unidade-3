from contextlib import redirect_stdout
import io
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
    poles_k,
    poles_l,
    use_plus_bk=True,
    use_minus_lc=True,
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

    print("=" * 75)
    print("DADOS DO SISTEMA FORNECIDOS NA ENTRADA")
    print("=" * 75)
    print("Matriz A:")
    sp.pprint(A_sym)
    print("\nMatriz B:")
    sp.pprint(B_sym)
    print("\nMatriz C:")
    sp.pprint(C_sym)
    print("\nPolos desejados para o Projeto de Realimentacao de Estados (Realimentacao K):")
    for idx, p in enumerate(poles_k, start=1):
        print(f"  p_c_{idx} = {p}")
    print("\nPolos desejados para Projeto de Observador de Estados (Ganho L):")
    for idx, p in enumerate(poles_l, start=1):
        print(f"  p_l_{idx} = {p}")
    print("=" * 75)

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
    #print("\nDeterminante det(sI - A) antes de expandir:")
    #print(" ", to_caret(det_expr))
    print("\nPolinomio caracteristico expandido p(s) = det(sI - A):")
    print(" ", to_caret(char_poly))
    print("\nPolos de malha aberta (raizes de p(s) = 0):")
    for idx, p_str in enumerate(format_poles(poles), start=1):
        print(f"  p_{idx} = {p_str}")
    print("\nConclusao sobre a estabilidade:")
    print(" ", classify_stability_from_poles(poles))

    print("\n2. MATRIZ DE CONTROLABILIDADE")
    print("-" * 75)
    ctrl_cols = []
    for k in range(n):
        if k == 0:
            col = B_sym
            label = "B"
        elif k == 1:
            col = A_sym * B_sym
            label = "AB"
        else:
            col = (A_sym ** k) * B_sym
            label = f"A^{k}B"
        ctrl_cols.append(col)
        print(f"Coluna {label}:")
        sp.pprint(col)
        print()

    U_ctrl = sp.Matrix.hstack(*ctrl_cols)
    rank_c = int(U_ctrl.rank())
    is_ctrl = (rank_c == n)

    print("Matriz de controlabilidade U = [B  AB  A^2B ...]:")
    sp.pprint(U_ctrl)
    print(f"\nOrdem do sistema: n = {n}")
    print(f"Posto da matriz de controlabilidade: posto(U) = {rank_c}")
    print("Conclusao:", "O sistema eh controlavel." if is_ctrl else "O sistema nao eh controlavel.")

    print("\n3. MATRIZ DE OBSERVABILIDADE")
    print("-" * 75)
    obs_rows = []
    for k in range(n):
        if k == 0:
            row = C_sym
            label = "C"
        elif k == 1:
            row = C_sym * A_sym
            label = "CA"
        else:
            row = C_sym * (A_sym ** k)
            label = f"CA^{k}"
        obs_rows.append(row)
        print(f"Linha {label}:")
        sp.pprint(row)
        print()

    V_obs = sp.Matrix.vstack(*obs_rows)
    rank_o = int(V_obs.rank())
    is_obs = (rank_o == n)

    print("Matriz de observabilidade V = [C; CA; CA^2; ...]:")
    sp.pprint(V_obs)
    print(f"\nOrdem do sistema: n = {n}")
    print(f"Posto da matriz de observabilidade: posto(V) = {rank_o}")
    print("Conclusao:", "O sistema eh observavel." if is_obs else "O sistema nao eh observavel.")

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
        "\u2500": "-",  # ─
        "\u2502": "|",  # │
        "\u250c": "+",  # ┌
        "\u2510": "+",  # ┐
        "\u2514": "+",  # └
        "\u2518": "+",  # ┘
        "\u251c": "+",  # ├
        "\u2524": "+",  # ┤
        "\u252c": "+",  # ┬
        "\u2534": "+",  # ┴
        "\u253c": "+",  # ┼
        "\u23a1": "[",  # ⎡
        "\u23a2": "|",  # ⎢
        "\u23a3": "[",  # ⎣
        "\u23a4": "]",  # ⎤
        "\u23a5": "|",  # ⎥
        "\u23a6": "]",  # ⎦
        "\u239b": "(",  # ⎛
        "\u239c": "|",  # ⎜
        "\u239d": "(",  # ⎝
        "\u239e": ")",  # ⎞
        "\u239f": "|",  # ⎟
        "\u23a0": ")",  # ⎠
        "\u00b7": "*",  # ·
        "\u2212": "-",  # −
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_text_to_pdf(text, filename="relatorio_controle.pdf"):
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Courier", size=8.5)

    clean_text = sanitize_for_pdf(text)
    line_height = 4.2

    for line in clean_text.splitlines():
        pdf.cell(0, line_height, txt=line, ln=True)

    pdf.output(filename)


if __name__ == "__main__":
    A_ex = [
        [-5, 1, 0],
        [2, 7, -1],
        [0, -3, 6],
    ]
    B_ex = [0, 0, 1]
    C_ex = [1, -2, 3]

    polos_controlador_ex = [-2 + 1j, -2 - 1j, -10]
    polos_observador_ex = [-1 + 1j, -1 - 1j, -10]

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        full_control_analysis_explicit(
            A=A_ex,
            B=B_ex,
            C_out=C_ex,
            poles_k=polos_controlador_ex,
            poles_l=polos_observador_ex,
            use_plus_bk=True,
            use_minus_lc=True,
        )

    saida_completa = buffer.getvalue()
    print(saida_completa)

    arquivo_pdf = "lista_3_1_controle.pdf"
    export_text_to_pdf(saida_completa, filename=arquivo_pdf)
    print(f"\nPDF gerado com sucesso: {arquivo_pdf}")