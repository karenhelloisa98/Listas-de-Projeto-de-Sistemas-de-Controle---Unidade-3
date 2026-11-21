from contextlib import redirect_stdout
import io
import os
import re
from fpdf import FPDF
import numpy as np
import sympy as sp


def to_caret(expr):
    return str(expr).replace("**", "^")


def format_number_clean(val, tol=1e-9):
    c = complex(val)
    re_val = 0.0 if abs(c.real) < tol else round(float(c.real), 6)
    im_val = 0.0 if abs(c.imag) < tol else round(float(c.imag), 6)
    if im_val == 0.0:
        return f"{re_val}"
    sinal = "+" if im_val > 0 else "-"
    return f"({re_val} {sinal} {abs(im_val)}j)"


def format_poles(poles, tol=1e-7):
    return [format_number_clean(p, tol=tol) for p in poles]


def format_poly_decimal_str(coeffs_desc, var_name="s", tol=1e-9):
    deg = len(coeffs_desc) - 1
    terms = []
    for idx, c in enumerate(coeffs_desc):
        power = deg - idx
        val = complex(c)
        re_val = 0.0 if abs(val.real) < tol else round(float(val.real), 6)
        im_val = 0.0 if abs(val.imag) < tol else round(float(val.imag), 6)

        if abs(re_val) < tol and abs(im_val) < tol:
            continue

        if abs(im_val) < tol:
            coeff_str = f"{re_val}"
        else:
            sinal = "+" if im_val > 0 else "-"
            coeff_str = f"({re_val} {sinal} {abs(im_val)}j)"

        if power == 0:
            term = coeff_str
        elif power == 1:
            term = f"{coeff_str}*{var_name}"
        else:
            term = f"{coeff_str}*{var_name}^{power}"

        terms.append(term)

    return " + ".join(terms).replace("+ -", "- ") if terms else "0.0"


def _fmt(val, casas=4):
    v = round(float(complex(val).real), casas)
    if v == 0:
        v = 0.0
    txt = f"{v:.{casas}f}".rstrip("0").rstrip(".")
    return txt if txt else "0"


def _matriz_limpa(M):
    return sp.Matrix(
        [[sp.Symbol(_fmt(M[i, j])) for j in range(M.cols)] for i in range(M.rows)]
    )


def classify_stability_from_poles(poles, tol=1e-7):
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


def realization_from_tf(num_coeffs, den_coeffs, form="controlavel"):
    """
    Constroi A, B e C a partir de G(s) = N(s)/D(s) em ordem decrescente de s.
    form: 'controlavel' ou 'observavel'.
    """
    s = sp.Symbol("s")
    num_poly = sp.Poly(
        sum(sp.nsimplify(c) * (s ** i) for i, c in enumerate(reversed(num_coeffs))), s
    )
    den_poly = sp.Poly(
        sum(sp.nsimplify(c) * (s ** i) for i, c in enumerate(reversed(den_coeffs))), s
    )

    a_n = sp.simplify(den_poly.LC())
    if a_n == 0:
        raise ValueError("Coeficiente do maior grau do denominador nao pode ser zero.")

    den_monic = sp.Poly(sp.expand(den_poly.as_expr() / a_n), s)
    num_scaled = sp.Poly(sp.expand(num_poly.as_expr() / a_n), s)

    n = den_monic.degree()
    if n < 1 or num_scaled.degree() >= n:
        raise ValueError("Sistema deve ter ordem n >= 1 e ser estritamente proprio.")

    den_all = [sp.simplify(c) for c in den_monic.all_coeffs()]
    alphas_asc = list(reversed(den_all[1:]))

    raw_num = [sp.simplify(c) for c in num_scaled.all_coeffs()]
    betas_desc = [sp.Integer(0)] * (n - len(raw_num)) + raw_num
    betas_asc = list(reversed(betas_desc))

    if form == "controlavel":
        A_rows = []
        for r in range(n - 1):
            row = [sp.Integer(0)] * n
            row[r + 1] = sp.Integer(1)
            A_rows.append(row)
        A_rows.append([-a for a in alphas_asc])
        A_sym = sp.Matrix(A_rows)
        B_sym = sp.Matrix([[0] * (n - 1) + [1]]).T
        C_sym = sp.Matrix([betas_asc])
        form_title = "Forma Canonica Controlavel"
    elif form == "observavel":
        A_rows = []
        for r in range(n):
            row = [sp.Integer(0)] * n
            if r > 0:
                row[r - 1] = sp.Integer(1)
            row[-1] = -alphas_asc[r]
            A_rows.append(row)
        A_sym = sp.Matrix(A_rows)
        B_sym = sp.Matrix([[b] for b in betas_asc])
        C_sym = sp.Matrix([[0] * (n - 1) + [1]])
        form_title = "Forma Canonica Observavel"
    else:
        raise ValueError("Forma invalida. Use 'controlavel' ou 'observavel'.")

    info = {
        "n": n,
        "form_title": form_title,
        "num_expr": num_scaled.as_expr(),
        "den_expr": den_monic.as_expr(),
        "alphas_asc": alphas_asc,
        "betas_asc": betas_asc,
    }
    return A_sym, B_sym, C_sym, info


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
    print(" ", to_caret(produto_fatores).replace("I", "j"))
    print(f"\nPolinomio expandido {label_poly}(s) (em decimais):")
    print(" ", format_poly_decimal_str(coeffs_desc, var_name="s"))

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


def tracker_reference_explicit(A_sym, B_sym, C_sym, desired_poles):
    """
    Seguidor de referencia para entrada degrau (estado aumentado [e; x], e' = y - r):
      A_a = [0  C]     B_a = [0]
            [0  A]           [B]
    Ackermann: K = -[0 ... 1] * U^(-1) * q_c(A_a) = [k1 | K2]
    Malha fechada (A_a + B_a K) e reescrita na ordem [x; e], como no livro.
    """
    n = A_sym.rows
    n_aug = n + 1

    # ---------- sistema aumentado ----------
    first_row_Aa = sp.Matrix.hstack(sp.zeros(1, 1), C_sym)      # [0, C]
    lower_rows_Aa = sp.Matrix.hstack(sp.zeros(n, 1), A_sym)     # [0, A]
    A_aug = sp.simplify(sp.Matrix.vstack(first_row_Aa, lower_rows_Aa))
    B_aug = sp.simplify(sp.Matrix.vstack(sp.zeros(1, 1), B_sym))  # [0; B]
    I_aug = sp.eye(n_aug)

    print("Sistema aumentado (estado [e; x], com e' = y - r):")
    print("Matriz aumentada A_a:")
    sp.pprint(A_aug)
    print("\nMatriz aumentada B_a:")
    sp.pprint(B_aug)

    # ---------- 1. controlabilidade aumentada ----------
    cols = [(A_aug ** k) * B_aug for k in range(n_aug)]
    U_aug = sp.simplify(sp.Matrix.hstack(*cols))
    rank_aug = int(U_aug.rank())

    print("\n--- 1. Matriz de controlabilidade aumentada U = [B_a  A_a*B_a  ...] ---")
    sp.pprint(U_aug)
    print(f"\nOrdem do sistema aumentado: n_a = {n_aug}")
    print(f"Posto da matriz de controlabilidade: posto(U) = {rank_aug}")
    if rank_aug == n_aug:
        print("Conclusao: O sistema aumentado eh controlavel, logo o rastreamento "
              "da referencia degrau eh possivel.")
    else:
        print("Conclusao: posto(U) < n_a -> sistema aumentado nao eh controlavel "
              "(provavel zero de G(s) em s = 0). Seguidor impossivel.")
        return

    # ---------- 2. inversa ----------
    print("\n--- 2. Inversa da matriz de controlabilidade U^(-1) ---")
    U_inv = sp.simplify(U_aug.inv())
    sp.pprint(U_inv)

    # ---------- 3. polinomio desejado ----------
    poles_complex = [complex(p) for p in desired_poles]
    fatores_str = []
    delta_poly_num = np.poly1d([1.0 + 0.0j])
    for p in poles_complex:
        re_p = 0.0 if abs(p.real) < 1e-12 else round(float(p.real), 6)
        im_p = 0.0 if abs(p.imag) < 1e-12 else round(float(p.imag), 6)
        p_clean = complex(re_p, im_p)
        fatores_str.append(f"(s - {format_number_clean(p_clean)})")
        delta_poly_num = delta_poly_num * np.poly1d([1.0 + 0.0j, -p_clean])

    coeffs_desc = list(delta_poly_num.coeffs)
    print("\n--- 3. Polinomio desejado delta(s) ---")
    print("Produto dos fatores (s - p_i):")
    print(" ", " * ".join(fatores_str))
    print("\nPolinomio expandido delta(s) (em decimais):")
    print(" ", format_poly_decimal_str(coeffs_desc, var_name="s"))

    # ---------- 4. q_c(A_a) ----------
    potencias = {k: sp.simplify(A_aug ** k) for k in range(1, n_aug + 1)}
    Q_mat = sp.zeros(n_aug, n_aug)
    for idx, c in enumerate(coeffs_desc):
        grau = n_aug - idx
        c_num = complex(c)
        c_sym = sp.Float(round(float(c_num.real), 8)) if abs(c_num.imag) < 1e-12 else sp.nsimplify(c_num)
        termo = sp.simplify(c_sym * (potencias[grau] if grau >= 1 else I_aug))
        Q_mat = sp.simplify(Q_mat + termo)

    print("\n--- 4. Matriz q_c(A_a) ---")
    sp.pprint(_matriz_limpa(Q_mat))

    # ---------- 5. Ackermann ----------
    print("\n--- 5. Formula de Ackermann: K = -[0 ... 0 1] * U^(-1) * q_c(A_a) ---")
    e_last = sp.Matrix([[0] * (n_aug - 1) + [1]])
    K_aug = sp.simplify(-(e_last * U_inv * Q_mat))

    k_int = float(sp.N(K_aug[0, 0]))
    K_x = [float(sp.N(K_aug[0, j])) for j in range(1, n_aug)]

    nomes_k2 = "  ".join(f"k2{j + 1}" for j in range(n))
    vals_k2 = "  ".join(_fmt(v) for v in K_x)
    print(f"\nK = [k1 | K2] = [k1 | {nomes_k2}] = [ {_fmt(k_int)} | {vals_k2} ]")

    # ---------- 6. equacoes de estado em malha fechada, ordem [x; e] ----------
    A_np = np.array(A_sym.tolist(), dtype=float)
    B_np = np.array(B_sym.tolist(), dtype=float).reshape(-1, 1)
    C_np = np.array(C_sym.tolist(), dtype=float).reshape(1, -1)
    K2_np = np.array(K_x, dtype=float).reshape(1, -1)

    A_cl = np.block([
        [A_np + B_np @ K2_np, B_np * k_int],
        [C_np,                np.zeros((1, 1))],
    ])
    B_r = np.vstack([np.zeros((n, 1)), -np.ones((1, 1))])

    estados = ", ".join([f"x{i + 1}" for i in range(n)] + ["e"])
    print(f"\n--- 6. Equacoes de estado em malha fechada (estado [{estados}]^T) ---")
    print("d/dt [x; e] = (A + B*K2) x + B*k1*e   ;   e' = C*x - r")
    print("\nMatriz de malha fechada:")
    sp.pprint(_matriz_limpa(sp.Matrix(A_cl.tolist())))
    print("\nMatriz de entrada (referencia r):")
    sp.pprint(_matriz_limpa(sp.Matrix(B_r.tolist())))

    # ---------- verificacao ----------
    polos_mf = np.linalg.eigvals(A_cl)
    print("\nVerificacao - autovalores de malha fechada:")
    for idx, p_str in enumerate(format_poles(polos_mf), start=1):
        print(f"  lambda_{idx} = {p_str}")


def full_control_analysis_explicit(
    A,
    B,
    C_out,
    poles_k=None,
    poles_l=None,
    poles_ref=None,
    calc_initial=True,
    calc_k=False,
    calc_l=False,
    calc_ref=False,
    use_plus_bk=True,
    use_minus_lc=True,
    show_header=True,
    tf_info=None,
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
        if tf_info is not None:
            print("Funcao de transferencia G(s):")
            print(f"  Num(s) = {to_caret(tf_info['num_expr'])}")
            print(f"  Den(s) = {to_caret(tf_info['den_expr'])}")
            print(f"Realizacao escolhida: {tf_info['form_title']} (ordem n = {n})")
            alphas_str = ", ".join(f"alpha_{i+1} = {val}" for i, val in enumerate(tf_info["alphas_asc"]))
            betas_str = ", ".join(f"beta_{i+1} = {val}" for i, val in enumerate(tf_info["betas_asc"]))
            print(f"Coeficientes do denominador: {alphas_str}")
            print(f"Coeficientes do numerador:   {betas_str}\n")
        print("Matriz A:")
        sp.pprint(A_sym)
        print("\nMatriz B:")
        sp.pprint(B_sym)
        print("\nMatriz C:")
        sp.pprint(C_sym)
        if calc_k and poles_k is not None:
            print("\nPolos desejados para Realimentacao de Estados (K):")
            for idx, p in enumerate(poles_k, start=1):
                print(f"  p_c_{idx} = {format_number_clean(p)}")
        if calc_l and poles_l is not None:
            print("\nPolos desejados para Observador de Estados (L):")
            for idx, p in enumerate(poles_l, start=1):
                print(f"  p_l_{idx} = {format_number_clean(p)}")
        if calc_ref and poles_ref is not None:
            print("\nPolos desejados para Seguidor de Referencia:")
            for idx, p in enumerate(poles_ref, start=1):
                print(f"  p_r_{idx} = {format_number_clean(p)}")
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

    if calc_ref:
        print("\n6. SEGUIDOR DE REFERENCIA (ENTRADA DEGRAU)")
        print("-" * 75)
        tracker_reference_explicit(A_sym, B_sym, C_sym, poles_ref)


def sanitize_for_pdf(text):
    replacements = {
        "\u2500": "-", "\u2502": "|", "\u250c": "+", "\u2510": "+",
        "\u2514": "+", "\u2518": "+", "\u251c": "+", "\u2524": "+",
        "\u252c": "+", "\u2534": "+", "\u253c": "+", "\u23a1": "[",
        "\u23a2": "|", "\u23a3": "[", "\u23a4": "]", "\u23a5": "|",
        "\u23a6": "]", "\u239b": "(", "\u239c": "|", "\u239d": "(",
        "\u239e": ")", "\u239f": "|", "\u23a0": ")", "\u00b7": "*",
        "\u2212": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_text_to_pdf(text, filename="lista_controle.pdf"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filename)

    linhas = sanitize_for_pdf(text).splitlines()
    margem = 10.0
    largura_util = 210.0 - 2 * margem
    max_chars = max((len(l) for l in linhas), default=1)
    # Courier: largura do caractere = 0.6 * tamanho da fonte (pt) * 0.3528 mm
    fonte = min(8.0, largura_util / (max_chars * 0.6 * 0.3528))
    fonte = max(fonte, 4.0)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(margem, 12, margem)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Courier", size=fonte)

    altura_linha = max(fonte * 0.5, 2.2)
    for line in linhas:
        pdf.cell(0, altura_linha, text=line, new_x="LMARGIN", new_y="NEXT")

    pdf.output(full_path)
    return full_path


def parse_number_list(raw_text, expected_len=None):
    tokens = [tok.strip() for tok in raw_text.replace(",", " ").split() if tok.strip()]
    if expected_len is not None and len(tokens) != expected_len:
        raise ValueError(f"Esperados {expected_len} valores, mas recebi {len(tokens)}.")
    values = []
    for tok in tokens:
        expr_str = tok.replace("i", "j")
        expr_str = re.sub(r"(?<![0-9a-zA-Z_.])j", "1*j", expr_str)
        # CORRIGIDO: 'j' passa a ser a unidade imaginaria no sympy
        val = complex(sp.N(sp.sympify(expr_str, locals={"j": sp.I})))
        if abs(val.imag) < 1e-12:
            values.append(float(val.real))
        else:
            values.append(complex(val))
    return values


def poles_are_conjugate_closed(poles, tol=1e-9):
    """True se todo polo complexo tem seu conjugado na lista (K/L reais)."""
    restantes = [complex(p) for p in poles]
    while restantes:
        p = restantes.pop(0)
        if abs(p.imag) < tol:
            continue
        conj = p.conjugate()
        achou = None
        for idx, q in enumerate(restantes):
            if abs(q - conj) < 1e-7:
                achou = idx
                break
        if achou is None:
            return False
        restantes.pop(achou)
    return True


def prompt_matrix_A(n):
    print(f"\nDigite a matriz A ({n}x{n}), uma linha por vez (elementos separados por espaco ou virgula):")
    return [parse_number_list(input(f"  Linha {i + 1}: "), expected_len=n) for i in range(n)]


def prompt_vector_B(n):
    print(f"\nDigite a matriz/vetor B ({n} linhas, 1 coluna):")
    return [parse_number_list(input(f"  B[{i + 1}]: "), expected_len=1)[0] for i in range(n)]


def prompt_vector_C(n):
    raw = input(f"\nDigite a matriz/vetor C (1 linha, {n} colunas separados por espaco ou virgula): ")
    return parse_number_list(raw, expected_len=n)


def prompt_poles(count, label):
    print(f"\nFormato para digitar os {count} polos de {label}:")
    print("  - Separe cada polo por espaco ou virgula (ex.: -1+0.1j -1-0.1j -5 -12)")
    print("  - Numeros complexos: use j ou i, com ou sem * (ex.: -1+1j, -1+1*j, -1+0.1i, -1+j)")
    print("  - Reais, decimais e fracoes tambem sao aceitos (ex.: -10, -2.5, -1/2)")
    print("  - Polos complexos devem vir em pares conjugados")
    while True:
        raw = input(f"Digite os {count} polos desejados: ")
        try:
            poles = parse_number_list(raw, expected_len=count)
        except Exception as exc:
            print(f"Entrada invalida ({exc}). Tente novamente.")
            continue
        if not poles_are_conjugate_closed(poles):
            print("Os polos complexos precisam estar em pares conjugados. Tente novamente.")
            continue
        return poles


def parse_options(raw_opts):
    """Interpreta uma ou mais opcoes como '1, 4', '2, 3', '4' ou '5' (tudo)."""
    tokens = [t.strip() for t in raw_opts.replace(",", " ").split() if t.strip()]
    if not tokens or "5" in tokens:
        return {"1", "2", "3", "4"}
    valid = {t for t in tokens if t in {"1", "2", "3", "4"}}
    return valid if valid else {"1"}


def run_interactive():
    print("=== ANALISE E PROJETO DE SISTEMAS DE CONTROLE (ESPACO DE ESTADOS) ===")
    print("\nComo voce deseja informar o sistema?")
    print("  [1] Digitar as matrizes A, B e C diretamente")
    print("  [2] Informar uma funcao de transferencia G(s) e escolher a realizacao canonica")
    modo_entrada = input("Escolha o modo de entrada [1/2] (padrao 1): ").strip() or "1"

    tf_info = None

    if modo_entrada == "2":
        print("\nDigite os coeficientes em ordem decrescente de s (separados por espaco ou virgula):")
        print("  Exemplo para G(s) = (3s^2 + 4s - 2) / (s^3 + 3s^2 + 7s + 5):")
        print("    Numerador:   3 4 -2")
        print("    Denominador: 1 3 7 5")
        num_coeffs = parse_number_list(input("Coeficientes do numerador: "))
        den_coeffs = parse_number_list(input("Coeficientes do denominador: "))

        print("\nEscolha a realizacao:")
        print("  [1] Forma Canonica Controlavel")
        print("  [2] Forma Canonica Observavel")
        forma_opt = input("Escolha [1/2] (padrao 1): ").strip() or "1"
        forma = "observavel" if forma_opt == "2" else "controlavel"

        A_sym, B_sym, C_sym, tf_info = realization_from_tf(num_coeffs, den_coeffs, form=forma)
        n = tf_info["n"]
        A = A_sym.tolist()
        B = [val[0] for val in B_sym.tolist()]
        C = C_sym.tolist()[0]
    else:
        n = int(input("\nOrdem do sistema n (ex.: 2, 3): ").strip())
        if n < 1:
            raise ValueError("A ordem do sistema deve ser >= 1.")
        A = prompt_matrix_A(n)
        B = prompt_vector_B(n)
        C = prompt_vector_C(n)

    pdf_name = input("\nNome do arquivo PDF (Enter para 'lista_controle.pdf'): ").strip() or "lista_controle.pdf"
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    buffer = io.StringIO()
    primeira_execucao = True

    while True:
        print("\nO que voce deseja calcular? (Voce pode digitar UMA ou VARIAS opcoes, ex.: 1, 4)")
        print("  [1] Analise inicial (polinomio caracteristico, estabilidade, controlabilidade e observabilidade)")
        print("  [2] Realimentacao de estados (K)")
        print("  [3] Observador de estados (L)")
        print("  [4] Seguidor de referencia (entrada degrau)")
        print("  [5] Todas as anteriores (1 + 2 + 3 + 4)")
        raw_opts = input("Digite a(s) opcao(oes) desejada(s): ").strip()
        escolhas = parse_options(raw_opts)

        calc_initial = "1" in escolhas
        calc_k = "2" in escolhas
        calc_l = "3" in escolhas
        calc_ref = "4" in escolhas

        poles_k = prompt_poles(n, "a realimentacao de estados (K)") if calc_k else None
        poles_l = prompt_poles(n, "o observador de estados (L)") if calc_l else None
        poles_ref = prompt_poles(n + 1, f"o seguidor de referencia (ordem {n + 1})") if calc_ref else None

        with redirect_stdout(buffer):
            full_control_analysis_explicit(
                A=A,
                B=B,
                C_out=C,
                poles_k=poles_k,
                poles_l=poles_l,
                poles_ref=poles_ref,
                calc_initial=calc_initial,
                calc_k=calc_k,
                calc_l=calc_l,
                calc_ref=calc_ref,
                use_plus_bk=True,
                use_minus_lc=True,
                show_header=primeira_execucao,
                tf_info=tf_info,
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