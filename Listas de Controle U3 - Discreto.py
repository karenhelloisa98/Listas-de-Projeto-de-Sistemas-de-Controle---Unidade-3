from contextlib import redirect_stdout
import io
import os
import re
from fpdf import FPDF
import numpy as np
import sympy as sp

# ---- Cabecalho do PDF (edite aqui se precisar) ----
PDF_UNIVERSIDADE = "Universidade Federal do Rio Grande do Norte"
PDF_DISCIPLINA = "Disciplina: Projeto de Sistemas de Controle"
PDF_ALUNO = "Aluno: Karen Helloisa Araújo"


class RelatorioPDF(FPDF):
    """PDF com cabecalho (universidade, disciplina e aluno) em todas as paginas."""

    def header(self):
        familia, estilo, tamanho = self.font_family, self.font_style, self.font_size_pt
        self.set_font("Helvetica", style="B", size=11)
        self.cell(0, 5, text=PDF_UNIVERSIDADE, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", style="", size=10)
        self.cell(0, 5, text=PDF_DISCIPLINA, align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 5, text=PDF_ALUNO, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        if familia:
            self.set_font(familia, style=estilo, size=tamanho)

# Sistema discreto no tempo (D = 0):
#   x(k+1) = G x(k) + H u(k)
#   y(k)   = C x(k)
# Tambem aceita um sistema continuo (A, B, C ou G(s)) + periodo T, que e
# discretizado com segurador de ordem zero (ZOH):
#   G = e^(AT)   ;   H = (integral_0^T e^(A tau) d tau) B


def to_caret(expr):
    return str(expr).replace("**", "^")


def format_number_clean(val, tol=1e-9):
    """Formata numero real ou complexo em decimal usando 'j' (nunca 'I')."""
    c = complex(val)
    re_val = 0.0 if abs(c.real) < tol else round(float(c.real), 6)
    im_val = 0.0 if abs(c.imag) < tol else round(float(c.imag), 6)
    if im_val == 0.0:
        return f"{re_val}"
    sinal = "+" if im_val > 0 else "-"
    return f"({re_val} {sinal} {abs(im_val)}j)"


def format_poles(poles, tol=1e-7):
    return [format_number_clean(p, tol=tol) for p in poles]


def format_poly_decimal_str(coeffs_desc, var_name="z", tol=1e-9):
    """Monta string do polinomio expandido com coeficientes decimais e '^'."""
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
    """Numero real em decimal limpo (sem zeros sobrando, sem -0)."""
    v = round(float(complex(val).real), casas)
    if v == 0:
        v = 0.0
    txt = f"{v:.{casas}f}".rstrip("0").rstrip(".")
    return txt if txt else "0"


def _matriz_limpa(M, casas=4):
    """Matriz sympy com os elementos formatados em decimal (so para imprimir)."""
    return sp.Matrix(
        [[sp.Symbol(_fmt(M[i, j], casas)) for j in range(M.cols)] for i in range(M.rows)]
    )


def _poly_sig(coeffs_desc, var_name="z", tol=1e-12):
    """Polinomio com 6 algarismos significativos (bom para coeficientes pequenos)."""
    deg = len(coeffs_desc) - 1
    termos = []
    for idx, c in enumerate(coeffs_desc):
        c = float(np.real(c))
        if abs(c) < tol:
            continue
        pot = deg - idx
        cs = f"{c:.6g}"
        if pot == 0:
            termos.append(cs)
        elif pot == 1:
            termos.append(f"{cs}*{var_name}")
        else:
            termos.append(f"{cs}*{var_name}^{pot}")
    return " + ".join(termos).replace("+ -", "- ") if termos else "0"


def expm_np(M):
    """Exponencial de matriz (escalonamento e quadrados + serie de Taylor), so com numpy."""
    M = np.array(M, dtype=float)
    norma = np.linalg.norm(M, np.inf)
    s = 0 if norma == 0 else max(0, int(np.ceil(np.log2(norma))) + 1)
    Ms = M / (2 ** s)
    n = M.shape[0]
    E = np.eye(n)
    termo = np.eye(n)
    for k in range(1, 30):
        termo = termo @ Ms / k
        E = E + termo
    for _ in range(s):
        E = E @ E
    return E


def discretize_zoh(A, B, C, T):
    """
    Discretizacao exata com segurador de ordem zero:
      exp([[A, B], [0, 0]] * T) = [[G, H], [0, I]]
      G = e^(AT) ,  H = (integral_0^T e^(A tau) d tau) B
    """
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float).reshape(-1, 1)
    C_np = np.array(C, dtype=float).reshape(1, -1)
    n = A_np.shape[0]
    bloco = np.zeros((n + 1, n + 1))
    bloco[:n, :n] = A_np
    bloco[:n, n:] = B_np
    E = expm_np(bloco * float(T))
    G_np = E[:n, :n]
    H_np = E[:n, n:]
    info = {
        "T": float(T),
        "A": A_np,
        "B": B_np,
        "C": C_np,
        "G": G_np,
        "H": H_np,
    }
    return G_np, H_np, C_np, info


def print_state_equation(G_np, H_np, C_np, casas=6):
    """Escreve a representacao em espaco de estados discreta (forma matricial e escalar)."""
    G_np = np.array(G_np, dtype=float)
    H_np = np.array(H_np, dtype=float).reshape(-1, 1)
    C_np = np.array(C_np, dtype=float).reshape(1, -1)
    n = G_np.shape[0]
    meio = n // 2

    def bloco(M):
        """Linhas '[ a  b ]' com colunas alinhadas."""
        txt = [[_fmt(M[i, j], casas) for j in range(M.shape[1])] for i in range(M.shape[0])]
        larg = [max(len(txt[i][j]) for i in range(M.shape[0])) for j in range(M.shape[1])]
        return ["[" + "  ".join(txt[i][j].rjust(larg[j]) for j in range(M.shape[1])) + "]"
                for i in range(M.shape[0])]

    esq = [f"x{i + 1}(k+1)" for i in range(n)]
    larg_esq = max(len(s) for s in esq)
    col_x = [f"x{i + 1}(k)" for i in range(n)]
    larg_x = max(len(s) for s in col_x)
    bG = bloco(G_np)
    bH = bloco(H_np)

    print("Equacao de estado:  x(k+1) = G x(k) + H u(k)")
    for i in range(n):
        sinal_igual = "=" if i == meio else " "
        sinal_mais = "+" if i == meio else " "
        u_txt = " u(k)" if i == meio else ""
        print(f"[{esq[i].center(larg_esq)}] {sinal_igual} {bG[i]} [{col_x[i].center(larg_x)}] "
              f"{sinal_mais} {bH[i]}{u_txt}")

    bC = bloco(C_np)[0]
    prefixo = f"y(k) = {bC} "
    print("\nEquacao de saida:  y(k) = C x(k)")
    for i in range(n):
        if i == 0:
            print(f"{prefixo}[{col_x[i].center(larg_x)}]")
        else:
            print(" " * len(prefixo) + f"[{col_x[i].center(larg_x)}]")

    def termo(c, nome):
        v = _fmt(c, casas)
        if v == "0":
            return None
        if v == "1":
            return nome
        if v == "-1":
            return "-" + nome
        return f"{v}*{nome}"

    def soma(termos):
        termos = [x for x in termos if x is not None]
        if not termos:
            return "0"
        s = termos[0]
        for x in termos[1:]:
            s += f" - {x[1:]}" if x.startswith("-") else f" + {x}"
        return s

    print("\nEm forma escalar:")
    for i in range(n):
        termos = [termo(G_np[i, j], f"x{j + 1}(k)") for j in range(n)] + [termo(H_np[i, 0], "u(k)")]
        print(f"  x{i + 1}(k+1) = {soma(termos)}")
    termos_y = [termo(C_np[0, j], f"x{j + 1}(k)") for j in range(n)]
    print(f"  y(k) = {soma(termos_y)}")


def print_discretization(info):
    """Imprime o passo a passo da discretizacao ZOH."""
    A_np, B_np, C_np = info["A"], info["B"], info["C"]
    G_np, H_np, T = info["G"], info["H"], info["T"]
    n = A_np.shape[0]
    z = sp.Symbol("z")

    print("\n--- DISCRETIZACAO (SEGURADOR DE ORDEM ZERO) ---")
    print("Sistema continuo: x'(t) = A x(t) + B u(t)   ;   y(t) = C x(t)")
    print(f"Periodo de amostragem: T = {_fmt(T, 6)} s")
    print("\nMatriz A:")
    sp.pprint(_matriz_limpa(sp.Matrix(A_np.tolist()), 6))
    print("\nMatriz B:")
    sp.pprint(_matriz_limpa(sp.Matrix(B_np.tolist()), 6))
    print("\nMatriz C:")
    sp.pprint(_matriz_limpa(sp.Matrix(C_np.tolist()), 6))

    print("\nFormulas: G = e^(A*T)   ;   H = (integral de 0 a T de e^(A*tau) d(tau)) * B")
    print("\nMatriz G = e^(A*T):")
    sp.pprint(_matriz_limpa(sp.Matrix(G_np.tolist()), 6))
    print("\nMatriz H:")
    sp.pprint(_matriz_limpa(sp.Matrix(H_np.tolist()), 6))

    if abs(np.linalg.det(A_np)) > 1e-12:
        H_alt = np.linalg.inv(A_np) @ (G_np - np.eye(n)) @ B_np
        print("\nConferencia (A inversivel): H = A^(-1) (G - I) B")
        sp.pprint(_matriz_limpa(sp.Matrix(H_alt.tolist()), 6))

    lam_c = np.linalg.eigvals(A_np)
    print("\nAutovalores continuos lambda_i de A e discretos e^(lambda_i*T):")
    for i, lam in enumerate(lam_c, start=1):
        print(f"  lambda_{i} = {format_number_clean(lam)}   ->   e^(lambda_{i}*T) = "
              f"{format_number_clean(np.exp(lam * T), tol=1e-12)}")

    # G(z) = C (zI - G)^(-1) H = Num(z) / Den(z)
    G_sym = sp.Matrix(G_np.tolist())
    H_sym = sp.Matrix(H_np.tolist())
    C_sym = sp.Matrix(C_np.tolist())
    adj = (z * sp.eye(n) - G_sym).adjugate()
    num_expr = sp.expand((C_sym * adj * H_sym)[0, 0])
    num_coeffs = [float(c) for c in sp.Poly(num_expr, z).all_coeffs()] if num_expr != 0 else [0.0]
    den_coeffs = [float(c) for c in np.poly(G_np)]
    print("\nFuncao de transferencia discreta G(z) = C (zI - G)^(-1) H:")
    print(f"  Num(z) = {_poly_sig(num_coeffs, 'z')}")
    print(f"  Den(z) = {_poly_sig(den_coeffs, 'z')}")


def classify_stability_discrete(poles, tol=1e-7):
    """Classifica a estabilidade pelo modulo dos autovalores de G (circulo unitario)."""
    modulos = np.abs(np.array(poles, dtype=complex))
    if np.any(modulos > 1 + tol):
        return (
            "INSTAVEL: existe pelo menos um autovalor com modulo maior que 1 "
            "(fora do circulo unitario)."
        )
    elif np.any(np.abs(modulos - 1) <= tol):
        return (
            "MARGINALMENTE ESTAVEL / CRITICO: ha autovalor(es) com modulo igual a 1 "
            "(sobre o circulo unitario)."
        )
    else:
        return (
            "ASSINTOTICAMENTE ESTAVEL: todos os autovalores estao dentro do circulo unitario."
        )


def realization_from_tf(num_coeffs, den_coeffs, form="controlavel", var="z"):
    """
    Constroi G, H e C a partir de G(z) = N(z)/D(z) em ordem decrescente de z.
    (com var="s" serve para G(s): devolve A, B e C do sistema continuo)
    form: 'controlavel' ou 'observavel'.
    """
    z = sp.Symbol(var)
    num_poly = sp.Poly(
        sum(sp.nsimplify(c) * (z ** i) for i, c in enumerate(reversed(num_coeffs))), z
    )
    den_poly = sp.Poly(
        sum(sp.nsimplify(c) * (z ** i) for i, c in enumerate(reversed(den_coeffs))), z
    )

    a_n = sp.simplify(den_poly.LC())
    if a_n == 0:
        raise ValueError("Coeficiente do maior grau do denominador nao pode ser zero.")

    den_monic = sp.Poly(sp.expand(den_poly.as_expr() / a_n), z)
    num_scaled = sp.Poly(sp.expand(num_poly.as_expr() / a_n), z)

    n = den_monic.degree()
    if n < 1 or num_scaled.degree() >= n:
        raise ValueError("Sistema deve ter ordem n >= 1 e ser estritamente proprio.")

    den_all = [sp.simplify(c) for c in den_monic.all_coeffs()]
    alphas_asc = list(reversed(den_all[1:]))

    raw_num = [sp.simplify(c) for c in num_scaled.all_coeffs()]
    betas_desc = [sp.Integer(0)] * (n - len(raw_num)) + raw_num
    betas_asc = list(reversed(betas_desc))

    if form == "controlavel":
        G_rows = []
        for r in range(n - 1):
            row = [sp.Integer(0)] * n
            row[r + 1] = sp.Integer(1)
            G_rows.append(row)
        G_rows.append([-a for a in alphas_asc])
        G_sym = sp.Matrix(G_rows)
        H_sym = sp.Matrix([[0] * (n - 1) + [1]]).T
        C_sym = sp.Matrix([betas_asc])
        form_title = "Forma Canonica Controlavel"
    elif form == "observavel":
        G_rows = []
        for r in range(n):
            row = [sp.Integer(0)] * n
            if r > 0:
                row[r - 1] = sp.Integer(1)
            row[-1] = -alphas_asc[r]
            G_rows.append(row)
        G_sym = sp.Matrix(G_rows)
        H_sym = sp.Matrix([[b] for b in betas_asc])
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
        "var": var,
    }
    return G_sym, H_sym, C_sym, info


def ackermann_gain_explicit(
    G_sym,
    H_sym,
    C_sym,
    I_sym,
    Wc_sym,
    Wo_sym,
    n,
    desired_poles,
    mode="controller",
):
    """
    Formula de Ackermann (discreto), passo a passo:
      controlador: K = -[0 ... 1] Wc^(-1) q_c(G)       (malha G + HK)
      observador : L = q_o(G) Wo^(-1) [0 ... 1]^T      (erro  G - LC)
    """
    z = sp.Symbol("z")
    fatores = [(z - sp.nsimplify(p)) for p in desired_poles]
    produto_fatores = sp.Mul(*fatores, evaluate=False)
    delta_poly = sp.expand(sp.Mul(*fatores))
    poly_obj = sp.Poly(delta_poly, z)
    coeffs_desc = [sp.simplify(c) for c in poly_obj.all_coeffs()]

    label_poly = "q_c" if mode == "controller" else "q_o"

    print(f"\n--- 1. Polinomio desejado {label_poly}(z) ---")
    print("Produto dos fatores (z - p_i):")
    print(" ", to_caret(produto_fatores).replace("I", "j"))
    print(f"\nPolinomio expandido {label_poly}(z) (em decimais):")
    print(" ", format_poly_decimal_str(coeffs_desc, var_name="z"))

    print("\n--- 2. Calculo explicito das potencias de G ---")
    potencias_G = {}
    for k in range(1, n + 1):
        G_pow = sp.simplify(G_sym ** k)
        potencias_G[k] = G_pow
        print(f"G^{k}:")
        sp.pprint(G_pow)
        print()

    Q_mat = sp.zeros(n, n)
    for idx, c in enumerate(coeffs_desc):
        grau = n - idx
        c_simp = sp.simplify(c)
        if grau >= 1:
            termo_sym = sp.simplify(c_simp * potencias_G[grau])
        else:
            termo_sym = sp.simplify(c_simp * I_sym)
        Q_mat = sp.simplify(Q_mat + termo_sym)

    print(f"--- 3. Matriz {label_poly}(G) ---")
    sp.pprint(Q_mat)

    G_np = np.array(G_sym.tolist(), dtype=float)

    if mode == "controller":
        Wc_inv = sp.simplify(Wc_sym.inv())
        print("\n--- 4. Inversa da matriz de controlabilidade Wc^(-1) ---")
        sp.pprint(Wc_inv)

        e_last_row = sp.Matrix([[0] * (n - 1) + [1]])
        K_sym = sp.simplify(-(e_last_row * Wc_inv * Q_mat))

        print("\n--- 5. Matriz de ganhos K = -[0 ... 1] Wc^(-1) q_c(G) ---")
        sp.pprint(K_sym)

        H_np = np.array(H_sym.tolist(), dtype=float)
        K_np = np.array(K_sym.tolist(), dtype=float).reshape(1, -1)
        polos_mf = np.linalg.eigvals(G_np + H_np @ K_np)

        print("\nVerificacao - autovalores obtidos de (G + HK):")
        for idx, p_str in enumerate(format_poles(polos_mf), start=1):
            print(f"  lambda_c_{idx} = {p_str}")
        return K_sym

    else:
        Wo_inv = sp.simplify(Wo_sym.inv())
        print("\n--- 4. Inversa da matriz de observabilidade Wo^(-1) ---")
        sp.pprint(Wo_inv)

        e_last_col = sp.Matrix([[0] * (n - 1) + [1]]).T
        L_sym = sp.simplify(Q_mat * Wo_inv * e_last_col)

        print("\n--- 5. Matriz de ganhos L = q_o(G) Wo^(-1) [0 ... 1]^T ---")
        sp.pprint(L_sym)

        C_np = np.array(C_sym.tolist(), dtype=float)
        L_np = np.array(L_sym.tolist(), dtype=float).reshape(-1, 1)
        polos_obs = np.linalg.eigvals(G_np - L_np @ C_np)

        print("\nVerificacao - autovalores obtidos de (G - LC):")
        for idx, p_str in enumerate(format_poles(polos_obs), start=1):
            print(f"  lambda_l_{idx} = {p_str}")
        return L_sym


def tracker_reference_explicit(G_sym, H_sym, C_sym, desired_poles):
    """
    Seguidor de referencia para entrada degrau (discreto), metodo do professor:
      v(k) = v(k-1) + r(k) - y(k)        u(k) = -k2 x(k) + k1 v(k)
      G_hat = [G H; 0 0],  H_hat = [0; 1]
      K_hat = [0 ... 1] Wc_hat^(-1) q_c(G_hat)            (w = -K_hat xi)
      [k2 k1] = (K_hat + [0 ... 0 1]) [G-I  H; CG  CH]^(-1)
      Malha: [x(k+1); u(k+1)] = [G  H; k2-k2G-k1CG  1-k2H-k1CH][x; u] + [0; k1] r(k+1)
    """
    n = G_sym.rows
    n_aug = n + 1
    I_n = sp.eye(n)
    I_aug = sp.eye(n_aug)

    # ---------- sistema aumentado ----------
    G_hat = sp.Matrix.vstack(sp.Matrix.hstack(G_sym, H_sym), sp.zeros(1, n_aug))
    H_hat = sp.Matrix.vstack(sp.zeros(n, 1), sp.Matrix([[1]]))

    print("Sistema aumentado (estado xi = [x_e; u_e]):")
    print("Matriz aumentada G_hat = [G H; 0 0]:")
    sp.pprint(G_hat)
    print("\nMatriz aumentada H_hat = [0; 1]:")
    sp.pprint(H_hat)

    # ---------- 1. controlabilidade aumentada ----------
    cols = [(G_hat ** k) * H_hat for k in range(n_aug)]
    Wc_hat = sp.simplify(sp.Matrix.hstack(*cols))
    rank_aug = int(np.linalg.matrix_rank(np.array(Wc_hat.tolist(), dtype=float)))

    print("\n--- 1. Matriz de controlabilidade aumentada Wc = [H_hat  G_hat*H_hat  ...] ---")
    sp.pprint(Wc_hat)
    print(f"\nOrdem do sistema aumentado: n_a = {n_aug}")
    print(f"Posto da matriz de controlabilidade: posto(Wc) = {rank_aug}")
    if rank_aug == n_aug:
        print("Conclusao: O sistema aumentado E controlavel.")
    else:
        print("Conclusao: posto(Wc) < n_a -> sistema aumentado NAO controlavel. "
              "Seguidor impossivel.")
        return

    # ---------- 2. inversa ----------
    print("\n--- 2. Inversa da matriz de controlabilidade Wc^(-1) ---")
    Wc_inv = sp.simplify(Wc_hat.inv())
    sp.pprint(Wc_inv)

    # ---------- 3. polinomio desejado ----------
    poles_complex = [complex(p) for p in desired_poles]
    fatores_str = []
    delta_poly_num = np.poly1d([1.0 + 0.0j])
    for p in poles_complex:
        re_p = 0.0 if abs(p.real) < 1e-12 else round(float(p.real), 6)
        im_p = 0.0 if abs(p.imag) < 1e-12 else round(float(p.imag), 6)
        p_clean = complex(re_p, im_p)
        fatores_str.append(f"(z - {format_number_clean(p_clean)})")
        delta_poly_num = delta_poly_num * np.poly1d([1.0 + 0.0j, -p_clean])

    coeffs_desc = list(delta_poly_num.coeffs)
    print("\n--- 3. Polinomio desejado delta(z) ---")
    print("Produto dos fatores (z - p_i):")
    print(" ", " * ".join(fatores_str))
    print("\nPolinomio expandido delta(z) (em decimais):")
    print(" ", format_poly_decimal_str(coeffs_desc, var_name="z"))

    # ---------- 4. q_c(G_hat) ----------
    potencias = {k: sp.simplify(G_hat ** k) for k in range(1, n_aug + 1)}
    Q_mat = sp.zeros(n_aug, n_aug)
    for idx, c in enumerate(coeffs_desc):
        grau = n_aug - idx
        c_num = complex(c)
        c_sym = sp.Float(round(float(c_num.real), 8)) if abs(c_num.imag) < 1e-12 else sp.nsimplify(c_num)
        termo = sp.simplify(c_sym * (potencias[grau] if grau >= 1 else I_aug))
        Q_mat = sp.simplify(Q_mat + termo)

    print("\n--- 4. Matriz q_c(G_hat) ---")
    sp.pprint(_matriz_limpa(Q_mat))

    # ---------- 5. Ackermann para K_hat ----------
    print("\n--- 5. Formula de Ackermann: K_hat = [0 ... 0 1] * Wc^(-1) * q_c(G_hat) ---")
    e_last = sp.Matrix([[0] * (n_aug - 1) + [1]])
    K_hat = sp.simplify(e_last * Wc_inv * Q_mat)
    print("Matriz K_hat (com w(k) = -K_hat * xi(k)):")
    sp.pprint(_matriz_limpa(K_hat))

    # ---------- 6. ganhos k2 e k1 ----------
    M_mat = sp.Matrix.vstack(
        sp.Matrix.hstack(G_sym - I_n, H_sym),
        sp.Matrix.hstack(C_sym * G_sym, C_sym * H_sym),
    )
    print("\n--- 6. Ganhos [k2  k1] = (K_hat + [0 ... 0 1]) * [G-I  H; CG  CH]^(-1) ---")
    print("Matriz [G-I  H; CG  CH]:")
    sp.pprint(M_mat)

    if np.linalg.matrix_rank(np.array(M_mat.tolist(), dtype=float)) < n_aug:
        print("\nAviso: a matriz [G-I  H; CG  CH] e singular (G(z) tem zero em z = 1). "
              "Seguidor impossivel.")
        return

    M_inv = sp.simplify(M_mat.inv())
    print("\nInversa [G-I  H; CG  CH]^(-1):")
    sp.pprint(M_inv)

    K_gain = sp.simplify((K_hat + e_last) * M_inv)
    k2_vals = [float(sp.N(K_gain[0, j])) for j in range(n)]
    k1_val = float(sp.N(K_gain[0, n]))

    nomes_k2 = "  ".join(f"k2{j + 1}" for j in range(n))
    vals_k2 = "  ".join(_fmt(v) for v in k2_vals)
    print(f"\n[k2 | k1] = [{nomes_k2} | k1] = [ {vals_k2} | {_fmt(k1_val)} ]")

    # ---------- 7. equacoes de estado em malha fechada, estado [x; u] ----------
    G_np = np.array(G_sym.tolist(), dtype=float)
    H_np = np.array(H_sym.tolist(), dtype=float).reshape(-1, 1)
    C_np = np.array(C_sym.tolist(), dtype=float).reshape(1, -1)
    k2_np = np.array(k2_vals, dtype=float).reshape(1, -1)

    bottom_left = k2_np - k2_np @ G_np - k1_val * (C_np @ G_np)
    bottom_right = np.ones((1, 1)) - k2_np @ H_np - k1_val * (C_np @ H_np)
    A_cl = np.block([
        [G_np, H_np],
        [bottom_left, bottom_right],
    ])
    B_r = np.vstack([np.zeros((n, 1)), [[k1_val]]])

    estados = ", ".join([f"x{i + 1}(k)" for i in range(n)] + ["u(k)"])
    print(f"\n--- 7. Equacoes de estado em malha fechada (estado [{estados}]^T) ---")
    print("[x(k+1); u(k+1)] = [G  H; k2-k2*G-k1*C*G  1-k2*H-k1*C*H] [x(k); u(k)] + [0; k1] r(k+1)")
    print("\nMatriz dinamica de malha fechada:")
    sp.pprint(_matriz_limpa(sp.Matrix(A_cl.tolist())))
    print("\nMatriz de entrada (referencia r(k+1)):")
    sp.pprint(_matriz_limpa(sp.Matrix(B_r.tolist())))

    # ---------- verificacao ----------
    polos_mf = np.linalg.eigvals(A_cl)
    print("\nVerificacao - autovalores de malha fechada:")
    for idx, p_str in enumerate(format_poles(polos_mf), start=1):
        print(f"  lambda_{idx} = {p_str}")


def full_control_analysis_explicit(
    G,
    H,
    C_out,
    poles_k=None,
    poles_l=None,
    poles_ref=None,
    calc_initial=True,
    calc_k=False,
    calc_l=False,
    calc_ref=False,
    show_header=True,
    tf_info=None,
    disc_info=None,
    calc_disc=False,
):
    G_np = np.array(G, dtype=float)
    H_np = np.array(H, dtype=float).reshape(-1, 1)
    C_np = np.array(C_out, dtype=float).reshape(1, -1)
    n = G_np.shape[0]

    z = sp.Symbol("z")
    G_sym = sp.Matrix(G_np)
    H_sym = sp.Matrix(H_np)
    C_sym = sp.Matrix(C_np)
    I_sym = sp.eye(n)

    if show_header:
        print("=" * 75)
        if disc_info is not None:
            print("DADOS DO SISTEMA DISCRETO (OBTIDO DE UM SISTEMA CONTINUO POR ZOH)")
            print(f"  Periodo de amostragem: T = {_fmt(disc_info['T'], 6)} s")
        else:
            print("DADOS DO SISTEMA DISCRETO FORNECIDOS NA ENTRADA")
        print("  x(k+1) = G x(k) + H u(k)   ;   y(k) = C x(k)")
        print("=" * 75)
        if tf_info is not None:
            v = tf_info.get("var", "z")
            print(f"Funcao de transferencia G({v}):")
            print(f"  Num({v}) = {to_caret(tf_info['num_expr'])}")
            print(f"  Den({v}) = {to_caret(tf_info['den_expr'])}")
            print(f"Realizacao escolhida: {tf_info['form_title']} (ordem n = {n})")
            alphas_str = ", ".join(f"alpha_{i+1} = {val}" for i, val in enumerate(tf_info["alphas_asc"]))
            betas_str = ", ".join(f"beta_{i+1} = {val}" for i, val in enumerate(tf_info["betas_asc"]))
            print(f"Coeficientes do denominador: {alphas_str}")
            print(f"Coeficientes do numerador:   {betas_str}\n")
        print("Matriz G:")
        sp.pprint(G_sym)
        print("\nMatriz H:")
        sp.pprint(H_sym)
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
            col = H_sym
        elif k == 1:
            col = G_sym * H_sym
        else:
            col = (G_sym ** k) * H_sym
        ctrl_cols.append(col)
    Wc = sp.Matrix.hstack(*ctrl_cols)
    rank_c = int(np.linalg.matrix_rank(np.array(Wc.tolist(), dtype=float)))
    is_ctrl = (rank_c == n)

    obs_rows = []
    for k in range(n):
        if k == 0:
            row = C_sym
        elif k == 1:
            row = C_sym * G_sym
        else:
            row = C_sym * (G_sym ** k)
        obs_rows.append(row)
    Wo = sp.Matrix.vstack(*obs_rows)
    rank_o = int(np.linalg.matrix_rank(np.array(Wo.tolist(), dtype=float)))
    is_obs = (rank_o == n)

    if calc_disc:
        print("\n0. REPRESENTACAO DISCRETA DO SISTEMA")
        print("-" * 75)
        if disc_info is not None:
            print_discretization(disc_info)
            print("\n>>> REPRESENTACAO DISCRETA EM ESPACO DE ESTADOS (T = "
                  f"{_fmt(disc_info['T'], 6)} s) <<<\n")
            print_state_equation(G_np, H_np, C_np)
        else:
            print("O sistema ja foi informado na forma discreta (G, H, C), nao ha o que discretizar.")
            print("(Para discretizar, escolha o modo de entrada [2] ou [3] ao iniciar o programa.)")
            print("\nRepresentacao discreta em espaco de estados:\n")
            print_state_equation(G_np, H_np, C_np)

    if calc_initial:
        zI_minus_G = z * I_sym - G_sym
        det_expr = sp.det(zI_minus_G)
        char_poly = sp.expand(det_expr)

        poly_obj = sp.Poly(char_poly, z)
        coeffs_desc = [complex(c) for c in poly_obj.all_coeffs()]
        poles = np.roots(coeffs_desc)

        print("\n1. POLINOMIO CARACTERISTICO, AUTOVALORES E ESTABILIDADE")
        print("-" * 75)
        print("Matriz (zI - G):")
        sp.pprint(zI_minus_G)
        print("\nPolinomio caracteristico expandido p(z) = det(zI - G):")
        print(" ", to_caret(char_poly))
        print("\nAutovalores de G (raizes de p(z) = 0) e seus modulos:")
        for idx, p_str in enumerate(format_poles(poles), start=1):
            print(f"  p_{idx} = {p_str}   |p_{idx}| = {_fmt(abs(poles[idx - 1]), casas=6)}")
        print("\nConclusao sobre a estabilidade:")
        print(" ", classify_stability_discrete(poles))

        print("\n2. MATRIZ DE CONTROLABILIDADE")
        print("-" * 75)
        for k, col in enumerate(ctrl_cols):
            label = "H" if k == 0 else ("GH" if k == 1 else f"G^{k}H")
            print(f"Coluna {label}:")
            sp.pprint(col)
            print()

        print("Matriz de controlabilidade Wc = [H  GH  G^2H ...]:")
        sp.pprint(Wc)
        print(f"\nOrdem do sistema: n = {n}")
        print(f"Posto da matriz de controlabilidade: posto(Wc) = {rank_c}")
        print("Conclusao:", "O sistema eh controlavel." if is_ctrl else "O sistema nao eh controlavel.")

        print("\n3. MATRIZ DE OBSERVABILIDADE")
        print("-" * 75)
        for k, row in enumerate(obs_rows):
            label = "C" if k == 0 else ("CG" if k == 1 else f"CG^{k}")
            print(f"Linha {label}:")
            sp.pprint(row)
            print()

        print("Matriz de observabilidade Wo = [C; CG; CG^2; ...]:")
        sp.pprint(Wo)
        print(f"\nOrdem do sistema: n = {n}")
        print(f"Posto da matriz de observabilidade: posto(Wo) = {rank_o}")
        print("Conclusao:", "O sistema eh observavel." if is_obs else "O sistema nao eh observavel.")

    if calc_k:
        print("\n4. REALIMENTACAO DE ESTADOS (FORMULA DE ACKERMANN)")
        print("-" * 75)
        if is_ctrl:
            ackermann_gain_explicit(
                G_sym=G_sym,
                H_sym=H_sym,
                C_sym=C_sym,
                I_sym=I_sym,
                Wc_sym=Wc,
                Wo_sym=Wo,
                n=n,
                desired_poles=poles_k,
                mode="controller",
            )
        else:
            print("Calculo de K cancelado: o sistema nao eh controlavel.")

    if calc_l:
        print("\n5. OBSERVADOR DE ESTADOS (FORMULA DE ACKERMANN)")
        print("-" * 75)
        if is_obs:
            ackermann_gain_explicit(
                G_sym=G_sym,
                H_sym=H_sym,
                C_sym=C_sym,
                I_sym=I_sym,
                Wc_sym=Wc,
                Wo_sym=Wo,
                n=n,
                desired_poles=poles_l,
                mode="observer",
            )
        else:
            print("Calculo de L cancelado: o sistema nao eh observavel.")

    if calc_ref:
        print("\n6. SEGUIDOR DE REFERENCIA (ENTRADA DEGRAU)")
        print("-" * 75)
        tracker_reference_explicit(G_sym, H_sym, C_sym, poles_ref)


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


def export_text_to_pdf(text, filename="lista_controle_discreto.pdf", questao=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, filename)

    linhas = sanitize_for_pdf(text).splitlines()
    margem = 10.0
    largura_util = 210.0 - 2 * margem
    max_chars = max((len(l) for l in linhas), default=1)
    # Courier: largura do caractere = 0.6 * tamanho da fonte (pt) * 0.3528 mm
    fonte = min(8.0, largura_util / (max_chars * 0.6 * 0.3528))
    fonte = max(fonte, 4.0)

    pdf = RelatorioPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(margem, 12, margem)
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    if questao:
        titulo = f"Questão {questao}".encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 9, text=titulo, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
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


def prompt_matrix_G(n):
    print(f"\nDigite a matriz G ({n}x{n}), uma linha por vez (elementos separados por espaco ou virgula):")
    return [parse_number_list(input(f"  Linha {i + 1}: "), expected_len=n) for i in range(n)]


def prompt_vector_H(n):
    print(f"\nDigite a matriz/vetor H ({n} linhas, 1 coluna):")
    return [parse_number_list(input(f"  H[{i + 1}]: "), expected_len=1)[0] for i in range(n)]


def prompt_vector_C(n):
    raw = input(f"\nDigite a matriz/vetor C (1 linha, {n} colunas separados por espaco ou virgula): ")
    return parse_number_list(raw, expected_len=n)


def prompt_poles(count, label):
    print(f"\nFormato para digitar os {count} polos de {label}:")
    print("  - Separe cada polo por espaco ou virgula (ex.: 0.5+0.2j 0.5-0.2j 0.3 0.1)")
    print("  - Numeros complexos: use j ou i, com ou sem * (ex.: 0.5+0.2j, 0.5+1*j, 0.5+0.2i)")
    print("  - Reais, decimais e fracoes tambem sao aceitos (ex.: 0.3, -0.5, 1/4)")
    print("  - Polos complexos devem vir em pares conjugados")
    print("  - Para estabilidade, os polos devem estar dentro do circulo unitario (|z| < 1)")
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
        if any(abs(complex(p)) >= 1 for p in poles):
            print("Aviso: ha polo(s) com |z| >= 1; a malha fechada nao sera assintoticamente estavel.")
        return poles


def prompt_period():
    print("\nPeriodo de amostragem T (em segundos; aceita decimais e fracoes, ex.: 0.1, 1/10):")
    while True:
        raw = input("Digite T: ")
        try:
            T = parse_number_list(raw, expected_len=1)[0]
            if isinstance(T, complex) or T <= 0:
                raise ValueError("T deve ser um numero real positivo.")
            return float(T)
        except Exception as exc:
            print(f"Entrada invalida ({exc}). Tente novamente.")


def parse_options(raw_opts):
    """Interpreta uma ou mais opcoes como '1, 4', '2, 3', '5' ou '6' (tudo)."""
    tokens = [t.strip() for t in raw_opts.replace(",", " ").split() if t.strip()]
    if not tokens or "6" in tokens:
        return {"1", "2", "3", "4", "5"}
    valid = {t for t in tokens if t in {"1", "2", "3", "4", "5"}}
    return valid if valid else {"2"}


def prompt_question():
    while True:
        q = input("Numero da questao (ex.: 1, 2, 3a): ").strip()
        if q:
            return q
        print("Digite o numero da questao.")


def run_interactive():
    print("=== ANALISE E PROJETO DE SISTEMAS DE CONTROLE DISCRETOS (ESPACO DE ESTADOS) ===")
    questao = prompt_question()
    print("Modelo: x(k+1) = G x(k) + H u(k) ; y(k) = C x(k)")
    print("\nComo voce deseja informar o sistema?")
    print("  [1] Digitar as matrizes G, H e C do sistema discreto")
    print("  [2] Informar o sistema CONTINUO (A, B, C) e o periodo T (discretizacao ZOH)")
    print("  [3] Informar G(s) CONTINUA, escolher a realizacao e o periodo T (discretizacao ZOH)")
    modo_entrada = input("Escolha o modo de entrada [1/2/3] (padrao 1): ").strip() or "1"

    tf_info = None
    disc_info = None

    if modo_entrada == "3":
        print("\nDigite os coeficientes de G(s) em ordem decrescente de s (separados por espaco ou virgula):")
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

        A_s, B_s, C_s, tf_info = realization_from_tf(num_coeffs, den_coeffs, form=forma, var="s")
        n = tf_info["n"]
        T = prompt_period()
        G_np, H_np, C_np, disc_info = discretize_zoh(
            np.array(A_s.tolist(), dtype=float),
            np.array(B_s.tolist(), dtype=float),
            np.array(C_s.tolist(), dtype=float),
            T,
        )
        G = G_np.tolist()
        H = [v[0] for v in H_np.tolist()]
        C = C_np.tolist()[0]
    elif modo_entrada == "2":
        n = int(input("\nOrdem do sistema n (ex.: 2, 3): ").strip())
        if n < 1:
            raise ValueError("A ordem do sistema deve ser >= 1.")
        print("\n--- Sistema CONTINUO: x'(t) = A x(t) + B u(t) ; y(t) = C x(t) ---")
        A_c = [parse_number_list(input(f"  A, linha {i + 1}: "), expected_len=n) for i in range(n)]
        B_c = [parse_number_list(input(f"  B[{i + 1}]: "), expected_len=1)[0] for i in range(n)]
        C_c = parse_number_list(input(f"  C (1 linha, {n} colunas): "), expected_len=n)
        T = prompt_period()
        G_np, H_np, C_np, disc_info = discretize_zoh(A_c, B_c, C_c, T)
        G = G_np.tolist()
        H = [v[0] for v in H_np.tolist()]
        C = C_np.tolist()[0]
    else:
        n = int(input("\nOrdem do sistema n (ex.: 2, 3): ").strip())
        if n < 1:
            raise ValueError("A ordem do sistema deve ser >= 1.")
        G = prompt_matrix_G(n)
        H = prompt_vector_H(n)
        C = prompt_vector_C(n)

    padrao = "questao_" + re.sub(r"[^0-9A-Za-z]+", "_", questao) + ".pdf"
    pdf_name = input(f"\nNome do arquivo PDF (Enter para '{padrao}'): ").strip() or padrao
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name += ".pdf"

    buffer = io.StringIO()
    primeira_execucao = True

    while True:
        print("\nO que voce deseja calcular? (Voce pode digitar UMA ou VARIAS opcoes, ex.: 1, 4)")
        print("  [1] Representacao discreta (discretizacao ZOH do sistema continuo, com T)")
        print("  [2] Analise inicial (polinomio caracteristico, estabilidade, controlabilidade e observabilidade)")
        print("  [3] Realimentacao de estados (K)")
        print("  [4] Observador de estados (L)")
        print("  [5] Seguidor de referencia (entrada degrau)")
        print("  [6] Todas as anteriores (1 + 2 + 3 + 4 + 5)")
        raw_opts = input("Digite a(s) opcao(oes) desejada(s): ").strip()
        escolhas = parse_options(raw_opts)

        calc_disc = "1" in escolhas
        calc_initial = "2" in escolhas
        calc_k = "3" in escolhas
        calc_l = "4" in escolhas
        calc_ref = "5" in escolhas

        poles_k = prompt_poles(n, "a realimentacao de estados (K)") if calc_k else None
        poles_l = prompt_poles(n, "o observador de estados (L)") if calc_l else None
        poles_ref = prompt_poles(n + 1, f"o seguidor de referencia (ordem {n + 1})") if calc_ref else None

        with redirect_stdout(buffer):
            full_control_analysis_explicit(
                G=G,
                H=H,
                C_out=C,
                poles_k=poles_k,
                poles_l=poles_l,
                poles_ref=poles_ref,
                calc_initial=calc_initial,
                calc_k=calc_k,
                calc_l=calc_l,
                calc_ref=calc_ref,
                show_header=primeira_execucao,
                tf_info=tf_info,
                disc_info=disc_info,
                calc_disc=calc_disc,
            )

        primeira_execucao = False

        mais = input("\nDeseja calcular outra combinacao para este mesmo sistema? [s/N]: ").strip().lower()
        if mais not in ("s", "sim", "y", "yes"):
            break

    saida_completa = buffer.getvalue()
    print(saida_completa)

    caminho_pdf = export_text_to_pdf(saida_completa, filename=pdf_name, questao=questao)
    print(f"\nPDF gerado com sucesso em: {caminho_pdf}")


if __name__ == "__main__":
    run_interactive()