import sympy as sp


# Symbols
N = sp.Symbol("N")
H1, H2 = sp.symbols("H_1 H_2")
h1, h2 = sp.symbols("h_1 h_2")


def basis_offset_sympy(kind):
    if kind == "m":
        return sp.Rational(1, 3), sp.Rational(1, 3)
    if kind == "M":
        return sp.Rational(2, 3), sp.Rational(2, 3)
    raise ValueError("kind must be 'm' or 'M'")


def coord_coeff_sympy(kind, index, cell):
    """
    Return coefficients (c1, c2) for

        x = c1 * a1 + c2 * a2

    based on Eqs. (147)-(150):

        x_m = H1 A1 + H2 A2 + h1 a1 + h2 a2 + 1/3(a1+a2)
        x_M = H1 A1 + H2 A2 + h1 a1 + h2 a2 + 2/3(a1+a2)

        A1 = (N+1)a1 - a2
        A2 = -a1 + N a2
    """
    h1_val, h2_val = index
    H1_val, H2_val = cell

    b1, b2 = basis_offset_sympy(kind)

    # H1*A1 + H2*A2
    # A1 = (N+1)a1 - a2
    # A2 = -a1 + N a2
    c1 = H1_val * (N + 1) + H2_val * (-1) + h1_val + b1
    c2 = H1_val * (-1) + H2_val * N + h2_val + b2

    return sp.simplify(c1), sp.simplify(c2)


def phase_diff_coeff(neighbor_kind, neighbor_index, neighbor_cell,
                     center_kind, center_index, center_cell):
    """
    Compute

        x_neighbor - x_center

    as coefficients of a1 and a2.
    """
    n1, n2 = coord_coeff_sympy(neighbor_kind, neighbor_index, neighbor_cell)
    c1, c2 = coord_coeff_sympy(center_kind, center_index, center_cell)

    d1 = sp.simplify(n1 - c1)
    d2 = sp.simplify(n2 - c2)

    return d1, d2

def coeff_to_latex_term(coeff, basis_latex):
    """
    Convert coefficient * basis vector into LaTeX.
    """
    coeff = sp.simplify(coeff)

    if coeff == 0:
        return ""

    if coeff == 1:
        return basis_latex

    if coeff == -1:
        return "-" + basis_latex

    return sp.latex(coeff) + basis_latex


def phase_diff_latex(neighbor_kind, neighbor_index, neighbor_cell,
                     center_kind, center_index, center_cell):
    """
    Return LaTeX string for

        x_neighbor - x_center

    such as:
        -\\frac{2}{3}\\mathbf{a}_1+\\frac{1}{3}\\mathbf{a}_2
    """
    d1, d2 = phase_diff_coeff(
        neighbor_kind, neighbor_index, neighbor_cell,
        center_kind, center_index, center_cell
    )

    term1 = coeff_to_latex_term(d1, r"\mathbf{a}_1")
    term2 = coeff_to_latex_term(d2, r"\mathbf{a}_2")

    terms = []

    if term1:
        terms.append(term1)

    if term2:
        if term2.startswith("-"):
            terms.append(term2)
        else:
            terms.append("+" + term2)

    if not terms:
        return "0"

    result = "".join(terms)

    # Remove leading plus if exists
    if result.startswith("+"):
        result = result[1:]

    return result

def amp_latex(amp, index, cell=None):
    h1, h2 = index

    if cell is None:
        return rf"\mathbf{{{amp}}}_{{[{h1},{h2}]}}"

    H1, H2 = cell
    return rf"\mathbf{{{amp}}}_{{[{h1},{h2}]}}^{{[{H1},{H2}]}}"


def coord_latex(kind, index, cell=None):
    """
    kind: 'm' 또는 'M'
    index: ('1', '1') 등
    cell: ('H_1', 'H_2') 등
    """
    h1, h2 = index

    if cell is None:
        return rf"(\mathbf{{x}}_{kind})_{{[{h1},{h2}]}}"

    H1, H2 = cell
    return rf"(\mathbf{{x}}_{kind})_{{[{h1},{h2}]}}^{{[{H1},{H2}]}}"


def phase_latex(coord):
    return rf"\exp\left(i\mathbf{{k}}\cdot {coord}\right)\exp(i\omega t)"
    
def spring_term_latex(
    spring_const,
    neighbor_amp,
    neighbor_index,
    neighbor_cell,
    center_amp,
    center_kind,
    center_index,
    center_cell,
    phase_diff,
    direction,
):
    neighbor = amp_latex(neighbor_amp, neighbor_index, neighbor_cell)
    center = amp_latex(center_amp, center_index, center_cell)

    x_center = coord_latex(center_kind, center_index, center_cell)

    common_phase = rf"\exp\left(i\mathbf{{k}}\cdot {x_center}\right)\exp(i\omega t)"

    body = rf"""
    =
    {spring_const}
    \left[
    \left(
    {neighbor}
    \exp\left(i\mathbf{{k}}\cdot\left({phase_diff}\right)\right)
    -
    {center}
    \right)
    \cdot {direction}
    \right]
    {direction}
    \times
    {common_phase}
    """

    return "\\[\n" + body.strip() + "\n\\]"

def spring_term_latex_auto_phase(
    spring_const,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction,
):
    """
    sympy index/cell are used for phase calculation.
    latex index/cell are used for pretty LaTeX printing.
    """

    phase_diff = phase_diff_latex(
        neighbor_kind=neighbor_kind,
        neighbor_index=neighbor_index_sympy,
        neighbor_cell=neighbor_cell_sympy,
        center_kind=center_kind,
        center_index=center_index_sympy,
        center_cell=center_cell_sympy,
    )

    return spring_term_latex(
        spring_const=spring_const,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index_latex,
        neighbor_cell=neighbor_cell_latex,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index_latex,
        center_cell=center_cell_latex,
        phase_diff=phase_diff,
        direction=direction,
    )



latex_15 = spring_term_latex_auto_phase(
    spring_const="C_1^{①}",
    neighbor_amp="V",
    neighbor_kind="M",
    neighbor_index_sympy=(N + 1, 0),
    neighbor_cell_sympy=(H1 - 1, H2),
    neighbor_index_latex=("N+1", "0"),
    neighbor_cell_latex=("H_1-1", "H_2"),
    center_amp="U",
    center_kind="m",
    center_index_sympy=(1, 1),
    center_cell_sympy=(H1, H2),
    center_index_latex=("1", "1"),
    center_cell_latex=("H_1", "H_2"),
    direction=r"\hat{\mathbf{n}}_1",
)
'''
print(latex_15)
'''
def print_one_unit():
    spring_const1= "C_1^{①}"
    spring_const2= "C_2^{①}"
    spring_const3= "C_3^{①}"
    direction1=r"\hat{\mathbf{n}}_1"
    direction2=r"\hat{\mathbf{n}}_2"
    direction3=r"\hat{\mathbf{n}}_3"
    
    neighbor_amp="V"
    neighbor_kind="M"
    center_amp="U"
    center_kind="m"
    
    center_index_latex=("h_1", "N")
    center_index_sympy=(h1, N)
    center_cell_latex=("H_1", "H_2")
    center_cell_sympy=(H1, H2)

    
    neighbor_index_latex=("h_1 - 1", "N")
    neighbor_index_sympy=(h1 - 1, N) 
    neighbor_cell_latex=("H_1", "H_2")
    neighbor_cell_sympy=(H1, H2)  

    latex1 = spring_term_latex_auto_phase(
    spring_const1,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction1,
    )

    print("라면\n\n", latex1)
    
    neighbor_index_latex=("h_1", "N - 1")
    neighbor_index_sympy=(h1, N - 1) 
    neighbor_cell_latex=("H_1", "H_2")
    neighbor_cell_sympy=(H1, H2)
        
    latex2 = spring_term_latex_auto_phase(
    spring_const2,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction2,
    )
    print("라면\n\n", latex2)
    
    neighbor_index_latex=("h_1", "N")
    neighbor_index_sympy=(h1, N) 
    neighbor_cell_latex=("H_1", "H_2")
    neighbor_cell_sympy=(H1, H2)   
        
    latex3 = spring_term_latex_auto_phase(
    spring_const3,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction3,
    )
    print("라면\n\n", latex3)
    
    neighbor_amp="U"
    neighbor_kind="m"
    center_amp="V"
    center_kind="M"

    neighbor_index_latex=("h_1 + 1", "N")
    neighbor_index_sympy=(h1 + 1, N) 
    neighbor_cell_latex=("H_1", "H_2")
    neighbor_cell_sympy=(H1, H2)    
    
    latex1 = spring_term_latex_auto_phase(
    spring_const1,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction1,
    )

    print("라면\n\n", latex1)
    
    neighbor_index_latex=("h_1 - 1", "1")
    neighbor_index_sympy=(h1 - 1, 1) 
    neighbor_cell_latex=("H_1", "H_2 + 1")
    neighbor_cell_sympy=(H1, H2 + 1)   
        
    latex2 = spring_term_latex_auto_phase(
    spring_const2,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction2,
    )
    print("라면\n\n", latex2)
    
    neighbor_index_latex=("h_1", "N")
    neighbor_index_sympy=(h1, N) 
    neighbor_cell_latex=("H_1", "H_2")
    neighbor_cell_sympy=(H1, H2)   
        
    latex3 = spring_term_latex_auto_phase(
    spring_const3,
    neighbor_amp,
    neighbor_kind,
    neighbor_index_sympy,
    neighbor_cell_sympy,
    neighbor_index_latex,
    neighbor_cell_latex,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction3,
    )
    print("라면\n\n", latex3)
    