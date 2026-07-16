import sympy as sp


# =========================
# Symbols
# =========================
N = sp.Symbol("N")
H1, H2 = sp.symbols("H_1 H_2")
h1, h2 = sp.symbols("h_1 h_2")


# =========================
# Small formatting utilities
# =========================
def _normalize_pair(pair, name="pair"):
    """
    Normalize a pair-like value.

    This also fixes a common bug:
        cell = ("H_1-1", "H_2"),
    which creates (("H_1-1", "H_2"),) instead of ("H_1-1", "H_2").
    """
    if pair is None:
        return None

    if isinstance(pair, tuple) and len(pair) == 1 and isinstance(pair[0], tuple):
        pair = pair[0]

    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError(f"{name} must be a 2-tuple, got {pair!r}")

    return pair


def _as_latex_text(value):
    """
    Convert a symbolic/index value to a readable LaTeX-like string for subscripts.

    Examples:
        H1 - 1 -> H_1 - 1
        h1 + 1 -> h_1 + 1
        "N+1" -> N+1
    """
    if isinstance(value, str):
        return value
    return sp.latex(value).replace(r"\left", "").replace(r"\right", "")


def _join_index(index):
    index = _normalize_pair(index, "index")
    return f"{_as_latex_text(index[0])},{_as_latex_text(index[1])}"


def _join_cell(cell):
    cell = _normalize_pair(cell, "cell")
    return f"{_as_latex_text(cell[0])},{_as_latex_text(cell[1])}"


# =========================
# Coordinate and phase calculation
# =========================
def basis_offset_sympy(kind):
    """
    Eq. (147), (148):
        x_m has + 1/3(a1+a2)
        x_M has + 2/3(a1+a2)
    """
    if kind == "m":
        return sp.Rational(1, 3), sp.Rational(1, 3)
    if kind == "M":
        return sp.Rational(2, 3), sp.Rational(2, 3)
    raise ValueError("kind must be 'm' or 'M'")


def coord_coeff_sympy(kind, index, cell):
    """
    Return coefficients (c1, c2) for

        x = c1*a1 + c2*a2

    based on Eqs. (147)-(150):

        x_m = H1*A1 + H2*A2 + h1*a1 + h2*a2 + 1/3(a1+a2)
        x_M = H1*A1 + H2*A2 + h1*a1 + h2*a2 + 2/3(a1+a2)

        A1 = (N+1)a1 - a2
        A2 = -a1 + N*a2
    """
    index = _normalize_pair(index, "index")
    cell = _normalize_pair(cell, "cell")

    h1_val, h2_val = index
    H1_val, H2_val = cell

    b1, b2 = basis_offset_sympy(kind)

    c1 = H1_val * (N + 1) + H2_val * (-1) + h1_val + b1
    c2 = H1_val * (-1) + H2_val * N + h2_val + b2

    return sp.simplify(c1), sp.simplify(c2)


def phase_diff_coeff(neighbor_kind, neighbor_index, neighbor_cell,
                     center_kind, center_index, center_cell):
    """
    Compute x_neighbor - x_center as coefficients of a1 and a2.
    """
    n1, n2 = coord_coeff_sympy(neighbor_kind, neighbor_index, neighbor_cell)
    c1, c2 = coord_coeff_sympy(center_kind, center_index, center_cell)

    return sp.simplify(n1 - c1), sp.simplify(n2 - c2)


def coeff_to_latex_term(coeff, basis_latex):
    """
    Convert coeff * basis vector into LaTeX.
    """
    coeff = sp.simplify(coeff)

    if coeff == 0:
        return ""
    if coeff == 1:
        return basis_latex
    if coeff == -1:
        return "-" + basis_latex

    return sp.latex(coeff).replace(" ", "") + basis_latex


def phase_diff_latex(neighbor_kind, neighbor_index, neighbor_cell,
                     center_kind, center_index, center_cell):
    """
    Return LaTeX string for x_neighbor - x_center.

    Example:
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
    return result[1:] if result.startswith("+") else result


# =========================
# LaTeX object builders
# =========================
def amp_latex(amp, index, cell=None):
    """
    Plane-wave amplitude notation:
        U_[h1,h2]^[H1,H2] or V_[h1,h2]^[H1,H2]
    """
    index = _normalize_pair(index, "index")

    if cell is None:
        return rf"\mathbf{{{amp}}}_{{[{_join_index(index)}]}}"

    cell = _normalize_pair(cell, "cell")
    return rf"\mathbf{{{amp}}}_{{[{_join_index(index)}]}}^{{[{_join_cell(cell)}]}}"


def disp_latex(disp, index, cell=None):
    """
    Real-space displacement notation:
        u_[h1,h2]^[H1,H2] or v_[h1,h2]^[H1,H2]

    Use this for original pre-plane-wave spring equations like:
        (u_neighbor - u_center) dot a_hat.
    """
    index = _normalize_pair(index, "index")

    if cell is None:
        return rf"\mathbf{{{disp}}}_{{[{_join_index(index)}]}}"

    cell = _normalize_pair(cell, "cell")
    return rf"\mathbf{{{disp}}}_{{[{_join_index(index)}]}}^{{[{_join_cell(cell)}]}}"


def coord_latex(kind, index, cell=None):
    """
    Coordinate notation:
        (x_m)_[h1,h2]^[H1,H2] or (x_M)_[h1,h2]^[H1,H2]
    """
    index = _normalize_pair(index, "index")

    if cell is None:
        return rf"(\mathbf{{x}}_{kind})_{{[{_join_index(index)}]}}"

    cell = _normalize_pair(cell, "cell")
    return rf"(\mathbf{{x}}_{kind})_{{[{_join_index(index)}]}}^{{[{_join_cell(cell)}]}}"


def phase_latex(coord):
    return rf"\exp\left(i\mathbf{{k}}\cdot {coord}\right)\exp(i\omega t)"


def exp_phase_diff(phase_diff):
    return rf"\exp\left(i\mathbf{{k}}\cdot\left({phase_diff}\right)\right)"


# =========================
# Macros for original spring equations
# =========================
def spring_original_term_latex(
    spring_const,
    neighbor_disp,
    neighbor_index,
    neighbor_cell,
    center_disp,
    center_index,
    center_cell,
    direction,
):
    """
    Make one original, pre-plane-wave spring term:

        C [ (u_neighbor - u_center) · e_hat ] e_hat
    """
    neighbor = disp_latex(neighbor_disp, neighbor_index, neighbor_cell)
    center = disp_latex(center_disp, center_index, center_cell)

    return rf"""{spring_const}
\left[
\left(
{neighbor}
-
{center}
\right)
\cdot {direction}
\right]
{direction}"""


def spring_original_multi_term_latex(
    term_name,
    spring_const,
    neighbors,
    center_disp,
    center_index,
    center_cell,
    direction,
    display=True,
):
    """
    Make a multi-neighbor original spring equation.

    Example target:
        f_1^(2) =
        C_1^(2)[(u_[h1-1,h2]-u_[h1,h2])·a1]a1
        + C_1^(2)[(u_[h1+1,h2]-u_[h1,h2])·a1]a1

    neighbors: list of dictionaries:
        {
            "disp": "u" or "v",
            "index": (..., ...),
            "cell": None or (..., ...)
        }
    """
    pieces = []
    for nb in neighbors:
        pieces.append(
            spring_original_term_latex(
                spring_const=spring_const,
                neighbor_disp=nb["disp"],
                neighbor_index=nb["index"],
                neighbor_cell=nb.get("cell"),
                center_disp=center_disp,
                center_index=center_index,
                center_cell=center_cell,
                direction=direction,
            ).strip()
        )

    body = "\n+\n".join(pieces)
    equation = "=\n" + body

    if display:
        return "\\[\n" + equation + "\n\\]"
    return equation


# =========================
# Macros for plane-wave-substituted equations
# =========================
def spring_term_latex(
    spring_const,
    neighbor_amp,
    neighbor_kind,
    neighbor_index,
    neighbor_cell,
    center_amp,
    center_kind,
    center_index,
    center_cell,
    phase_diff,
    direction,
    form="factored",
):
    """
    Make one plane-wave-substituted spring term.

    form:
        "raw"      : no factoring
        "factored" : factor out center phase, but do not cancel it
        "reduced"  : cancel common center phase
    """
    neighbor = amp_latex(neighbor_amp, neighbor_index, neighbor_cell)
    center = amp_latex(center_amp, center_index, center_cell)

    x_neighbor = coord_latex(neighbor_kind, neighbor_index, neighbor_cell)
    x_center = coord_latex(center_kind, center_index, center_cell)
    common_phase = phase_latex(x_center)

    if form == "raw":
        neighbor_part = rf"{neighbor}\exp\left(i\mathbf{{k}}\cdot {x_neighbor}\right)\exp(i\omega t)"
        center_part = rf"{center}\exp\left(i\mathbf{{k}}\cdot {x_center}\right)\exp(i\omega t)"
        return rf"""{spring_const}
\left[
\left(
{neighbor_part}
-
{center_part}
\right)
\cdot {direction}
\right]
{direction}"""

    core = rf"""{spring_const}
\left[
\left(
{neighbor}
{exp_phase_diff(phase_diff)}
-
{center}
\right)
\cdot {direction}
\right]
{direction}"""

    if form == "factored":
        return core + "\n" + rf"\times {common_phase}"

    if form == "reduced":
        return core

    raise ValueError("form must be 'raw', 'factored', or 'reduced'")


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
    form="factored",
):
    """
    Make one plane-wave-substituted spring term.
    phase_diff is automatically computed using Eqs. (147)-(150).
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
        neighbor_kind=neighbor_kind,
        neighbor_index=neighbor_index_latex,
        neighbor_cell=neighbor_cell_latex,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index_latex,
        center_cell=center_cell_latex,
        phase_diff=phase_diff,
        direction=direction,
        form=form,
    )


def spring_multi_term_latex_auto_phase(
    term_name,
    spring_const,
    neighbors,
    center_amp,
    center_kind,
    center_index_sympy,
    center_cell_sympy,
    center_index_latex,
    center_cell_latex,
    direction,
    form="factored",
    display=True,
):
    """
    Make a multi-neighbor plane-wave-substituted spring equation.

    neighbors: list of dictionaries:
        {
            "amp": "U" or "V",
            "kind": "m" or "M",
            "index_sympy": (..., ...),
            "cell_sympy": (..., ...),
            "index_latex": (..., ...),
            "cell_latex": None or (..., ...)
        }
    """
    pieces = []
    for nb in neighbors:
        pieces.append(
            spring_term_latex_auto_phase(
                spring_const=spring_const,
                neighbor_amp=nb["amp"],
                neighbor_kind=nb["kind"],
                neighbor_index_sympy=nb["index_sympy"],
                neighbor_cell_sympy=nb["cell_sympy"],
                neighbor_index_latex=nb["index_latex"],
                neighbor_cell_latex=nb.get("cell_latex"),
                center_amp=center_amp,
                center_kind=center_kind,
                center_index_sympy=center_index_sympy,
                center_cell_sympy=center_cell_sympy,
                center_index_latex=center_index_latex,
                center_cell_latex=center_cell_latex,
                direction=direction,
                form=form,
            ).strip()
        )

    body = "\n+\n".join(pieces)
    equation = "=\n" + body

    if display:
        return "\\[\n" + equation + "\n\\]"
    return equation


# =========================
# Examples
# =========================
if __name__ == "__main__":
    # Example 1: Original equation like the uploaded image.
    term_name = "f_1^{②}"

    center_amp="U"
    center_kind="m"
    center_index_sympy=(h1, N)
    center_cell_sympy=(H1, H2)
    center_index_latex=("h_1", "N")
    center_cell_latex=("H_1", "H_2")
    form="factored"
    
    spring_const = "C_1^{②}"
    direction=r"\hat{\mathbf{a}}_1"
    eq_f1_2_reduced = spring_multi_term_latex_auto_phase(
        term_name=term_name,
        spring_const=spring_const,
        neighbors=[
            {
                "amp": "U",
                "kind": "m",
                "index_sympy": (h1 -1, N),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1 -1", "N"),
                "cell_latex": ("H_1", "H_2"),
            },
            {
                "amp": "U",
                "kind": "m",
                "index_sympy": (h1 + 1, N),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1 + 1", "N"),
                "cell_latex": ("H_1", "H_2"),
            },
        ],
        center_amp=center_amp,
        center_kind=center_kind,
        center_index_sympy=center_index_sympy,
        center_cell_sympy=center_cell_sympy,
        center_index_latex=center_index_latex,
        center_cell_latex=center_cell_latex,
        direction=direction,
        form=form
    )
    print("1면\n\n", eq_f1_2_reduced)
    
    spring_const = "C_2^{②}"
    direction=r"\hat{\mathbf{a}}_2"
    eq_f1_2_reduced = spring_multi_term_latex_auto_phase(
        term_name=term_name,
        spring_const=spring_const,
        neighbors=[
            {
                "amp": "U",
                "kind": "m",
                "index_sympy": (h1, N -1),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1", "N - 1"),
                "cell_latex": ("H_1", "H_2"),
            },
            {
                "amp": "U",
                "kind": "m",
                "index_sympy": (h1 - 1, 1),
                "cell_sympy": (H1, H2 + 1),
                "index_latex": ("h_1 - 1", "1"),
                "cell_latex": ("H_1", "H_2 + 1"),
            },
        ],
        center_amp=center_amp,
        center_kind=center_kind,
        center_index_sympy=center_index_sympy,
        center_cell_sympy=center_cell_sympy,
        center_index_latex=center_index_latex,
        center_cell_latex=center_cell_latex,
        direction=direction,
        form=form
    )

    print("2면\n\n", eq_f1_2_reduced)
    
    spring_const = "C_3^{②}"
    direction=r"\hat{\mathbf{a}}_{1\bar{2}}"
    eq_f1_2_reduced = spring_multi_term_latex_auto_phase(
        term_name=term_name,
        spring_const=spring_const,
        neighbors=[
            {
                "amp": "U",
                "kind": "m",
                "index_sympy": (h1 -2,1),
                "cell_sympy": (H1, H2 + 1),
                "index_latex": ("h_1 -2", "1"),
                "cell_latex": ("H_1", "H_2 + 1"),
            },
            {
                "amp": "U",
                "kind": "m",
                "index_sympy": (h1 + 1, N-1),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1 + 1", "N -1"),
                "cell_latex": ("H_1", "H_2"),
            },
        ],
        center_amp=center_amp,
        center_kind=center_kind,
        center_index_sympy=center_index_sympy,
        center_cell_sympy=center_cell_sympy,
        center_index_latex=center_index_latex,
        center_cell_latex=center_cell_latex,
        direction=direction,
        form=form
    )

    print("3면\n\n", eq_f1_2_reduced)
    
    center_amp="V"
    center_kind="M"
    
    spring_const = "C_1^{②}"
    direction=r"\hat{\mathbf{a}}_1"
    eq_f1_2_reduced = spring_multi_term_latex_auto_phase(
        term_name=term_name,
        spring_const=spring_const,
        neighbors=[
            {
                "amp": "V",
                "kind": "M",
                "index_sympy": (h1 -1, N),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1 - 1", "N"),
                "cell_latex": ("H_1", "H_2"),
            },
            {
                "amp": "V",
                "kind": "M",
                "index_sympy": (h1 + 1, N),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1 + 1", "N"),
                "cell_latex": ("H_1", "H_2"),
            },
        ],
        center_amp=center_amp,
        center_kind=center_kind,
        center_index_sympy=center_index_sympy,
        center_cell_sympy=center_cell_sympy,
        center_index_latex=center_index_latex,
        center_cell_latex=center_cell_latex,
        direction=direction,
        form=form
    )

    print("1면\n\n", eq_f1_2_reduced)
    
    spring_const = "C_2^{②}"
    direction=r"\hat{\mathbf{a}}_2" 
    eq_f1_2_reduced = spring_multi_term_latex_auto_phase(
        term_name=term_name,
        spring_const=spring_const,
        neighbors=[
            {
                "amp": "V",
                "kind": "M",
                "index_sympy": (h1, N - 1),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1", "N - 1"),
                "cell_latex": ("H_1", "H_2"),
            },
            {
                "amp": "V",
                "kind": "M",
                "index_sympy": (h1-1, 1),
                "cell_sympy": (H1, H2 + 1),
                "index_latex": ("h_1 -1", "1"),
                "cell_latex": ("H_1", "H_2 + 1"),
            },
        ],
        center_amp=center_amp,
        center_kind=center_kind,
        center_index_sympy=center_index_sympy,
        center_cell_sympy=center_cell_sympy,
        center_index_latex=center_index_latex,
        center_cell_latex=center_cell_latex,
        direction=direction,
        form=form
    )

    print("2면\n\n", eq_f1_2_reduced)

    spring_const = "C_3^{②}"
    direction=r"\hat{\mathbf{a}}_{1\bar{2}}"
    eq_f1_2_reduced = spring_multi_term_latex_auto_phase(
        term_name=term_name,
        spring_const=spring_const,
        neighbors=[
            {
                "amp": "V",
                "kind": "M",
                "index_sympy": (h1 -2, 1),
                "cell_sympy": (H1, H2 + 1),
                "index_latex": ("h_1 - 2", "1"),
                "cell_latex": ("H_1", "H_2 + 1"),
            },
            {
                "amp": "V",
                "kind": "M",
                "index_sympy": (h1 + 1, N - 1),
                "cell_sympy": (H1, H2),
                "index_latex": ("h_1 + 1", "N - 1"),
                "cell_latex": ("H_1", "H_2"),
            },
        ],
        center_amp=center_amp,
        center_kind=center_kind,
        center_index_sympy=center_index_sympy,
        center_cell_sympy=center_cell_sympy,
        center_index_latex=center_index_latex,
        center_cell_latex=center_cell_latex,
        direction=direction,
        form=form
    )

    print("3면\n\n", eq_f1_2_reduced)
    # Simple sanity checks.
    assert phase_diff_latex("m", (h1 - 1, h2), (H1, H2), "m", (h1, h2), (H1, H2)) == r"-\mathbf{a}_1"
    assert phase_diff_latex("m", (h1 + 1, h2), (H1, H2), "m", (h1, h2), (H1, H2)) == r"\mathbf{a}_1"
