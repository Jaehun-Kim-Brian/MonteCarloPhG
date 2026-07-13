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

latex_15_raw = spring_term_latex(
    spring_const="C_1^{①}",
    neighbor_amp="V",
    neighbor_index=("N+1", "0"),
    neighbor_cell=("H_1-1", "H_2"),
    center_amp="U",
    center_kind="m",
    center_index=("1", "1"),
    center_cell=("H_1", "H_2"),
    phase_diff=r"-\frac{2}{3}\mathbf{a}_1+\frac{1}{3}\mathbf{a}_2",
    direction=r"\hat{\mathbf{n}}_1",
)


def print_one_unit():
    center_amp="U"
    center_index=("1", "1")
    neighbor_amp="V"
    center_kind="m"
    
    spring_const1= "C_1^{①}"
    neighbor_cell=("H_1-1", "H_2"),
    neighbor_index=("N+1", "0")    
    phase_diff=r"-\frac{2}{3}\mathbf{a}_1+\frac{1}{3}\mathbf{a}_2"
    direction1=r"\hat{\mathbf{n}}_1"
    
    latex1 = spring_term_latex(
        spring_const=spring_const1,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index,
        neighbor_cell=neighbor_cell,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index,
        center_cell=None,
        phase_diff=phase_diff,
        direction=direction1
    )
    print("라면\n\n", latex1)
    
    spring_const1= "C_2^{①}"
    neighbor_cell=("H_1", "H_2 -1"),
    neighbor_index=("2", "N")
    phase_diff=r"\frac{1}{3}\mathbf{a}_1-\frac{2}{3}\mathbf{a}_2"
    direction2=r"\hat{\mathbf{n}}_2"
    latex2 = spring_term_latex(
        spring_const=spring_const1,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index,
        neighbor_cell=neighbor_cell,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index,
        center_cell=None,
        phase_diff=phase_diff,
        direction=direction2
    )
    print("라면\n\n", latex2)
    
    spring_const1= "C_3^{①}"
    neighbor_cell=None,
    neighbor_index=("1", "1")
    phase_diff=r"\frac{1}{3}\mathbf{a}_1+\frac{1}{3}\mathbf{a}_2"
    direction3=r"\hat{\mathbf{n}}_3"
    latex3 = spring_term_latex(
        spring_const=spring_const1,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index,
        neighbor_cell=neighbor_cell,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index,
        center_cell=None,
        phase_diff=phase_diff,
        direction=direction3
    )
    print("라면\n\n", latex3)
    
    
    center_amp="V"
    neighbor_amp="U"
    center_kind="M"
    
    spring_const1= "C_1^{①}"
    neighbor_cell=None,
    neighbor_index=("2", "1")    
    phase_diff=r"\frac{2}{3}\mathbf{a}_1-\frac{1}{3}\mathbf{a}_2"
    direction1=r"\hat{\mathbf{n}}_1"
    
    latex1 = spring_term_latex(
        spring_const=spring_const1,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index,
        neighbor_cell=neighbor_cell,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index,
        center_cell=None,
        phase_diff=phase_diff,
        direction=direction1
    )
    print("라면\n\n", latex1)
    
    spring_const1= "C_2^{①}"
    neighbor_cell=None,
    neighbor_index=("1", "2")
    phase_diff=r"-\frac{1}{3}\mathbf{a}_1+\frac{2}{3}\mathbf{a}_2"
    direction2=r"\hat{\mathbf{n}}_2"
    latex2 = spring_term_latex(
        spring_const=spring_const1,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index,
        neighbor_cell=None,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index,
        center_cell=None,
        phase_diff=phase_diff,
        direction=direction2
    )
    print("라면\n\n", latex2)
    
    spring_const1= "C_3^{①}"
    neighbor_cell=None,
    neighbor_index=("1", "1")
    phase_diff=r"\frac{1}{3}\mathbf{a}_1+\frac{1}{3}\mathbf{a}_2"
    direction3=r"\hat{\mathbf{n}}_3"
    latex3 = spring_term_latex(
        spring_const=spring_const1,
        neighbor_amp=neighbor_amp,
        neighbor_index=neighbor_index,
        neighbor_cell=neighbor_cell,
        center_amp=center_amp,
        center_kind=center_kind,
        center_index=center_index,
        center_cell=None,
        phase_diff=phase_diff,
        direction=direction3
    )
    print("라면\n\n", latex3)
    
print_one_unit()
    
    