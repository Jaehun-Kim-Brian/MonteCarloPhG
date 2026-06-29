
* [수정 코드 다운로드](sandbox:/mnt/data/sim_class_pnas_reproduction_fixed.py)
* [패치 diff 다운로드](sandbox:/mnt/data/sim_class_pnas_reproduction_fix.patch)

내가 본 핵심 원인은 **Mie form factor 수식 자체의 큰 오류라기보다, PNAS/structcol 모델의 convention과 현재 새 코드의 Monte Carlo 적용 방식이 몇 군데 어긋난 것**입니다.

## 핵심 진단

### 1. Fine roughness가 “첫 step length”뿐 아니라 “첫 산란각”까지 Mie-only로 바뀌고 있었습니다

PNAS SI/structcol의 fine roughness 정의는 엄밀히 말해 다음입니다.

$\text{fine roughness} =
\text{첫 step size만 structure factor 없이 Mie-only scattering length에서 뽑는 trajectory fraction}$

즉, **첫 step의 평균 자유행로만 Mie-only**가 되고, 이후 산란 방향 샘플링은 bulk의 form factor × structure factor phase function을 따라야 합니다. SI도 step size 평균을

$l_\mathrm{sca}=\frac{1}{\rho C_\mathrm{sca}^{\mathrm{sample}}}$

로 두고, fine roughness에서는 첫 step size만 Mie theory 기반으로 계산한다고 설명합니다.

그런데 새 코드에서는 fine roughness trajectory에서

```python
current_l_scat = l_scat_surf
current_cdf = cdf_surf
```

처럼 **첫 step length와 첫 산란각 CDF가 모두 Mie-only**가 되었습니다. 이러면 표면에서 structure factor가 빠진 “순수 form-factor 산란 방향”이 실제보다 크게 반영되어, 특히 단파장 쪽 form-factor 산란이 과대해질 수 있습니다. 업로드된 새 코드의 `_get_phase_func_ginoza()`는 Mie differential cross-section에 (S(q))를 곱해 bulk phase function을 만들고 있었지만, fine roughness branch에서는 별도 Mie-only CDF를 쓰고 있었습니다. 

수정본에서는 기본값을 다음처럼 바꿨습니다.

```python
surface_phase_mode="bulk"
```

그래서 fine roughness가 걸려도 기본 동작은

```python
current_l_scat = l_scat_surf
current_cdf = cdf_norm
```

입니다. 기존 결과를 재현하고 싶을 때만

```python
surface_phase_mode="mie"
```

로 돌릴 수 있게 남겼습니다.

---

### 2. Monodisperse 비교에서도 항상 Ginoza/Schulz polydisperse structure factor를 쓰고 있었습니다

Manoharan/structcol 쪽 기본 monodisperse photonic glass structure factor는 Percus–Yevick hard-sphere `glass` kernel입니다. 원래 `structure.py`에는 monodisperse `factor_py(qd, phi)`와 polydisperse Ginoza–Yasutomi `factor_poly(...)`가 분리되어 있습니다.

그런데 재헌 코드의 `_get_phase_func_ginoza()`는 `use_polydispersity=False`인 경우에도 기본적으로 `_get_structure_factor_ginoza(q)`를 호출했습니다. 즉, 단분산 Manoharan 결과와 비교하면서도 실제로는 Schulz 분포가 들어간 measurable structure factor를 쓰고 있었습니다. 이 경우 structure peak가 monodisperse PY 대비 낮고 넓어질 수 있어서, 사용자가 말한 **“structural factor 부분 산란이 Manoharan 대비 작다”**는 증상과 잘 맞습니다.

수정본에서는 기본값을 다음처럼 바꿨습니다.

```python
structure_model="auto"
```

동작은 다음입니다.

```python
use_polydispersity=False  ->  structure_model="py"
use_polydispersity=True   ->  structure_model="ginoza"
```

명시적으로 바꿀 수도 있습니다.

```python
structure_model="py"       # monodisperse Percus–Yevick / structcol glass
structure_model="ginoza"   # Schulz polydisperse measurable S_M(q)
structure_model="none"     # form factor only
```

---

### 3. Ginoza structure factor의 low-q 쪽 수치 불안정이 있었습니다

Ginoza–Yasutomi 식은 (q \to 0) 근처에서 복소수 고차항들의 차로 계산되기 때문에 cancellation이 심합니다. 새 코드에서는 `q_safe = 1e-8`로만 처리하고 있어서, (q)가 아주 작은 forward-scattering 영역에서 (S(q))가 물리적인 compressibility limit와 맞지 않는 값으로 튈 수 있습니다.

이 영역은 ($\sin\theta$) 때문에 총 적분 기여가 아주 크지는 않지만, Mie form factor가 forward lobe를 강하게 갖는 단파장/큰 size parameter 조건에서는 총 ($C_\mathrm{sca}$), ($g$), ($l^*$)에 미세하게 영향을 줄 수 있습니다.

수정본에서는 monodisperse Percus–Yevick에 대해

$S(0)=\frac{(1-\phi)^4}{(1+2\phi)^2}$

compressibility limit를 쓰도록 했고, Ginoza branch에서도 비정상 low-q 값은 안정화했습니다.

---

### 4. Surface Mie branch의 medium convention을 분리했습니다

PNAS/structcol의 bulk scattering은 effective medium 안에서 form factor와 structure factor를 계산하지만, fine roughness의 첫 Mie-only scattering length는 원래 structcol 구현에서 particle-in-matrix branch로 계산됩니다. `montecarlo.py`의 fine roughness branch도 `n_matrix`를 써서 Mie-only scattering coefficient를 따로 계산하고, `sample_step()`에서 첫 step size에만 적용합니다. 

수정본에서는 이를 명시적으로 분리했습니다.

```python
bulk_form_medium="effective"   # PNAS bulk default
surface_form_medium="matrix"   # structcol fine roughness default
```

비교 진단용으로는 아래처럼 바꿀 수 있습니다.

```python
surface_form_medium="effective"
```

---

### 5. `n_matrix`와 `n_medium`을 분리했습니다

원 코드에서는 `n_m`이 matrix index이면서 동시에 외부 medium index처럼 쓰이는 부분이 있었습니다. 공기 matrix sample에서는 차이가 없지만, **PS-in-water film을 air에서 측정하는 경우**에는 matrix는 water이고 external medium은 air입니다. PNAS 모델도 film/detector geometry에서 sample material parameter와 external medium을 분리합니다. 

수정본에서는 `n_medium`을 따로 넣을 수 있게 했습니다. 예를 들어 물 matrix sample을 air에서 측정하는 경우:

```python
sim = PhotonicGlassMCSimulator(
    film_thickness=80,
    phi=0.40,
    fine_roughness=0.28,
    coarse_roughness=0.2,
    r_i=0.101,
    n_m=1.33,       # matrix: water
    n_medium=1.0,   # outside: air
)
```

---

## 수정본 기본 사용법

PNAS/Manoharan reproduction 기준으로는 이렇게 시작하면 됩니다.

```python
import numpy as np
from sim_class_pnas_reproduction_fixed import PhotonicGlassMCSimulator

theta_array = np.linspace(0.01, np.pi, 800)
wvls = np.linspace(0.4, 0.8, 81)  # micrometer unit

sim = PhotonicGlassMCSimulator(
    film_thickness=77.0,
    phi=0.56,
    fine_roughness=0.5,
    coarse_roughness=0.9,
    r_i=0.138,
    n_m=1.0,
    n_medium=1.0,
    k_p=2e-5,
    pdi=0.03,
    structure_model="auto",
    bulk_form_medium="effective",
    surface_form_medium="matrix",
    surface_phase_mode="bulk",
    random_seed=1,
)

R, T, A, diagnostics = sim.run_simulation(
    wvls=wvls,
    theta_array=theta_array,
    N_photons=20000,
    backend="internal",
    use_polydispersity=False,
    return_diagnostics=True,
)
```

비교용으로 기존 코드와 가까운 동작을 보고 싶으면:

```python
sim_old_like = PhotonicGlassMCSimulator(
    ...,
    structure_model="ginoza",
    surface_form_medium="effective",
    surface_phase_mode="mie",
)
```

---

## 빠른 kernel 진단 함수도 추가했습니다

Monte Carlo noise 없이 form factor와 structure factor 쪽만 먼저 비교하려고 다음 helper를 넣었습니다.

```python
rows = sim.kernel_diagnostics(
    wavelengths=[0.4, 0.5, 0.8],
    theta_array=theta_array,
    backend="internal",
)

for row in rows:
    print(row)
```

출력에는 예를 들어 다음 항목들이 들어갑니다.

```python
csca_mie_effective
csca_mie_matrix
csca_sample_py
csca_sample_ginoza
weighted_S_py
weighted_S_ginoza
l_scat_py
l_star_py
Smax_py
Smax_ginoza
```

이걸로 “Mie form factor가 큰 문제인지”, “PY와 Ginoza 중 어느 structure factor가 peak를 낮추는지”, “(l^*)가 PNAS SI Fig. S2와 같은 order인지”를 Monte Carlo 전에 먼저 확인할 수 있습니다. PNAS 논문도 scattering strength와 (l^*) regime을 중요하게 보고, 모델은 (r/\lambda)가 대략 0.5 이하이고 transport length가 wavelength보다 충분히 큰 weak multiple-scattering regime에서 가장 잘 맞는다고 설명합니다. 

## 내가 수정하면서 확인한 것

수정본은 `py_compile`로 문법 검사를 통과했고, 소규모 sanity run도 정상 동작했습니다. 다만 아직 전체 파장 400–800 nm, 20,000 photons/wavelength 수준의 긴 재현 계산은 수행하지 않았습니다. 따라서 이번 파일은 **physics-kernel과 Monte Carlo branching correction 1차 수정본**으로 보시면 됩니다.

남은 주의점은 하나 있습니다. 현재 `detect_angle` 파라미터는 코드에 저장되어 있지만, 원 structcol `detector.py`처럼 exit angle filter로 완전히 적용되지는 않았습니다. PNAS의 대부분 integrating sphere 계산처럼 90° 전체 반사 hemisphere를 보는 경우에는 큰 문제가 아니지만, SI에서 언급된 75°/80° detection-angle sample까지 정확히 재현하려면 detector-angle correction을 다음 패치에서 넣는 것이 좋습니다.

