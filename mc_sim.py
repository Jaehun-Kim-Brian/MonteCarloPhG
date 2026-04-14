import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as ps
from tqdm import tqdm
from scipy.stats import gamma
from scipy.special import spherical_jn, spherical_yn


def get_q(n_eff, theta_array, wavelength):
    q_array = 4 * np.pi * n_eff * np.sin(theta_array / 2.0) / wavelength
    return q_array

def get_np_real(wavelength):
    # wavelength in um scale
    n_p_real = np.sqrt(1 + (1.4435 * wavelength**2) / (wavelength**2 - 0.020216))
    return n_p_real

# --- Bruggeman 유효 굴절률 계산 함수 ---
def get_effective_index_complex(n_p_complex, n_med_complex, phi):
    eps_p = n_p_complex**2
    eps_med = n_med_complex**2
    
    H_b = (3*phi-1) * eps_p + (3*(1-phi)-1)*eps_med
    eps_eff = H_b + np.sqrt(H_b**2 + 8*eps_p*eps_med)
    eps_eff /= 4.0
    
    n_eff = np.sqrt(eps_eff)
    if n_eff.imag < 0: # 올바른 감쇠 방향 보정
        n_eff = -n_eff
    return n_eff

def riccati_bessel(n, z):
    # 복소수 z에 대한 Riccati-Bessel 함수와 도홤수 계산
    jn = spherical_jn(n, z)
    yn = spherical_yn(n, z)
    jn_1 = spherical_jn(n-1, z)
    yn_1 = spherical_yn(n-1, z)
    
    hn = jn + 1j * yn
    hn_1 = jn_1 + 1j * yn_1
    
    psi = z * jn
    psi_prime = z * jn_1 - n * jn
    
    xi = z * hn
    xi_prime = z * hn_1 - n * hn
    
    return psi, psi_prime, xi, xi_prime

def get_mie_absorbing_old(wavelength, d_i, n_p_complex, n_eff_complex, theta_array):
    # 흡수성 매질(Complex n_eff) 내에 존재하는 입자의 표면 적분(Surface Integration) 미 산란을 계산
    a = d_i / 2.0
    k0 = 2.0 * np.pi / wavelength
    
    m_med = n_eff_complex
    m_p = n_p_complex
    m = m_p / m_med
    
    x = k0 * a * m_med
    y = k0 * a * m_p
    
    # 수렴을 위한 최대 차수 (Wiscombe's rule)
    x_abs = np.abs(x)
    N_stop = int(x_abs + 4.0 * x_abs**(1/3) + 2.0) + 5
    
    a_n = np.zeros(N_stop, dtype=complex)
    b_n = np.zeros(N_stop, dtype=complex)
    
    # 1. 미 계수 (Mie Coefficients) 계산
    for n in range(1, N_stop + 1):
        psi_x, psi_x_prime, xi_x, xi_x_prime = riccati_bessel(n, x)
        psi_y, psi_y_prime, _, _ = riccati_bessel(n, y)
        
        # B&H 수식을 복소수 영역으로 확장 적용
        num_a = m * psi_y * psi_x_prime - psi_x * psi_y_prime
        den_a = m * psi_y * xi_x_prime - xi_x * psi_y_prime
        a_n[n-1] = num_a / den_a
        
        num_b = psi_y * psi_x_prime - m * psi_x * psi_y_prime
        den_b = psi_y * xi_x_prime - m * xi_x * psi_y_prime
        b_n[n-1] = num_b / den_b

    # 2. 중심 입사광 진폭 보정 (Incident Amplitude Correction)
    k_med_im = np.imag(k0 * m_med)
    correction = np.exp(-2.0 * k_med_im * a)

    # 3. 각도별 산란 진폭 (S1, S2) 계산
    mu = np.cos(theta_array)
    S1 = np.zeros_like(theta_array, dtype=complex)
    S2 = np.zeros_like(theta_array, dtype=complex)
    
    pi_n = np.ones_like(mu)
    pi_n_minus_1 = np.zeros_like(mu)
    
    for n in range(1, N_stop + 1):
        if n == 1:
            pi_n_curr = np.ones_like(mu)
            tau_n = mu
        else:
            pi_n_curr = ((2*n - 1) * mu * pi_n - n * pi_n_minus_1) / (n - 1)
            tau_n = n * mu * pi_n_curr - (n + 1) * pi_n
            pi_n_minus_1 = pi_n
            pi_n = pi_n_curr
            
        coeff = (2*n + 1) / (n * (n + 1))
        S1 += coeff * (a_n[n-1] * pi_n_curr + b_n[n-1] * tau_n)
        S2 += coeff * (a_n[n-1] * tau_n + b_n[n-1] * pi_n_curr)
        
    # 4. 미분 산란 단면적 (표면 적분 기반)
    F_theta = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2) / np.abs(k0 * m_med)**2
    F_theta *= correction
    
    return F_theta

def get_mie_absorbing(wavelength, d_i, n_p_complex, n_eff_complex, theta_array):
    # 흡수성 매질(Complex n_eff) 내에 존재하는 입자의 표면 적분(Surface Integration) 미 산란을 계산
    a = d_i / 2.0
    k0 = 2.0 * np.pi / wavelength
    
    m_med = n_eff_complex
    m_p = n_p_complex
    m = m_p / m_med
    
    x = k0 * a * m_med
    y = k0 * a * m_p
    
    # 수렴을 위한 최대 차수 (Wiscombe's rule)
    x_abs = np.abs(x)
    N_stop = int(x_abs + 4.0 * x_abs**(1/3) + 2.0) + 5
    
    a_n = np.zeros(N_stop, dtype=complex)
    b_n = np.zeros(N_stop, dtype=complex)
    
    # 1. 미 계수 (Mie Coefficients) 계산
    for n in range(1, N_stop + 1):
        psi_x, psi_x_prime, xi_x, xi_x_prime = riccati_bessel(n, x)
        psi_y, psi_y_prime, _, _ = riccati_bessel(n, y)
        
        # B&H 수식을 복소수 영역으로 확장 적용
        num_a = m * psi_y * psi_x_prime - psi_x * psi_y_prime
        den_a = m * psi_y * xi_x_prime - xi_x * psi_y_prime
        a_n[n-1] = num_a / den_a
        
        num_b = psi_y * psi_x_prime - m * psi_x * psi_y_prime
        den_b = psi_y * xi_x_prime - m * xi_x * psi_y_prime
        b_n[n-1] = num_b / den_b

    # 2. 중심 입사광 진폭 보정 (Incident Amplitude Correction)
    k_med_im = np.imag(k0 * m_med)
    correction = np.exp(-2.0 * k_med_im * a)

    # 3. 각도별 산란 진폭 (S1, S2) 계산
    mu = np.cos(theta_array)
    S1 = np.zeros_like(theta_array, dtype=complex)
    S2 = np.zeros_like(theta_array, dtype=complex)
    
    pi_n = np.ones_like(mu)
    pi_n_minus_1 = np.zeros_like(mu)
    
    for n in range(1, N_stop + 1):
        if n == 1:
            pi_n_curr = np.ones_like(mu)
            tau_n = mu
        else:
            pi_n_curr = ((2*n - 1) * mu * pi_n - n * pi_n_minus_1) / (n - 1)
            tau_n = n * mu * pi_n_curr - (n + 1) * pi_n
            pi_n_minus_1 = pi_n
            pi_n = pi_n_curr
            
        coeff = (2*n + 1) / (n * (n + 1))
        S1 += coeff * (a_n[n-1] * pi_n_curr + b_n[n-1] * tau_n)
        S2 += coeff * (a_n[n-1] * tau_n + b_n[n-1] * pi_n_curr)
        
    # 4. 미분 산란 단면적 (표면 적분 기반)
    F_theta = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2) / np.abs(k0 * m_med)**2
    F_theta *= correction
    
    return F_theta

def get_mie_absorbing_old(wavelength, d_i, n_p_complex, n_eff_complex, theta_array):
    a = d_i / 2.0
    k0 = 2.0 * np.pi / wavelength

    m_med = n_eff_complex
    m_p = n_p_complex
    m = m_p / m_med

    x = k0 * a * m_med
    y = k0 * a * m_p

    x_abs = np.abs(x)
    N_stop = int(x_abs + 4.0 * x_abs**(1/3) + 2.0) + 5

    a_n = np.zeros(N_stop, dtype=complex)
    b_n = np.zeros(N_stop, dtype=complex)

    for n in range(1, N_stop + 1):
        psi_x, psi_x_prime, xi_x, xi_x_prime = riccati_bessel(n, x)
        psi_y, psi_y_prime, _, _ = riccati_bessel(n, y)

        num_a = m * psi_y * psi_x_prime - psi_x * psi_y_prime
        den_a = m * psi_y * xi_x_prime - xi_x * psi_y_prime
        a_n[n-1] = num_a / den_a

        num_b = psi_y * psi_x_prime - m * psi_x * psi_y_prime
        den_b = psi_y * xi_x_prime - m * xi_x * psi_y_prime
        b_n[n-1] = num_b / den_b

    mu = np.cos(theta_array)
    S1 = np.zeros_like(theta_array, dtype=complex)
    S2 = np.zeros_like(theta_array, dtype=complex)

    pi_nm1 = np.zeros_like(mu)   # pi_{n-1}
    pi_n = np.ones_like(mu)      # pi_1 for recurrence base

    for n in range(1, N_stop + 1):
        if n == 1:
            pi_curr = np.ones_like(mu)
            tau_n = mu
        else:
            pi_curr = ((2*n - 1) * mu * pi_n - n * pi_nm1) / (n - 1)
            tau_n = n * mu * pi_curr - (n + 1) * pi_n
            pi_nm1 = pi_n
            pi_n = pi_curr

        coeff = (2*n + 1) / (n * (n + 1))
        S1 += coeff * (a_n[n-1] * pi_curr + b_n[n-1] * tau_n)
        S2 += coeff * (a_n[n-1] * tau_n + b_n[n-1] * pi_curr)

    # absorbing medium surface integration
    k_med = k0 * m_med
    k_real = np.real(k_med)
    k_imag = np.imag(k_med)

    denom = (k_real * a)**2 + (k_imag * a)**2
    denom = max(denom, 1e-30)

    # far-field amplitude at r=a with absorbing-medium decay
    radial_factor = np.exp(-2.0 * k_imag * a) / denom

    # SI Eq. (16): incident amplitude correction
    twoak = 2.0 * a * k_imag
    if abs(twoak) < 1e-10:
        I0_corr = 1.0
    else:
        I0_corr = 2.0 * (
            np.exp(twoak) / twoak
            + (1.0 - np.exp(twoak)) / (twoak**2)
        )

    F_theta = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2) * radial_factor * I0_corr
    return np.real_if_close(F_theta)

def get_schulz_distribution(r_mean, pdi, num_bins=21):
    """
    Schulz 분포(Gamma 분포)를 사용하여 다분산(Polydisperse) 입자 크기 배열과 확률 가중치를 반환합니다.
    pdi: Polydispersity Index (표준편차 / 평균 반경). 예: 0.05 (크기 오차 5%)
    """
    if pdi == 0.0:
        return np.array([r_mean]), np.array([1.0])
        
    shape = 1.0 / (pdi**2)
    scale = r_mean / shape
    
    sigma = r_mean * pdi
    # 평균을 중심으로 +- 3 표준편차 범위를 쪼갭니다.
    r_min = max(0.001, r_mean - 3*sigma)
    r_max = r_mean + 3*sigma
    
    r_array = np.linspace(r_min, r_max, num_bins)
    
    # 각 크기별 확률(Weight) 계산 및 총합을 1로 정규화
    weights = gamma.pdf(r_array, a=shape, scale=scale)
    weights = weights / np.sum(weights)
    
    return r_array, weights

def get_structure_factor_ginoza(q, phi, r_mean, pdi):
    """
    1999년 Ginoza 논문의 해석적 수식을 사용하여 
    다분산(Schulz 분포) 시스템의 '측정 가능한 유효 구조 인자 S_M(q)'를 한 번에 계산합니다.
    """
    d = r_mean * 2.0
    
    # 헬퍼 함수 1: fm
    def fm(x, t, tm, m):
        return tm * (1.0 + x/(t+1.0))**(-(t+1.0+m))
        
    # 헬퍼 함수 2: tm
    def get_tm(m, t):
        if m == 0: return 1.0
        prod = 1.0
        for i in range(1, m+1):
            prod *= (t + i)
        return prod / ((t + 1.0)**m)
        
    pdi_val = max(pdi, 1e-5) # 0으로 나누기 방지
    Dsigma = pdi_val**2
    Delta = 1.0 - phi
    t = 1.0/Dsigma - 1.0
    
    # Ginoza 계수 계산
    t0 = get_tm(0, t)
    t1 = get_tm(1, t)
    t2 = Dsigma + 1.0
    t3 = (Dsigma + 1.0) * (2.0*Dsigma + 1.0)
    
    rho = 6.0 * phi / (t3 * np.pi * d**3)
    sigma0 = (6.0 * phi / (np.pi * rho))**(1.0/3.0)
    
    q_safe = np.maximum(q, 1e-8)
    
    s = 1j * q_safe
    x = s * d
    F0 = rho
    zeta2 = rho * sigma0**2
    
    f0, f1, f2 = fm(x, t, t0, 0), fm(x, t, t1, 1), fm(x, t, t2, 2)
    f0_inv, f1_inv, f2_inv = fm(-x, t, t0, 0), fm(-x, t, t1, 1), fm(-x, t, t2, 2)
    
    # 방정식 파라미터 (Eqs 29a-29d)
    fa = 1.0/x**3 * (1.0 - x/2.0 - f0 - x/2.0 * f1)
    fb = 1.0/x**3 * (1.0 - x/2.0 * t2 - f1 - x/2.0 * f2)
    fc = 1.0/x**2 * (1.0 - x - f0)
    fd = 1.0/x**2 * (1.0 - x*t2 - f1)
    
    Ialpha1 = 24.0/s**3 * (F0 * (-0.5*(1.0-f0) + x/4.0 * (1.0 + f1)))
    Ialpha2 = 24.0/s**3 * (F0 * (-d/2.0 * (1.0-f1) + s*d**2/4.0 * (t2 + f2)))
    
    Iw1 = 2.0*np.pi*rho/(Delta*s**3) * (Ialpha1 + s/2.0*Ialpha2)
    Iw2 = (np.pi*rho/(Delta*s**2) * (1.0 + np.pi*zeta2/(Delta*s))*Ialpha1 + 
           np.pi**2*zeta2*rho/(2.0*Delta**2*s**2) * Ialpha2)
           
    F11 = 2.0*np.pi*rho*d**3/Delta * fa
    F12 = 1.0/d * ((np.pi/Delta)**2 * rho * zeta2 * d**4 * fa + np.pi*rho*d**3/Delta * fc)
    F21 = d * 2.0*np.pi*rho*d**3/Delta * fb
    F22 = ((np.pi/Delta)**2 * rho*zeta2*d**4*fb + np.pi*rho*d**3/Delta * fd)
    
    FF11, FF12, FF21, FF22 = 1.0 - F11, -F12, -F21, 1.0 - F22
    
    # 행렬식(Determinant) 및 역행렬 성분
    det = FF11 * FF22 - FF12 * FF21
    G11, G12 = FF22 / det, -FF12 / det
    G21, G22 = -FF21 / det, FF11 / det
    
    I0 = -9.0/2.0 * (2.0/s)**6 * (F0**2 * (1.0 - 0.5*(f0_inv + f0) + x/2.0*(f1_inv - f1) - (s**2*d**2)/8.0*(f2_inv + f2 + 2.0*t2)))
    
    term1 = Iw1 * G11 * Ialpha1 / I0
    term2 = Iw1 * G12 * Ialpha2 / I0
    term3 = Iw2 * G21 * Ialpha1 / I0
    term4 = Iw2 * G22 * Ialpha2 / I0
    
    h2 = (term1 + term2 + term3 + term4).real
    SM = 1.0 - 2.0*h2
    
    return np.maximum(SM, 0.0)

def get_phase_func_ginoza(wavelength, theta, r_mean, pdi, phi, k_p, n_p_real, n_m):
    """
    Ginoza 다성분계 구조 인자(S_M_q)와 Schulz 분포가 적용된 폼 팩터(F_theta)를 결합하여
    최종 위상 함수(Phase Function)를 반환합니다.
    """
        # 1. 유효 굴절률 및 파수 벡터(q) 계산
    n_p_complex = n_p_real + 1j * k_p
    n_med_complex = n_m + 0j
    n_eff_complex = get_effective_index_complex(n_p_complex, n_med_complex, phi)
    
    # 2. Ginoza 해석적 수식을 통한 유효 구조 인자 S_M(q) 추출
    # (앞서 정의한 get_structure_factor_ginoza 함수 호출)
    q_array = get_q(n_eff_complex.real, theta, wavelength)  
    S_M_q = get_structure_factor_ginoza(q_array, phi=phi, r_mean=r_mean, pdi=pdi)
    
    # 3. Schulz 분포를 따르는 크기 배열(r_array)과 확률 가중치(weights) 생성
    r_array, weights = get_schulz_distribution(r_mean, pdi, num_bins=21)
    F_theta_norm_avg = np.zeros_like(theta, dtype=float)
    F_theta_surf_avg = np.zeros_like(theta, dtype=float)
    
    # 4. 크기별 Form Factor(F_theta)를 계산하고 가중 평균(Ensemble Average) 누적
    for r, w in zip(r_array, weights):
        d_i = r * 2.0
        F_theta_norm = get_mie_absorbing(wavelength, d_i, n_p_complex, n_eff_complex, theta)
        F_theta_surf = get_mie_absorbing(wavelength, d_i, n_p_complex, n_med_complex, theta)
        F_theta_norm_avg += F_theta_norm * w
        F_theta_surf_avg += F_theta_surf * w
        
    # 5. 최종 Phase Function 조립
    # 필름 내부(Normal): S_M_q의 얽힘(Coupling) 간섭 효과와 다분산 폼 팩터의 완벽한 결합
    Phase_Function_Normal = S_M_q * F_theta_norm_avg
    # 필름 표면(Surface): 구조 인자 간섭(S_M_q) 없이 순수한 다분산 미 산란(Mie Scattering) 적용
    Phase_Function_Surface = F_theta_surf_avg
    
    return Phase_Function_Normal, Phase_Function_Surface

def get_norm_phase(Phase_Function, theta):
    # 확률 밀도 함수로 쓰기 위해 면적을 1로 정규화 (optional)
    normalization_factor = np.trapezoid(Phase_Function * np.sin(theta), theta)
    return Phase_Function / normalization_factor 

def cdf_phase(phase, theta_array):
    # 확률 밀도 함수인 phase function을 누적 확률 함수로 변환
    # sin(theta) 가중치를 곱해서 누적합 해야 3D 공간의 확률이 됨
    norm_phase = get_norm_phase(phase, theta_array)
    pdf = norm_phase * np.sin(theta_array)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1] # 0 ~ 1 사이로 정규화
    return cdf
    
def get_l_scat(Phase_Function, theta, phi, r_i):
    scat_crossec = 2*np.pi * np.trapezoid(Phase_Function * np.sin(theta), theta)
    number_density = phi / (4/3*np.pi * r_i**3)
    
    l_scat = 1 / (scat_crossec * number_density)
    return l_scat

def check_transport_length(Phase_Function, theta, phi, r_i):
    # 1. 총 산란 단면적 및 Scattering Length 계산
    scat_crossec = 2 * np.pi * np.trapezoid(Phase_Function * np.sin(theta), theta)
    number_density = phi / ((4/3) * np.pi * r_i**3)
    l_scat = 1.0 / (scat_crossec * number_density)
    
    # 2. Anisotropy factor (g) 계산
    # g = ∫ P(θ)*cos(θ)*sin(θ)dθ / ∫ P(θ)*sin(θ)dθ
    numerator = np.trapezoid(Phase_Function * np.cos(theta) * np.sin(theta), theta)
    denominator = np.trapezoid(Phase_Function * np.sin(theta), theta)
    g = numerator / denominator
    
    # 3. Transport Length (l*) 도출
    l_star = l_scat / (1.0 - g)
    return l_star, l_scat, g

def get_mu_a(wavelength, k_eff): # 흡수 계수 계산
    return 4*np.pi*k_eff/wavelength
    
# 굴절률이 큰 필름과 공기의 인터페이스에서의 프레넬 반사와 전반사를 고려해준다.
# ref : Fresnel Equation    
def calculate_fresnel(n_i, n_t, cos_theta_i): 
    # n_i : 현재 매질의 굴절률(필름 내부)
    # n_t : 넘어갈 매질 굴절률(공기)
    # cos_theta_i : 입사각의 cos 값
    
    if cos_theta_i == 0.0:
        cos_theta_i = 1e-6
        
    sin_theta_i = np.sqrt(max(0.0, 1.0 - cos_theta_i**2))
    sin_theta_t = (n_i / n_t) * sin_theta_i # 스넬의 법칙
    
    # 1. 전반사 조건 --> 반사율 100%
    if sin_theta_t >= 1.0: 
        return 1.0 
    
    # 2. Fresnel equation
    cos_theta_t = np.sqrt(max(0.0, 1.0 - sin_theta_t**2))
    # wave impedence와 refractive inex가 역수 관계이므로 성립
    r_s = (n_i * cos_theta_i - n_t * cos_theta_t) / (n_i * cos_theta_i + n_t * cos_theta_t)
    r_p = (n_i * cos_theta_t - n_t * cos_theta_i) / (n_i * cos_theta_t + n_t * cos_theta_i)
    
    R_f = 0.5 * (r_s**2 + r_p**2)
    return R_f

# always returns z positive norm vector
def get_norm_vec(coarse_roughness): 
    slope_x = np.random.normal(0.0, coarse_roughness)
    slope_y = np.random.normal(0.0, coarse_roughness)
    
    x_vector = [1.0, 0.0, slope_x]
    y_vector = [0.0, 1.0, slope_y]
    
    norm_vec = np.cross(x_vector, y_vector)
    length = np.sqrt(norm_vec[0]**2 + norm_vec[1]**2 + norm_vec[2]**2)
    return norm_vec/length

# for the incident light, 
def interface_enter(u_x, u_y, u_z, coarse_roughness, n_i, n_t):      
    norm_vec = get_norm_vec(coarse_roughness)
    cos_i = u_x * norm_vec[0] + u_y * norm_vec[1] + u_z * norm_vec[2]
    if cos_i < 0: # as, norm vector always faces inside of the film, inner poduct of inc_vec and norm_vec should be positive
        norm_vec = -1 * norm_vec
        cos_i = abs(cos_i)
        
        
    R_f = calculate_fresnel(n_i, n_t, cos_i) # return the fraction of reflected light. If TIR, return 1.0
    # In case of ligth entering into film, we have to return only transmitted direction vector and R_f
    
    if R_f >= 1.0: # as light reflected before it tranmitted to the film, we don't have to return actual direction vector value
        return 0.0, 0.0, 0.0, True
    
    if np.random.rand() <= R_f:
        return 0.0, 0.0, 0.0, True
    
    else: 
        eta = n_i / n_t
        
        # Snell's Law
        # It doesn't occur error, because function 'calculate_fresnel' already confirmed TIR condition
        theta_i = np.arccos(min(1.0, cos_i))
        sin_t = np.sin(theta_i) * eta
        
        vec_t_p_x = eta * (u_x - norm_vec[0] * cos_i)
        vec_t_p_y = eta * (u_y - norm_vec[1] * cos_i)
        vec_t_p_z = eta * (u_z - norm_vec[2] * cos_i)
        vec_t_parallel = [vec_t_p_x, vec_t_p_y, vec_t_p_z]
        vec_t_vertical = np.sqrt(1 - sin_t ** 2) * norm_vec
        vec_trans = vec_t_parallel + vec_t_vertical
        
        return vec_trans[0], vec_trans[1], abs(vec_trans[2]), False

# n_i is index of film and n_t is index of medium(Air)
def interface_infilm(u_x, u_y, u_z, coarse_roughness, is_top, n_i, n_t):
    sign = -1 if is_top else 1
    # norm vector is positive when bottom,  negative when top
    norm_vec = get_norm_vec(coarse_roughness) * sign # norm vector always heads outside of the film
    cos_i = u_x * norm_vec[0] + u_y * norm_vec[1] + u_z * norm_vec[2]
    if cos_i < 0: # as, norm vector always faces inside of the film, inner poduct of inc_vec and norm_vec should be positive
        cos_i = abs(cos_i)
        norm_vec = norm_vec * -1
    
    # as n_i > n_t for this case, we should consider TIR actually
    R_f = calculate_fresnel(n_i, n_t, cos_i) # return the fraction of reflected light. If TIR, return 1.0
    # And the direction vector we should return is the reflected direction vector
    
    if R_f >= 1.0 or np.random.rand() < R_f:  
        ref_x = u_x - 2 * norm_vec[0] * cos_i
        ref_y = u_y - 2 * norm_vec[1] * cos_i
        ref_z = u_z - 2 * norm_vec[2] * cos_i
        ref_z = abs(ref_z) * sign * -1
        return ref_x, ref_y, ref_z, True
    
    else:
        return 0.0, 0.0, 0.0, False


# --- Monte Carlo 포톤 패킷날리기 함수 ---
def run_mc(N_photons, film_thickness, mu_a,
                    l_scat_norm, cdf_norm, l_scat_surf, cdf_surf,
                    theta_array, n_eff, n_med,
                    fine_roughness, coarse_roughness):
    # N_photons : 쏠 광자 수
    # film_thickness : 필름 두께
    # wavelength : photon의 파장
    # k_p : polystyrene의 imaginary refractive index
    # l_scat : 평균 산란 거리(Mean scattering path, um)
    # cdf : # phase function을 cdf로 변환하여 로드
    # theta_array : 0 ~ pi 각도 배열
    # fine_roughness : 표면에 입자가 튀어나와 있을 확률
    
    #n_glass = 1.52 # 유리 굴절률
    
    # 결과 저장
    reflected = 0
    transmitted = 0
    top_touch = 0 
    bot_touch = 0
    # 2. Photon 추적 루프
    for _ in range(N_photons):
        # 광자 초기화 : +z 방향으로 수직 입사
        x, y, z = 0.0, 0.0, 0.0
        inc_angle = np.radians(8.0)
        u_x = np.sin(inc_angle)
        u_y = 0.0
        u_z = np.cos(inc_angle)
        # 입사각에 따른 방향 벡터 (u_x, u_y, u_z)

        # 밖(n_m)에서 안(n_eff)으로 들어올 때 표면이 기울어져 있으면 빛이 꺾임
        u_x, u_y, u_z, is_reflected = interface_enter(u_x, u_y, u_z, coarse_roughness, n_med, n_eff)
        
        if is_reflected:
            reflected += 1.0
            continue
        
        is_first_step = True
        total_moving = 0
        
        while True:
            if is_first_step:
                is_first_step = False
                if np.random.rand() < fine_roughness: current_l_scat, current_cdf = l_scat_surf, cdf_surf
                else: current_l_scat, current_cdf = l_scat_norm, cdf_norm
            else: current_l_scat, current_cdf = l_scat_norm, cdf_norm

            # 1) step size 뽑기 (exponential distribution)
            # step_size = -ln(random) * l_scat, random은 0과 1 사이의 수
            step_size = -np.log(np.random.rand()) * current_l_scat
            
            # 2) photon 이동
            next_x = x + step_size * u_x
            next_y = y + step_size * u_y
            next_z = z + step_size * u_z
            
            # 3) 경계면 도달 시 프레넬/전반사 체크 로직
            hit_boundary = False
            is_top = False
            if next_z <= 0:
                hit_boundary = True
                is_top = True
                top_touch += 1
            elif next_z >= film_thickness:
                hit_boundary = True
                is_top = False
                bot_touch += 1
                       
            if hit_boundary:
                n_out = n_med
                # 내부(n_eff)에서 외부(n_med)로 나가려는 시도
                u_x_new, u_y_new, u_z_new, is_reflected = interface_infilm(u_x, u_y, u_z, coarse_roughness, is_top, n_eff, n_out)
                dz_total = abs(u_z * step_size)
                
                if dz_total < 1e-15:
                    path_to_boundary = 0.0
                    frac_before = 0.0
                else:
                    if is_top: 
                        dz_to_boundary = abs(z)
                    else: 
                        dz_to_boundary = abs(film_thickness - z)
                    frac_before = np.clip(dz_to_boundary/dz_total, 0.0, 1.0)
                    path_to_boundary = step_size * frac_before
                x_hit = x + step_size * frac_before * u_x
                y_hit = y + step_size * frac_before * u_y
                    
                remaining_path = step_size * (1.0 - frac_before)
                
                if is_reflected:
                    z_hit = 0.0 if is_top else film_thickness
                    u_x, u_y, u_z = u_x_new, u_y_new, u_z_new
                    x = x_hit + remaining_path * u_x
                    y = y_hit + remaining_path * u_y
                    z = z_hit + remaining_path * u_z
                
                else:
                    W_final = np.exp(-mu_a * (total_moving + path_to_boundary))
                    if is_top: reflected += W_final
                    else: transmitted += W_final
                    break
                    
                    
                
                # 에러 디버깅용
                if hit_boundary == True:
                    if is_top == True and u_z_new <0:
                        print('error')
                    if is_top == False and u_z_new > 0:
                        print('error') 
                
            else: 
                x, y, z = next_x, next_y, next_z

            total_moving += step_size
                    
                
            # 4) 매질 안에서 새로운 산란 방향(theta, phi=azimuth) 뽑기
            # 랜덤 난수로 CDF를 기반한 theta 결정
            cdf_value = np.random.rand()
            scat_theta = np.interp(cdf_value, current_cdf, theta_array) # cdf_value 값일때 광자가 산란하는 각도를 역연산
            azimuth = 2.0 * np.pi * np.random.rand() # 방위각 랜덤으로 뽑아주기
            
            # 5) 3D 공간에서 새로운 방향 벡터
            sin_t, cos_t = np.sin(scat_theta), np.cos(scat_theta)
            sin_p, cos_p = np.sin(azimuth), np.cos(azimuth)
            
            # 6) 좌표 회전
            if abs(u_z) > 0.99999:
                sign = 1 if u_z > 0 else -1
                u_x_new = sin_t * cos_p * sign
                u_y_new = sin_t * sin_p * sign
                u_z_new = cos_t * sign
                
            else:
                temp = np.sqrt(1.0 - u_z**2)
                u_x_new = sin_t * (u_x * u_z * cos_p - u_y * sin_p) / temp + u_x * cos_t
                u_y_new = sin_t * (u_y * u_z * cos_p + u_x * sin_p) / temp + u_y * cos_t
                u_z_new = -sin_t * cos_p * temp + u_z * cos_t
    
            u_x, u_y, u_z = u_x_new, u_y_new, u_z_new
            
        
    R = reflected / N_photons
    T = transmitted / N_photons
    print(top_touch, bot_touch)
    return R, T

def run_all_mc(filename, wvls, r_i,
               N_photons, film_thickness, k_p,
                theta_array, n_med,
                phi, fine_roughness, coarse_roughness):
    
    reflectance = []
    transmittance = []
    absorb = []
    
    for wavelength in  tqdm(wvls):       
        # Sellmeier equation을 이용한 n_p 값 계산
        n_p_real = get_np_real(wavelength)         
        n_p_complex = n_p_real + 1j * k_p
        n_med_complex = n_med + 0j
        
        # 2. Bruggeman을 통한 엄밀한 유효 복소 굴절률 도출
        n_eff_complex = get_effective_index_complex(n_p_complex, n_med_complex, phi)
        n_eff_real = n_eff_complex.real
        k_eff = abs(n_eff_complex.imag) # 허수부 절대값 추출
        
        # 3. 엄밀한 흡수 계수(mu_a) 계산
        mu_a = get_mu_a(wavelength, k_eff)
        
        # Ginoza 이용한 phase function
        Phase_norm, Phase_surf = get_phase_func_ginoza(wavelength, theta_array, r_mean=r_i, pdi=0.03, phi=phi, k_p=2e-5, n_p_real=n_p_real, n_m=n_med)
        cdf_norm, cdf_surf = cdf_phase(Phase_norm, theta_array), cdf_phase(Phase_surf, theta_array)
        l_scat_norm, l_scat_surf = get_l_scat(Phase_norm, theta_array, phi, r_i), get_l_scat(Phase_surf, theta_array, phi, r_i)

        R, T =run_mc(N_photons=N_photons, film_thickness=film_thickness,
                        mu_a=mu_a, l_scat_norm=l_scat_norm, cdf_norm=cdf_norm, l_scat_surf=l_scat_surf, cdf_surf=cdf_surf,
                        theta_array=theta_array, n_eff=n_eff_real, n_med=n_med,
                        fine_roughness=fine_roughness, coarse_roughness=coarse_roughness)
        reflectance.append(R)
        transmittance.append(T)
        absorb.append(1-(R+T))
        np.savez(f"{filename}.npz", 
                wavelengths=wvls,
                reflectance=reflectance, 
                transmittance=transmittance,
                absorbance = absorb,
                )
    
    print(f"{filename} is successfully saved!")
    
    return reflectance, transmittance

if __name__=="__main__":
    #film = [6.0, 0.58, 0.5, 0.9]
    #film = [13.0, 0.58, 0.5, 0.9]
    film = [77.0, 0.56, 0.5, 0.9]
    #film = [3930.0, 0.5, 1, 0.9]
    
    wvls = np.linspace(0.400, 0.800, 100) # 400nm - 800nm
    #wvls = [0.400]
    theta = np.linspace(0.01, np.pi-0.01, 500)  # 0 ~ 180도
    r_i = 0.138
    n_m = 1.0              # 공기 매질 굴절률
    
    film_thickness = film[0]
    phi = film[1]
    fine_roughness = film[2]
    coarse_roughness = film[3]
    filename = f"phg_mc/file/mc_{int(film_thickness)}um_ver12"
    print(filename)
    run_all_mc(filename=filename, wvls=wvls, r_i = r_i,
               N_photons=20000, film_thickness=film_thickness, k_p=2e-5,   
                theta_array=theta, n_med=n_m, 
                phi=phi, fine_roughness=fine_roughness, coarse_roughness=coarse_roughness)
    '''
    wavelength = 0.500
    k_p = 2e-5
    n_p_real = get_np_real(wavelength)         
    n_p_complex = n_p_real + 1j * k_p
    n_med_complex = n_m + 0j
    
    n_eff_complex = get_effective_index_complex(n_p_complex, n_med_complex, phi)
    n_eff_real = n_eff_complex.real
    
    inc_angle = np.radians(8.0)
    u_x = np.sin(inc_angle)
    u_y = 0.0
    u_z = np.cos(inc_angle)
    #interface_enter(u_x, u_y, u_z, coarse_roughness, n_m, n_eff_real) --> 통과
    u_x = np.random.rand()
    u_y = np.random.rand()
    u_z = np.random.rand()
    length = np.sqrt(u_x **2 + u_y ** 2 + u_z ** 2)
    is_top = True if u_z < 0 else False
    print(u_x/length, u_y/length, u_z/length)
    interface_infilm(u_x/length, u_y/length, u_z/length, coarse_roughness, is_top, n_eff_real, n_m)'''
    