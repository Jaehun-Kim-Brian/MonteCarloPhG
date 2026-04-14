import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.stats import gamma
from scipy.special import spherical_jn, spherical_yn
import datetime


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

def get_effective_index(n_p, n_med, phi):
    eps_p = n_p**2
    eps_med = n_med**2
    
    H_b = (3*phi-1) * eps_p + (3*(1-phi)-1)*eps_med
    eps_eff = H_b + np.sqrt(H_b**2 + 8*eps_p*eps_med)
    eps_eff /= 4.0
    
    n_eff = np.sqrt(eps_eff)
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
    
    x_threshold = 0.1
    q_safe_threshold = x_threshold / d
    
    # q가 threshold보다 작으면 threshold 값으로 고정하여 평평하게 덮어씌움
    q_safe = np.where(q < q_safe_threshold, q_safe_threshold, q)
    
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
    #F_theta_surf_avg = np.zeros_like(theta, dtype=float)
    
    # 4. 크기별 Form Factor(F_theta)를 계산하고 가중 평균(Ensemble Average) 누적
    for r, w in zip(r_array, weights):
        d_i = r * 2.0
        F_theta_norm = get_mie_absorbing(wavelength, d_i, n_p_complex, n_eff_complex, theta)
        #F_theta_surf = get_mie_absorbing(wavelength, d_i, n_p_complex, n_med_complex, theta)
        F_theta_norm_avg += F_theta_norm * w
        #F_theta_surf_avg += F_theta_surf * w
        
    # 5. 최종 Phase Function 조립
    # 필름 내부(Normal): S_M_q의 얽힘(Coupling) 간섭 효과와 다분산 폼 팩터의 완벽한 결합
    Phase_Function_Normal = S_M_q * F_theta_norm_avg
    # 필름 표면(Surface): 구조 인자 간섭(S_M_q) 없이 순수한 다분산 미 산란(Mie Scattering) 적용
    Phase_Function_Surface = F_theta_norm_avg
    
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

# coarse roughness에 따라 경사진 표면에 입사하니 입사각이 0이 아님
# 입사 및 인터페이스 반사 시에도 활용해주어야 함
def get_tilted_normal(coarse_roughness, is_top):
    # coarse_roughness : root mean square of the slope of the surface
    # RMS는 평균이 0인 집단의 표준편차와 동일하므로 아래와 같이 coarse_roughness를 이용한 갑 추출 가능
    
    slope_x = np.random.normal(0, coarse_roughness)
    slope_y = np.random.normal(0, coarse_roughness)
    
    # 법선 벡터
    n_x = -slope_x
    n_y = -slope_y
    n_z = 1.0 if is_top else -1.0
    # is_top(z=0) 이면 양수, else 음수 --> 항상 필름 내부로 법선벡터가 향하게 설계됨
    
    length = np.sqrt(n_x **2 + n_y**2 + n_z**2)
    return n_x/length, n_y/length, n_z/length

def interact_enter(u_x, u_y, u_z, coarse_roughness, n_air, n_film):
    """
    외부(Air)에서 윗면(Top, z=0)을 통해 필름 내부로 진입할 때의 상호작용.
    광자는 위에서 아래(+z 방향)로 쏘아집니다.
    """
    n_x, n_y, n_z = get_tilted_normal(coarse_roughness, is_top=True)
    
    cos_i = u_x*n_x + u_y*n_y + u_z*n_z #내적
    if cos_i > 0: # 법선과 방향이 같은 방향임!
        n_x, n_y, n_z = -n_x, -n_y, -n_z # 법선을 뒤집어주자~
    cos_i = abs(cos_i) # 프레넬 계산을 위해 입사각을 항상 0-90도로 만들자
    
    R_f = calculate_fresnel(n_air, n_film, cos_i) # 프레넬 반사가 인터페이스에서 반사될 확률
    
    if np.random.rand() < R_f:
        # [반사] 필름에 들어가지 못하고 밖으로 튕김
        u_x_new = u_x + 2.0*cos_i*n_x
        u_y_new = u_y + 2.0*cos_i*n_y
        u_z_new = u_z + 2.0*cos_i*n_z
        
        u_z_new = -abs(u_z_new) 
        
        length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
        return u_x_new/length, u_y_new/length, u_z_new/length, True
    
    else:
        # [투과] 필름 내부로 진입 성공
        eta = n_air / n_film
        sin_t2 = eta**2 * (1.0 - cos_i**2)
        
        if sin_t2 >= 1.0: # 공기->매질이라 발생 안 하지만 안전장치, 투과하는 과정에서의 계산으로 전반사 쳐내기
            u_x_new = u_x + 2.0*cos_i*n_x
            u_y_new = u_y + 2.0*cos_i*n_y
            u_z_new = u_z + 2.0*cos_i*n_z
            u_z_new = -abs(u_z_new) # 튕기면 외부
            length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
            return u_x_new/length, u_y_new/length, u_z_new/length, True
            
        cos_t = np.sqrt(1.0 - sin_t2)
        factor = eta * cos_i - cos_t
        u_x_new = eta * u_x + factor * n_x
        u_y_new = eta * u_y + factor * n_y
        u_z_new = eta * u_z + factor * n_z
        
        # 🚨 강제 부호화: 투과 시 반드시 필름 내부(+z 방향)를 향해야 함
        u_z_new = abs(u_z_new)
        
        length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
        return u_x_new/length, u_y_new/length, u_z_new/length, False
        
def interact_exit(u_x, u_y, u_z, coarse_roughness, is_top, n_film, n_out):
    """
    필름 내부에서 외부(Top: 공기, Bottom: 유리/공기)로 탈출하려 할 때의 상호작용.
    """
    n_x, n_y, n_z = get_tilted_normal(coarse_roughness, is_top)
    
    # 법선 마주보기
    cos_i = u_x*n_x + u_y*n_y + u_z*n_z 
    if cos_i > 0: 
        cos_i = -cos_i
        n_x, n_y, n_z = -n_x, -n_y, -n_z 
    cos_i = abs(cos_i)
    
    R_f = calculate_fresnel(n_film, n_out, cos_i)
    
    # [핵심 로직] is_top 여부에 따른 부호 하드코딩
    sign_inside = 1.0 if is_top else -1.0   # 내부로 튕길 때: Top이면 +z, Bottom이면 -z
    sign_outside = -1.0 if is_top else 1.0  # 외부로 나갈 때: Top이면 -z, Bottom이면 +z

    if np.random.rand() < R_f:
        # [반사] 필름 내부로 다시 갇힘
        u_x_new = u_x + 2.0*cos_i*n_x
        u_y_new = u_y + 2.0*cos_i*n_y
        u_z_new = u_z + 2.0*cos_i*n_z
        
        # 🚨 강제 부호화: 완벽하게 내부 방향 고정
        u_z_new = abs(u_z_new) * sign_inside
            
        length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
        return u_x_new/length, u_y_new/length, u_z_new/length, True
        
    else:
        # [투과/굴절] 탈출 시도
        eta = n_film / n_out
        sin_t2 = eta**2 * (1.0 - cos_i**2)
        
        if sin_t2 >= 1.0: 
            # [전반사] TIR 당첨 -> 내부로 반사!
            u_x_new = u_x + 2.0*cos_i*n_x
            u_y_new = u_y + 2.0*cos_i*n_y
            u_z_new = u_z + 2.0*cos_i*n_z
            
            # 🚨 강제 부호화: 내부 방향 고정
            u_z_new = abs(u_z_new) * sign_inside
                
            length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
            return u_x_new/length, u_y_new/length, u_z_new/length, True
            
        # [진짜 투과 성공] 밖으로 탈출
        cos_t = np.sqrt(1.0 - sin_t2)
        factor = eta * cos_i - cos_t
        u_x_new = eta * u_x + factor * n_x
        u_y_new = eta * u_y + factor * n_y
        u_z_new = eta * u_z + factor * n_z
        
        # 🚨 강제 부호화: 완벽하게 외부 방향 고정! (블랙홀 차단)
        u_z_new = abs(u_z_new) * sign_outside
            
        length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
        return u_x_new/length, u_y_new/length, u_z_new/length, False

def interact_boundary(u_x, u_y, u_z, n_x, n_y, n_z, n_i, n_t):
    """
    3D 벡터 스넬의 법칙 & 프레넬 주사위 로직
    반환값: (새로운 u_x, u_y, u_z), is_reflected (True면 반사, False면 투과/탈출)
    """
    # 법선을 광자의 진행방향과 항상 마주보게 바꾸어준다.(방향 바꾸기)
    cos_i = u_x*n_x + u_y*n_y + u_z*n_z # 광자 방향과 법선 벡터를 내적하여 입사각의 cos 값 얻음
    if cos_i > 0: # 같은 방향 보고 있음
        cos_i = -cos_i
        n_x, n_y, n_z = -n_x, -n_y, -n_z 
    
    cos_i = abs(cos_i) # 입사각을 계산할때는 마주봐서는 안되니까 다시 양수로
    
    # 프레넬 반사율 계산
    R_f = calculate_fresnel(n_i, n_t, cos_i)
    
    if np.random.rand() < R_f: # 반사하겠지
        # 3D 반사 벡터 계산: u_new = u + 2*cos_i*n
        u_x_new = u_x + 2.0*cos_i*n_x
        u_y_new = u_y + 2.0*cos_i*n_y
        u_z_new = u_z + 2.0*cos_i*n_z
        
        if np.sign(u_z_new) != np.sign(n_z):
            u_z_new = -u_z_new
            
        length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
        return u_x_new/length, u_y_new/length, u_z_new/length, True

    else:
        # 투과(굴절) : 3D 벡터 스넬의 법칙 적용
        eta = n_i / n_t  # 상대굴절률, 입사각 cos을 아니까 굴절각 sin 구할 수 있음
        sin_t2 = eta**2 * (1.0 - cos_i**2) 
        
        # (안전장치) 논리상 R_f=1.0에서 다 걸러지지만, 수치 오차 대비
        if sin_t2 >= 1.0: # 전반사, 안전장치
            u_x_new = u_x + 2.0*cos_i*n_x
            u_y_new = u_y + 2.0*cos_i*n_y
            u_z_new = u_z + 2.0*cos_i*n_z
            return u_x_new, u_y_new, u_z_new, True
        
        cos_t = np.sqrt(1.0 - sin_t2)
        factor = eta * cos_i - cos_t
        
        u_x_new = eta * u_x + factor * n_x
        u_y_new = eta * u_y + factor * n_y
        u_z_new = eta * u_z + factor * n_z
        
        if np.sign(u_z_new) != np.sign(n_z):
            u_z_new = -u_z_new
        
        # normalization
        length = np.sqrt(u_x_new**2 + u_y_new**2 + u_z_new**2)
        return u_x_new/length, u_y_new/length, u_z_new/length, False
    

# 500 nm --> 하나의 파장에서 포톤 하나가 움직이는것만 봐볼거임
def run_one_mc(wavelength, r_i,
               film_thickness, k_p,
                theta_array, n_med,
                phi, fine_roughness, coarse_roughness):
    
    n_p_real = get_np_real(wavelength) # 현재 파장에 해당하는 PS 굴절률   
    n_p_complex = n_p_real + 1j * k_p
    n_med_complex = n_med + 0j
    
    # 2. Bruggeman을 통한 엄밀한 유효 복소 굴절률 도출
    n_eff_complex = get_effective_index_complex(n_p_complex, n_med_complex, phi)
    n_eff_real = n_eff_complex.real
    k_eff_strict = abs(n_eff_complex.imag) # 허수부 절대값 추출
    
    # 3. 엄밀한 흡수 계수(mu_a) 계산
    #mu_a = get_mu_a(wavelength, k_eff_strict)
    
    # Ginoza 이용한 phase function
    Phase_norm, Phase_surf = get_phase_func_ginoza(wavelength, theta_array, r_mean=r_i, pdi=0.03, phi=phi, k_p=2e-5, n_p_real=n_p_real, n_m=n_med)
    cdf_norm, cdf_surf = cdf_phase(Phase_norm, theta_array), cdf_phase(Phase_surf, theta_array)
    l_scat_norm, l_scat_surf = get_l_scat(Phase_norm, theta_array, phi, r_i), get_l_scat(Phase_surf, theta_array, phi, r_i)
    
    # 광자 초기화 : +z 방향으로 수직 입사
    x,y,z = 0.0, 0.0, 0.0
    inc_angle = np.radians(8.0)
    u_x = np.sin(inc_angle)
    u_y = 0.0
    u_z = np.cos(inc_angle)
    # 입사각에 따른 방향 벡터 (u_x, u_y, u_z)
    path_x, path_y, path_z = [x], [y], [z] 
    logs = ["Start: Photon entered from surface (z=0.0) -> interact with boundary"]
    step_count = 0
    # 밖(n_m)에서 안(n_eff)으로 들어올 때 표면이 기울어져 있으면 빛이 꺾임
    u_x, u_y, u_z, is_reflected = interact_enter(u_x, u_y, u_z, coarse_roughness, n_med, n_eff_real)
   
    
    if is_reflected:
        event = "Reflected when it enters the film!"
        logs.append(f"Step {step_count}: {event} | Pos: ({x:.1f}, {y:.1f}, {z:.1f})")
        return path_x, path_y, path_z, logs
        
    event = "Transmitted into the film"
    logs.append(f"Step {step_count}: {event} | Pos: ({x:.1f}, {y:.1f}, {z:.1f})")
    alive = True
    is_first_step = True
    total_moving_length = 0
    
    while alive:
        step_count += 1
        if is_first_step:
            is_first_step = False
            if np.random.rand() < fine_roughness: 
                current_l_scat, current_cdf = l_scat_surf, cdf_surf
                event = "Fine Roughness (Surface Scattering)"
            else: 
                current_l_scat, current_cdf = l_scat_norm, cdf_norm
                event = "Normal Bulk Scattering"
        else: 
            current_l_scat, current_cdf = l_scat_norm, cdf_norm
            event = "Bulk Scattering"

        # 1) step size 뽑기 (exponential distribution)
        # step_size = -ln(random) * l_scat, random은 0과 1 사이의 수
        step_size = -np.log(np.random.rand()) * current_l_scat
        total_moving_length += step_size
        
        next_x = x + step_size * u_x # boundary를 건들지 않는다면 얘네를 다음 x, y, z로 추진할 것
        next_y = y + step_size * u_y
        next_z = z + step_size * u_z
        
        # 3) 경계면 도달 시 프레넬/전반사 체크 로직
        hit_boundary = False
        is_top = False
        if next_z <= 0:
            hit_boundary = True
            is_top = True
            event += f" -> Hit TOP boundary"
        elif next_z >= film_thickness:
            hit_boundary = True
            is_top = False
            event += f" -> Hit BOTTOM boundary"
                    
        if hit_boundary:
            # 탈출하려는 면의 기울어진 법선 벡터 뽑기
            n_out = n_med
            # 내부(n_eff)에서 외부(n_med)로 나가려는 시도
            u_x_new, u_y_new, u_z_new, is_reflected = interact_exit(u_x, u_y, u_z, coarse_roughness, is_top, n_eff_real, n_out)
            
            if is_reflected:
                # 내부로 튕겨짐
                dz_total = abs(u_z * step_size)
                if is_top: 
                    dz_to_boundary = abs(z)
                    z_hit = 0.0
                    event += f" -> Bounced Back(TOP)"
                else: 
                    dz_to_boundary = abs(film_thickness - z)
                    z_hit = film_thickness
                    event += f" -> Bounced Back(BOTTOM)"
                
                fraction_before = dz_to_boundary / dz_total
                fraction_after = 1.0 - fraction_before
                
                x_hit = x + u_x * fraction_before * step_size
                y_hit = y + u_y * fraction_before * step_size
                
                next_x = x_hit + fraction_after * step_size * u_x_new # 튀기는거 확정이니까 바로 x, y, z로 이식해줘야지
                next_y = y_hit + fraction_after * step_size * u_y_new
                next_z = z_hit + fraction_after * step_size * u_z_new
                x, y, z = next_x, next_y, next_z
                u_x, u_y, u_z = u_x_new, u_y_new, u_z_new
                logs.append(f"Step {step_count}: {event} | Hit: ({x_hit:.2f}, {y_hit:.2f}, {z_hit:.2f}) -> Pos: ({x:.2f}, {y:.2f}, {z:.2f})")
        
                    
            else:
                # 탈출 성공
                event += " -> Escaped! (Transmitted out)"
                logs.append(f"Step {step_count}: {event} | Final Hit: (err, err, err)")
                #x, y, z = x_hit, y_hit, z_hit
                alive = False
                # 입사각의 cos은 z 방향의 절댓값
        else:  # 안부딫힌 경우이므로 x, y, z를 이식
            x, y, z = next_x, next_y, next_z
            logs.append(f"Step {step_count}: {event} | Pos: ({x:.2f}, {y:.2f}, {z:.2f})")
        path_x.append(x)
        path_y.append(y)
        path_z.append(z)
        
        
        # 4) 매질 안에서 새로운 산란 방향(theta, phi) 뽑기
        # 랜덤 난수로 CDF를 기반한 theta 결정
        cdf_value = np.random.rand()
        scat_theta = np.interp(cdf_value, current_cdf, theta_array) # cdf_value 값일때 광자가 산란하는 각도를 역연산
        phi = 2.0 * np.pi * np.random.rand() # 방위각 랜덤으로 뽑아주기
        
        # 5) 3D 공간에서 새로운 방향 벡터
        sin_t, cos_t = np.sin(scat_theta), np.cos(scat_theta)
        sin_p, cos_p = np.sin(phi), np.cos(phi)
        
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
        
    return path_x, path_y, path_z, logs
    



# --- 애니메이션 실행부 ---
if __name__ == "__main__":
    theta_array = theta = np.linspace(0, np.pi, 500)  # 0 ~ 180도
    n_m = 1.0
    phi = 0.56
    fine_roughness = 0.5
    coarse_roughness = 0.9
    path_x, path_y, path_z, logs = run_one_mc(wavelength=0.550, r_i=0.138,
               film_thickness=77, k_p=2e-5,
                theta_array=theta_array, n_med=n_m,
                phi=phi, fine_roughness=fine_roughness, coarse_roughness=coarse_roughness)

    print(logs)
    
    timestamp =datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"phg_mc/file/mc_log_{timestamp}.txt"
    
    with open(log_filename, "w", encoding="utf-8") as f:
        for log in logs:
                f.write(log + "\n")
    print(f"✅ 디버그 로그가 '{log_filename}' 파일로 저장되었습니다.")
    # 2. 3D 애니메이션 셋업
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 광자가 미친듯이 튀어나가는 걸 잡기 위해 축 범위를 동적으로 할당!
    max_x = max(np.max(np.abs(path_x)), 10)
    max_y = max(np.max(np.abs(path_y)), 10)
    min_z = min(np.min(path_z), -10)
    max_z = max(np.max(path_z), 87) # 77 + 10 여유

    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, max_y)
    ax.set_zlim(min_z, max_z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title("Photon Trajectory Escape Debugger")

    # 필름 경계면 그리기 (0 ~ 77)
    xx, yy = np.meshgrid(np.linspace(-max_x, max_x, 2), np.linspace(-max_y, max_y, 2))
    ax.plot_surface(xx, yy, np.full_like(xx, 77.0), alpha=0.2, color='gray') # Bottom
    ax.plot_surface(xx, yy, np.full_like(xx, 0.0), alpha=0.2, color='blue')  # Top

    line, = ax.plot([], [], [], marker='o', markersize=4, color='r', linestyle='-', linewidth=1.5)
    log_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.8), va='top')

    def update(num):
        # 궤적 업데이트
        line.set_data(np.array(path_x[:num+1]), np.array(path_y[:num+1]))
        line.set_3d_properties(np.array(path_z[:num+1]))
        
        # 우측 상단에 최근 5개의 로그만 실시간 출력
        current_logs = "\n".join(logs[max(0, num-4):num+1])
        log_text.set_text(current_logs)
        return line, log_text

    ani = animation.FuncAnimation(fig, update, frames=len(path_x), interval=500, blit=False)

    # 3. 애니메이션 파일로 저장 (.mp4)
    # (주의: mp4로 저장하려면 컴퓨터에 'ffmpeg'가 설치되어 있어야 합니다.)
    ani_filename = f"phg_mc/file/mc_ani_{timestamp}.mp4"
    try:
        print("⏳ 애니메이션 렌더링 및 저장 중... (시간이 조금 걸릴 수 있습니다)")
        ani.save(ani_filename, writer='ffmpeg', fps=2)
        print(f"✅ 애니메이션 영상이 '{ani_filename}' 파일로 저장되었습니다!")
    except Exception as e:
        print(f"❌ mp4 저장 실패 (ffmpeg 미설치 등): {e}")
        # mp4 저장이 안 될 경우 gif로 대체 시도
        gif_filename = f"phg_mc/file/mc_ani_{timestamp}.gif"
        try:
            ani.save(gif_filename, writer='pillow', fps=2)
            print(f"✅ 대신 '{gif_filename}' (GIF) 파일로 저장했습니다!")
        except Exception as e2:
            print(f"❌ GIF 저장도 실패: {e2}")

    plt.show()
    
    