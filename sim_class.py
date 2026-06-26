import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gamma
from scipy.special import spherical_jn, spherical_yn
from scipy.optimize import fsolve

class PhotonicGlassMCSimulator:
    """
    Photonic Glass 환경에서의 빛의 산란 및 투과/반사를 모사하는 Monte Carlo 시뮬레이터 클래스.
    """
    def __init__(self, film_thickness, phi, fine_roughness, coarse_roughness, 
                 r_i=0.138, n_m=1.0, k_p=2e-5, pdi=0.03, detect_angle=90, polydispersity=0):
        # 1. 시뮬레이션 환경 및 물리적 파라미터 초기화
        # wvls, N_photons, theta_array 는 아직 안들어옴
        self.film_thickness = film_thickness
        self.phi = phi
        self.fine_roughness = fine_roughness
        self.coarse_roughness = coarse_roughness
        self.r_i = r_i
        self.n_m = n_m
        self.k_p = k_p
        self.pdi = pdi
        self.detect_angle = detect_angle


    # ==========================================
    # [그룹 1] 광학 상수 및 매질 특성 계산 메서드
    # ==========================================
    @staticmethod
    def get_q(n_eff, theta_array, wavelength):
        # Ref: Eq. (1), Magkiriadou, Park, Kim, and Manoharan,
        # "Absence of red structural color in photonic glasses, bird feathers, and certain beetles,"
        # Phys. Rev. E 90, 062302 (2014). DOI: 10.1103/PhysRevE.90.062302
        return 4 * np.pi * n_eff * np.sin(theta_array / 2.0) / wavelength

    @staticmethod
    def get_n_p_real_sellmeier(wavelength):
        # Ref: Sultanova, Kasarova, and Nikolov, "Dispersion Properties of Optical Polymers,"
        # Acta Physica Polonica A 116(4), 585-587 (2009). DOI: 10.12693/APhysPolA.116.585
        return np.sqrt(1 + (1.4435 * wavelength**2) / (wavelength**2 - 0.020216))

    def _get_n_eff_ps_matrix(self, wavelength):
        # Ref: SI Eq. (8), Hwang, Stephenson, Barkley, Brandt, Xiao, Aizenberg, and Manoharan,
        # "Designing angle-independent structural colors using Monte Carlo simulations of multiple scattering,"
        # Proc. Natl. Acad. Sci. U.S.A. 118(4), e2015551118 (2021). DOI: 10.1073/pnas.2015551118
        n_p_complex = self.get_n_p_real_sellmeier(wavelength) + 1j * self.k_p
        n_m_complex = self.n_m + 1j * 0.0
        
        phi_p = self.phi
        phi_m = 1 - self.phi
        
        if not (0.0 <= phi_p <= 1.0):
            raise ValueError("self.phi must be between 0 and 1.")
        
        def bruggeman_equations(x):
            # x = [Re(n_eff), Im(n_eff)]
            n_eff = x[0] + 1j * x[1]

            val = (
                phi_p * (n_p_complex**2 - n_eff**2) / (n_p_complex**2 + 2.0 * n_eff**2)
                + phi_m * (n_m_complex**2 - n_eff**2) / (n_m_complex**2 + 2.0 * n_eff**2)
            )

            return [val.real, val.imag]
        
        n_eff_guess_real = phi_p * n_p_complex.real + phi_m * n_m_complex.real
        n_eff_guess_imag = max(
            0.0,
            phi_p * n_p_complex.imag + phi_m * n_m_complex.imag
        )
        
        sol = fsolve(bruggeman_equations, x0=[n_eff_guess_real, n_eff_guess_imag])
        n_eff = sol[0] + 1j * sol[1]
        
        if n_eff.real < 0: 
            n_eff = -n_eff
            
        if abs(n_eff.imag) < 1e-14:
            n_eff = complex(n_eff.real, 0.0)
            
        if n_eff.imag < 0:
            raise ValueError(f"Computed n_eff has negative imaginary part: {n_eff}")
        
        return n_eff


    def _riccati_bessel(self, n, z): 
        # Ref: SI Eq. (12)-(13), Hwang, Stephenson, Barkley, Brandt, Xiao, Aizenberg, and Manoharan,
        # "Designing angle-independent structural colors using Monte Carlo simulations of multiple scattering,"
        # Proc. Natl. Acad. Sci. U.S.A. 118(4), e2015551118 (2021). DOI: 10.1073/pnas.2015551118
        jn = spherical_jn(n, z)
        jn_1 = spherical_jn(n-1, z)
        yn = spherical_yn(n, z)
        yn_1 = spherical_yn(n-1, z)
        
        hn = jn + 1j * yn
        hn_1 = jn_1 + 1j * yn_1
        
        psi_n = z * jn
        psi_n_prime = z * jn_1 - n * jn
        
        xi_n = z * hn
        xi_n_prime = z * hn_1 - n *hn
        return psi_n, psi_n_prime, xi_n, xi_n_prime 

    def _get_mie_absorbing(self, wavelength, n_p_complex, n_eff_complex, theta_array, radius=None, backend="internal", return_components=False):    
        """
        Returns
        -------
        diff_csca : ndarray
            dCsca/dOmega for unpolarized light.
        csca_total : float
            Total scattering cross section.
        S1, S2, a_n, b_n : optional
            Returned only for internal backend with return_components=True.
        """
        theta_array = np.asarray(theta_array, dtype=float)
        if theta_array.ndim != 1:
            raise ValueError("theta_array must be 1D.")
        if np.any(theta_array < 0.0) or np.any(theta_array > np.pi):
            raise ValueError("theta_array must lie in [0, pi].")

        if backend not in ("internal", "pymie"):
            raise ValueError("backend must be either 'internal' or 'pymie'.")

        a = float(self.r_i if radius is None else radius)
        if a <= 0.0:
            raise ValueError("radius must be positive.")

        # ----------------------------------
        # BACKEND 1: pymie
        # ----------------------------------
        if backend == "pymie":
            """
            Assumptions based on the uploaded model.py:
            - model.py imports:
                from pymie import index_ratio, mie
                from pymie import size_parameter
            - absorbing systems use:
                mie.diff_scat_intensity_complex_medium(...)
                mie.integrate_intensity_complex_medium(...)
            - distance is taken at the particle surface r = a
            """
            try:
                from pymie import mie, size_parameter, index_ratio
            except Exception as exc:
                raise ImportError(
                    "backend='pymie' requested, but pymie could not be imported."
                ) from exc

            # particle-medium index ratio and size parameter
            m = index_ratio(n_p_complex, n_eff_complex)
            x = size_parameter(wavelength, n_eff_complex, a)

            # complex wavevector in the effective medium
            k = 2.0 * np.pi * n_eff_complex / wavelength

            # pymie returns parallel/perpendicular differential intensities
            # in the absorbing medium, evaluated at k*distance
            ff = mie.diff_scat_intensity_complex_medium(
                m,
                x,
                theta_array,
                k * a,
                coordinate_system="scattering plane",
                incident_vector=None,
                phis=None,
            )

            diff_par = ff[0]
            diff_perp = ff[1]

            # If pymie returns Quantity-like objects, strip magnitudes
            try:
                diff_par = diff_par.magnitude
            except AttributeError:
                pass
            try:
                diff_perp = diff_perp.magnitude
            except AttributeError:
                pass

            diff_par = np.asarray(diff_par, dtype=float)
            diff_perp = np.asarray(diff_perp, dtype=float)

            # unpolarized average
            diff_csca = 0.5 * (diff_par + diff_perp)
            diff_csca = np.clip(diff_csca, 0.0, None)

            # total cross section by direct pymie surface integration
            csca_tuple = mie.integrate_intensity_complex_medium(
                diff_par,
                diff_perp,
                a,
                theta_array,
                k
            )
            csca_total = csca_tuple[0]
            try:
                csca_total = csca_total.magnitude
            except AttributeError:
                pass
            csca_total = float(np.real_if_close(csca_total))

            if not np.isfinite(csca_total) or csca_total <= 0.0:
                raise ValueError(
                    f"Invalid csca_total from pymie backend: {csca_total}"
                )

            if return_components:
                return diff_csca, csca_total, None, None, None, None
            return diff_csca, csca_total, None, None

        # ----------------------------------
        # BACKEND 2: internal
        # ----------------------------------
        k = 2.0 * np.pi * n_eff_complex / wavelength
        x = k * a
        m = n_p_complex / n_eff_complex
        mx = m * x

        x_mag = np.abs(x)
        n_stop = int(np.round(x_mag + 4.0 * x_mag ** (1.0 / 3.0) + 2.0))
        n_stop = max(n_stop, 1)
        n = np.arange(1, n_stop + 1, dtype=int)

        def _psi(order, z):
            return z * spherical_jn(order, z)

        def _xi(order, z):
            return z * (spherical_jn(order, z) + 1j * spherical_yn(order, z))

        def _Dn(order, z):
            jn = spherical_jn(order, z)
            jn_p = spherical_jn(order, z, derivative=True)
            psi = z * jn
            psi_p = jn + z * jn_p
            tiny = 1e-300
            denom = np.where(np.abs(psi) < tiny, tiny + 0j, psi)
            return psi_p / denom

        psi_n_x = _psi(n, x)
        psi_nm1_x = _psi(n - 1, x)
        xi_n_x = _xi(n, x)
        xi_nm1_x = _xi(n - 1, x)
        D_n_mx = _Dn(n, mx)

        tiny = 1e-300
        x_safe = x if np.abs(x) >= tiny else (tiny + 0j)

        # SI-stable coefficients
        A_n = D_n_mx / m + n / x_safe
        B_n = m * D_n_mx + n / x_safe

        denom_a = A_n * xi_n_x - xi_nm1_x
        denom_b = B_n * xi_n_x - xi_nm1_x
        denom_a = np.where(np.abs(denom_a) < tiny, tiny + 0j, denom_a)
        denom_b = np.where(np.abs(denom_b) < tiny, tiny + 0j, denom_b)

        a_n = (A_n * psi_n_x - psi_nm1_x) / denom_a
        b_n = (B_n * psi_n_x - psi_nm1_x) / denom_b

        mu = np.cos(theta_array)
        S1 = np.zeros_like(theta_array, dtype=complex)
        S2 = np.zeros_like(theta_array, dtype=complex)

        pi_0 = np.zeros_like(theta_array, dtype=float)
        pi_1 = np.ones_like(theta_array, dtype=float)

        pi_nm2 = pi_0
        pi_nm1 = pi_1

        for idx, ni in enumerate(n):
            if ni == 1:
                pi_n = pi_1
                pi_n_minus_1 = pi_0
            else:
                pi_n = (
                    ((2 * ni - 1) / (ni - 1)) * mu * pi_nm1
                    - (ni / (ni - 1)) * pi_nm2
                )
                pi_n_minus_1 = pi_nm1

            tau_n = ni * mu * pi_n - (ni + 1) * pi_n_minus_1
            prefac = (2 * ni + 1) / (ni * (ni + 1))

            S1 += prefac * (a_n[idx] * pi_n + b_n[idx] * tau_n)
            S2 += prefac * (a_n[idx] * tau_n + b_n[idx] * pi_n)

            if ni >= 2:
                pi_nm2, pi_nm1 = pi_nm1, pi_n

        # absorbing-medium radial factor at r = a
        # SI description: exp(-2 k'' r) / [ (k' r)^2 + (k'' r)^2 ]
        k_real = np.real(k)
        k_imag = np.imag(k)

        '''radial_denom = (k_real * a) ** 2 + (k_imag * a) ** 2
        if radial_denom <= 0.0:
            raise ValueError("Non-physical radial denominator in internal backend.")
        radial_factor = np.exp(-2.0 * k_imag * a) / radial_denom'''

        k_abs_sq = k_real**2 + k_imag**2
        if k_abs_sq <= 0.0:
            raise ValueError("Non-physical |k|^2 in internal backend.")

        radial_factor = np.exp(-2.0 * k_imag * a) / k_abs_sq
        # SI Eq. (16): incident-intensity correction
        zeta = 2.0 * a * k_imag
        if np.abs(zeta) < 1e-10:
            I0_corr_ratio = 1.0
        else:
            I0_corr_ratio = 2.0 * (
                np.exp(zeta) / zeta + (1.0 - np.exp(zeta)) / (zeta ** 2)
            )

        diff_csca = 0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2)
        diff_csca = radial_factor * diff_csca / np.real(I0_corr_ratio)
        diff_csca = np.asarray(np.real_if_close(diff_csca), dtype=float)
        diff_csca = np.clip(diff_csca, 0.0, None)

        csca_total = 2.0 * np.pi * np.trapezoid(
            diff_csca * np.sin(theta_array),
            theta_array
        )
        csca_total = float(np.real_if_close(csca_total))

        if not np.isfinite(csca_total) or csca_total <= 0.0:
            raise ValueError(
                f"Invalid total scattering cross section in internal backend: {csca_total}"
            )

        if return_components:
            return diff_csca, csca_total, S1, S2, a_n, b_n
        return diff_csca, csca_total, S1, S2
        
    def _get_polydisperse_form_factor_absorbing(self, wavelength, n_p_complex, n_eff_complex, theta_array, radius_samples, size_pdf, backend="internal"):
        """
        Size-averaged absorbing-medium form factor:
        F(theta) = ∫ p(r) dCsca/dOmega(theta; r) dr
        """
        theta_array = np.asarray(theta_array, dtype=float)
        radius_samples = np.asarray(radius_samples, dtype=float)
        size_pdf = np.asarray(size_pdf, dtype=float)

        if theta_array.ndim != 1:
            raise ValueError("theta_array must be 1D.")
        if radius_samples.ndim != 1 or size_pdf.ndim != 1:
            raise ValueError("radius_samples and size_pdf must be 1D.")
        if radius_samples.size != size_pdf.size:
            raise ValueError("radius_samples and size_pdf must have the same length.")
        if np.any(radius_samples <= 0.0):
            raise ValueError("All radius samples must be positive.")

        size_pdf = np.clip(size_pdf, 0.0, None)
        norm = np.trapezoid(size_pdf, radius_samples)
        if norm <= 0:
            raise ValueError("size_pdf must integrate to a positive value.")
        size_pdf = size_pdf / norm

        diff_stack = []
        csca_stack = []

        for r in radius_samples:
            diff_r, csca_r, _, _ = self._get_mie_absorbing(
                wavelength=wavelength,
                n_p_complex=n_p_complex,
                n_eff_complex=n_eff_complex,
                theta_array=theta_array,
                radius=r,
                backend=backend,
                return_components=False,
            )
            diff_stack.append(diff_r)
            csca_stack.append(csca_r)

        diff_stack = np.asarray(diff_stack)   # shape (Nr, Ntheta)
        csca_stack = np.asarray(csca_stack)   # shape (Nr,)

        diff_csca_avg = np.trapezoid(diff_stack * size_pdf[:, None], radius_samples, axis=0)
        csca_avg = np.trapezoid(csca_stack * size_pdf, radius_samples)

        diff_csca_avg = np.asarray(np.real_if_close(diff_csca_avg), dtype=float)
        diff_csca_avg = np.clip(diff_csca_avg, 0.0, None)
        csca_avg = float(np.real_if_close(csca_avg))

        if not np.isfinite(csca_avg) or csca_avg <= 0.0:
            raise ValueError(f"Invalid polydisperse csca_avg: {csca_avg}")

        return diff_csca_avg, csca_avg
    
    def _get_l_scat(self, csca_sample):
        rho = (self.phi * 3.0) / (4.0* np.pi * self.r_i**3)
        return 1.0 / (rho * csca_sample)
        
    # Calculate transport length
    def _get_l_star(self, l_scat, theta_pdf, theta_array):
        g = np.trapezoid(np.cos(theta_array) * theta_pdf, theta_array)
        return l_scat / (1.0 - g)
    
    def _get_phase_func_ginoza(self, wavelength, theta_array, backend="internal", radius_samples=None, size_pdf=None):
        theta_array = np.asarray(theta_array, dtype=float)
        if theta_array.ndim != 1:
            raise ValueError("theta_array must be a 1D array.")
        if np.any(theta_array < 0.0) or np.any(theta_array > np.pi):
            raise ValueError("theta_array must be in radians and lie in [0, pi].")
        if backend not in ("internal", "pymie"):
            raise ValueError("backend must be either 'internal' or 'pymie'.")
        
        # 1) Complex particle index and effective medium index
        n_p_complex = self.get_n_p_real_sellmeier(wavelength) + 1j * self.k_p
        n_eff_complex = self._get_n_eff_ps_matrix(wavelength)
        
        # form factor / single-particle diff cross section
        use_poly = (radius_samples is not None) and (size_pdf is not None)
        
        # 2) Single-particle absorbing Mie differential cross section 
        if use_poly:
            diff_csca_mie, csca_mie = self._get_polydisperse_form_factor_absorbing(
                wavelength=wavelength,
                n_p_complex=n_p_complex,
                n_eff_complex=n_eff_complex,
                theta_array=theta_array,
                radius_samples=radius_samples,
                size_pdf=size_pdf,
                backend=backend,
            )
           
        else: 
            diff_csca_mie, csca_mie, _, _ = self._get_mie_absorbing(
                wavelength=wavelength,
                n_p_complex=n_p_complex,
                n_eff_complex=n_eff_complex,
                theta_array=theta_array,
                backend=backend,
            )
        
        # 3) Scattering wavevector q
        # Use the real part for q, consistent with the usual structural-color definition.
        q = self.get_q(np.real(n_eff_complex), theta_array, wavelength)
        
        # 4) Structure factor from Ginoza model
        S_q = self._get_structure_factor_ginoza(q)
        S_q = np.asarray(np.real_if_close(S_q), dtype=float)
        # Numerical safeguard: measurable structure factor should not be negative
        S_q = np.clip(S_q, 0.0, None)

        # 5) Sample differential scattering cross section
        # Form-factor-like angular scattering from Mie multiplied by structure factor
        diff_csca_sample = diff_csca_mie * S_q
        diff_csca_sample = np.asarray(np.real_if_close(diff_csca_sample), dtype=float)

        # 6) Total sample scattering cross section
        csca_sample = 2.0 * np.pi * np.trapezoid(diff_csca_sample * np.sin(theta_array), theta_array)
        csca_sample = float(np.real_if_close(csca_sample))
        
        if not np.isfinite(csca_sample) or csca_sample <= 0.0:
            raise ValueError(
                f"Invalid sample scattering cross section in _get_phase_func_ginoza: {csca_sample}"
            )

        # 7) Phase function normalization:
        # p(theta) = (dCsca_sample/dOmega) / Csca_sample
        phase_function = diff_csca_sample / csca_sample

        # Optional: renormalize once more to suppress small numerical drift
        norm_check = 2.0 * np.pi * np.trapezoid(phase_function * np.sin(theta_array), theta_array)
        if not np.isfinite(norm_check) or norm_check <= 0.0:
            raise ValueError(f"Invalid phase-function normalization: {norm_check}")

        phase_function = phase_function / norm_check

        return phase_function, csca_sample, diff_csca_sample
    
    def _get_structure_factor_ginoza(self, q):
        """
        Monospecies polydisperse measurable structure factor S_M(q)
        using the Ginoza-Yasutomi formulation with a Schulz distribution.

        Parameters
        ----------
        q : ndarray
            Scattering wavevector magnitude [1/length].

        Returns
        -------
        SM : ndarray
            Measurable structure factor.
        """
        q = np.asarray(q, dtype=float)
        d = 2.0 * self.r_i
        phi = float(self.phi)

        # Avoid division by zero at q=0
        q_safe = np.where(np.abs(q) < 1e-8, 1e-8, q)

        # Schulz polydispersity parameter
        pdi = max(float(self.pdi), 1e-5)
        Dsigma = pdi**2
        Delta = 1.0 - phi
        t = 1.0 / Dsigma - 1.0

        def tm(m, t):
            if m == 0:
                return 1.0
            num_array = np.arange(m, 0, -1, dtype=float) + t
            return np.prod(num_array) / (t + 1.0)**m

        def fm(x, t, tm_val, m):
            return tm_val * (1.0 + x / (t + 1.0))**(-(t + 1.0 + m))

        t0 = tm(0, t)
        t1 = tm(1, t)
        t2 = Dsigma + 1.0
        t3 = (Dsigma + 1.0) * (2.0 * Dsigma + 1.0)

        # number density with Schulz correction
        rho = 6.0 * phi / (np.pi * d**3 * t3)

        # effective mean spacing
        sigma0 = (6.0 * phi / (np.pi * rho))**(1.0 / 3.0)

        s = 1j * q_safe
        x = s * d
        F0 = rho
        zeta2 = rho * sigma0**2

        f0 = fm(x, t, t0, 0)
        f1 = fm(x, t, t1, 1)
        f2 = fm(x, t, t2, 2)

        f0_inv = fm(-x, t, t0, 0)
        f1_inv = fm(-x, t, t1, 1)
        f2_inv = fm(-x, t, t2, 2)

        fa = (1.0 - x/2.0 - f0 - x/2.0 * f1) / x**3
        fb = (1.0 - x/2.0 * t2 - f1 - x/2.0 * f2) / x**3
        fc = (1.0 - x - f0) / x**2
        fd = (1.0 - x * t2 - f1) / x**2

        Ialpha1 = 24.0 / s**3 * F0 * (-0.5 * (1.0 - f0) + x/4.0 * (1.0 + f1))
        Ialpha2 = 24.0 / s**3 * F0 * (-d/2.0 * (1.0 - f1) + s * d**2 / 4.0 * (t2 + f2))

        Iw1 = 2.0 * np.pi * rho / (Delta * s**3) * (Ialpha1 + s/2.0 * Ialpha2)
        Iw2 = (
            np.pi * rho / (Delta * s**2) * (1.0 + np.pi * zeta2 / (Delta * s)) * Ialpha1
            + np.pi**2 * zeta2 * rho / (2.0 * Delta**2 * s**2) * Ialpha2
        )

        F11 = 2.0 * np.pi * rho * d**3 / Delta * fa
        F12 = 1.0 / d * ((np.pi / Delta)**2 * rho * zeta2 * d**4 * fa + np.pi * rho * d**3 / Delta * fc)
        F21 = d * 2.0 * np.pi * rho * d**3 / Delta * fb
        F22 = (np.pi / Delta)**2 * rho * zeta2 * d**4 * fb + np.pi * rho * d**3 / Delta * fd

        FF11 = 1.0 - F11
        FF12 = -F12
        FF21 = -F21
        FF22 = 1.0 - F22

        det = FF11 * FF22 - FF12 * FF21
        G11 = FF22 / det
        G12 = -FF12 / det
        G21 = -FF21 / det
        G22 = FF11 / det

        I0 = (
            -9.0 / 2.0 * (2.0 / s)**6 * F0**2
            * (
                1.0
                - 0.5 * (f0_inv + f0)
                + x/2.0 * (f1_inv - f1)
                - (s**2 * d**2) / 8.0 * (f2_inv + f2 + 2.0 * t2)
            )
        )

        h2 = np.real(
            (Iw1 * G11 * Ialpha1 + Iw1 * G12 * Ialpha2
            + Iw2 * G21 * Ialpha1 + Iw2 * G22 * Ialpha2) / I0
        )

        SM = 1.0 - 2.0 * h2
        SM = np.real_if_close(SM)
        SM = np.clip(SM, 0.0, None)
        return SM
        
    def _get_schulz_distribution(self, n_points=101, n_std=4.0, pdi=None, mean_radius=None):
        """
        Schulz distribution for particle radius.

        Returns
        -------
        radius_samples : ndarray
            Radius grid, same length unit as self.r_i.
        size_pdf : ndarray
            Probability density with respect to radius.
            Normalized so that ∫ size_pdf dr = 1.

        Notes
        -----
        The Schulz distribution is conventionally written for diameter.
        Since d = 2r, the radius density is:
            p_r(r) = 2 p_d(2r)
        """
        pdi = self.pdi if pdi is None else float(pdi)
        mean_radius = self.r_i if mean_radius is None else float(mean_radius)

        if mean_radius <= 0.0:
            raise ValueError("mean_radius must be positive.")
        if pdi < 0.0:
            raise ValueError("pdi must be non-negative.")

        # Nearly monodisperse fallback
        pdi_eff = max(float(pdi), 1e-5)

        mean_diameter = 2.0 * mean_radius

        # For Schulz/Gamma:
        # mean = D_bar
        # std / mean = pdi
        # shape = 1 / pdi^2
        # scale = mean / shape
        shape = 1.0 / (pdi_eff ** 2)
        scale = mean_diameter / shape
        std_diameter = pdi_eff * mean_diameter

        d_min = max(mean_diameter - n_std * std_diameter, 1e-12)
        d_max = mean_diameter + n_std * std_diameter

        diameter_samples = np.linspace(d_min, d_max, n_points)
        pdf_diameter = gamma.pdf(diameter_samples, a=shape, scale=scale)

        # Transform diameter density to radius density:
        # d = 2r, so p_r(r) = p_d(2r) * 2
        radius_samples = diameter_samples / 2.0
        size_pdf = 2.0 * pdf_diameter

        # Normalize over radius grid
        norm = np.trapezoid(size_pdf, radius_samples)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("Invalid Schulz distribution normalization.")

        size_pdf = size_pdf / norm

        return radius_samples, size_pdf
    
    def _cdf_phase(self, phase, theta_array):
        """
        Build CDF for sampling scattering angle theta from phase function.

        Parameters
        ----------
        phase : ndarray
            Phase function p(theta) = (1/Csca) dCsca/dOmega.
            It should satisfy ∫ 2π p(theta) sin(theta) dtheta = 1.
        theta_array : ndarray
            Scattering angle grid in radians.

        Returns
        -------
        cdf : ndarray
            Monotonic CDF over theta_array, normalized from 0 to 1.
        theta_pdf : ndarray
            Marginal PDF for theta:
                p_theta(theta) = 2π p(theta) sin(theta)
        """
        phase = np.asarray(phase, dtype=float)
        theta_array = np.asarray(theta_array, dtype=float)

        if phase.ndim != 1 or theta_array.ndim != 1:
            raise ValueError("phase and theta_array must be 1D arrays.")
        if phase.size != theta_array.size:
            raise ValueError("phase and theta_array must have the same length.")
        if np.any(theta_array < 0.0) or np.any(theta_array > np.pi):
            raise ValueError("theta_array must lie in [0, pi].")

        # Marginal density for theta
        theta_pdf = 2.0 * np.pi * phase * np.sin(theta_array)
        theta_pdf = np.asarray(np.real_if_close(theta_pdf), dtype=float)
        theta_pdf = np.clip(theta_pdf, 0.0, None)

        norm = np.trapezoid(theta_pdf, theta_array)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("Invalid phase CDF normalization.")

        theta_pdf = theta_pdf / norm

        # cumulative trapezoid without requiring scipy.integrate.cumulative_trapezoid
        cdf = np.zeros_like(theta_array, dtype=float)
        dtheta = np.diff(theta_array)
        cdf[1:] = np.cumsum(0.5 * (theta_pdf[1:] + theta_pdf[:-1]) * dtheta)

        # Numerical cleanup
        cdf = cdf / cdf[-1]
        cdf[0] = 0.0
        cdf[-1] = 1.0

        # Ensure monotonicity against tiny numerical wiggles
        cdf = np.maximum.accumulate(cdf)

        return cdf, theta_pdf

    def _get_mu_a(self, wavelength, n_eff_complex):
        """
        Beer-Lambert absorption coefficient of the effective medium.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength, same length unit as simulation geometry.
        n_eff_complex : complex
            Complex effective refractive index.

        Returns
        -------
        mu_a : float
            Absorption coefficient in 1 / length.
        """
        if wavelength <= 0.0:
            raise ValueError("wavelength must be positive.")

        mu_a = 4.0 * np.pi * np.imag(n_eff_complex) / wavelength

        # Tiny negative values can appear from numerical solving noise.
        if mu_a < 0 and abs(mu_a) < 1e-14:
            mu_a = 0.0

        if mu_a < 0:
            raise ValueError(f"Computed negative absorption coefficient: {mu_a}")

        return float(mu_a)


    # ==========================================
    # [그룹 2] 경계면 및 표면 거칠기(Fresnel & Roughness) 로직
    # ==========================================
    def _calculate_fresnel(self, n_i, n_t, cos_theta_i):
        """
        Unpolarized Fresnel reflectance for an interface from medium i to medium t.

        Parameters
        ----------
        n_i : float
            Refractive index of incident medium.
        n_t : float
            Refractive index of transmitted medium.
        cos_theta_i : float
            Cosine of incident angle with respect to interface normal.
            Should be non-negative.

        Returns
        -------
        R_f : float
            Fresnel reflection probability. If TIR occurs, returns 1.0.
        """
        n_i = float(np.real(n_i))
        n_t = float(np.real(n_t))

        if n_i <= 0.0 or n_t <= 0.0:
            raise ValueError("Refractive indices must be positive.")
        
        cos_theta_i = abs(float(cos_theta_i))
        cos_theta_i = np.clip(cos_theta_i, 0.0, 1.0)
        
        if cos_theta_i  < 1e-8:
            cos_theta_i = 1e-8
        
        # Snell's Law
        sin_theta_i = np.sqrt(max(0.0, 1 - cos_theta_i**2))
        sin_theta_t = n_i / n_t * sin_theta_i
        
        # TIR condition
        if sin_theta_t >= 1.0:
            return 1.0
        
        cos_theta_t = np.sqrt(max(0.0, 1 - sin_theta_t ** 2))
        r_s = (n_i * cos_theta_i - n_t * cos_theta_t) / (n_i * cos_theta_i + n_t * cos_theta_t)
        r_p = (n_i * cos_theta_t - n_t * cos_theta_i) / (n_i * cos_theta_t + n_t * cos_theta_i)
        
        R_f = 0.5 * (r_s**2 + r_p**2)
        return R_f
    
    def _get_norm_vec(self, upward=True):
        c = float(self.coarse_roughness)
        
        if c < 0.0:
            raise ValueError("coarse_roughness must be non-negative.")
        if c == 0.0:
            n = np.array([0.0, 0.0, 1.0])
        else:    
            slope_x = np.random.normal(0.0, c/np.sqrt(2.0))
            slope_y = np.random.normal(0.0, c/np.sqrt(2.0))
            
            n = np.array([-slope_x, -slope_y, 1.0], dtype=float)
            n /= np.linalg.norm(n)
            
        
        if not upward:
            n = -n
        return n
     
    def _interface_enter(self, u_x, u_y, u_z, is_top, n_i, n_t, max_trial=5):
        u = np.array([u_x, u_y, u_z], dtype=float)
        norm_u = np.linalg.norm(u)
        
        if norm_u <= 0:
            raise ValueError("Incident direction vector has zero norm.")
        u = u / norm_u
        
        n_i = float(np.real(n_i))
        n_t = float(np.real(n_t))
        
        for _ in range(max_trial):
            norm_vec = self._get_norm_vec(upward= is_top)
            cos_i = np.dot(u, norm_vec)
            
            if cos_i < 0:
                norm_vec = -norm_vec
                cos_i = -cos_i
                
            cos_i = np.clip(cos_i, 0.0, 1.0)
            R_f = self._calculate_fresnel(n_i, n_t, cos_i)
            
            if R_f >= 1.0 or np.random.rand() < R_f:
                return 0.0, 0.0, 0.0, True

            eta = n_i / n_t
            sin_i = np.sqrt(max(0.0, 1.0 - cos_i ** 2))
            sin_t = eta * sin_i
            
            if sin_t >= 1.0:
                continue
            cos_t = np.sqrt(max(0.0, 1- sin_t**2))
            
            u_parallel = u - norm_vec * cos_i
            u_trans = eta * u_parallel + cos_t * norm_vec
            
            norm_trans = np.linalg.norm(u_trans)
            if norm_trans <= 0:
                continue
            u_trans = u_trans/ norm_trans
            
            if u_trans[2] > 1e-8:
                return u_trans[0], u_trans[1], u_trans[2], False
        
        return 0.0, 0.0, 0.0, True
        
    def _interface_infilm(self, u_x, u_y, u_z, is_top, n_i, n_t, max_trial=5):
        # 필름 내부에서 외부로 나갈 때의 전반사/투과 로직 (interface_infilm 대체)
        
        u = np.array([u_x, u_y, u_z], dtype=float)
        norm_u = np.linalg.norm(u)
        
        if norm_u <= 0:
            raise ValueError("Incident direction vector has zero norm.")
        u = u / norm_u
        
        n_i = float(np.real(n_i))
        n_t = float(np.real(n_t))
        
        for _ in range(max_trial): # 이상한 표면에 반사해서, 반사해도 바깥쪽으로 진행하고 있을 경우 대비
            norm_vec = self._get_norm_vec(upward= not is_top)
            cos_i = np.dot(u, norm_vec)
            
            if cos_i < 0:
                norm_vec = -norm_vec
                cos_i = -cos_i
                
            cos_i = np.clip(cos_i, 0.0, 1.0)
            R_f = self._calculate_fresnel(n_i, n_t, cos_i)
            
            if R_f >= 1.0 or np.random.rand() < R_f:
                u_ref = u -2.0 * cos_i * norm_vec
                norm_ref = np.linalg.norm(u_ref)
                
                if norm_ref <= 0:
                    continue
                
                u_ref = u_ref / norm_ref
                if is_top and u_ref[2] > 1e-8:
                    return u_ref[0], u_ref[1], u_ref[2], True
                
                if (not is_top) and u_ref[2] < -1e-8:
                    return u_ref[0], u_ref[1], u_ref[2], True  
                
                continue
                
            else:
                return 0.0, 0.0, 0.0, False

            
        if is_top:
            u_ref = np.array([u[0], u[1], abs(u[2])], dtype=float)
        else:
            u_ref = np.array([u[0], u[1], -abs(u[2])], dtype=float)
            
        u_ref = u_ref/ np.linalg.norm(u_ref)
        return u_ref[0], u_ref[1], u_ref[2], True  


    # ==========================================
    # [그룹 3] 몬테카를로 핵심 엔진 및 실행 메서드
    # ==========================================
    
    def _scatter_direction(self, u, scat_theta, azimuth):
        u = np.asarray(u, dtype=float)
        u = u / np.linalg.norm(u)
        
        ux, uy, uz = u
        sin_t = np.sin(scat_theta)
        cos_t = np.cos(scat_theta)
        sin_p = np.sin(azimuth)
        cos_p = np.cos(azimuth)
        
        if abs(uz) > 0.9999:
            sign = 1.0 if uz > 0 else -1.0
            u_new = np.array([
                sin_t * cos_p * sign,
                sin_t * sin_p * sign,
                cos_t * sign
            ], dtype=float)
        else:
            temp = np.sqrt(1.0 - uz**2)
            u_new = np.array([
                sin_t * (ux * uz * cos_p - uy * sin_p) / temp + ux * cos_t,
                sin_t * (uy * uz * cos_p + ux * sin_p) / temp + uy * cos_t,
                -sin_t * cos_p * temp + uz * cos_t
            ], dtype=float)

        return u_new / np.linalg.norm(u_new)
        
    def _track_single_photon(self, mu_a, l_scat_norm, cdf_norm,
                             l_scat_surf, cdf_surf, theta_array, 
                             n_eff_real, incident_angle=8.0,
                             max_scatters=10000, min_weight=1e-8, trace=False):
        
        L = float(self.film_thickness)
        eps = 1e-10

        if L <= 0:
            raise ValueError("film_thickness must be positive.")
        if l_scat_norm <= 0 or l_scat_surf <= 0:
            raise ValueError("scattering lengths must be positive.")
    
        theta_array = np.asarray(theta_array, dtype=float)
        cdf_norm = np.asarray(cdf_norm, dtype=float)
        cdf_surf = np.asarray(cdf_surf, dtype=float)
        
        # Initial Position at top surface
        x, y, z = 0.0, 0.0, 0.0
        
        # Incindent from air to film
        inc_angle = np.radians(incident_angle)
        u = np.array([np.sin(inc_angle), 0.0, np.cos(inc_angle)], dtype=float)
        u = u / np.linalg.norm(u)
        
        W = 1.0             # primary weight
        total_path = 0.0
        n_scat = 0          # scattering number
        
        records = []
        if trace:
            records.append((x, y, z, W, "start"))
            
        ux, uy, uz, reflected_at_entry = self._interface_enter(u[0], u[1], u[2], is_top=True, n_i=self.n_m, n_t=n_eff_real)
        
        if reflected_at_entry:
            if trace:
                records.append((x, y, z, W, "entry_reflected"))
            return {
                "status": "reflected",
                "R": W,
                "T": 0.0,
                "W": W,
                "n_scat": 0,
                "path": 0.0,
                "trace": records if trace else None,
                }
        
        u = np.array([ux, uy, uz], dtype=float)
        u = u / np.linalg.norm(u)
        
        # 경계값에서 오류 날까봐, 위치를 필름의 아주 조금 내부로 변경함
        z = eps 
        
        if trace:
            records.append((x, y, z, W, "entered"))
        
        is_first_step = True
        
        for _ in range(max_scatters):
            # Select first-step surface scattering or normal bulk scattering
            if is_first_step:
                is_first_step = False
                if np.random.rand() < self.fine_roughness:
                    current_l_scat = l_scat_surf
                    current_cdf = cdf_surf
                    step_type = "surface_step"
                else:
                    current_l_scat = l_scat_norm
                    current_cdf = cdf_norm
                    step_type = "bulk_step"
            else:
                current_l_scat = l_scat_norm
                current_cdf = cdf_norm
                step_type = "bulk_step"
                
            # Sample exponential step size
            # 여기 한번 제대로 봐바
            rnd = max(np.random.rand(), 1e-300)
            remaining = -np.log(rnd) * current_l_scat
            
            while remaining > 1e-12:
                ux, uy, uz = u
                
                if uz > 1e-12:
                    dist_to_boundary = (L - z) / uz
                    is_top_hit = False
                elif uz < -1e-12:
                    dist_to_boundary = (0.0 - z) / uz
                    is_top_hit = True
                else:
                    dist_to_boundary = np.inf
                    is_top_hit = False
                    
                if dist_to_boundary < 0 : # 이 경우는, 이미 photon이 경계 밖으로 나간 상태, 따라서 그냥 0으로 처리
                    dist_to_boundary = 0.0
                    
                # Case 1: no boundary hit before next scattering event
                if dist_to_boundary >= remaining:
                    x += remaining * ux
                    y += remaining * uy
                    z += remaining * uz

                    W *= np.exp(-mu_a * remaining)
                    total_path += remaining
                    remaining = 0.0
                    
                    if trace:
                        records.append((x, y, z, W, step_type + "_move"))
                        
                    break
                
                path_to_boundary = dist_to_boundary
                
                x += path_to_boundary * ux
                y += path_to_boundary * uy
                z = 0.0 if is_top_hit else L
                
                W *= np.exp(-mu_a * path_to_boundary)
                remaining -= path_to_boundary
                
                if trace:
                    records.append((x, y, z, W, "hit_top" if is_top_hit else "hit_bottom"))
                
                if W < min_weight:
                    return {
                    "status": "absorbed",
                    "R": 0.0,
                    "T": 0.0,
                    "W": W,
                    "n_scat": n_scat,
                    "path": total_path,
                    "trace": records if trace else None,
                }
                
                ux_new, uy_new, uz_new, reflected_inside = self._interface_infilm(ux, uy, uz, is_top=is_top_hit, n_i=n_eff_real, n_t=self.n_m)
                
                if not reflected_inside:
                    if is_top_hit:
                        if trace:
                            records.append((x, y, z, W, "exit_top_reflected"))
                        return {
                            "status": "reflected",
                            "R": W,
                            "T": 0.0,
                            "W": W,
                            "n_scat": n_scat,
                            "path": total_path,
                            "trace": records if trace else None,
                        }
                    else:
                        if trace:
                            records.append((x, y, z, W, "exit_bottom_transmitted"))
                        return {
                            "status": "transmitted",
                            "R": 0.0,
                            "T": W,
                            "W": W,
                            "n_scat": n_scat,
                            "path": total_path,
                            "trace": records if trace else None,
                        }
                
                u = np.array([ux_new, uy_new, uz_new], dtype=float)
                u = u / np.linalg.norm(u)
                
                z = eps if is_top_hit else L - eps
                
                if trace:
                    records.append((x, y, z, W, "internal_reflection"))
                    
            if W < min_weight:
                return {
                    "status": "absorbed",
                    "R": 0.0,
                    "T": 0.0,
                    "W": W,
                    "n_scat": n_scat,
                    "path": total_path,
                    "trace": records if trace else None,
                }

            rand_cdf = np.random.rand()    
            scat_theta = np.interp(rand_cdf, current_cdf, theta_array)
            azimuth = 2.0 * np.pi * np.random.rand()     
            
            u = self._scatter_direction(u, scat_theta, azimuth)
            n_scat += 1
            
            if trace:
                records.append((x, y, z, W, "scatter"))
        
        return {
            "status": "max_events",
            "R": 0.0,
            "T": 0.0,
            "W": W,
            "n_scat": n_scat,
            "path": total_path,
            "trace": records if trace else None,
        }        
    def _run_single_wavelength(self, N_photons, theta_array, 
                               mu_a, l_scat_norm, cdf_norm, l_scat_surf, cdf_surf, n_eff_real,
                               return_diagnostics=False):
        """
        단일 파장에 대한 Photon Tracking 루프 (기존 run_mc 역할)
        """
        reflected = 0.0
        transmitted = 0.0
        
        status_count = {
            "reflected": 0,
            "transmitted": 0,
            "absorbed": 0,
            "max_events": 0,
        }
        scat_counts = []
        path_lengths = []
        final_weights = []
            
        for _ in range(N_photons):
            result = self._track_single_photon(mu_a, l_scat_norm, cdf_norm,
                             l_scat_surf, cdf_surf, theta_array, 
                             n_eff_real, incident_angle=8.0, trace=False)
            
            reflected += result["R"]
            transmitted += result["T"]
            status_count[result["status"]] = status_count.get(result["status"], 0) + 1
            scat_counts.append(result["n_scat"])
            path_lengths.append(result["path"])
            final_weights.append(result["W"])
        
        R = reflected / N_photons
        T = transmitted / N_photons
        A = max(0.0, 1.0 - R - T)
        
        if return_diagnostics:
            diagnostics = {
                "status_count": status_count,
                "mean_scatters": float(np.mean(scat_counts)),
                "median_scatters": float(np.median(scat_counts)),
                "mean_path": float(np.mean(path_lengths)),
                "median_path": float(np.median(path_lengths)),
                "mean_final_weight": float(np.mean(final_weights)),
            }
            return R, T, A, diagnostics

        return R, T

    def run_simulation(self, wvls, theta_array, N_photons,
                        save_filename=None, backend="internal", use_polydispersity=False, return_diagnostics=False):
        """
        Run Monte Carlo simulation over wavelengths.
        """
        reflectance = []
        transmittance = []
        absorbance = []
        diagnostics_all = []
        
        theta_array = np.asarray(theta_array, dtype=float)
        
        # Optional polydisperse radius grid
        if use_polydispersity:
            radius_samples, size_pdf = self._get_schulz_distribution()
        else:
            radius_samples, size_pdf = None, None
            
        for wavelength in tqdm(wvls, desc="MC Simulation"):
            # 1. 광학 파라미터 세팅 (n_eff, mu_a 등 계산)
            n_eff_complex = self._get_n_eff_ps_matrix(wavelength)
            n_eff_real = float(np.real(n_eff_complex))
            mu_a = self._get_mu_a(wavelength, n_eff_complex)
            
            # 2. Phase function 및 l_scat 도출
            phase_norm, csca_sample, diff_sample = self._get_phase_func_ginoza(
                wavelength=wavelength,
                theta_array=theta_array,
                backend=backend,
                radius_samples=radius_samples,
                size_pdf=size_pdf,
            )
            cdf_norm, theta_pdf_norm = self._cdf_phase(phase_norm, theta_array)
            l_scat_norm = self._get_l_scat(csca_sample)
            
            n_p_complex = self.get_n_p_real_sellmeier(wavelength) + 1j * self.k_p

            if use_polydispersity:
                diff_mie, csca_mie = self._get_polydisperse_form_factor_absorbing(
                    wavelength=wavelength,
                    n_p_complex=n_p_complex,
                    n_eff_complex=n_eff_complex,
                    theta_array=theta_array,
                    radius_samples=radius_samples,
                    size_pdf=size_pdf,
                    backend=backend,
                )
            else:
                diff_mie, csca_mie, _, _ = self._get_mie_absorbing(
                    wavelength=wavelength,
                    n_p_complex=n_p_complex,
                    n_eff_complex=n_eff_complex,
                    theta_array=theta_array,
                    backend=backend,
                )

            phase_surf = diff_mie / csca_mie
            phase_surf_norm = 2.0 * np.pi * np.trapezoid(
                phase_surf * np.sin(theta_array),
                theta_array
            )
            phase_surf = phase_surf / phase_surf_norm

            cdf_surf, theta_pdf_surf = self._cdf_phase(phase_surf, theta_array)
            l_scat_surf = self._get_l_scat(csca_mie)
            
            if return_diagnostics:
                R, T, A, diag = self._run_single_wavelength(
                    N_photons=N_photons,
                    theta_array=theta_array,
                    mu_a=mu_a,
                    l_scat_norm=l_scat_norm,
                    cdf_norm=cdf_norm,
                    l_scat_surf=l_scat_surf,
                    cdf_surf=cdf_surf,
                    n_eff_real=n_eff_real,
                    return_diagnostics=True,
                )
                diag.update({
                    "wavelength": wavelength,
                    "n_eff_real": n_eff_real,
                    "n_eff_imag": float(np.imag(n_eff_complex)),
                    "mu_a": mu_a,
                    "csca_sample": csca_sample,
                    "csca_mie": csca_mie,
                    "l_scat_norm": l_scat_norm,
                    "l_scat_surf": l_scat_surf,
                })
                diagnostics_all.append(diag)
            else:
                R, T = self._run_single_wavelength(
                    N_photons=N_photons,
                    theta_array=theta_array,
                    mu_a=mu_a,
                    l_scat_norm=l_scat_norm,
                    cdf_norm=cdf_norm,
                    l_scat_surf=l_scat_surf,
                    cdf_surf=cdf_surf,
                    n_eff_real=n_eff_real,
                    return_diagnostics=False,
                )
                A = max(0.0, 1.0 - R - T)

            reflectance.append(R)
            transmittance.append(T)
            absorbance.append(A)

        reflectance = np.asarray(reflectance)
        transmittance = np.asarray(transmittance)
        absorbance = np.asarray(absorbance)

        if save_filename:
            np.savez(
                f"{save_filename}.npz",
                wavelengths=np.asarray(wvls),
                reflectance=reflectance,
                transmittance=transmittance,
                absorbance=absorbance,
                diagnostics=np.array(diagnostics_all, dtype=object) if return_diagnostics else None,
            )
            print(f"{save_filename}.npz is successfully saved!")

        if return_diagnostics:
            return reflectance, transmittance, absorbance, diagnostics_all

        return reflectance, transmittance, absorbance
                

if __name__=="__main__":
    sim = PhotonicGlassMCSimulator(
        film_thickness = 77.0, #um
        phi = 0.56, 
        fine_roughness=0.5,
        coarse_roughness=0.9,
        r_i=0.138, 
        n_m=1.0, 
        k_p=2e-5, 
        pdi=0.03, 
        detect_angle=90
    )
    
    wvl_array = np.linspace(0.400, 0.800, 1000)
    theta_array = np.linspace(1e-4, np.pi, 1000)    # radians
    theta_deg = np.degrees(theta_array)  

    R, T, A, diags = sim.run_simulation(
        wvls=wvl_array,
        theta_array=theta_array,
        N_photons=1000,
        backend="internal",
        use_polydispersity=False,
        return_diagnostics=True,
    )

    plt.figure(figsize=(7, 4))
    plt.plot(wvl_array * 1000, R, label="R")
    plt.plot(wvl_array * 1000, T, label="T")
    plt.plot(wvl_array * 1000, A, label="A")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Fraction")
    plt.title("MC spectrum test: 77 µm film")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()