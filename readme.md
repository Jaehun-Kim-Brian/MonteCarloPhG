# Reproduction of Hwang et al. (2021) - Figure 3.C

본 프로젝트는 다중 산란이 발생하는 무질서한 나노 구조의 구조색을 예측하는 Monte Carlo 시뮬레이션 모델을 구현 및 재현합니다.
특히, 논문의 **Figure 3.C (두께 변화에 따른 반사율 스펙트럼)** 재현을 목표로 합니다.

---

## 1. 핵심 물리 모델 체크리스트 (구현 시 필수 반영)

### A. 유효 매질 이론 (Effective Medium Theory)
- [ ] 매질의 굴절률 계산 시 비대칭적인 Maxwell-Garnett 대신 대칭적인 **Bruggeman 근사법**을 사용해야 함.
- [ ] 굴절률은 흡수를 반영하기 위해 반드시 **복소수(Complex index)** 형태로 계산되어야 함.

### B. 미 이론(Mie Theory) 보정 (흡수 매질 대응)
- [ ] **대수적 도함수(Logarithmic derivatives):** 복소수 파라미터 연산 시 발산을 막기 위해 Riccati-Bessel 함수의 대수적 도함수 형태를 사용하여 산란 계수($a_n, b_n$)를 계산해야 함.
- [ ] **표면 적분(Surface Integration):** 흡수 매질에서는 먼 거리(Far-field) 근사가 성립하지 않으므로, 미분 산란 단면적은 반드시 입자의 표면($r=a$)에서 적분해야 함.
- [ ] **입사광 진폭 보정:** 매질 흡수로 인한 입자 내 위치별 빛 강도 차이를 보정하는 계수($I_{0, corrected}$)를 곱해야 함.

### C. 구조 팩터와 다분산성 (Structure Factor & Polydispersity)
- [ ] **Schulz 분포:** 입자 크기 분포는 Schulz 분포를 따른다고 가정함. 다분산 지수(PDI) 0.03을 기준으로 확률을 계산함.
- [ ] **가중합 계산:** 단일 크기가 아닌, 분포 확률에 따른 가중치를 곱해 폼 팩터 평균값 $\overline{F(q)}$를 도출하고, 이를 다분산 구조 팩터 $S_M(q)$와 곱하여 산란 강도를 구해야 함.

### D. 몬테카를로 엔진 및 표면 거칠기 (Monte Carlo & Roughness)
- [ ] **가중치 감소:** 광자가 매질을 이동할 때 Beer-Lambert 법칙에 따라 흡수되는 만큼 Weight를 감소시킴.
- [ ] **전반사(TIR) 트릭:** 임계각보다 작아 전반사되는 경우, 궤적을 새로 그리지 않고 '거울 이미지(Mirror image)' 필름을 생성하여 통과시킨 뒤 반사광으로 카운트함.
- [ ] **큰 거칠기(Coarse roughness):** 필름 경계에서 빛이 굴절/반사될 때, 국소적인 표면 기울기(Surface tilt)를 반영하여 Snell의 법칙과 Fresnel 공식을 적용함.
- [ ] **미세 거칠기(Fine roughness):** 광자의 '첫 번째 스텝(First step)'은 표면 나노 입자 구조의 영향을 받으므로, 구조 팩터(Structure factor)를 배제하고 오직 Mie 이론만을 이용해 샘플링함. 이후의 스텝부터는 원래대로 계산함.

---

##  2. Figure 3.C 재현을 위한 실험 파라미터 (Parameters)

모델 시뮬레이션을 실행할 때 아래 파라미터 값들이 정확히 입력되어야 합니다.

| Parameter | Value | 
| :--- | :--- | 
| **Particle Radius** (반지름) | 138 nm |
| **Matrix** (매질) | Air (공기) |
| **Polydispersity Index** (PDI) | 0.03 | 
| **Trajectories** (파장당 궤적 수) | 20,000 |
| **Incident Angle** (입사각) | 8° |
| **Detection Angle** (탐지각) | 90° |

### 두께(Thickness)별 세부 세팅

Figure 3.C는 두께에 따라 4개의 데이터 라인을 출력합니다. 각 두께마다 **부피비(Volume fraction)**와 **거칠기(Roughness)** 세팅이 다름에 극도로 주의하십시오.

| Thickness ($t$) | Volume Fraction ($v$) | Fine Roughness | Coarse Roughness |
| :--- | :--- | :--- | :--- |
| **3930 μm** | 0.50 | 1.0 | 0.9 |
| **77 μm** | 0.56 | 0.5 | 0.9 |
| **13 μm** | 0.58 | 0.5 | 0.9 |
| **6 μm** | 0.58 | 0.5 | 0.9 |

> **Note:** 3930 μm 필름은 최대 전송 길이(Transport length)인 47 μm보다 훨씬 두꺼워 강한 다중 산란을 보장하며, 6 μm 필름은 최소 전송 길이인 8 μm보다 얇아 다중 산란을 최소화하도록 설계된 값입니다.