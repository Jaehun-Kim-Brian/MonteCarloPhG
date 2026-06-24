# SI 분석

## Materials and Methods

### Fabrication of samples. 실제 공정

- 필름 제작 시 변수는 np size, volume fracton, film thickness, components 수, carbon black 농도
- 사용 np는 항상 PS(지름이 190 - 280nm)

- 실험 내용... 

### Measurement of reflectance spectra.

pass

### Measurement of sample parameters.
- 모델이 사용할 수 있도록 실험 조건들을 파라미터화함
- 모델 파라미터 : partice radius, polydispersity, volume fraction, film thickness, particle complex refractive index, matrix complex refractive index

- polydispersity = root-mean-square of deviation / (mean particle diameter)
- 위와 같이 particle radius와 polydispersity, volume fraction, film thickness, particle과 matrix의 complex refractive indices를 실제로 측정함

### Implementation of absorption. 
- 흡수 효과를 반영해주기 위해 complex refractive index를 사용함
- size parameter x, index ratio m, wavevector k도 복소수가 되는 효과가 있음!
- 
### Implementation of boundary effects.

- Interface에서 Fresnel equation으로 반사 계산 
- 경우에 따라 위 아래에 유리 넣는 것도 고려
- Snell 법칙으로 enter, exit 때 굴절각 계산

- Critical Angle에 의한 전반사 고려해주기
- 시간 최적화를 위한 Mirror Image로 trajectory 계산

$$\text{Si(OCH}_2\text{CH}_3)_4 + 4\text{H}_2\text{O} \xrightarrow{\text{NH}_3} \text{Si(OH)}_4 + 4\text{CH}_3\text{CH}_2\text{OH}$$

\(2\text{Si(OH)}_4 \rightarrow \text{(OH)}_3\text{Si-O-Si(OH)}_3 + \text{H}_2\text{O}\)

$$\text{Si(OC}_2\text{H}_5)_4 + 2\text{H}_2\text{O} \xrightarrow{\text{NH}_3} \text{SiO}_2 \downarrow + \;4\text{C}_2\text{H}_5\text{OH}$$

제공해주신 실험 과정 파일과 변경된 실험 조건을 반영하여, 실험 보고서의 **‘3. 실험 방법 (Procedure)’** 섹션에 바로 삽입할 수 있도록 학술적이고 간결한 줄글 형태로 정리해 드립니다.

보고서용 양식에 맞춰 불필요한 구어체나 기구 목록은 생략하고, 변경 요청하신 TEOS 시차 분할 주입 공정(Semi-batch process)을 정확한 타임라인으로 녹여냈습니다.

---

## 3. 실험 방법 (Procedure)

### 3.1. 전구체 및 촉매 혼합 용액 제조

1. $60,^\circ\text{C}$로 예열된 Water bath에 $600,\text{mL}$의 물을 채워 온도를 유지한다.


2. 마그네틱 바(Magnetic stirring bar)가 구비된 바이알에 피펫과 주사기를 이용하여 에탄올(Ethanol) $70\,\text{mL}$, DI Water $3\,\text{mL}$, 25% 암모니아수($\text{NH}_4\text{OH}$) $8,\text{mL}$를 순차적으로 투입하고 교반한다.


3. 휘발성이 강한 암모니아 가스로 인한 용기 내 압력 상승을 방지하기 위해, 바이알 입구에 주사기 바늘과 풍선을 장착한 밀폐 시스템을 구성한다.


4. 제조된 혼합 용액 유리를 $60\,^\circ\text{C}$ Water bath에 30분간 침지시켜 열적 평형(Thermal equilibrium) 상태를 유도한다.



### 3.2. 전구체(TEOS) 시차 분할 주입 및 축합 반응 (Semi-batch Process)

1. 열적 평형이 완료된 혼합 용액에 주사기를 이용하여 1차 전구체인 TEOS $2.33,\text{mL}$를 일시 주입한 후 $60,^\circ\text{C}$에서 30분간 가열 및 교반한다.


2. 이후 핵 성장(Growth)의 제어를 위해, 잔여 TEOS $2.0,\text{mL}$를 $1,\text{mL}$씩 30분 간격으로 총 2회에 걸쳐 시차 분할 주입(Semi-batch injection)한다.


3. 최종 전구체 주입이 완료된 시점으로부터 총 2시간 동안 지속적으로 교반하며 실리카 나노입자의 가수분해 및 축합 반응을 유도한다.



### 3.3. 원심분리 및 세척 공정 (Washing Cycle)

1. 반응이 완료된 용액을 꺼내어 Falcon tube에 약 $10,\text{mL}$씩 소분(Aliquot)하여 파지한다.


2. 소분된 튜브를 원심분리기(Centrifuge)에 대칭으로 장착한 후, **$900\times g$ 조건에서 1시간 동안 1차 원심분리**를 수행한다.


3. 분리가 끝난 후 상층액(Supernatant)인 에탄올을 조심스럽게 데칸테이션(Decantation)하여 제거한다.


4. 튜브 바닥에 침전된 실리카 펠릿(Pellet)을 물리적으로 파쇄한 뒤, 신선한 에탄올 $10,\text{mL}$를 첨가하여 완전히 재분산(Redispersion)시킨다.


5. 재분산된 용액을 다시 원심분리기에 넣어 **30분간 회전시키는 세척 과정을 총 2회 반복**하여 잔류 불순물을 완전히 제거한다.



### 3.4. 건조 및 최종 파우더 회수

1. 세척 공정이 완료된 모든 튜브의 실리카 펠릿(Pellet)을 하나의 바이알로 유실 없이 모은다.


2. 해당 바이알을 건조 오븐(Drying Oven)에 배치하여 잔류 용매를 완전히 증발시킨 후, 최종적으로 백색의 단분산성 실리카 나노입자 파우더(Silica Powder)를 회수한다.
