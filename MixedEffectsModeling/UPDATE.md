Normative modeling에서 Z(=당신 엔진의 RQR)는 HC에서 정확히 N(0,1)이어야 하고, 그렇지 않으면 모델 오지정(misspecification)입니다 (Dinga & Fraza et al. 2021, GAMLSS normative modeling; Fraza et al. 2021 NeuroImage). 이 분야가 skew/kurtosis를 만나면 먼저 shape 파라미터를 도입해 warping(SHASH)으로 분포 자체를 교정합니다:

Fraza et al. 2021 (NeuroImage 118715) — Warped BLR: 가우시안 가정이 skewed/kurtotic 데이터에서 Z를 miscalibrate 시키므로 SHASH warping으로 Z를 정규화.
de Boer et al. 2024 (Imaging Neuroscience) — non-Gaussian hierarchical Bayesian normative modeling, 같은 논리.
Rutherford et al. 2023 (eLife 85082) / 2022 (Nat Protocols) — QC 표준으로 held-out Z의 skewness·kurtosis를 직접 보고.

당신 엔진의 NBI+RQR은 이미 count용 location-scale 모델이라, "Z가 HC에서 N(0,1)인가"가 곧 calibration 진단입니다 (RQR의 정의 자체가 correct model 하에서 N(0,1): Dunn & Smyth 1996). 필터링은 warping/shape 확장으로도 Z를 정규화 못 하는 유전자(예: mito batch-fold군)에만 적용하는 게 문헌적으로 방어됩니다.


Per-gene 기준 ("이상도" = 주변부 null 보정)
대상 질문: 유전자 g의 held-out HC Z가 N(0,1)에서 얼마나 벗어나는가.

측정 지표 (문헌 표준):

skewness, excess kurtosis of held-out HC Z — normative QC의 직접 표준 (Fraza 2021, Rutherford 2022/23).
정규성 검정: D'Agostino-Pearson(skew+kurtosis 결합), Anderson-Darling(꼬리 민감), KS — 유전자별 RQR ~ N(0,1) 검정.

empirical null (Efron): 중앙부를 N(δ₀, σ₀)로 적합해 (δ₀,σ₀)가 (0,1)에서 벗어나는 정도. 이게 FDR과의 직접 연결고리입니다.
FDR로의 연결 (당신 관찰의 근거): |Z|>c 임계로 BH-FDR을 돌릴 때 p-value는 Φ(theoretical null) 꼬리를 씁니다. Z가 skewed/heavy-tailed면 그 꼬리 확률이 틀려 theoretical-null FDR이 무효가 됩니다 — 이게 Efron 2007 (Annals of Statistics "Size, power and false discovery rates")과 fdrtool/locfdr (Strimmer 2008)의 핵심. 즉 당신의 "quantile matching 왜곡 → skew/kurtosis → FDR 영향" 우려는 large-scale inference 문헌에서 정확히 다루는 실재 현상입니다.

**임계값을 어떻게 잡나 (중요): 문헌은 지표(skew/kurtosis/empirical null)와 원리(null이 보정돼야 FDR 유효)는 주지만 보편 cutoff 숫자는 표준화돼 있지 않습니다. 임의 |kurtosis|<7 같은 rule-of-thumb에 기대지 말고, cutoff를 downstream FDR calibration에 앵커하는 게 가장 엄격합니다: held-out HC에서 임계 c의 실제 false-positive 비율이 nominal(예: 1−Φ(c))과 맞는지로 유전자별/집단 skew·kurtosis 컷을 역산. HC가 곧 null이므로 이 데이터-주도 앵커가 가능합니다.**