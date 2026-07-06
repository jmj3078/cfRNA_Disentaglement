# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
항상 CLAUDE.md에 존재할 가치가 있는 줄만 유지하도록 할 것.

# cfRNA Normative Modeling — 공통 프로젝트 요구사항
### 분석 핵심 가정 및 목적
- 기존 cfRNA 전사체 분석은 주로 정상군과 질병군 간의 집단 수준 비교(Group-wise comparison)에 의존해 왔다. 그러나 생물학적 및 기술적 공변량(Covariates)에 의한 분산이 질병 고유의 신호를 압도하는 경우가 많다. 이로 인해 집단 단위의 일괄적인 공변량 보정은 질병 신호의 소실이나 교란 요인의 잔존을 초래하는 근본적인 한계가 존재한다. 이를 극복하기 위해, 본 연구는 전장 전사체(Whole-Transcriptome) 기반의 대규모 정상군(Healthy Control) 데이터를 활용한 규범적 모델링(Normative Modeling)을 도입한다. 개별 샘플의 공변량을 반영하여 정상 상태의 예상 분포를 추정하고, 이를 통계적 편차(Z-score)로 산출함으로써 교란 요인의 영향 없이 질병 특이적 신호를 정밀하게 정량화.

### 코드 작성 요령
- **간결성**: 최소한의 코드를 지향. 불필요한 추상화·방어 로직 금지.
- **타입 힌트 금지**: 함수/메서드의 입력 인자 dtype, 반환값 dtype 모두 표기하지 않는다.
- **정렬용 공백 금지**: 줄·등호를 맞추기 위한 인위적 띄어쓰기 금지 (`a = 1` O / `a   = 1` X).
- **이모지 금지**: 코드·주석·출력·문서 어디에도 사용하지 않는다.
- **주석**: 영어로만 작성. 단, 사용자가 명시적으로 요청하기 전에는 주석을 달지 않는다 (작업 최종 완료 후 일괄 추가 예정).
- **import 순서**: 알파벳 순.
- **경로·재사용 변수**: 절대 재선언하지 말고 루트 `config.py`에서 전역 import (구조는 아래 디렉토리 트리 참조).
- **캐시 우선 로딩**: 시각화·분석 스크립트에서 재계산 비용이 큰 중간 산출물(CV 결과, gene-wise 통계 등)은 항상 저장된 캐시 파일(csv/pkl)이 있으면 그걸 먼저 불러오고, 없을 때만 재계산 후 저장하는 로직을 기본으로 넣는다 (`modeling_criteria_eda.ipynb` Section 4의 `if os.path.isfile(...): load else: compute+save` 패턴 참고).
- **시각화**: 모든 그림은 `apply_style()`로 공통 테마 적용. 노트북/스크립트 위치 기준으로 아래 패턴 사용.
  ```python
  if parent_dir not in sys.path:
      sys.path.insert(0, parent_dir)
  from viz_style import apply_style
  apply_style()
  ```

### 프로젝트 디렉토리 구조 (경로 = 간단설명 단일 소스)
유지 규칙: 모든 경로/구조 정보는 아래 트리에 **경로 : 한 줄 설명** 형식으로만 기록한다. 별도 상세 섹션을 만들어 중복시키지 말 것. config.py나 디렉토리가 바뀌면 본 트리만 갱신.
작업 환경 : conda env "scRNA" 에서 작업
./config.py : 경로·파라미터 단일 소스. 모든 코드가 재선언 없이 import. ROOT/DATA_DIR/PIPELINE_DIR · MODELING_DIR/ENGINE_DIR/CV_RESULTS_DIR/CV_FIG_DIR/Z_SCORES_DIR/GSEA_DIR/BENCHMARK_DIR(DESEQ2_RESULTS_DIR/DESEQ2_GSEA_DIR/DESEQ2_COV_*) · DISPERSION_TREND_PATH · H5AD_PATH · Z_DISEASE/Z_HC/Z_RARE_DISEASE/Z_RARE_HC/Z_RARE_GENE_NAMES 등 Z 경로 · RARE_GLM · EXCLUDED_GENES(downstream 분석에서만 제외할 유전자 set, 기본 7개 intercept-stage 예외 유전자) · PATHS{merged_raw/biases/qc} · BIAS_COLUMNS · MODELING_PARAMS{분석키:ood_percentile=95/min_samples=5/z_flag=3.0/stratify_col/n_splits=5/gsea_*/emap_sim_thr=0.50 + 엔진키:nz_a_max=7/trend_min_nz=30/alpha_floor·cap/ridge_lambda_sigma/outlier_z/max_outlier_iter/max_remove_frac/beta_explode_thr/gaic_k/rare_overdisp_thr/rare_z_cap} · PARAMS(EDA용)
./viz_style.py : apply_style() 공통 matplotlib 테마 (모든 시각화 필수)
./CLAUDE.md · ./README.md : 공용 문서
./_legacy/ : 폐기 파일·임시파일. 참고·수정·읽기 금지
./Data/ : 분석 핵심 데이터 (권한 없이 수정 금지)
./OpenAccess_nfcore/ : 주 분석 데이터(adata) 원본·전처리본 (권한 없이 수정 금지). config.PATHS.merged_qc = 주 h5ad
./RPM_nfcore/ : Validation 실험실 데이터 (추가 예정)
./Saved_Pipeline/ : config.PIPELINE_DIR (LogisticGP·Z matrix, 생성 예정)
./EDA/ : 코호트 QC·batch/bias 교란분석. cwd=EDA 가정, 루트 config/viz_style를 Path.cwd().parent로 sys.path 등록해 import
./EDA/analysis_cfrna_cohorts.ipynb : QC→PCA→RDA 교란분석 (helper+plot 호출)
./EDA/analysis_helper.py : QC·bias 정량화 + RDA 분산분해 엔진 (자체적으로 root를 sys.path 등록)
./EDA/analysis_plot.py : analysis_helper 전용 시각화
./EDA/VariousNormalizationMethods_OpenAccess.R : 정규화 레이어 생성 (R, PROJECT_ROOT 절대경로)
./EDA/Analysis_Results/ : 위 노트북 출력 (노트북에서 ./Analysis_Results/ 상대경로)
./Modeling/ : cfRNA Normative Modeling 본체
./Modeling/*.py : model_engine(NZ 게이팅 demotion chain 엔진 본체, 아래 핵심 아키텍처 참조) · run_model_engine(→engine_state/, demotion 통계+funnel figure, --limit smoke test) · cv_model_engine(엔진 5-fold CV → CV_Results/, cv_stats.csv·cv_zscores.pkl·cv_ppc.pkl) · dispersion_trend(Phase 0 covariate-free MoM dispersion trend) · sample_filter(MahalanobisFilter OOD) · gene_selectors(proportion/effect_size/svd + effect_size_specific[방안1 질병간대조+방안2 ubiquity damping] + l1_logistic[OVR L1 판별]) · build_disease_reference(Open Targets 질병별 참조 유전자 JSON 재생성). 경로/BIAS_COLUMNS/임계값 전부 config import
./Modeling/pipeline/ : 분석·시각화 모듈 패키지. data_prep(공통 전처리:load_adata/study-split/OOD·MIN_SAMPLES/Z 로드/EXCLUDED_GENES 제외)·scoring·selection·enrichment·signatures(THEMES+heuristic/emap 군집)·cv_diagnostics(엔진 CV 진단: calibration·PPC 시각화+요약 CSV, _model_diagnostics.ipynb의 로직 본체)·benchmark(DESeq2 vs Normative 유전자 단위 비교)·gsea_compare(GSEA term-level 비교: with_rare↔no_filter↔DESeq2 겹침 통계+diff+Open Targets DB 교차검증. load_sets/compare_all/validate_rare_novel/deseq2_coverage/db_hit_rates) + plots(시각화 전용). __init__이 root를 sys.path 등록. 노트북은 thin runner(import+호출)로 동일 산출물 재현
./Modeling/engine_state/ : run_model_engine.py 산출 (학습된 engine) = config.ENGINE_DIR. genes.pkl(GeneRecord dict) · scaler.pkl · config.pkl · rare_glm.pkl(pooled rare GLM 계수) · training_summary.csv(route/stage/nz/fail_reason) · dispersion_trend.json(=config.DISPERSION_TREND_PATH) · route_demotion_summary.png
./Modeling/Z_scores/ : normative model Z-score 산출물 전용 (Z_disease/sample/gene/hc/hc_names.npy = engine-only canonical · Z_rare_disease/hc/gene_names.npy = rare 공변량 GLM 별도 아티팩트 · disease_scores_flagged.parquet = 전 분기 통합 long표) = config.Z_SCORES_DIR
./Modeling/CV_Results/ : 엔진 CV·진단 출력 = config.CV_RESULTS_DIR. cv_stats.csv(per-gene held-out calibration) · cv_zscores.pkl · cv_ppc.pkl(per-point y/mu/sigma) · cv_summary_by_stage.csv · ppc_summary_stats.csv · discrimination_summary.csv/discrimination_by_disease.csv(selection.discrimination_control 산출: random floor/batch-null/per-disease AUC) · Figures/(cv_diagnostics.py + plots.plot_discrimination_control 산출). Z 행렬은 여기 아님(→Z_scores/)
./Modeling/GSEA/ : GSEA 산출 (조건별 하위폴더 = _gene_enrichment.ipynb CONDITIONS의 label, 각 gsea_result_*.csv · Clusters/ · Figures/) + 해석 리포트. 조건 = 포함할 엔진 stage/route 집합(scoring.stage_masked_z로 그 stage 외 유전자 열 0). no_filter=nbi+nb_fixed+intercept(engine-only count route, rare 미포함, 기존 이름 유지=gsea_compare/downstream 호환) · with_rare=nbi+pool(full-NBI+rare covariate GLM만) · nbi_only=nbi만. ubiquity/artifact 필터는 폐기. no_filter/GSEA_Master_Report.md · with_rare/GSEA_Master_Report.md(DB+20질병 문헌검증) · Analysis_Provenance.md(rare/DESeq2 비교에 쓴 DB 엔드포인트·쿼리·질병ID·PubMed 기록)
./Modeling/Benchmark/ : Normative Modeling vs DESeq2 정성/정량 비교 전용 = config.BENCHMARK_DIR. deseq2_results/(deseq2_*.csv·gsea_result_*.csv per phenotype, PyDESeq2 within-study 공변량미보정=config.DESEQ2_RESULTS_DIR) · deseq2_gsea/(공변량미보정 GSEA=config.DESEQ2_GSEA_DIR) · deseq2_covariate_results/(공변량보정 DESeq2 결과=config.DESEQ2_COV_RESULTS_DIR, run_deseq2_covariate.py 산출) · deseq2_covariate_gsea/(공변량보정 GSEA=config.DESEQ2_COV_GSEA_DIR) · disease_reference/(질병별 Open Targets association 상위 300 유전자 JSON = DB 교차검증 참조) · gsea_compare/(gsea_compare.py 산출: overlap_stats.csv · rare_novel_validated/summary.csv · deseq2_coverage.csv · deseq2_cov_vs_nocov_overlap.csv · deseq2_cov_db_hits.csv · db_hit_rates*.csv · {comparison}__{which}__{pheno}.csv diff 리스트) · rescued_genes_*.csv(분석1 산출) · DESeq2_vs_Normative_Report.md(term 커버리지/방향불일치 비교) · Figures/
./Modeling/노트북 (모두 thin runner) : disease_scoring(→Z_scores/ 재생성) → gene_selection → gene_enrichment → gsea_heuristic_signatures(수동 theme,PPT용) · model_diagnostics(cv_diagnostics.run_all 호출: 엔진 CV calibration·PPC 그림+CSV) · gsea_rare_deseq2_comparison(rare 포함/미포함/DESeq2 3자 GSEA term 비교, gsea_compare 호출)
./Modeling/dispersion_trend.py : Phase 0. 공변량 무시하고 raw count에서 gene별 NB2 MoM dispersion(sigma=(var-mean)/mean^2) 계산 → nz>=trend_min_nz(30)만 신뢰 → log(mu) 구간별 nonzero-가중 중앙값 → lowess(log-log) 평활. load_trend()가 alpha_of(mean)->dispersion 클로저 반환(alpha_floor~alpha_cap 클립). stage nb_fixed/intercept 분산 고정에 사용
./Modeling/model_engine.py : `NormativeModelEngine`(엔진 본체). NZ 게이팅 + demotion chain (아래 핵심 아키텍처 참조)
./Modeling/run_model_engine.py : 엔진 학습 스크립트 → config.ENGINE_DIR(engine_state/). demotion-chain 통계+funnel figure 출력. `--limit N`(smoke test) · `--nz-a-max`

### pipeline/ 주요 진입점 (노트북에서 실제 호출되는 함수)
- `data_prep.load_disease_filtered()` → `DiseaseData` dataclass (Z_dis · dis_pheno · dis_names · gene_names · gene_syms · adata · is_hc · X_raw) 반환. 분석 노트북의 공통 진입점.
- `data_prep.ood_min_samples_filter()` → OOD + min_samples 양방향 필터 적용 후 (Z, pheno, names, ood, keep, excluded) 반환.
- `scoring.load_engine()` → engine_state/가 있으면 NormativeModelEngine.load(), 없으면 학습 후 저장.
- `scoring.score_full/score_hc(engine,...)` → `engine.score(..., as_dict=True)`로 canonical Z_scores/ 산출물 저장(Z_disease.npy=engine-only rare=0 placeholder · Z_rare_disease.npy · disease_scores_flagged.parquet).
- `cv_diagnostics.run_all()` → CV_Results/의 cv_stats/cv_zscores/cv_ppc 읽어 calibration·PPC 그림+요약 CSV 산출.
- `selection.discrimination_control(Z_dis,dis_pheno,Z_hc,hc_batch,gene_names)` → (summary_df, disease_df). healthy-null calibration control 대체(지도학습이 iid N(0,1)과의 미세상관을 miscalibration으로 오인하는 문제 때문). selector별 3종 held-out AUC: RANDOM split(HC 반분→pseudo-disease, batch 구조 파괴, ~0.5=위양성 없음) / BATCH-group split(batch 통째, batch-confound null) / DISEASE(질병별 vs HC, batch null 대비 z). `data_prep.hc_batch_ids(hc_names)`가 HC row에 정렬된 Batch_ID 제공. _gene_selection.ipynb 말미에서 호출. 결론: 엔진 HC calibration 무결함, 유일한 위양성원은 batch confound, 질병신호(median AUC~0.95)는 batch-null(~0.7~0.79) 초과([[project-hc-calibration-batch-confound]]).
- `disease_scores_flagged.parquet` (Z_scores/) → 샘플×유전자 Z-score를 z_flag(3.0) 기준으로 이진화한 플래그 표(branch=pool→rare else count, score_type=<stage>_z/rare_glm).

### 핵심 아키텍처 (여러 파일을 읽어야 파악되는 큰 그림)
- **Normative Model = 단일 엔진 (`NormativeModelEngine`, model_engine.py) = NZ 게이팅 + demotion chain**. HC nonzero 샘플 수(NZ) 기준 게이팅은 **단 하나**뿐:
  - **Route pool** (`nz < nz_a_max`=7): 풀링 GLM(offset=log(mean_hc+eps), shared beta, Poisson→deviance/df>`rare_overdisp_thr`시 NB). 이 route는 demotion chain에 진입하지 않고 항상 성공(별도 `train_rare()`). 순수 파이썬(statsmodels), R 불필요.
  - **Route model** (`nz >= nz_a_max`): NZ로 더 세분하지 않고 전부 stage "nbi"부터 시도 → 실패하면 실제 fit 결과에 따라 한 단계씩 강등(demotion), 고정 NZ 컷오프 없음.
- **Demotion chain (정보 손실 우선순위, 한 단계씩)**:
  1. **stage nbi**: R `gamlss` full NBI(mu·sigma 모두 공변량 회귀, sigma는 `ridge_lambda_sigma` L2). R 비수렴/예외/계수폭발(`beta_explode_thr`=3.0, mu 또는 sigma) → 강등.
  2. **stage nb_fixed**: 순수 파이썬 IRLS(`_nb_irls`)로 mean-only NB, dispersion은 Phase 0 trend에서 고정(`alpha_of(mean)`, 공변량은 mean에만 사용). outlier 반복 제거(`outlier_z`=5.0, `max_remove_frac`=0.05) 후 full vs intercept-only를 GAIC(`gaic_k`=2.0=AIC)로 비교해 `mean_model_chosen` 결정. IRLS 발산 → 강등.
  3. **stage intercept**: 닫힌 형태(mu=mean(y), dispersion=trend) 최종 폴백. 유한 양수 평균이면 사실상 항상 성공; 실패 시 해당 유전자 **excluded**.
- **Phase 0 (dispersion_trend.py)**: 공변량 무시, raw count MoM dispersion을 log(mu) 구간별 lowess로 평활한 covariate-free 트렌드(edgeR/DESeq2 trended-dispersion과 유사 형태). stage nb_fixed/intercept가 이 트렌드로 dispersion을 고정해 공변량 자유도를 전부 mean에 씀. stage nbi는 trend 미사용(sigma도 직접 회귀).
- **Z-score = randomized quantile residual (RQR)**. HC 규범 분포가 맞으면 z ~ N(0,1). stage별 순수 파이썬 함수: nbi→`_nbi_rqr_from_coeffs`, nb_fixed/intercept→`_nb_rqr`, pool→`_poisson_rqr`/`_nb_rqr`(rare_z_cap=10 클립). scoring 시 R 불필요(stage nbi 학습 시에만 R 필요). `GeneRecord`가 gene마다 (initial_route, route, stage, 계수, fail_reason)을 보관해 `training_summary.csv`로 전체 demotion 이력 추적. `.branch` 프로퍼티(pool→'rare' else 'count')로 downstream taxonomy 제공.
- **공변량(X) = BIAS_COLUMNS 10개**(config). HC로 fit한 StandardScaler로 표준화 후 모델 입력. disease 샘플은 학습된 scaler/계수로 score만.
- **score()의 두 모드**: 기본은 bare Z 배열(CV용). `as_dict=True`면 downstream 계약 dict 반환(combined=pool열 0으로 zero한 engine-only · combined_all=전체 · rare/rare_gene_names=pool 서브행렬 · gene_names). scoring.py가 이걸로 canonical Z_scores/ 산출물 저장.
- **데이터 흐름**: h5ad(config.H5AD_PATH=merged_qc) → 엔진 학습(engine_state/) → disease scoring(Z_scores/ 의 Z_disease.npy=engine-only canonical + Z_rare_disease.npy=rare 별도) → OOD(Mahalanobis, HC-fit) + MIN_SAMPLES 필터 + EXCLUDED_GENES 제외(data_prep.load_disease_filtered) → gene_selection/enrichment/GSEA/comparison. **EXCLUDED_GENES는 scoring이 아니라 downstream 진입점에서만 적용**(scoring은 전 유전자 저장).
- **rare 저장/사용은 통합 + opt-in**. canonical Z_disease.npy는 engine-only(pool 컬럼 0 placeholder) 유지로 기존 downstream 불변. rare(pool) 공변량 GLM z는 Z_rare_*.npy로 따로 저장되고, disease_scores_flagged.parquet에는 전 route 통합 long으로 들어감. GSEA/signature 등에서 rare를 합쳐 쓸지는 `scoring.score_disease_with_rare(dd)` / `scoring.load_z(with_rare=True)`로 분석 단계에서 선택(기본은 미포함).
- **pipeline/ 패키지가 분석 로직, 노트북은 thin runner**. 같은 분석을 재현하려면 노트북이 아니라 pipeline 모듈을 수정. `pipeline/__init__.py`와 각 엔트리 스크립트가 자체적으로 root를 sys.path에 등록하므로 config/모듈 import는 재선언 없이 동작.
- **v1(3-way detection-rate 분기 엔진)은 폐기 → `_legacy/Modeling_v1/`** (model_engine·run_model_engine·cv_gamlss_nb/zinb·cv_logistic·cv_rare·comparison.py·_model_validation.ipynb + engine_state/CV_Results/Z_scores 구 산출물). git branch가 실 백업.

### 실행 명령어
- **R 의존성**: 엔진 학습(run_model_engine.py의 stage nbi)과 cv_model_engine.py는 R + `gamlss` 패키지 + rpy2 필요(gamlss.r를 source). pool/nb_fixed/intercept 및 모든 scoring(RQR)은 순수 파이썬으로 R 불필요.
- 엔진 학습 → engine_state/: `python Modeling/run_model_engine.py` (smoke test는 `--limit N`, 절대 실 산출물 덮어쓰지 말 것)
- 엔진 CV → CV_Results/(cv_stats.csv · cv_zscores.pkl · cv_ppc.pkl): `python Modeling/cv_model_engine.py`
- 분석 노트북 실행 순서(모두 thin runner): disease_scoring → gene_selection → gene_enrichment → gsea_heuristic_signatures, 그리고 model_diagnostics(엔진 CV calibration·PPC). EDA는 cwd=EDA 가정.
- 테스트 스위트·린터·빌드 시스템 없음(연구 코드). 검증은 노트북 재실행/스크립트 산출물 확인으로 수행.

### 신규 분석 노트북 추가 시 체크리스트
1. `pipeline/` 모듈에 로직 구현 → 노트북은 import+호출만 (thin runner 원칙)
2. `config.py`의 경로/파라미터 import, 재선언 금지
3. `apply_style()` 호출 확인

### 데이터베이스 참조/논문참조
skill 중 /paper-lookup, /database-lookup, /scientific-critical-thinking 을 효율적이게 활용하여, 사용자가 결과의 해석을 요청한 경우 반드시 fetching과 skill을 적절히 활용하여 기존 연구결과의 엄격한 검증을 통해 해석을 수행할 것. 반드시 과학적인 근거가 있는 내용만을 보수적으로 제공할 것.
