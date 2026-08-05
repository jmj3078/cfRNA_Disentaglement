# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
항상 CLAUDE.md에 존재할 가치가 있는 줄만 유지하도록 할 것.

# cfRNA Normative Modeling — 공통 프로젝트 요구사항
### 분석 핵심 가정 및 목적
- 기존 cfRNA 전사체 분석은 주로 정상군과 질병군 간의 집단 수준 비교(Group-wise comparison)에 의존해 왔다. 그러나 생물학적 및 기술적 공변량(Covariates)에 의한 분산이 질병 고유의 신호를 압도하는 경우가 많다. 이로 인해 집단 단위의 일괄적인 공변량 보정은 질병 신호의 소실이나 교란 요인의 잔존을 초래하는 근본적인 한계가 존재한다. 이를 극복하기 위해, 본 연구는 전장 전사체(Whole-Transcriptome) 기반의 대규모 정상군(Healthy Control) 데이터를 활용한 규범적 모델링(Normative Modeling)을 도입한다. 개별 샘플의 공변량을 반영하여 정상 상태의 예상 분포를 추정하고, 이를 통계적 편차(Z-score)로 산출함으로써 교란 요인의 영향 없이 질병 특이적 신호를 정밀하게 정량화.

### 코드 작성 요령
- **토큰절약**: 되도록 ponytail 같은 스킬들을 활용해서 효율적이고 짧게 코드를 작성하도록 할 것
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
- **대규모 변경** 대규모 코드 리팩토링 진행시 반드시 새로운 branch에서 코드를 고치고 반드시 소규모 테스트를 통해서 동작과 재현을 모두 확인할 것.

### 신규 분석 노트북 추가 시 체크리스트
1. `pipeline/` 모듈에 로직 구현 → 노트북은 import+호출만 (thin runner 원칙)
2. `config.py`의 경로/파라미터 import, 재선언 금지
3. `apply_style()` 호출 확인

### 데이터베이스 참조/논문참조
skill 중 /paper-lookup, /database-lookup, /scientific-critical-thinking 을 효율적이게 활용하여, 사용자가 결과의 해석을 요청한 경우 반드시 fetching과 skill을 적절히 활용하여 기존 연구결과의 엄격한 검증을 통해 해석을 수행할 것. 반드시 과학적인 근거가 있는 내용만을 보수적으로 제공할 것. 교차검증을 위한 문서와 인용 링크를 남겨서 사용자가 직접 결과를 검증할 수 있도록 할 것
