# BiAG 재현 문제와 수정 내역

분석 기준: `e13755e76a64d026d047d9749a6ed58c6c59c82f`.
논문: [arXiv v1](https://arxiv.org/html/2503.21258v1), 특히 식 (4), (7), (10), (16)과 Table III.

이 변경은 코드의 정확성과 논문에 명시된 연결 구조를 수정한다. 전체 데이터셋의
학습을 다시 수행한 결과는 아니며, 논문 정확도 회복을 보장하지 않는다.

## 확인된 버그

| 문제 | 이전 동작과 영향 | 수정 | 검증 |
|---|---|---|---|
| 에피소드 차원 | `(5,1,D)`로 학습하여 클래스를 별도 배치로 처리 | `(1,5,D)`로 학습·평가 통일 | 출력 및 클래스 축 검사 |
| 손실 broadcasting | `(5,1,D)`와 `(5,D)`를 비교하여 `(5,5)`의 모든 클래스 조합을 평균 | 동일한 `(K,D)`만 허용하는 `classwise_cosine` | 정답 순열·붕괴 예제 및 잘못된 차원 거부 |
| 학습되지 않는 블록 | singleton WSA와 `SCM(W_s)` 경로로 앞의 세 블록 gradient가 0 | 클래스 토큰 축 및 query 경로 수정 | 모든 블록과 WSA Q/K에 nonzero gradient |
| 미등장 클래스 입력 | 비어 있는 100-class table의 행도 old knowledge로 전달 | 이전 세션까지 관측한 ID만 선택 | 빈 슬롯 대신 큰 값을 넣어도 generator 입력에서 제외 |
| 미등장 클래스 예측 | 아직 배우지 않은 classifier 행도 argmax 후보 | seen class 열에서만 argmax, global ID로 복원 | 미등장 logit이 가장 높아도 예측에서 제외 |
| support augmentation | CIFAR 분기가 `do_augment=False`를 무시 | 학습 split은 유지하면서 crop/flip 비활성화 | 실제 dataset 변환과 호출 경로 검사 |
| CutMix 정답 비율 | 경계에서 잘린 패치 면적을 label 비율에 반영하지 않음 | 실제 패치 면적으로 lambda 재계산 | 픽셀 평균과 target 혼합 비율 일치 |
| 실행 설정 | 데이터셋 변경 후 backbone 기본값, depth, worker 수가 반영되지 않는 경로 | 실행 시점 설정을 적용하고 depth를 평가 CLI에도 제공 | parser/config 및 모델 구성 검사 |
| 파일 경로 | base CIFAR는 `~/datasets`, 평가는 `DATA_ROOT`; `--biag` 학습 저장 경로 무시 | 데이터 경로 통일; `--biag`를 학습 시 last 경로 alias로 지원 | 세 CLI 단계의 저장/재로드 통합 검사 |
| base loader의 증분 데이터 | 사용하지 않는 support를 생성하며 miniImageNet session index도 어긋남 | base/prototype/test만 생성; 증분은 기존 고정 split loader 전담 | base-only 로딩 검사 |
| seed | 학습 entrypoint와 CutMix 독립 RNG에서 seed 누락 | Python/NumPy/PyTorch seed 적용, CutMix는 worker가 seed하는 NumPy RNG 사용 | 동일 seed의 CutMix 일치 |

분류기 scale은 체크포인트에서 복원한다. 공통 양수 scale 누락 자체는 top-1 하락의
원인으로 판단하지 않았지만, 저장된 모델을 정확하게 복원하기 위해 수정했다.

## 논문과의 구조·설정 차이

- 각 층의 별도 SCM을 하나의 공유 SCM으로 바꿨다.
- `q_P = SCM(W_s)`를 `q_P = SCM(q_L)`로 수정했다.
- query 갱신에 추가되어 있던 학습 가능한 gamma 및 별도 MLP를 제거하고
  `q_next = q + SCM(new_w)`를 사용한다.
- 기본 base 학습을 200 epoch, SGD, learning-rate milestones `[100,150]`, gamma `0.1`로 맞췄다.
- 두 데이터셋 모두 base를 포함한 총 9세션임을 명확히 했다.

구조 차이 각각의 성능 기여는 아직 ablation으로 측정하지 않았다. decoder embedding의
초기화, attention projection/head 구성 등 논문이 충분히 명시하지 않은 세부 사항은
공식 구현과 동일하다고 주장하지 않는다. 기존 기본 동작인 prototype 기반 decoder
초기화 및 multihead attention/정규화를 유지하고, 사용되지 않던 decoder offset
파라미터는 제거했다. BiAG의 AdamW, learning rate, epoch 수 역시 구현 선택이며
논문에서 검증된 최적 설정이라는 의미가 아니다.

## 결과표 및 평가 지표

- 논문 CIFAR-100: base **84.00**, final **57.95**, session average **68.93**.
- 기존 README의 재현 보고값: base **82.92**, final **49.74**. 최종 격차는 **8.21%p**.
- 기존 `63.88`의 집계 의미는 확인되지 않아 논문 수치로 재사용하지 않는다.
- 수정 이후의 정확도는 **아직 미측정**이다.
- `summary.json`과 CSV에 base/novel 정확도를 별도로 기록한다. 각 그룹의 정확도는
  해당 그룹의 샘플을 **전체 seen class와 경쟁시켜** 측정한다.
- 기존 `forgetting`은 base accuracy에서 누적 전체 accuracy를 뺀 값이었다.
  이제는 같은 base 샘플 집단의 `session-0 base accuracy - current base accuracy`다.
  과거 로그의 `forgetting`과 직접 비교하면 안 된다. 이는 모든 task에 대한 표준
  average forgetting이 아니라 base retention drop이다.

## 체크포인트와 재실험

기존 BiAG 체크포인트는 공유 SCM 구조와 호환되지 않는다. 평가 시 명시적으로 오류를
내며, `strict=False`로 부분 로딩해서 사용하지 않는다. **BiAG는 반드시 재학습한다.**
backbone/classifier/prototype 세 파일은 같은 base 학습에서 생성된 묶음이면 재사용할
수 있다. 공개 release에는 prototype 파일이 없으므로, 그 파일만으로 재학습에 필요한
묶음이 완성되지 않는다. 전체 설정을 맞추려면 base 단계부터 재학습한다.

새 출력 폴더를 사용하면 기존 실험을 보존할 수 있다. 아래 명령은 repository root에서 실행한다.

```bash
python main.py base --dataset cifar100 --data_root ./code/data --output_dir ./checkpoints/corrected --epochs 200 --seed 1
python main.py biag --dataset cifar100 --data_root ./code/data --output_dir ./checkpoints/corrected --biag_epochs 50 --biag_depth 4 --seed 1
python main.py incremental_run --dataset cifar100 --data_root ./code/data --output_dir ./checkpoints/corrected --biag_depth 4 --seed 1
```

miniImageNet은 `--dataset miniimagenet`과 데이터 위치를 지정한다. 기본 backbone은
ResNet-12로 선택된다. CIFAR-100은 ResNet-18이다. 다른 depth로 학습했다면 평가에서도
같은 `--biag_depth`를 지정한다. `--epochs`는 base 단계용이며 BiAG는 `--biag_epochs`를 쓴다.

재검증은 같은 base checkpoint로 기존 결과, 수정 후 BiAG, prototype 직접 분류 기준선을
비교하고 base/novel 정확도를 함께 기록한다. 이후 base 설정과 CutMix까지 수정하여
재학습한 결과를 별도로 기록해야 영향 요인을 구분할 수 있다.

## 검증 범위

```bash
python -m unittest discover -s tests -v
```

CPU 테스트는 loss, gradient, 데이터 변환, seen-class 평가, 설정 전달, 세 CLI 단계의
학습/저장/로드 및 legacy checkpoint 거부를 검사한다. 통합 테스트는 작은 합성 데이터와
작은 backbone을 사용하며, CIFAR-100/miniImageNet 벤치마크를 대체하지 않는다.
