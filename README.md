# 송련00 진화판 (SongRyeon00-evo)

역전파 없이, **진화**가 구조를 고르고 **국소 학습 규칙**만으로 살면서 배우는 다음 자모 예측 실험.
한국어 글을 자모 28개(자음 14, 모음 10, 글자 경계, 띄어쓰기, 줄 바꿈, 말 끝)로 쪼개 다음 자모를 맞힌다.

*Evolving instruction lists that pick an operator, read/write addresses into a shared memory (weights and activations are the same cells), a write mode and a local plasticity rule — for online next-symbol prediction on Korean jamo, without backpropagation.*

## 구조

- **정보**: 한 줄로 이은 312칸 = 입력 28(지금 자모만 1) + 상태 256 + 출력 28.
  가중치 칸과 활성값 칸의 구분이 없다. 모든 값이 같은 정보 칸이다.
- **교류(유전자 한 줄)**: 진화가 다섯 가지를 고른다.
  1. 어떤 연산을 쓸지 (뉴런식 `x·w + b`, ReLU, 국소 정규화)
  2. 어디서 읽을지 (x 구간, w 구간, b 칸)
  3. 어디에 쓸지 (상태나 출력, 입력 칸은 못 씀)
  4. 어떻게 쓸지 (덮어쓰기, 더하기, 섞기)
  5. 쓴 직후 읽은 수를 어떻게 조정할지 (헵, 역헵, 델타(오차로 볼 칸도 고름), 예측 부호화)
- **한 턴**: 입력 칸에 지금 자모를 넣고, 출력 칸을 비우고, 교류를 차례로 돌린 뒤 출력에서 가장 큰 칸이 예측.
- **한 삶**: 고정된 시작 상태에서 자모 2000개를 처음부터 한 번 읽으며 먼저 맞히고 배운다. 뒤 절반 정답률이 점수.
- **진화**: 30마리, 위 10마리가 부모, 칸 다시 뽑기와 교류 넣기·빼기 돌연변이. 잘함 = 정답률.

## 결과 (2026-11-19, 일상대화 100만 자모, CPU 몇 시간)

확인용 글(학습에 안 쓴 뒤 10%)의 2000자모 조각 4개, 같은 조건(처음부터 한 번 읽기, 뒤 절반 채점):

| 모델 | 다음 자모 정답률 | 한 자모당 계산(대략) |
|---|---|---|
| 늘 가장 흔한 자모 | 24.7% | - |
| 2-gram (같은 조각에서 셈) | 39.0% | - |
| 저수지 40 + 델타 출력 | 41.0% | 약 2,800 |
| **송련00 진화판 (약 15,000세대, 교류 40줄)** | **42.6%** | **약 250** |
| 저수지 256 + 델타 출력 | 42.8% | 약 73,000 |
| 저수지 200 + 델타 출력 | 43.3% | 약 46,000 |

- 글 순서를 섞으면 14.8%로 떨어짐 → 답을 몰래 보는 곳 없음.
- 한 삶 앞 절반 31.3% → 뒤 절반 42.6%: 살면서 배움.
- 글자 경계 98.9%, 띄어쓰기 100%, 모음 24.2%, 자음 5.4%를 맞힘.
- 계산량 잣대는 송련(읽는 칸 수)과 저수지(곱셈·덧셈 수)가 조금 달라 대략 비교.
- **실패에서 배운 것**: 턴마다 출력 칸을 비우지 않으면 한 번 커진 칸이 계속 1등이라 25%(늘 같은 자모)에 갇힘. 잘함에 비용을 넣으면 정답률이 다 같은 초기에 줄 지우기가 이겨 교류 1줄로 쪼그라듦.

## 선행 연구와의 관계

부품은 모두 이미 있다. 조립한 것은 찾지 못했다(조사 확신 중간, 작은 학회 논문·개인 저장소까지 다 보지 못함).

- 주소를 고르는 명령어 목록 진화: 선형 유전 프로그래밍, [AutoML-Zero](https://arxiv.org/abs/2003.03384) (가장 가까움, 다만 학습 규칙이 별도 함수이고 진화가 역전파 비슷한 것을 다시 찾음)
- 진화된 국소 규칙으로 살면서 배움: Chalmers 1990, [Najarro & Risi 2020](https://arxiv.org/abs/2007.02686), [Jordan et al. 2021](https://elifesciences.org/articles/66273), [Soltoggio et al. 개관](https://arxiv.org/abs/1703.10371)
- 역전파 없는 문자 순서 온라인 학습 규칙 진화: "Evolving Neural Update Rules for Sequence Learning", [Finding online neural update rules by learning to remember](https://arxiv.org/abs/2003.03124)
- 가중치와 활성값이 같은 저장 공간: Schmidhuber 1993, [현대판](https://arxiv.org/abs/2202.05780)

정답률은 같은 조건의 저수지와 거의 같다. 내세울 수 있는 점은 계산량이지만, 같은 계산량으로 맞춘 비교는 아직 안 했다.

## 돌리기

```bash
pip install numpy
python make_npy.py 글.txt data.npy                      # 한국어 글 → 자모 번호
python run.py --data data.npy --life 2000 --hours 8 --workers 8 --out out
python autopsy.py data.npy out/best.json                 # 1등 부검
python reservoir_baseline.py data.npy                    # 같은 조건 저수지 비교
```

- `out/상황판.txt`: 10세대마다 확인용 정답률. `out/gen_XXXXXX.json`: 30세대마다 무리 전체. `--resume`으로 이어 돌림.
- 실험에 쓴 일상대화 데이터(AI Hub)는 배포 조건 때문에 올리지 않았다. 아무 한국어 글이나 쓰면 된다.

## 만든 사람

- 설계: 송련 (정보·연산자·교류, 진화가 고르는 다섯 가지)
- 코드: Claude. 코드 주석의 `[Claude 임시]`는 설계자가 정하지 않고 Claude가 고른 값이다.

MIT License
