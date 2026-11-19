# 송련00 밤 실험 (2026-11-19). 설계: 정후 / 코드: 하얀(Claude)
#
# 정후가 정한 것 (원문 요약, 송련_결정.md 11-19 밤)
#   진화가 고르는 것: 1 어떤 연산을 쓸지  2 어디서 읽을지  3 어디에 쓸지  4 어떻게 쓸지
#                     5 쓴 직후 읽은 수를 어떻게 조정할지 (헵, 역헵, 델타(무엇을 오차로 볼지도), 예측 부호화)
#   정보: 입력 28, 상태 256, 출력 28 / 조정의 세기: 임의 / 개체 수 30
#   고르는 방식 (정후 11-19 밤): 잘함 = 정답률 - 비용/4
# 그 밖의 값과 방식은 전부 [Claude 임시]. 목록은 맨 아래.

import numpy as np, argparse, json, os, time
from multiprocessing import Pool

# ---------- 정보: 한 줄로 이은 312칸 ----------
IN, ST, OUT = 28, 256, 28
S0 = IN                 # 상태 시작 = 28
O0 = IN + ST            # 출력 시작 = 284
TOTAL = IN + ST + OUT   # 312

ENTITY = 30
OPS = ['뉴런식', 'ReLU', '국소 정규화']        # [Claude 임시] 신경 연산 세 가지
HOWS = ['덮어쓰기', '더하기', '섞기']           # [Claude 임시] 쓰는 방식 세 가지
LEARNS = ['헵', '역헵', '델타', '예측 부호화']   # 정후

MAXN = 8       # [Claude 임시] 한 번에 읽는 구간 크기 1~8
ETA = 0.01     # [Claude 임시] 조정의 세기 (정후: 임의)
CLIP = 10.0    # [Claude 임시] 정보 칸 값은 -10~10으로 자름 (터짐 방지)

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)        # 자모 번호(0~27) .npy
ap.add_argument('--life', type=int, default=2000)   # [Claude 임시] 한 삶 = 자모 2000개, 뒤 절반만 채점
ap.add_argument('--hours', type=float, default=8)
ap.add_argument('--gens', type=int, default=10**9)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='out')
ap.add_argument('--fit', default='acc')       # [Claude 임시, 정후 "네가 생각해봐"] acc = 정답률만 / cost4 = 정답률 - 비용/4
ap.add_argument('--clear_out', type=int, default=1)   # [Claude 임시] 1이면 턴마다 출력 칸을 0으로 지우고 시작
ap.add_argument('--resume', default='')       # gen_XXXXXX.json 에서 무리를 불러와 이어 돌림
MAXG = 200     # [Claude 임시] 교류 줄 수 상한
A = ap.parse_args()


# ---------- 유전자 한 줄 = 교류 하나 ----------
def new_gene(rng):
    n = int(rng.integers(1, MAXN + 1))
    return {
        'op': int(rng.integers(len(OPS))),            # 1 어떤 연산
        'n': n,                                       # 2 읽기: 구간 크기
        'x': int(rng.integers(0, TOTAL - n + 1)),     # 2 읽기: x 구간 시작
        'w': int(rng.integers(0, TOTAL - n + 1)),     # 2 읽기: w 구간 시작 (뉴런식만)
        'b': int(rng.integers(0, TOTAL)),             # 2 읽기: b 칸 (뉴런식만)
        'write': int(rng.integers(S0, TOTAL)),        # 3 어디에 쓸지 (입력 칸 제외)
        'how': int(rng.integers(len(HOWS))),          # 4 어떻게 쓸지
        'learn': int(rng.integers(len(LEARNS))),      # 5 어떻게 조정할지
        'err': int(rng.integers(0, TOTAL)),           # 5 델타·예측 부호화가 "정답"으로 볼 칸
    }


def clamp(v):
    return -CLIP if v < -CLIP else CLIP if v > CLIP else v


def step(g, info):
    """교류 하나: 읽고 → 연산 → 쓰고 → 바로 조정. (info는 파이썬 리스트, 빠르게 하려고)"""
    n, op, x0, w0 = g['n'], g['op'], g['x'], g['w']
    # 읽기 + 연산. par = 파라미터 쪽 (칸, 기울기), inp = 입력 쪽 (칸, 기울기) [Claude 임시: 조정 방향에 씀]
    if op == 0:                                   # 뉴런식 y = x·w + b
        x = info[x0:x0 + n]; w = info[w0:w0 + n]
        y = sum(a * b for a, b in zip(x, w)) + info[g['b']]
        par = [(w0 + k, x[k]) for k in range(n)] + [(g['b'], 1.0)]   # w, b
        inp = [(x0 + k, w[k]) for k in range(n)]                     # x
    elif op == 1:                                 # ReLU y = max(0, x), 한 칸 [Claude 임시]
        xv = info[x0]
        y = xv if xv > 0 else 0.0
        par = inp = [(x0, 1.0 if xv > 0 else 0.0)]
    else:                                         # 국소 정규화 y = x0 / 구간의 제곱평균 제곱근 [Claude 임시]
        x = info[x0:x0 + n]
        r = (sum(v * v for v in x) / n + 1e-6) ** 0.5
        y = x[0] / r
        par = inp = [(x0, 1.0 / r)]               # [Claude 임시] 첫 칸만 조정

    # 쓰기
    a = g['write']; old = info[a]; how = g['how']
    info[a] = clamp(y if how == 0 else old + y if how == 1 else 0.5 * (old + y))

    # 쓴 직후 조정 (입력 칸은 안 고침)
    L = g['learn']
    if L == 0:   s, target = y, par                            # 헵: 결과 방향으로 더 키움
    elif L == 1: s, target = -y, par                           # 역헵: 반대
    else:
        e = info[g['err']] - y                                 # 오차 = 진화가 고른 칸 - 결과
        s, target = (e, par) if L == 2 else (e, inp)           # 델타: 파라미터 / 예측 부호화: 입력 쪽 [Claude 임시]
    k = ETA * s
    for i, gr in target:
        if i >= S0:
            info[i] = clamp(info[i] + k * gr)


def cost_of(genes):
    """비용 = 자모 하나에 읽는 칸 수 합 [Claude 임시]."""
    c = 0
    for g in genes:
        c += (2 * g['n'] + 1) if g['op'] == 0 else 1 if g['op'] == 1 else g['n']
        c += 1 if g['learn'] >= 2 else 0
    return max(c, 1)


def life(args):
    """한 삶: 처음부터 한 번 읽으며 다음 자모를 맞힘. 뒤 절반 정답률(%)."""
    genes, seq = args
    info = [0.0] * TOTAL
    info[S0:O0] = np.random.default_rng(0).normal(0, 0.1, ST).tolist()   # [Claude 임시] 상태 시작값
    seq = seq.tolist()
    half, hits = len(seq) // 2, 0
    for t in range(len(seq) - 1):
        info[:IN] = [0.0] * IN; info[seq[t]] = 1.0               # 입력: 그 자모만 1
        if A.clear_out:
            info[O0:] = [0.0] * OUT
        for g in genes:
            step(g, info)
        if t >= half:
            o = info[O0:]
            hits += int(o.index(max(o)) == seq[t + 1])             # 출력: 가장 큰 칸
    return 100.0 * hits / (len(seq) - 1 - half)


def bigram(seq):
    """비교군: 같은 조각을 처음부터 세는 2-gram, 뒤 절반 정답률."""
    c, half, hits = np.zeros((28, 28)), len(seq) // 2, 0
    for t in range(len(seq) - 1):
        if t >= half:
            hits += int(c[seq[t]].argmax() == seq[t + 1])
        c[seq[t], seq[t + 1]] += 1
    return 100.0 * hits / (len(seq) - 1 - half)


def mutate(genes, rng):
    """[Claude 임시] 자식 = 부모 복사 + 칸마다 5% 다시 뽑기 + 교류 추가/삭제."""
    out = []
    for g in genes:
        g = dict(g)
        if rng.random() < 0.1:
            fresh = new_gene(rng)
            for k in g:
                if rng.random() < 0.3:
                    g[k] = fresh[k]
            # 구간이 끝을 넘지 않게
            g['x'] = min(g['x'], TOTAL - g['n']); g['w'] = min(g['w'], TOTAL - g['n'])
        out.append(g)
    if rng.random() < 0.3:
        if len(out) < MAXG:
            out.insert(int(rng.integers(len(out) + 1)), new_gene(rng))
    if len(out) > 1 and rng.random() < 0.3:
        out.pop(int(rng.integers(len(out))))
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(A.seed)
    data = np.load(A.data).astype(np.int64)
    cut = int(len(data) * 0.9)                    # [Claude 임시] 앞 90% 학습 조각, 뒤 10% 확인용
    # 확인용 글: 한 삶과 같은 길이 조각 4개 평균 (11-19 저녁 고침: 5000자모 한 판은 2000자모 삶에 맞춰진 개체가
    # 2000자모 뒤에 무너져서 실력보다 낮게 나왔음) [Claude 임시]
    vals = [data[cut + k * A.life: cut + (k + 1) * A.life] for k in range(4)]
    os.makedirs(A.out, exist_ok=True)
    log = open(os.path.join(A.out, 'log.jsonl'), 'a', encoding='utf-8')

    pop = [[new_gene(rng) for _ in range(40)] for _ in range(ENTITY)]   # [Claude 임시] 처음 교류 40줄
    gen0 = 0
    if A.resume:                                                         # 이어 돌리기
        saved = json.load(open(A.resume))
        pop, gen0 = saved['pop'], saved['gen']
        rng = np.random.default_rng(A.seed + gen0)
    best_val = (0.0, -1)
    t0, val_bi = time.time(), float(np.mean([bigram(v) for v in vals]))
    with Pool(A.workers) as pool:
        for gen in range(gen0, A.gens):
            s = int(rng.integers(0, cut - A.life))
            chunk = data[s:s + A.life]                                     # 세대마다 모두 같은 조각
            acc = np.array(pool.map(life, [(p, chunk) for p in pop]))
            cost = np.array([cost_of(p) for p in pop], float)
            # 정후 (11-19 밤): 잘함 = 정답률 - 비용/4 → 25% 벽에서 쪼그라듦 → 정후 "네가 생각해봐"
            fit = acc - cost / 4 if A.fit == 'cost4' else acc.copy()      # [Claude 임시] 오늘 밤은 정답률만
            order = np.argsort(-fit)
            best = pop[order[0]]
            rec = {'gen': gen, 'sec': round(time.time() - t0), 'acc_best': acc[order[0]], 'acc_mean': acc.mean(),
                   'acc_max': acc.max(), 'cost_best': cost[order[0]], 'len_best': len(best), 'bigram': bigram(chunk),
                   'ops': np.bincount([g['op'] for g in best], minlength=3).tolist(),
                   'learns': np.bincount([g['learn'] for g in best], minlength=4).tolist()}
            if gen % 10 == 0:
                rec['val'] = float(np.mean(pool.map(life, [(best, v) for v in vals]))); rec['val_bigram'] = val_bi
                json.dump(best, open(os.path.join(A.out, 'best.json'), 'w'))
            if gen % 30 == 0:                                              # 정후: 30세대마다 부검할 수 있게
                json.dump({'gen': gen, 'acc': acc.tolist(), 'cost': cost.tolist(), 'fit': fit.tolist(), 'pop': pop},
                          open(os.path.join(A.out, f'gen_{gen:06d}.json'), 'w'))
            log.write(json.dumps(rec, ensure_ascii=False) + '\n'); log.flush()
            print(rec, flush=True)
            if 'val' in rec:                                               # 사람이 보는 상황판
                if rec['val'] > best_val[0]:
                    best_val = (rec['val'], gen)
                    json.dump(best, open(os.path.join(A.out, 'best_val.json'), 'w'))
                h = int(time.time() - t0) // 60
                line = (f"{time.strftime('%m-%d %H:%M')} | {gen}세대 | {h // 60}시간 {h % 60}분 | 확인용 글 {rec['val']:.1f}% "
                        f"(최고 {best_val[0]:.1f}%, {best_val[1]}세대) | 1등 {len(best)}줄 | 같은 글 2-gram {rec['val_bigram']:.1f}%\n")
                open(os.path.join(A.out, '상황판.txt'), 'a', encoding='utf-8').write(line)
            if time.time() - t0 > A.hours * 3600:
                break
            parents = [pop[i] for i in order[:10]]                         # [Claude 임시] 위 10마리가 부모
            pop = parents + [mutate(parents[int(rng.integers(10))], rng) for _ in range(ENTITY - 10)]

# ---------- [Claude 임시] 목록 ----------
# 정보 칸을 한 줄(312)로 이음 / 신경 연산 세 가지(뉴런식·ReLU·국소 정규화)와 그 식 / 쓰는 방식 세 가지
# 구간 크기 1~8 / 조정 세기 0.01 / 값 자르기 ±10 / 헵·역헵·델타·예측 부호화의 구체 식
# 비용 = 읽는 칸 수 / 한 삶 2000자모, 뒤 절반 채점 / 상태 시작값 / 출력 칸은 안 지움
# 처음 40줄 / 위 10마리 부모, 돌연변이 방식 / 학습 90%·확인용 5000자모
