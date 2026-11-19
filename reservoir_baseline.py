# 저수지 비교 (11-19 밤): 송련00 a 판과 같은 조건 = 확인용 글 2000자모 조각 4개, 처음부터 한 번 읽으며 먼저 맞히고 배움, 뒤 절반 채점
# 저수지: 지금 자모(28칸 원-핫) + 고정 무작위 되먹임(크기 0.9) -> tanh 은닉 h개. 출력 28칸 = W h + b, softmax, 델타 (정답 - 확률) x h
import sys, numpy as np
d = np.load(sys.argv[1])  # 사용: python reservoir_baseline.py 자모.npy
d = d.astype(np.int64); c = int(len(d) * .9)
P = [d[c + k * 2000: c + (k + 1) * 2000] for k in range(4)]
def res(seq, h, lr, seed):
    r = np.random.default_rng(seed)
    Win = r.normal(0, 1, (h, 28)); Wr = r.normal(0, 1, (h, h)); Wr *= 0.9 / max(abs(np.linalg.eigvals(Wr)))
    W = np.zeros((28, h)); b = np.zeros(28); s = np.zeros(h); half = len(seq) // 2; hit = 0
    for t in range(len(seq) - 1):
        x = np.zeros(28); x[seq[t]] = 1; s = np.tanh(Win @ x + Wr @ s)
        z = W @ s + b; p = np.exp(z - z.max()); p /= p.sum()
        if t >= half: hit += p.argmax() == seq[t + 1]
        e = -p; e[seq[t + 1]] += 1; W += lr * np.outer(e, s); b += lr * e
    return 100 * hit / (len(seq) - 1 - half)
for h in (40, 200, 256):
    for lr in (0.01, 0.03, 0.1, 0.3):
        v = [res(p, h, lr, sd) for p in P for sd in (0, 1)]
        print(f'저수지 {h:3d} lr {lr:<5} 평균 {np.mean(v):.1f}  (조각별 {[round(x,1) for x in v[::2]]})', flush=True)