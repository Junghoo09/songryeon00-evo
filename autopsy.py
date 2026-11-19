# 송련00 밤 실험 부검 (Claude 작성, 11-19 밤, 정후 "지금 부검 ㄱㄱ")
# 사용: python autopsy.py 자모.npy best.json
import sys, json, collections
import numpy as np
data_path, best_path = sys.argv[1], sys.argv[2]
sys.argv = ['run.py', '--data', data_path]          # run.py의 함수를 그대로 씀 (출력 지우기 켬)
import run

NAMES = list('ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ') + ['|', '_', '↵', '■']
data = np.load(data_path).astype(np.int64)
cut = int(len(data) * 0.9)
chunks = [data[cut + k * 5000: cut + (k + 1) * 5000] for k in range(3)]   # 확인용 글 3조각
genes = json.load(open(best_path))


def predict(genes, seq):
    """life와 같은 계산인데 예측을 전부 돌려줌."""
    info = [0.0] * run.TOTAL
    info[run.S0:run.O0] = np.random.default_rng(0).normal(0, 0.1, run.ST).tolist()
    seq = seq.tolist(); pr = []
    for t in range(len(seq) - 1):
        info[:run.IN] = [0.0] * run.IN; info[seq[t]] = 1.0
        info[run.O0:] = [0.0] * run.OUT
        for g in genes:
            run.step(g, info)
        o = info[run.O0:]; pr.append(o.index(max(o)))
    return seq, pr


def score(genes):
    tot = []
    for c in chunks:
        seq, pr = predict(genes, c); h = len(seq) // 2
        tot.append(100 * np.mean([pr[t] == seq[t + 1] for t in range(h, len(seq) - 1)]))
    return float(np.mean(tot))


base = score(genes)
print(f'교류 {len(genes)}줄, 비용 {run.cost_of(genes)}, 확인용 3조각 평균 {base:.2f}%')

# 1. 무엇을 맞히나
seq, pr = predict(genes, chunks[0]); h = len(seq) // 2
said = collections.Counter(NAMES[pr[t]] for t in range(h, len(seq) - 1))
hit = collections.Counter(NAMES[seq[t + 1]] for t in range(h, len(seq) - 1) if pr[t] == seq[t + 1])
n = len(seq) - 1 - h
print('\n[1] 예측한 자모 (비율) :', ', '.join(f'{k} {100*v/n:.1f}' for k, v in said.most_common(8)))
print('    맞힌 것 (전체 맞힘 중 비율):', ', '.join(f'{k} {100*v/sum(hit.values()):.1f}' for k, v in hit.most_common(8)))

# 2. 교류 모양
print('\n[2] 연산:', dict(collections.Counter(run.OPS[g['op']] for g in genes)),
      '/ 조정:', dict(collections.Counter(run.LEARNS[g['learn']] for g in genes)),
      '/ 쓰기:', dict(collections.Counter(run.HOWS[g['how']] for g in genes)))
outw = [g for g in genes if g['write'] >= run.O0]
print('    출력 칸에 쓰는 줄', len(outw), '→ 칸:', sorted(NAMES[g['write'] - run.O0] for g in outw))
reads_in = sum(1 for g in genes if g['x'] < run.IN or (g['op'] == 0 and (g['w'] < run.IN or g['b'] < run.IN)))
print('    입력 칸을 읽는 줄', reads_in)

# 3. 조정 끄기
eta = run.ETA; run.ETA = 0.0
print(f'\n[3] 조정(학습) 전부 끄기: {score(genes):.2f}% (원래 {base:.2f})')
run.ETA = eta

# 4. 한 줄씩 빼기
print('\n[4] 한 줄씩 빼기 (차이가 0.3%p 넘는 줄만)')
for i, g in enumerate(genes):
    d = score(genes[:i] + genes[i + 1:]) - base
    if abs(d) > 0.3:
        w = g['write']; wn = ('출력 ' + NAMES[w - run.O0]) if w >= run.O0 else f'상태 {w}'
        print(f'  {i:3d}줄 {d:+6.2f} | {run.OPS[g["op"]]} n={g["n"]} x={g["x"]} w={g["w"]} b={g["b"]} → {wn} {run.HOWS[g["how"]]}, '
              f'{run.LEARNS[g["learn"]]} err={g["err"]}')
