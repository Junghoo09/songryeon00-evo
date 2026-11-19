# 한국어 글(.txt) → 자모 번호(0~27) .npy
# 사용: python make_npy.py 글.txt 자모.npy
import sys, numpy as np, jamo
nums = jamo.encode(open(sys.argv[1], encoding='utf-8').read())[0]
a = np.array([x - 1 for x in nums], dtype=np.int8)
np.save(sys.argv[2], a)
print(len(a), '자모')
