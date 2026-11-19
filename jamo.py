"""
번호표 (2026-11-06~08 정후 설계)
- 자음 1~14, 모음 15~24, 글자 경계 25, 띄어쓰기 26, 줄 바꿈 27, 말 끝내기 28
- 겹자모는 쪼갠다. 자모는 자리와 상관없이 같은 번호 (초성 ㄴ = 받침 ㄴ).
- 글자가 끝날 때마다 글자 경계 25.
"""
CONS = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ'
VOWS = 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ'
TABLE = {c: i + 1 for i, c in enumerate(CONS + VOWS)}   # ㄱ=1 … ㅣ=24
BOUND, SPACE, NEWLINE, END = 25, 26, 27, 28
N_SYMBOLS = 28

SPLIT = {  # 겹자모 → 쪼갠 자모
    'ㄲ': 'ㄱㄱ', 'ㄸ': 'ㄷㄷ', 'ㅃ': 'ㅂㅂ', 'ㅆ': 'ㅅㅅ', 'ㅉ': 'ㅈㅈ',
    'ㅐ': 'ㅏㅣ', 'ㅒ': 'ㅑㅣ', 'ㅔ': 'ㅓㅣ', 'ㅖ': 'ㅕㅣ', 'ㅘ': 'ㅗㅏ', 'ㅙ': 'ㅗㅏㅣ',
    'ㅚ': 'ㅗㅣ', 'ㅝ': 'ㅜㅓ', 'ㅞ': 'ㅜㅓㅣ', 'ㅟ': 'ㅜㅣ', 'ㅢ': 'ㅡㅣ',
    'ㄳ': 'ㄱㅅ', 'ㄵ': 'ㄴㅈ', 'ㄶ': 'ㄴㅎ', 'ㄺ': 'ㄹㄱ', 'ㄻ': 'ㄹㅁ', 'ㄼ': 'ㄹㅂ',
    'ㄽ': 'ㄹㅅ', 'ㄾ': 'ㄹㅌ', 'ㄿ': 'ㄹㅍ', 'ㅀ': 'ㄹㅎ', 'ㅄ': 'ㅂㅅ',
}
# 완성형 글자(가~힣)를 초성·중성·종성으로 나눌 때 쓰는 순서
CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
JUNG = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
JONG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')


def letter(ch):
    """한 글자 → 자모 번호들 (끝에 글자 경계 25). 번호표에 없는 글자면 None."""
    c = ord(ch) - 0xAC00
    if 0 <= c < 11172:                                   # 완성형: 가~힣
        parts = CHO[c // 588] + JUNG[c % 588 // 28] + JONG[c % 28]
    elif ch in TABLE or ch in SPLIT:                     # 낱자 (ㅋㅋ, ㅠㅠ 등)도 한 글자로 본다
        parts = ch
    else:
        return None
    return [TABLE[j] for p in parts for j in SPLIT.get(p, p)] + [BOUND]


def encode(text):
    """글 → 번호 목록. 번호표에 없는 글자(영어·숫자·문장부호·이모지)는 버린다.
    버린 글자 수도 돌려준다."""
    out, dropped = [], 0
    for ch in text:
        if ch == ' ':
            if out and out[-1] != SPACE:                 # 띄어쓰기가 겹치면 하나만
                out.append(SPACE)
        elif ch == '\n':
            out.append(NEWLINE)
        else:
            nums = letter(ch)
            if nums is None:
                dropped += 1
            else:
                out += nums
    return out, dropped


def decode(nums):
    """번호 목록 → 사람이 읽는 글 (자모를 그대로 늘어놓는다. 확인용)."""
    names = {v: k for k, v in TABLE.items()}
    names.update({BOUND: '|', SPACE: ' _ ', NEWLINE: ' ⏎ ', END: ' ■ '})
    return ''.join(names.get(n, '?') for n in nums)


if __name__ == '__main__':
    print(encode('닭'))           # ([3, 15, 4, 1, 25], 0)
    print(decode(encode('투룸에서 살아보고싶다 ㅋㅋ!')[0]))
