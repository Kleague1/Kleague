# -*- coding: utf-8 -*-
"""
K리그1 최종순위 예측 — 옵타식 몬테카를로 (대전 하나 시티즌 아시아 진출 확률)
==============================================================================
- 실력 : (1) 경기별 Elo (Opta Power Rankings식) — 2024~2026 K1+K2 전 경기를 시간순으로
             돌려 상대강도·최근성·득점차(MOV)를 반영 (승강팀 이력까지 연속 반영)
         (2) 포아송 득점 모델 — Elo차 → 홈/원정 기대득점(λ) → 스코어 샘플
         (3) 맞대결(H2H) 보정 (선택)
- 데이터: kleague.com 공식 JSON 실시간 수집 (kleague_data.py)

K리그1 스플릿 구조 (이 파일은 두 국면을 라운드로 자동 판별)
  33R까지(스플릿 전): 33R 종료 시 상위 6팀=파이널 A / 하위 6팀=파이널 B.
                     핵심은 "6위 vs 7위" 경계(파이널 A 진입).
  34~38R(스플릿 후) : 그룹 내 5경기. 상위 6팀=1~6위, 하위 6팀=7~12위로 고정.
                     핵심은 "파이널 A 안에서 5위 이상(아시아행) vs 6위".

단독 실행: python k1_analysis.py   (heatmap.py 가 이 모듈을 import 해 그림을 만든다)
"""

import io
import sys
import math
import random
from collections import defaultdict

import kleague_data as kl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# =========================================================
# [설정]
# =========================================================
LEAGUE = "1"
LEAGUE_NAME = "하나은행 K리그1 2026"
N_SIMS = 100_000
RANDOM_SEED = None

TARGET_TEAM = "대전"
LIVE_YEAR = 2026
ELO_YEARS = (2024, 2025, 2026)
H2H_YEARS = (2024, 2025, 2026)

TOTAL_REGULAR_ROUNDS = 33      # 이 라운드까지 정규, 그 다음이 파이널 스플릿
FINAL_A_SIZE = 6               # 파이널 A(상위 그룹) 크기
HAS_SPLIT = True

# Opta식 경기별 Elo
ELO_START, ELO_K, SEASON_REGRESS, HFA_ELO, USE_MOV = 1500, 20, 0.30, 55, True
# 포아송 득점 모델
K_GOALS, LEAGUE_TOTAL_GOALS = 0.0040, None
# 맞대결 보정
USE_H2H, H2H_SCALE, H2H_BASE_PPG, H2H_SHRINK_K, H2H_CAP = True, 80, 1.4, 5, 60

QUALIFICATION = {
    1: "ACLE 본선 직행", 2: "ACLE 본선 직행", 3: "ACLE 본선 직행",
    4: "ACLE 플레이오프", 5: "ACL Two",
}

# =========================================================
# [데이터 수집]
# =========================================================
try:
    TEAMS = kl.fetch_standings(LEAGUE)                 # 공식 순위 {팀:(pts,g,gf,ga,w)}
    _remaining = kl.fetch_remaining_fixtures(LIVE_YEAR, LEAGUE)
    # Elo 훈련용: K1+K2 전 경기 통합(승강팀 이력 연속) — 시간순
    MATCHES = sorted(kl.fetch_completed_matches(ELO_YEARS, "1")
                     + kl.fetch_completed_matches(ELO_YEARS, "2"),
                     key=lambda m: (m["year"], m["date"]))
    HEAD_TO_HEAD = kl.fetch_h2h(TARGET_TEAM, H2H_YEARS, LEAGUE)
except Exception as e:
    print(f"[데이터 수집 실패] {e}\n인터넷 연결을 확인하세요.")
    sys.exit(1)

NUM_TEAMS = len(TEAMS)
REGULAR_FIXTURES = [(h, a) for h, a, r in _remaining if r <= TOTAL_REGULAR_ROUNDS]
FINALS_FIXTURES = [(h, a) for h, a, r in _remaining if r > TOTAL_REGULAR_ROUNDS]
PRE_SPLIT = bool(REGULAR_FIXTURES)
HEAD_TO_HEAD = {k: v for k, v in HEAD_TO_HEAD.items() if k[1] in TEAMS}

if LEAGUE_TOTAL_GOALS is None:
    _cur = [m for m in MATCHES if m["year"] == LIVE_YEAR and m["home"] in TEAMS]
    LEAGUE_TOTAL_GOALS = (sum(m["hg"] + m["ag"] for m in _cur) / len(_cur)) if _cur else 2.6
LEAGUE_HALF = LEAGUE_TOTAL_GOALS / 2.0


# =========================================================
# [모델]
# =========================================================
def train_elo(matches):
    """전 경기를 시간순으로 처리해 팀별 현재 Elo 산출 (Opta Power Rankings식)."""
    ratings = {}
    cur_year = None
    for m in matches:
        if cur_year is None:
            cur_year = m["year"]
        if m["year"] != cur_year:                      # 시즌 경계 -> 평균 회귀
            for t in ratings:
                ratings[t] = ELO_START + (ratings[t] - ELO_START) * (1 - SEASON_REGRESS)
            cur_year = m["year"]
        home, away = m["home"], m["away"]
        ratings.setdefault(home, ELO_START)
        ratings.setdefault(away, ELO_START)
        diff = (ratings[home] + HFA_ELO) - ratings[away]
        exp_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        actual = 1.0 if m["hg"] > m["ag"] else (0.5 if m["hg"] == m["ag"] else 0.0)
        k = ELO_K
        if USE_MOV and m["hg"] != m["ag"]:             # 538식 득점차 가중
            k *= math.log(abs(m["hg"] - m["ag"]) + 1) * (2.2 / (0.001 * abs(diff) + 2.2))
        delta = k * (actual - exp_home)
        ratings[home] += delta
        ratings[away] -= delta
    return ratings


ELO = train_elo(MATCHES)
for t in TEAMS:
    ELO.setdefault(t, ELO_START)


def h2h_delta(home, away):
    if (home, away) in HEAD_TO_HEAD:
        w, d, l = HEAD_TO_HEAD[(home, away)]
    elif (away, home) in HEAD_TO_HEAD:
        l, d, w = HEAD_TO_HEAD[(away, home)]
    else:
        return 0.0
    g = w + d + l
    if g == 0:
        return 0.0
    ppg = (3 * w + d) / g
    shrink = g / (g + H2H_SHRINK_K)
    return max(-H2H_CAP, min(H2H_CAP, (ppg - H2H_BASE_PPG) * H2H_SCALE * shrink))


def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def expected_goals(home, away):
    diff = (ELO[home] + HFA_ELO) - ELO[away]
    if USE_H2H:
        diff += h2h_delta(home, away)
    sup = diff * K_GOALS
    return max(0.12, LEAGUE_HALF + sup / 2), max(0.12, LEAGUE_HALF - sup / 2)


def round_robin(teams):
    pairs = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            if random.random() < 0.5:
                a, b = b, a
            pairs.append((a, b))
    return pairs


def rank(teams, pts, gf, ga, wins):
    """K리그 규정 순: 승점 -> 다득점 -> 득실차 -> 다승 (동률 시 무작위)."""
    return sorted(teams,
                  key=lambda t: (pts[t], gf[t], gf[t] - ga[t], wins[t], random.random()),
                  reverse=True)


def simulate_once():
    """시즌 1회 -> {팀: 최종순위}. 스플릿 전/후 자동 처리."""
    pts = {t: TEAMS[t][0] for t in TEAMS}
    gf = {t: TEAMS[t][2] for t in TEAMS}
    ga = {t: TEAMS[t][3] for t in TEAMS}
    wins = {t: TEAMS[t][4] for t in TEAMS}

    def play(home, away):
        lh, la = expected_goals(home, away)
        hg, ag = poisson(lh), poisson(la)
        gf[home] += hg; ga[home] += ag
        gf[away] += ag; ga[away] += hg
        if hg > ag:
            pts[home] += 3; wins[home] += 1
        elif hg == ag:
            pts[home] += 1; pts[away] += 1
        else:
            pts[away] += 3; wins[away] += 1

    if PRE_SPLIT:
        for home, away in REGULAR_FIXTURES:
            play(home, away)
        order = rank(list(TEAMS), pts, gf, ga, wins)
        group_a, group_b = order[:FINAL_A_SIZE], order[FINAL_A_SIZE:]
        finals = round_robin(group_a) + round_robin(group_b)
    else:
        # 스플릿 후: 공식 순위 상위 6팀이 파이널 A(1~6위 고정)
        group_a = list(TEAMS)[:FINAL_A_SIZE]
        group_b = list(TEAMS)[FINAL_A_SIZE:]
        finals = FINALS_FIXTURES

    for home, away in finals:
        play(home, away)

    final = rank(group_a, pts, gf, ga, wins) + rank(group_b, pts, gf, ga, wins)
    return {t: i + 1 for i, t in enumerate(final)}


def simulate_position_matrix(n_sims, seed=None):
    """전 팀 × 최종순위 카운트 -> {팀: [_, pos1수, ...]} (index 1..N)."""
    if seed is not None:
        random.seed(seed)
    counts = {t: [0] * (NUM_TEAMS + 1) for t in TEAMS}
    for _ in range(n_sims):
        for t, pos in simulate_once().items():
            counts[t][pos] += 1
    return counts


# =========================================================
# [실행] — 대전 진출 확률 리포트
# =========================================================
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    phase = "스플릿 전(정규 진행중)" if PRE_SPLIT else "스플릿 후(파이널 라운드)"
    print(f"[{LEAGUE_NAME}] {NUM_TEAMS}팀 / 남은경기 정규 {len(REGULAR_FIXTURES)}"
          f"·파이널 {len(FINALS_FIXTURES)} / Elo훈련 {len(MATCHES)}경기 / "
          f"평균득점 {LEAGUE_TOTAL_GOALS:.2f}")
    print(f"[국면] {phase}")

    print("\n[Opta식 Power Ranking (경기별 Elo)]")
    for t in sorted(TEAMS, key=lambda x: ELO[x], reverse=True):
        mark = f"  <-- {TARGET_TEAM}" if t == TARGET_TEAM else ""
        print(f"  {ELO[t]:6.0f}  {t}{mark}")

    pos_count = defaultdict(int)
    for _ in range(N_SIMS):
        pos_count[simulate_once()[TARGET_TEAM]] += 1

    print("\n" + "=" * 46)
    print(f" {TARGET_TEAM} 진출 확률 (옵타식 Elo+포아송, {N_SIMS:,}회)")
    print("=" * 46)
    print("\n[최종 순위 분포]")
    for pos in range(1, NUM_TEAMS + 1):
        p = pos_count[pos] / N_SIMS * 100
        if p >= 0.05:
            print(f"  {pos:2d}위 : {p:5.1f}%  {'#' * int(p / 2)}")

    def prob(lo, hi):
        return sum(pos_count[p] for p in range(lo, hi + 1)) / N_SIMS * 100

    p_final_a, p_asia = prob(1, 6), prob(1, 5)
    print("\n[핵심 지표]")
    if PRE_SPLIT:
        print(f"  파이널 A 진출(1~6위) : {p_final_a:5.1f}%   <- 33R 종료 시 분기점")
    print(f"  아시아행(1~5위)      : {p_asia:5.1f}%")
    print(f"  ACLE 본선 직행(1~3위): {prob(1, 3):5.1f}%")
    print(f"  4위 (ACLE PO)        : {prob(4, 4):5.1f}%")
    print(f"  5위 (ACL Two)        : {prob(5, 5):5.1f}%")
    if PRE_SPLIT and p_final_a > 0:
        print(f"  * 파이널 A 진입 가정 시 아시아행 조건부: {p_asia / p_final_a * 100:4.1f}%")


if __name__ == "__main__":
    main()
