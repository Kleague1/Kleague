# -*- coding: utf-8 -*-
"""
K리그2 최종순위 예측 — 옵타식 몬테카를로 (승격 확률)
==============================================================================
- 실력 : (1) 경기별 Elo (Opta Power Rankings식) — 2024~2026 K1+K2 전 경기를 시간순으로
             돌려 상대강도·최근성·득점차(MOV)를 반영 (승강팀 이력까지 연속 반영)
         (2) 포아송 득점 모델 — Elo차 → 홈/원정 기대득점(λ) → 스코어 샘플
- 데이터: kleague.com 공식 JSON 실시간 수집 (kleague_data.py)

K리그2 구조: 스플릿 없음. 단일 리그(2026년 17팀·34R)로 끝까지 진행 후 순위 확정.
  승격: 1·2위 자동승격 / 3~6위 승격 플레이오프(3-6위·4-5위 준PO → PO 승자가 K1 11위와 승강PO).

단독 실행: python k2_analysis.py   (heatmap.py 가 이 모듈을 import 해 그림을 만든다)
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
LEAGUE = "2"
LEAGUE_NAME = "K리그2 2026"
N_SIMS = 100_000
RANDOM_SEED = None

TARGET_TEAM = None             # K2는 특정 대상팀 없음 (전 팀 승격 확률)
LIVE_YEAR = 2026
ELO_YEARS = (2024, 2025, 2026)

HAS_SPLIT = False              # K2는 스플릿 없음
PROMO_AUTO = 2                 # 1~2위: 자동승격
PROMO_PO_LO = 3               # 3~6위: 승격 플레이오프권 (4팀)
PROMO_PO_HI = 6

# Opta식 경기별 Elo
ELO_START, ELO_K, SEASON_REGRESS, HFA_ELO, USE_MOV = 1500, 20, 0.30, 55, True
# 포아송 득점 모델
K_GOALS, LEAGUE_TOTAL_GOALS = 0.0040, None

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
except Exception as e:
    print(f"[데이터 수집 실패] {e}\n인터넷 연결을 확인하세요.")
    sys.exit(1)

NUM_TEAMS = len(TEAMS)
REMAINING_FIXTURES = [(h, a) for h, a, r in _remaining]   # K2는 전부 정규(스플릿 없음)

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
        if m["year"] != cur_year:
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
        if USE_MOV and m["hg"] != m["ag"]:
            k *= math.log(abs(m["hg"] - m["ag"]) + 1) * (2.2 / (0.001 * abs(diff) + 2.2))
        delta = k * (actual - exp_home)
        ratings[home] += delta
        ratings[away] -= delta
    return ratings


ELO = train_elo(MATCHES)
for t in TEAMS:
    ELO.setdefault(t, ELO_START)


def poisson(lam):
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def expected_goals(home, away):
    sup = ((ELO[home] + HFA_ELO) - ELO[away]) * K_GOALS
    return max(0.12, LEAGUE_HALF + sup / 2), max(0.12, LEAGUE_HALF - sup / 2)


def rank(teams, pts, gf, ga, wins):
    """K리그 규정 순: 승점 -> 다득점 -> 득실차 -> 다승 (동률 시 무작위)."""
    return sorted(teams,
                  key=lambda t: (pts[t], gf[t], gf[t] - ga[t], wins[t], random.random()),
                  reverse=True)


def simulate_once():
    """시즌 1회 -> {팀: 최종순위}. K2는 스플릿 없이 남은 경기 시뮬 후 전체 순위."""
    pts = {t: TEAMS[t][0] for t in TEAMS}
    gf = {t: TEAMS[t][2] for t in TEAMS}
    ga = {t: TEAMS[t][3] for t in TEAMS}
    wins = {t: TEAMS[t][4] for t in TEAMS}

    for home, away in REMAINING_FIXTURES:
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

    final = rank(list(TEAMS), pts, gf, ga, wins)
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
# [실행] — 승격 확률 리포트
# =========================================================
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print(f"[{LEAGUE_NAME}] {NUM_TEAMS}팀 / 남은경기 {len(REMAINING_FIXTURES)} / "
          f"Elo훈련 {len(MATCHES)}경기 / 평균득점 {LEAGUE_TOTAL_GOALS:.2f}")

    print("\n[Opta식 Power Ranking (경기별 Elo)]")
    for t in sorted(TEAMS, key=lambda x: ELO[x], reverse=True):
        print(f"  {ELO[t]:6.0f}  {t}")

    counts = simulate_position_matrix(N_SIMS)

    print("\n" + "=" * 52)
    print(f" K리그2 승격 확률 (옵타식 Elo+포아송, {N_SIMS:,}회)")
    print("=" * 52)
    print(f"\n  {'순위 팀':<10} {'승점':>4}   {'우승(1위)':>8} {'자동승격(1~2위)':>12} {'플레이오프권(3~6위)':>16}")
    for i, t in enumerate(TEAMS, 1):                    # 현재 순위순
        pts = TEAMS[t][0]
        p1 = counts[t][1] / N_SIMS * 100
        p_auto = sum(counts[t][1:PROMO_AUTO + 1]) / N_SIMS * 100
        p_po = sum(counts[t][PROMO_PO_LO:PROMO_PO_HI + 1]) / N_SIMS * 100
        print(f"  {i:2d} {t:<7} {pts:>4}   {p1:>7.1f}% {p_auto:>12.1f}% {p_po:>15.1f}%")


if __name__ == "__main__":
    main()
