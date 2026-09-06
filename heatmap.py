# -*- coding: utf-8 -*-
"""
heatmap.py — 옵타식 "Predicted Final League Position" 히트맵 (K리그1 · K리그2)
==============================================================================
k1_analysis.py / k2_analysis.py 의 시뮬레이션을 재사용해 전 팀 × 최종순위 확률
매트릭스를 만들고, Opta Analyst 스타일 표로 렌더링한다. 두 리그를 각각 PNG로 저장.

  python heatmap.py   ->  heatmap_k1.png , heatmap_k2.png
"""

import io
import sys
import importlib
from datetime import date

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

N_SIMS = 100_000
SEED = 1                        # 이미지 수치 안정화용

CMAP = LinearSegmentedColormap.from_list(
    "opta_pink", ["#ffffff", "#fde4ee", "#f7a8c4", "#e0538a", "#b0106a"])

# 옵타 방법 설명 (그림 하단에 표기)
METHOD_LINES = [
    "데이터: kleague.com  ·  getScheduleList.do (일정·결과 JSON), teamRank.do (순위 JSON)",
    "옵타식 방법  ①경기별 Elo(Power Rankings): 2024~26 K1·K2 전 경기를 시간순 처리, 상대강도·최근성·득점차(MOV) 반영",
    "               ②포아송 득점모델: Elo차 → 홈/원정 기대득점(λ) → 스코어 샘플링    "
    f"③몬테카를로 {N_SIMS:,}회로 최종순위 분포 산출",
]


def render(mod, out, cut_lines):
    """mod = 리그 분석 모듈(k1_analysis/k2_analysis). cut_lines = 경계선 정의 리스트."""
    teams = list(mod.TEAMS.keys())          # 현재 순위순
    n = len(teams)
    counts = mod.simulate_position_matrix(N_SIMS, seed=SEED)
    P = np.array([[counts[t][pos] / N_SIMS * 100 for pos in range(1, n + 1)]
                  for t in teams])

    norm = PowerNorm(gamma=0.65, vmin=0, vmax=max(50.0, P.max()))
    cell_fs = 7.3 if n <= 12 else 6.1
    fig_w = 2.8 + 0.82 * n
    fig_h = 2.4 + 0.52 * n
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(P, cmap=CMAP, norm=norm, aspect="auto")

    # 셀 격자
    for g in np.arange(-0.5, n, 1):
        ax.vlines(g, -0.5, n - 0.5, color="white", lw=1.3)
        ax.hlines(g, -0.5, n - 0.5, color="white", lw=1.3)

    # 경계선 (승격/스플릿/아시아행 컷)
    for c in cut_lines:
        x = c["after"] - 0.5
        ax.vlines(x, -0.5, n - 0.5, color=c["color"], lw=c["lw"],
                  linestyles=(0, (4, 2)) if c.get("dashed") else "solid")
        if c.get("label"):
            if c.get("pos", "top") == "top":
                ax.text(x, -0.44, c["label"], ha=c.get("ha", "right"), va="bottom",
                        fontsize=8, color=c["color"], weight="bold")
            else:
                ax.text(x, n - 0.44, c["label"], ha=c.get("ha", "left"), va="top",
                        fontsize=8, color=c["color"], weight="bold")

    # 셀 확률 텍스트 (0.1% 이상만)
    for i in range(n):
        for j in range(n):
            v = P[i, j]
            if v >= 0.1:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=cell_fs,
                        color="white" if norm(v) > 0.55 else "#40232f")

    # 왼쪽 라벨 (클럽 / POS / GP / PTS)
    X_TEAM, X_POS, X_GP, X_PTS, HEAD_Y = -2.85, -2.15, -1.55, -0.92, -1.12
    for x, h in ((X_TEAM, "클럽"), (X_POS, "POS"), (X_GP, "GP"), (X_PTS, "PTS")):
        ax.text(x, HEAD_Y, h, ha="right" if x == X_TEAM else "center", va="center",
                fontsize=8.5, weight="bold", color="#555")
    for j in range(n):
        ax.text(j, HEAD_Y, str(j + 1), ha="center", va="center",
                fontsize=8.3, weight="bold", color="#555")
    for i, t in enumerate(teams):
        pts, gp, gf, ga, wins = mod.TEAMS[t]
        is_target = (t == mod.TARGET_TEAM)
        ax.text(X_TEAM, i, t, ha="right", va="center", fontsize=9.3,
                color="#5b1a8b" if is_target else "#222",
                weight="bold" if is_target else "normal")
        ax.text(X_POS, i, str(i + 1), ha="center", va="center", fontsize=8.3, color="#666")
        ax.text(X_GP, i, str(gp), ha="center", va="center", fontsize=8.3, color="#666")
        ax.text(X_PTS, i, str(pts), ha="center", va="center", fontsize=8.3,
                color="#333", weight="bold")

    ax.set_xlim(-3.25, n - 0.5)
    ax.set_ylim(n - 0.5, -1.7)
    ax.axis("off")

    # 제목/부제
    fig.text(0.012, 0.965, "Predicted Final League Position",
             fontsize=19, weight="bold", color="#1a1a1a")
    fig.text(0.012, 0.932,
             f"{mod.LEAGUE_NAME}  |  {date.today():%Y-%m-%d}  ·  "
             f"옵타식 모델(경기별 Elo + 포아송)",
             fontsize=10.5, color="#777")
    fig.text(0.988, 0.965, "Opta식 재현", fontsize=13, weight="bold",
             color="#c2185b", ha="right")

    # 하단: 데이터 출처 + 옵타 방법 설명
    y0 = 0.008
    dy = 0.020
    for k, line in enumerate(reversed(METHOD_LINES)):
        ax.figure.text(0.012, y0 + k * dy, line, fontsize=7.4,
                       color="#8a8a8a" if k else "#aaaaaa")

    bottom_margin = 0.02 + dy * len(METHOD_LINES) + 0.02
    plt.subplots_adjust(left=0.02, right=0.995, top=0.90, bottom=bottom_margin)
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"저장 완료: {out}")
    return P, teams


def main():
    print("K리그1 분석 로드/시뮬레이션...")
    k1 = importlib.import_module("k1_analysis")
    render(k1, "heatmap_k1.png", cut_lines=[
        {"after": 5, "color": "#7b1fa2", "lw": 2.2, "dashed": True,
         "label": "← 아시아행(1~5위)", "pos": "top", "ha": "right"},
        {"after": 6, "color": "#1a1a1a", "lw": 2.6, "dashed": False,
         "label": "파이널 A | B →", "pos": "bottom", "ha": "left"},
    ])

    print("K리그2 분석 로드/시뮬레이션...")
    k2 = importlib.import_module("k2_analysis")
    render(k2, "heatmap_k2.png", cut_lines=[
        {"after": 2, "color": "#1565c0", "lw": 2.6, "dashed": False,
         "label": "← 자동승격(1~2위)", "pos": "top", "ha": "right"},
        {"after": 6, "color": "#7b1fa2", "lw": 2.2, "dashed": True,
         "label": "플레이오프권(3~6위) |", "pos": "bottom", "ha": "left"},
    ])


if __name__ == "__main__":
    main()
