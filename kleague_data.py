# -*- coding: utf-8 -*-
"""
K League 공식 JSON 표면에서 일정/결과를 가져오는 데이터 모듈.
- 일정·결과 JSON: https://www.kleague.com/getScheduleList.do   (월 단위, JSON 본문)

순위는 teamRank.do 대신 '경기 결과'로 직접 계산한다(standings_from_matches).
teamRank.do 는 K2 를 안정적으로 주지 않기 때문. getScheduleList.do 하나면
K1/K2 모두 순위·남은일정·과거결과(스코어)를 얻을 수 있다.

leagueId: "1" = K리그1, "2" = K리그2.
"""

import json
import urllib.request

BASE = "https://www.kleague.com"
SCHEDULE_API = "/getScheduleList.do"        # 일정·결과 JSON (POST, body에 year/month/leagueId)
RANK_API = "/record/teamRank.do"            # 순위 JSON (leagueId 는 '쿼리스트링'으로!)
_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/json",
}
_CACHE = {}                                  # (year, league) -> 경기 리스트 (프로세스 캐시)


def _post(path, body):
    req = urllib.request.Request(
        BASE + path, method="POST", headers=_HEADERS, data=json.dumps(body).encode()
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _season_games(year, league="1"):
    """해당 연도 전체 경기 (월별 조회 후 gameId 중복 제거). 프로세스 내 캐시."""
    key = (str(year), str(league))
    if key in _CACHE:
        return _CACHE[key]
    games = {}
    for m in range(1, 13):
        j = _post(SCHEDULE_API,
                  {"year": str(year), "month": f"{m:02d}", "leagueId": str(league)})
        for x in j["data"].get("scheduleList", []):
            games[x["gameId"]] = x
    _CACHE[key] = list(games.values())
    return _CACHE[key]


def fetch_completed_matches(years, league="1"):
    """지정 연도들의 종료 경기 -> [{year,date,home,away,hg,ag}, ...] (시간순)."""
    out = []
    for y in years:
        for x in _season_games(y, league):
            if x.get("endYn") != "Y":
                continue
            out.append({
                "year": x["year"], "date": x["gameDate"],
                "home": x["homeTeamName"], "away": x["awayTeamName"],
                "hg": x["homeGoal"], "ag": x["awayGoal"],
            })
    out.sort(key=lambda m: (m["year"], m["date"]))
    return out


def fetch_remaining_fixtures(year, league="1"):
    """아직 안 끝난 경기(endYn!=Y) -> [(홈, 원정, 라운드), ...] (라운드순)."""
    up = [x for x in _season_games(year, league) if x.get("endYn") != "Y"]
    up.sort(key=lambda x: (x["roundId"], x["gameId"]))
    return [(x["homeTeamName"], x["awayTeamName"], x["roundId"]) for x in up]


def fetch_standings(league="1"):
    """공식 순위 -> {팀: (승점, 경기수, 득점, 실점, 승수)} (공식 순위순).

    주의: teamRank.do 는 leagueId 를 JSON body 가 아니라 '쿼리스트링'으로 받는다.
    (POST body 의 leagueId 는 무시되고 항상 K1 을 준다.)
    """
    rows = _post(f"{RANK_API}?leagueId={league}", {})["data"]["teamRank"]
    return {
        t["teamName"]: (t["gainPoint"], t["gameCount"],
                        t["gainGoal"], t["lossGoal"], t["winCnt"])
        for t in rows
    }


def standings_from_matches(matches):
    """종료 경기 리스트 -> {팀: (승점, 경기수, 득점, 실점, 승수)} (현재 순위순).

    정렬(=현재 순위)은 K리그 규정순: 승점 -> 다득점 -> 득실차 -> 다승.
    """
    tbl = {}
    for m in matches:
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        for t in (h, a):
            tbl.setdefault(t, [0, 0, 0, 0, 0])   # pts, games, gf, ga, wins
        tbl[h][1] += 1; tbl[a][1] += 1
        tbl[h][2] += hg; tbl[h][3] += ag
        tbl[a][2] += ag; tbl[a][3] += hg
        if hg > ag:
            tbl[h][0] += 3; tbl[h][4] += 1
        elif hg == ag:
            tbl[h][0] += 1; tbl[a][0] += 1
        else:
            tbl[a][0] += 3; tbl[a][4] += 1
    ordered = sorted(tbl.items(),
                     key=lambda kv: (kv[1][0], kv[1][2], kv[1][2] - kv[1][3], kv[1][4]),
                     reverse=True)
    return {t: tuple(v) for t, v in ordered}


def fetch_h2h(target, years, league="1"):
    """target의 상대별 전적 -> {(target, 상대): (승, 무, 패)} (target 관점)."""
    rec = {}
    for m in fetch_completed_matches(years, league):
        h, a = m["home"], m["away"]
        if target not in (h, a):
            continue
        opp = a if h == target else h
        tg, og = (m["hg"], m["ag"]) if h == target else (m["ag"], m["hg"])
        w, d, l = rec.get(opp, (0, 0, 0))
        if tg > og:
            w += 1
        elif tg == og:
            d += 1
        else:
            l += 1
        rec[opp] = (w, d, l)
    return {(target, opp): v for opp, v in rec.items()}


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    for lg, name in (("1", "K리그1"), ("2", "K리그2")):
        official = fetch_standings(lg)
        computed = standings_from_matches(fetch_completed_matches((2026,), lg))
        rem = fetch_remaining_fixtures(2026, lg)
        match = "일치" if list(official) == list(computed) else "불일치!"
        print(f"\n[{name}] 공식 {len(official)}팀 / 남은경기 {len(rem)} / 순위검증(공식 vs 경기계산): {match}")
        for i, (t, (p, g, gf, ga, w)) in enumerate(list(official.items())[:6], 1):
            print(f"  {i:2d} {t:6s} {p:3d}pt {g}G {gf}-{ga}")
