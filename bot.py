import os
import math
import random
import itertools
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import discord
from discord.ext import tasks
from discord import app_commands
from dotenv import load_dotenv

# ========= env & client =========
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
HEARTBEAT_CH = os.getenv("HEARTBEAT_CHANNEL_ID")
AUTO_POST = os.getenv("AUTO_POST", "0") == "1"       # turn on in Railway to automate
POST_INTERVAL_MIN = int(os.getenv("POST_INTERVAL_MIN", "15"))
DISCORD_GUILD_ID = os.getenv("DISCORD_GUILD_ID")
GUILD_OBJ = discord.Object(id=int(DISCORD_GUILD_ID)) if DISCORD_GUILD_ID and DISCORD_GUILD_ID.isdigit() else None

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# channel routes by name (edit if you renamed)
ROUTES = {
    "nba": "nba-alerts",
    "nfl": "nfl-alerts",
    "mlb": "mlb-alerts",
    "nhl": "nhl-alerts",
    "soccer": "soccer-alerts",
    "updates": "ev-updates",
    "parlays": "top-5-parlays",
}

# ========= small utils =========
def tz_now():
    return dt.datetime.now(dt.timezone.utc)

def amer_to_decimal(amer: int) -> float:
    return (1 + 100/abs(amer)) if amer < 0 else (1 + amer/100)

def fmt_pct(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return str(x)

def brand_footer():
    return f"house edge • {tz_now().strftime('%Y-%m-%d %H:%M UTC')}"

def find_text_channel(guild: discord.Guild, name: str) -> Optional[discord.TextChannel]:
    name = name.strip().lower()
    for ch in guild.text_channels:
        if ch.name.lower() == name:
            return ch
    return None

async def route(guild: discord.Guild, key: str) -> Optional[discord.TextChannel]:
    target = ROUTES.get(key)
    return find_text_channel(guild, target) if target else None

# ========= EV model types =========
@dataclass
class Play:
    sport: str             # "nba", "nfl", etc.
    event: str             # "LAL @ BOS"
    market: str            # "assists", "spread", "total", "anytime TD", ...
    selection: str         # "LeBron o8.5", "LAL -3.5"
    american: int          # +115, -120, etc.
    book: str              # "fanduel"
    p_true: float          # model/consensus true probability (0..1)
    game_time: Optional[str] = None
    id_hint: Optional[str] = None  # for dedupe/hash

    @property
    def dec(self) -> float:
        return amer_to_decimal(self.american)

    @property
    def ev(self) -> float:
        # EV as expected ROI: p_true * payout - (1 - p_true)
        # which simplifies to p_true * dec - 1
        return self.p_true * self.dec - 1.0

# ========= sample data / provider hook =========
# NOTE: replace this with your real odds provider calls later.
BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]

import os, httpx, math

ODDS_API_KEY = os.getenv("THEODDS_API_KEY")
TARGET_BOOKS = [b.strip().lower() for b in os.getenv("TARGET_BOOKS", "fanduel,draftkings").split(",")]
REGION = os.getenv("REGION", "us")

# Which markets to pull (add more later: player_assists, player_rebounds, player_threes, etc.)
ODDS_MARKETS = "h2h,spreads,totals"

# optional: treat these as "sharper" when present
SHARP_BOOKS = {"pinnacle", "circa", "betonline", "william hill", "lowvig"}

def dec_from_american(a: int) -> float:
    return 1 + (a/100 if a > 0 else 100/abs(a))

def strip_vig_two_way(d1: float, d2: float) -> tuple[float, float]:
    # raw implied probs
    p1_raw, p2_raw = 1/d1, 1/d2
    total = p1_raw + p2_raw
    if total <= 0:
        return 0.5, 0.5
    return p1_raw/total, p2_raw/total

def odds_get(sport_key: str) -> list[dict]:
    # single source of truth for markets (NO player props here)
    ODDS_MARKETS = "h2h,spreads,totals"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": ODDS_MARKETS,
        "oddsFormat": "american",
    }
    try:
        print(f"[odds_get] sport_key={sport_key} params={params}")  # <- verify in Railway logs
        r = httpx.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
            params=params,
            timeout=20.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        print(f"[odds_get] HTTP {e.response.status_code}: {e.response.text}")
        return []
    except Exception as e:
        print(f"[odds_get] error: {e}")
        return []

SHARP_BOOKS = {"pinnacle", "circa", "betonline", "william hill", "lowvig"}

def dec_from_american(a: int) -> float:
    return 1 + (a / 100 if a > 0 else 100 / abs(a))

def pick_fair_two_way(bookmakers: list[dict], market_key: str) -> tuple[float, float] | None:
    """
    Build a vig-stripped *consensus* fair prob for a two-way market using any available books.
    If sharp books are present, weight them higher.
    Returns (p1_fair, p2_fair) or None if not enough data.
    """
    weights: list[float] = []
    pairs: list[tuple[float, float]] = []

    for bm in bookmakers:
        title = (bm.get("title") or "").lower()
        for m in bm.get("markets", []):
            if m.get("key") != market_key:  # "h2h" | "spreads" | "totals" 
                continue
            outcomes = m.get("outcomes", [])
            if len(outcomes) != 2:
                continue
            d1 = dec_from_american(int(outcomes[0]["price"]))
            d2 = dec_from_american(int(outcomes[1]["price"]))

            # vig-stripped probs for a 2-way market
            p1_raw, p2_raw = 1 / d1, 1 / d2
            overround = p1_raw + p2_raw
            if overround <= 0:
                continue
            p1, p2 = p1_raw / overround, p2_raw / overround

            w = 2.0 if title in SHARP_BOOKS else 1.0
            pairs.append((p1, p2))
            weights.append(w)

    if not pairs:
        return None

    total_w = sum(weights)
    p1 = sum(p[0] * w for p, w in zip(pairs, weights)) / total_w
    p2 = 1 - p1
    return p1, p2

def plays_from_two_way_market(
    sport: str,
    event_name: str,
    market_key: str,
    label1: str, label2: str,
    fair_p1: float, fair_p2: float,
    target_offers: list[tuple[str, dict]]
) -> list["Play"]:
    """
    Build Play objects for each side from the target-book offers provided.
    label1/label2 are hints for aligning 'over/under' when possible.
    """
    out: list["Play"] = []
    for book_name, market in target_offers:
        outcomes = market.get("outcomes", [])
        if len(outcomes) != 2:
            continue

        names_lower = [o.get("name", "").lower() for o in outcomes]
        # Try to align to Over/Under or known labels when present
        if label1.lower() in names_lower and label2.lower() in names_lower:
            i1, i2 = names_lower.index(label1.lower()), names_lower.index(label2.lower())
        else:
            i1, i2 = 0, 1  # fallback

        o1, o2 = outcomes[i1], outcomes[i2]
        a1, a2 = int(o1["price"]), int(o2["price"])

        # EV = p_fair * decimal - 1
        d1 = dec_from_american(a1)
        d2 = dec_from_american(a2)

        # Construct plays; keep all, we'll filter by EV afterwards
        out.append(Play(
            sport=sport, event=event_name, market=market_key,
            selection=o1.get("name", "side1"), american=a1, book=book_name, p_true=fair_p1
        ))
        out.append(Play(
            sport=sport, event=event_name, market=market_key,
            selection=o2.get("name", "side2"), american=a2, book=book_name, p_true=fair_p2
        ))

    # rank by EV and keep positive edges
    out.sort(key=lambda p: p.ev, reverse=True)
    return [p for p in out if p.ev >= 0.02]  # require >= +2% EV

SPORT_MAP = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "soccer": "soccer_epl",  # adjust to your preferred league(s)
}

def fetch_plays(sport: str) -> list["Play"]:
    """
    Fetch live odds from The Odds API and return EV-ranked plays for one sport.
    Depends on:
      - ODDS_API_KEY (env)
      - REGION (env, default "us")
      - TARGET_BOOKS (env list: "fanduel,draftkings,betmgm,caesars")
    """
    sport_key = SPORT_MAP.get(sport)
    if not sport_key:
        return []

    games = odds_get(sport_key)  # <- keep your existing odds_get
    if not games:
        return []  # nothing fetched; keep bot stable

    plays: list["Play"] = []
    # ... (rest of your function stays the same)


    for g in games:
    event_name = f"{g.get('home_team', '')} vs {g.get('away_team', '')}".strip()
    bookmakers = g.get("bookmakers", [])

    # collect target-book offers by market key
    target_markets: dict[str, list[tuple[str, dict]]] = {}
    for bm in bookmakers:
        bm_title = (bm.get("title") or "").lower()
        if bm_title not in TARGET_BOOKS:
            continue
        for m in bm.get("markets", []):
            key = m.get("key")  # "h2h" | "spreads" | "totals"
            target_markets.setdefault(key, []).append((bm_title, m))

    # moneyline (h2h)
    if "h2h" in target_markets:
        fair = pick_fair_two_way(bookmakers, "h2h")
        if fair:
            p1, p2 = fair
            plays += plays_from_two_way_market(
                sport, event_name, "h2h", "home", "away", p1, p2, target_markets["h2h"]
            )

    # totals (Over/Under)
    if "totals" in target_markets:
        fair = pick_fair_two_way(bookmakers, "totals")
        if fair:
            p_over, p_under = fair
            plays += plays_from_two_way_market(
                sport, event_name, "totals", "over", "under", p_over, p_under, target_markets["totals"]
            )

    # spreads
    if "spreads" in target_markets:
        fair = pick_fair_two_way(bookmakers, "spreads")
        if fair:
            p1, p2 = fair
            plays += plays_from_two_way_market(
                sport, event_name, "spreads", "side1", "side2", p1, p2, target_markets["spreads"]
            )


        # totals (Over/Under)
        if "totals" in target_markets:
            fair = pick_fair_two_way(bookmakers, "totals")
            if fair:
                p_over, p_under = fair
                plays += plays_from_two_way_market(
                    sport, event_name, "totals", "over", "under", p_over, p_under, target_markets["totals"]
                )

        # spreads (two-way against a number)
        if "spreads" in target_markets:
            fair = pick_fair_two_way(bookmakers, "spreads")
            if fair:
                p1, p2 = fair
                plays += plays_from_two_way_market(
                    sport, event_name, "spreads", "side1", "side2", p1, p2, target_markets["spreads"]
                )


    # light dedupe + final rank
    uniq: dict[tuple, Play] = {}
    for p in plays:
        k = (p.sport, p.event, p.market, p.selection, p.book, p.american)
        if k not in uniq or p.ev > uniq[k].ev:
            uniq[k] = p

    ranked = sorted(uniq.values(), key=lambda x: x.ev, reverse=True)
    return ranked[:50]


def pick_fair_two_way(bookmakers: list[dict], market_key: str) -> tuple[float, float] | None:
    """
    Build a vig-stripped *consensus* fair prob for a two-way market using any available books.
    If sharp books are present, weight them higher.
    Returns (p1_fair, p2_fair) or None if not enough data.
    """
    weights = []
    pairs = []
    for bm in bookmakers:
        title = (bm.get("title") or "").lower()
        for m in bm.get("markets", []):
            if m.get("key") != market_key:
                continue
            outcomes = m.get("outcomes", [])
            if len(outcomes) != 2:
                continue
            d1 = dec_from_american(int(outcomes[0]["price"]))
            d2 = dec_from_american(int(outcomes[1]["price"]))
            p1_raw, p2_raw = 1/d1, 1/d2
            total = p1_raw + p2_raw
            if total <= 0:
                continue
            p1, p2 = p1_raw/total, p2_raw/total
            w = 2.0 if title in {"pinnacle", "circa", "betonline", "william hill", "lowvig"} else 1.0
            pairs.append((p1, p2))
            weights.append(w)

    if not pairs:
        return None
    wsum = sum(weights)
    p1 = sum(p[0]*w for p, w in zip(pairs, weights)) / wsum
    p2 = 1 - p1
    return p1, p2

def pick_fair_two_way(bookmakers: list[dict], market_key: str) -> tuple[float, float] | None:
    """
    Build a vig-stripped *consensus* fair prob for a two-way market using any available books.
    If sharp books are present, weight them higher.
    Returns (p1_fair, p2_fair) or None if not enough data.
    """
    weights = []
    pairs = []
    for bm in bookmakers:
        title = (bm.get("title") or "").lower()
        for m in bm.get("markets", []):
            if m.get("key") != market_key:  # example: "h2h", "spreads", "totals"
                continue
            outcomes = m.get("outcomes", [])
            if len(outcomes) != 2:
                continue
            d1 = dec_from_american(int(outcomes[0]["price"]))
            d2 = dec_from_american(int(outcomes[1]["price"]))
            p1, p2 = strip_vig_two_way(d1, d2)
            w = 2.0 if title in SHARP_BOOKS else 1.0
            pairs.append((p1, p2))
            weights.append(w)

    if not pairs:
        return None
    # weighted average
    wsum = sum(weights)
    p1 = sum(p[0]*w for p, w in zip(pairs, weights)) / wsum
    p2 = 1 - p1
    return p1, p2

def plays_from_two_way_market(
    sport: str,
    event_name: str,
    market_key: str,
    label1: str, label2: str,
    fair_p1: float, fair_p2: float,
    target_offers: list[tuple[str, dict]]
) -> list["Play"]:
    """Build Play objects for each side from the target book offers (if present)."""
    out: list["Play"] = []
    for book_name, market in target_offers:
        outcomes = market.get("outcomes", [])
        if len(outcomes) != 2:
            continue
        names_lower = [o.get("name","").lower() for o in outcomes]
        if label1.lower() in names_lower and label2.lower() in names_lower:
            i1, i2 = names_lower.index(label1.lower()), names_lower.index(label2.lower())
        else:
            i1, i2 = 0, 1
        o1, o2 = outcomes[i1], outcomes[i2]
        a1, a2 = int(o1["price"]), int(o2["price"])

        fair_probs = [fair_p1, fair_p2]
        dec_odds = [1 + (a/100 if a > 0 else 100/abs(a)) for a in [a1, a2]]
        evs = [fair_probs[i] * dec_odds[i] - 1 for i in range(2)]

        out.append(Play(
            sport=sport, event=event_name, market=market_key,
            selection=o1.get("name","side1"), american=a1, book=book_name, p_true=fair_p1
        ))
        out.append(Play(
            sport=sport, event=event_name, market=market_key,
            selection=o2.get("name","side2"), american=a2, book=book_name, p_true=fair_p2
        ))

    out.sort(key=lambda p: p.ev, reverse=True)
    return [p for p in out if p.ev >= 0.02]

SPORT_MAP = {
    "nba": "basketball_nba",
    "nfl": "americanfootball_nfl",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "soccer": "soccer_epl",
}

def fetch_plays(sport: str) -> list["Play"]:
    """
    Fetch live odds from The Odds API and return EV-ranked plays for one sport.
    Depends on: ODDS_API_KEY, REGION, TARGET_BOOKS
    """
    sport_key = SPORT_MAP.get(sport)
    if not sport_key:
        return []

    games = odds_get(sport_key)  # <- keep your existing odds_get
    if not games:
        return []  # nothing fetched; keep bot stable

    plays: list["Play"] = []
    # ... keep the rest of your function below unchanged ...


    for g in games:
        event_name = f"{g.get('home_team','')} vs {g.get('away_team','')}"
        bookmakers = g.get("bookmakers", [])
        target_markets: dict[str, list[tuple[str, dict]]] = {}

        for bm in bookmakers:
            bm_title = (bm.get("title") or "").lower()
            if bm_title not in {"fanduel", "draftkings", "betmgm", "caesars"}:
                continue
            for m in bm.get("markets", []):
                key = m.get("key")
                target_markets.setdefault(key, []).append((bm_title, m))

        if "h2h" in target_markets:
            fair = pick_fair_two_way(bookmakers, "h2h")
            if fair:
                p1, p2 = fair
                plays += plays_from_two_way_market(sport, event_name, "h2h", "home", "away", p1, p2, target_markets["h2h"])

        if "totals" in target_markets:
            fair = pick_fair_two_way(bookmakers, "totals")
            if fair:
                p1, p2 = fair
                plays += plays_from_two_way_market(sport, event_name, "totals", "over", "under", p1, p2, target_markets["totals"])

        if "spreads" in target_markets:
            fair = pick_fair_two_way(bookmakers, "spreads")
            if fair:
                p1, p2 = fair
                plays += plays_from_two_way_market(sport, event_name, "spreads", "side1", "side2", p1, p2, target_markets["spreads"])

    uniq = {}
    for p in plays:
        key = (p.sport, p.event, p.market, p.selection, p.book, p.american)
        if key not in uniq or p.ev > uniq[key].ev:
            uniq[key] = p

    ranked = sorted(uniq.values(), key=lambda x: x.ev, reverse=True)
    return ranked[:50]

def plays_from_two_way_market(
    sport: str,
    event_name: str,
    market_key: str,
    label1: str, label2: str,
    fair_p1: float, fair_p2: float,
    target_offers: list[tuple[str, dict]]
) -> list[Play]:
    """Build Play objects for each side from the target book offers (if present)."""
    out: list[Play] = []
    for book_name, market in target_offers:
        outcomes = market.get("outcomes", [])
        if len(outcomes) != 2:
            continue
        # map fair probs to the correct outcome order by name
        names_lower = [o.get("name","").lower() for o in outcomes]
        # try to align: assume first outcome corresponds to label1 in most cases
        # If labels exist in names, match them
        if label1.lower() in names_lower and label2.lower() in names_lower:
            i1, i2 = names_lower.index(label1.lower()), names_lower.index(label2.lower())
        else:
            i1, i2 = 0, 1
        o1, o2 = outcomes[i1], outcomes[i2]

        a1, a2 = int(o1["price"]), int(o2["price"])
        d1, d2 = dec_from_american(a1), dec_from_american(a2)

        ev1 = fair_p1 * d1 - 1.0
        ev2 = fair_p2 * d2 - 1.0

        out.append(Play(
            sport=sport, event=event_name, market=market_key,
            selection=f"{o1.get('name','side1')}", american=a1, book=book_name, p_true=fair_p1
        ))
        out.append(Play(
            sport=sport, event=event_name, market=market_key,
            selection=f"{o2.get('name','side2')}", american=a2, book=book_name, p_true=fair_p2
        ))
    # keep high EV only
    out.sort(key=lambda p: p.ev, reverse=True)
    return [p for p in out if p.ev >= 0.02]  # require >= +2% EV


# ========= parlay generation =========
def correlated(a: Play, b: Play) -> bool:
    # basic rule: avoid multiple legs from the exact same event
    return a.event == b.event

def parlay_score(legs: List[Play]) -> Tuple[float, float, float]:
    """returns (net_ev, combined_prob, combined_decimal) with mild correlation penalty."""
    prob = 1.0
    dec = 1.0
    # correlation penalty: if any pair shares event, apply mild penalty
    penalty = 1.0
    for i, leg in enumerate(legs):
        prob *= leg.p_true
        dec *= leg.dec
        for j in range(i):
            if correlated(legs[i], legs[j]):
                penalty *= 0.97  # 3% off per conflicted pair
    prob *= penalty
    net_ev = prob * dec - 1.0
    return net_ev, prob, dec

def generate_top_parlays(candidates: List[Play], max_rank=5) -> List[Tuple[float, List[Play], Tuple[float, float]]]:
    # choose from top N EV single legs to control combinatorics
    base = sorted(candidates, key=lambda p: p.ev, reverse=True)[:12]
    best: List[Tuple[float, List[Play], Tuple[float, float]]] = []
    seen_hashes = set()

    for k in (2, 3, 4):  # 2–4 legs
        for combo in itertools.combinations(base, k):
            # quick hard rule: no more than 1 leg per event
            if len({c.event for c in combo}) < len(combo):
                continue
            net_ev, prob, dec = parlay_score(list(combo))
            if net_ev <= 0.02:  # require at least +2% EV
                continue
            # uniqueness hash
            h = tuple(sorted(f"{c.sport}-{c.event}-{c.selection}" for c in combo))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            best.append((net_ev, list(combo), (prob, dec)))

    best.sort(key=lambda x: x[0], reverse=True)
    return best[:max_rank]

# ========= EMBEDS =========
def ev_embed(play: Play) -> discord.Embed:
    title = f"{play.sport.upper()} – {play.event} ({play.market})"
    desc = (
        f"**pick**: {play.selection}  •  **{play.american:+d} @ {play.book}**\n"
        f"**edge**: {fmt_pct(play.ev)}  •  **true win**: {fmt_pct(play.p_true)}"
    )
    if play.game_time:
        desc += f"\n**game time**: {play.game_time}"
    emb = discord.Embed(title=title, description=desc, color=0x1f2937)
    emb.set_footer(text=brand_footer())
    return emb

def parlay_embed(rank: int, legs: List[Play], prob: float, dec: float, net_ev: float) -> discord.Embed:
    legs_text = "\n".join(
        f"• {p.sport.upper()} {p.event} — {p.selection} ({p.american:+d} @ {p.book}; p*={fmt_pct(p.p_true)})"
        for p in legs
    )
    desc = (
        f"**legs:**\n{legs_text}\n\n"
        f"**true win:** {fmt_pct(prob)}  •  **combined (dec):** {dec:.3f}\n"
        f"**net ev:** {fmt_pct(net_ev)}"
    )
    emb = discord.Embed(title=f"top parlay #{rank}", description=desc, color=0x1f8b4c)
    emb.set_footer(text=brand_footer())
    return emb

def revision_embed(old: Play, new: Play) -> discord.Embed:
    desc = (
        f"**old → new**: {old.selection} ({old.american:+d}) → **{new.selection} ({new.american:+d})**\n"
        f"**edge Δ**: {fmt_pct(old.ev)} → **{fmt_pct(new.ev)}**"
    )
    emb = discord.Embed(title=f"revision – {new.sport.upper()} {new.event}", description=desc, color=0xC27C0E)
    emb.set_footer(text=brand_footer())
    return emb

# ========= state (simple dedupe in-memory) =========
posted_hashes: Dict[str, set] = {"singles": set(), "parlays": set()}
best_single: Dict[str, Play] = {}  # sport -> best play so far

def day_key() -> str:
    return tz_now().strftime("%Y%m%d")

def hash_play(p: Play) -> str:
    return f"{day_key()}::{p.sport}::{p.event}::{p.selection}::{p.american}"

def hash_parlay(legs: List[Play]) -> str:
    parts = " | ".join(sorted(hash_play(p) for p in legs))
    return f"{day_key()}::PARLAY::{parts}"

# ========= schedulers =========
@tasks.loop(minutes=5)
async def heartbeat():
    if HEARTBEAT_CH:
        ch = client.get_channel(int(HEARTBEAT_CH))
        if ch:
            await ch.send("✅ heartbeat: still running")

@tasks.loop(minutes=POST_INTERVAL_MIN)
async def autopost():
    """Main loop: fetch, compute EV, post singles and Top-5 parlays."""
    if not AUTO_POST:
        return  # disabled unless AUTO_POST=1
    guilds = list(client.guilds)
    if not guilds:
        return
    guild = guilds[0]

    # --- fetch/update plays by sport ---
    sports = ["nba", "nfl", "mlb"]  # add more when ready
    all_candidates: List[Play] = []
    for g in games:
        try:
        plays = fetch_plays(s)
    except Exception as e:
        print(f"[autopost] fetch_plays error for {s}: {e}")
        continue
        all_candidates.extend(plays)

        # post the top single EV alert per sport
        top = max(plays, key=lambda p: p.ev)
        old = best_single.get(s)
        if old is None or top.ev > old.ev + 0.005:  # ~0.5% better
            best_single[s] = top
            ch = await route(guild, s)
            if ch:
                await ch.send(embed=ev_embed(top))
            # also notify in updates channel if it’s a revision
            if old is not None:
                up = await route(guild, "updates")
                if up:
                    await up.send(embed=revision_embed(old, top))

    # --- build and post Top-5 parlays ---
    if all_candidates:
        top_parlays = generate_top_parlays(all_candidates, max_rank=5)
        pchan = await route(guild, "parlays")
        if pchan:
            for rank, (net_ev, legs, (prob, dec)) in enumerate(top_parlays, start=1):
                h = hash_parlay(legs)
                if h in posted_hashes["parlays"]:
                    continue
                posted_hashes["parlays"].add(h)
                await pchan.send(embed=parlay_embed(rank, legs, prob, dec, net_ev))

# ========= slash commands =========
@tree.command(name="force_parlays", description="immediately post sample Top-5 parlays (for testing)")
@app_commands.checks.has_permissions(administrator=True)
async def force_parlays(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    guild = interaction.guild
    candidates = []
    for s in ["nba", "nfl", "mlb"]:
        candidates.extend(fetch_plays(s))
    parlays = generate_top_parlays(candidates, max_rank=5)
    pchan = await route(guild, "parlays")
    if not pchan:
        await interaction.followup.send("no #top-5-parlays channel found.", ephemeral=True)
        return
    for rank, (net_ev, legs, (prob, dec)) in enumerate(parlays, start=1):
        await pchan.send(embed=parlay_embed(rank, legs, prob, dec, net_ev))
    await interaction.followup.send(f"posted {len(parlays)} sample parlays to #top-5-parlays", ephemeral=True)

# ======= /setup_server (unchanged, keeps your professional layout) =======
@tree.command(name="setup_server", description="create the full house edge server layout")
@app_commands.checks.has_permissions(administrator=True)
async def setup_server(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    guild = interaction.guild
    structure = {
        "information": {"channels": [("welcome","read_only"), ("how-it-works","read_only"), ("announcements","read_only"), ("rules","read_only")]},
        "daily ev alerts": {"channels": [("nba-alerts","read_only"), ("nfl-alerts","read_only"), ("mlb-alerts","read_only"), ("nhl-alerts","read_only"), ("soccer-alerts","read_only")]},
        "featured parlays": {"channels": [("top-5-parlays","read_only"), ("touchdown-parlays","read_only"), ("first-basket-parlays","read_only"), ("community-parlays","open_chat")]},
        "live updates": {"channels": [("ev-updates","read_only"), ("injury-reports","read_only"), ("cancellations","read_only")]},
        "analytics & data": {"channels": [("daily-performance","read_only"), ("leaderboards","read_only"), ("suggestions","open_chat")]},
        "admin": {"channels": [("bot-logs","read_only"), ("dev-notes","read_only"), ("todo","read_only")]},
    }

    async def get_or_create_category(g: discord.Guild, name: str):
        for c in g.categories:
            if c.name == name:
                return c
        return await g.create_category(name)

    async def get_or_create_text_channel(g: discord.Guild, name: str, category: discord.CategoryChannel, read_only=False):
        for ch in category.text_channels:
            if ch.name == name:
                channel = ch
                break
        else:
            channel = await g.create_text_channel(name, category=category)
        everyone = g.default_role
        me = g.me
        # bot perms
        bot_over = channel.overwrites_for(me); bot_over.send_messages = True; bot_over.embed_links = True
        await channel.set_permissions(me, overwrite=bot_over)
        # lock members for read_only
        if read_only:
            over = channel.overwrites_for(everyone); over.send_messages = False; over.add_reactions = False
            await channel.set_permissions(everyone, overwrite=over)
        return channel

    created = 0
    for cat_name, cfg in structure.items():
        category = await get_or_create_category(guild, cat_name)
        for ch_name, mode in cfg["channels"]:
            await get_or_create_text_channel(guild, ch_name, category, read_only=(mode=="read_only"))
            created += 1
    await interaction.followup.send(f"✅ setup complete. created/verified {created} channels.", ephemeral=True)

# ========= startup =========
@client.event
async def setup_hook():
    await tree.sync()
    if HEARTBEAT_CH and not heartbeat.is_running():
        heartbeat.start()
    if AUTO_POST and not autopost.is_running():
        autopost.start()

@client.event
async def on_ready():
    print(f"✅ logged in as {client.user} (id: {client.user.id})")
    if HEARTBEAT_CH:
        ch = client.get_channel(int(HEARTBEAT_CH))
        if ch:
            await ch.send("👋 bot is online and ready!")

if GUILD_OBJ:
    @app_commands.guilds(GUILD_OBJ)
@tree.command(
    name="ev_scan",
    description="scan live odds for a sport (nba/nfl/mlb/nhl/soccer) and post the top N EV singles"
)
@app_commands.describe(
    sport="nba, nfl, mlb, nhl, or soccer",
    top_n="how many to post (default 5)"
)
async def ev_scan(interaction: discord.Interaction, sport: str, top_n: int = 5):
    # only the command runner will see status messages
    await interaction.response.defer(ephemeral=True, thinking=True)

    # normalize sport input
    sport = sport.lower().strip()
    if sport not in {"nba", "nfl", "mlb", "nhl", "soccer"}:
        await interaction.followup.send(
            "invalid sport. use one of: nba, nfl, mlb, nhl, soccer.",
            ephemeral=True
        )
        return

    # fetch plays using the Odds API pipeline you added
    try:
        plays = fetch_plays(sport)
    except Exception as e:
        await interaction.followup.send(f"error fetching odds: {e}", ephemeral=True)
        return

    if not plays:
        await interaction.followup.send(f"no plays found for {sport}.", ephemeral=True)
        return

    # find the correct channel (e.g., nba → #nba-alerts)
    channel = await route(interaction.guild, sport)
    if not channel:
        await interaction.followup.send(
            f"couldn’t find a channel for {sport} (expected something like #{sport}-alerts).",
            ephemeral=True
        )
        return

    # cap posts and send embeds
    top_n = max(1, min(top_n, 10))
    for p in plays[:top_n]:
        await channel.send(embed=ev_embed(p))

    await interaction.followup.send(
        f"posted top {top_n} {sport.upper()} singles to #{channel.name}.",
        ephemeral=True
    )
import os
GUILD_ID_ENV = os.getenv("DISCORD_GUILD_ID")
GUILD_OBJ = discord.Object(id=int(GUILD_ID_ENV)) if GUILD_ID_ENV and GUILD_ID_ENV.isdigit() else None

@client.event
async def on_ready():
    print(f"✅ logged in as {client.user} (id: {client.user.id})")
    # force guild sync so commands show instantly
    try:
        if GUILD_OBJ:
            synced = await tree.sync(guild=GUILD_OBJ)
            print(f"🔄 synced {len(synced)} commands to guild {GUILD_OBJ.id}")
        else:
            for g in client.guilds:
                s = await tree.sync(guild=g)
                print(f"🔄 synced {len(s)} commands to guild {g.id}")
        await tree.sync()
        print("🌐 global sync requested")
    except Exception as e:
        print(f"❌ slash sync error: {e}")

@client.event
async def setup_hook():
    try:
        if GUILD_OBJ:
            synced = await tree.sync(guild=GUILD_OBJ)
            print(f"🔄 synced {len(synced)} commands to guild {GUILD_OBJ.id}")
        else:
            # fallback: sync to every guild the bot is in
            for g in client.guilds:
                s = await tree.sync(guild=g)
                print(f"🔄 synced {len(s)} commands to guild {g.id}")
        await tree.sync()  # optional global publish
        print("🌐 global sync requested")
    except Exception as e:
        print(f"❌ slash sync error: {e}")

    # start your tasks after sync (keep your existing logic)
    if HEARTBEAT_CH and not heartbeat.is_running():
        heartbeat.start()
    if AUTO_POST and not autopost.is_running():
        autopost.start()


client.run(TOKEN)
