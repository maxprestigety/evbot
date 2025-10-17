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

def sample_plays_for(sport: str, n=12) -> List[Play]:
    """temporary generator so you can see real embeds immediately."""
    plays = []
    for i in range(n):
        american = random.choice([+110, +120, +135, -105, -110, +145, +100])
        # symmetric true prob around fair for a bit of spread (+EV & small -EV)
        fair_p = 1/amer_to_decimal(american)
        p_true = min(max(fair_p + random.uniform(-0.04, 0.06), 0.05), 0.95)
        event = f"event {i+1}"
        market = random.choice(["spread", "total", "player prop"])
        sel = random.choice(["team a", "team b", "player x o/u"])
        plays.append(Play(
            sport=sport, event=event, market=market,
            selection=f"{sel}", american=american,
            book=random.choice(BOOKS),
            p_true=p_true,
            game_time="7:30 pm est",
            id_hint=f"{sport}-{i}"
        ))
    # keep top 15 by EV
    plays.sort(key=lambda p: p.ev, reverse=True)
    return plays[:15]

def fetch_plays(sport: str) -> List[Play]:
    """Plug your API here. For now uses sample generator."""
    return sample_plays_for(sport)

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
    for s in sports:
        plays = fetch_plays(s)
        if not plays:
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

client.run(TOKEN)
