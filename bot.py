import os
import discord
from discord.ext import tasks
from dotenv import load_dotenv

# Load env vars from .env when running locally
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
HEARTBEAT_CH = os.getenv("HEARTBEAT_CHANNEL_ID")

intents = discord.Intents.default()
client = discord.Client(intents=intents)

@tasks.loop(minutes=5)
async def heartbeat():
    """Sends a periodic 'still running' ping to your chosen channel."""
    if HEARTBEAT_CH:
        ch = client.get_channel(int(HEARTBEAT_CH))
        if ch:
            await ch.send("✅ Heartbeat: still running")

@client.event
async def setup_hook():
    """Runs before the bot connects; safe place to start background tasks."""
    if not heartbeat.is_running():
        heartbeat.start()

@client.event
async def on_ready():
    print(f"✅ Logged in as {client.user} (ID: {client.user.id})")
    if HEARTBEAT_CH:
        ch = client.get_channel(int(HEARTBEAT_CH))
        if ch:
            await ch.send("👋 Bot is online and ready!")

client.run(TOKEN)
