import discord
from discord.ext import commands
from datetime import datetime, timedelta
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from zoneinfo import ZoneInfo

MODEL_ID = "Franklin001/sentimental"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load sentiment model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()


# -------- TIME PARSER & FORMATTER --------
def parse_time_input(time_str: str):
    """Accepts 5m, 5mn, 5 min, 2h, 1d etc., returns total minutes as int"""
    time_str = time_str.lower().strip()

    # Minutes
    minute_patterns = [
        r"^(\d+)\s*m$", r"^(\d+)\s*mn$", r"^(\d+)\s*min$",
        r"^(\d+)\s*mins$", r"^(\d+)\s*minute$", r"^(\d+)\s*minutes$"
    ]
    for p in minute_patterns:
        match = re.match(p, time_str)
        if match:
            return int(match.group(1))

    # Hours
    hour_patterns = [
        r"^(\d+)\s*h$", r"^(\d+)\s*hr$", r"^(\d+)\s*hrs$",
        r"^(\d+)\s*hour$", r"^(\d+)\s*hours$"
    ]
    for p in hour_patterns:
        match = re.match(p, time_str)
        if match:
            return int(match.group(1)) * 60

    # Days
    day_patterns = [
        r"^(\d+)\s*d$", r"^(\d+)\s*day$", r"^(\d+)\s*days$"
    ]
    for p in day_patterns:
        match = re.match(p, time_str)
        if match:
            return int(match.group(1)) * 24 * 60

    return None


def format_time(minutes: int):
    """Convert minutes → user-friendly format, e.g., 2 hours 5 minutes"""
    hours = minutes // 60
    mins = minutes % 60
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if mins > 0:
        parts.append(f"{mins} minute" + ("s" if mins != 1 else ""))
    return " ".join(parts) if parts else "0 minutes"


# -------- SENTIMENT HELPER --------
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, label = torch.max(probs, dim=1)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[label.item()], confidence.item()


def generate_thick_bar(percentage, length=10):
    """Create a thick unicode bar using ▰ and ▱"""
    filled_units = int(round(length * percentage / 100))
    empty_units = length - filled_units
    return "▰" * filled_units + "▱" * empty_units


# -------- MAIN COG --------
class Sentiment(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.local_tz = ZoneInfo("Asia/Phnom_Penh")

    @commands.command()
    async def analyze(self, ctx, *, time_input: str = "24h"):
        """Analyze messages from a custom time window."""
        minutes = parse_time_input(time_input)
        if minutes is None:
            await ctx.send(
                "❌ Invalid time format.\n"
                "Examples:\n"
                "`5m`, `5mn`, `5 mins`, `5 minutes`\n"
                "`2h`, `2hr`, `2 hours`\n"
                "`1d`, `2 days`"
            )
            return

        # Minimum/Maximum limits
        if minutes < 5:
            await ctx.send("⚠️ Minimum allowed is **5 minutes**.")
            return
        if minutes > 7 * 24 * 60:
            await ctx.send("⚠️ Maximum allowed is **7 days**.")
            return

        now_local = datetime.now(self.local_tz)
        since_local = now_local - timedelta(minutes=minutes)
        friendly_time = format_time(minutes)

        await ctx.send(f"🕒 Fetching messages from the last **{friendly_time}**...")

        # Fetch messages
        messages = []
        async for message in ctx.channel.history(limit=None):
            # Convert UTC -> Local
            local_time = message.created_at.replace(tzinfo=ZoneInfo("UTC")).astimezone(self.local_tz)
            if local_time < since_local:
                continue
            if message.author.bot or message.content.startswith("!"):
                continue
            messages.append([local_time.isoformat(), message.author.name, message.content])

        if not messages:
            await ctx.send("⚠️ No messages found in that time range.")
            return

        # Analyze messages
        results = []
        for time_str, author, text in messages:
            sentiment, confidence = analyze_sentiment(text)
            results.append((time_str, author, text, sentiment, confidence))

        # Summary stats
        total = len(results)
        pos = sum(1 for _, _, _, s, _ in results if s == "Positive")
        neu = sum(1 for _, _, _, s, _ in results if s == "Neutral")
        neg = sum(1 for _, _, _, s, _ in results if s == "Negative")
        avg_conf = sum(c for _, _, _, _, c in results) / total * 100

        pos_pct = pos / total * 100
        neu_pct = neu / total * 100
        neg_pct = neg / total * 100

        # Decide word
        pos_word = "messages" if pos > 1 else "message"
        neu_word = "messages" if neu > 1 else "message"
        neg_word = "messages" if neg > 1 else "message"

        # Fix width for alignment
        max_count_len = max(len(str(pos)), len(str(neu)), len(str(neg)))
        word_width = len("messages")  # the longest word
        count_word_format = f"{{count:>{max_count_len}}} {{word:<{word_width}}}"

        lines = [
            f"📊 Sentiment Summary (Last {friendly_time})\n",
            f"🟢 Positive  {count_word_format.format(count=pos, word=pos_word)}  │  {generate_thick_bar(pos_pct)}  {pos_pct:5.1f}%",
            f"⚪ Neutral   {count_word_format.format(count=neu, word=neu_word)}  │  {generate_thick_bar(neu_pct)}  {neu_pct:5.1f}%",
            f"🔴 Negative  {count_word_format.format(count=neg, word=neg_word)}  │  {generate_thick_bar(neg_pct)}  {neg_pct:5.1f}%",
            f"\n📦 Total Messages : {total}",
            f"🎯 Avg Confidence : {avg_conf:.2f}%",
        ]


        await ctx.send("```\n" + "\n".join(lines) + "\n```")

        # Save CSV
        csv_file = os.path.join(DATA_DIR, "analyzed_messages.csv")
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "author", "message", "sentiment", "confidence"])
            writer.writerows(results)

        await ctx.send("✅ Sentiment analysis complete.")


async def setup(bot):
    await bot.add_cog(Sentiment(bot))
