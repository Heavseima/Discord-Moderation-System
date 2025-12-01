import discord
from discord.ext import commands
import asyncio
import os
import csv
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from datetime import datetime

# ---- CONFIG ----
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
FILTERED_FILE = os.path.join(DATA_DIR, "filtered_messages.csv")

# Load model from Hugging Face Hub
MODEL_ID = "Franklin001/topic_classifier"

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_ID)
model = RobertaForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

# ---- LABEL MAP (canonical) ----
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Reverse lookup for case-insensitive mapping
canonical_from_name = {v.lower(): v for v in label_map.values()}

# Optional: only delete when model confidence >= this (0.0 = always)
CONFIDENCE_THRESHOLD = 0.0  # set to 0.6 to be more conservative

class TopicFilter(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        # store canonical topic values (e.g. "Sports")
        self.channel_topics = {}

    def _canonicalize(self, topic_str: str):
        """Return canonical topic string (from label_map) or None if invalid."""
        if not topic_str:
            return None
        candidate = topic_str.strip().lower()
        return canonical_from_name.get(candidate)

    # ---- SET TOPIC ----
    @commands.command(name="topicset")
    async def set_topic(self, ctx, *, topic: str):
        """Set the allowed topic for this channel."""
        canonical = self._canonicalize(topic)
        if not canonical:
            await ctx.send(f"❌ Invalid topic. Choose from: {', '.join(label_map.values())}")
            return
        self.channel_topics[ctx.channel.id] = canonical
        await ctx.send(f"✅ Topic for this channel set to **{canonical}**.")

    # ---- GET CURRENT TOPIC ----
    @commands.command(name="topicget")
    async def get_topic(self, ctx):
        topic = self.channel_topics.get(ctx.channel.id)
        if topic:
            await ctx.send(f"ℹ️ Current topic for this channel is **{topic}**.")
        else:
            await ctx.send("ℹ️ No topic is set for this channel yet. Use `!topicset <topic>` to set one.")

    # ---- LIST AVAILABLE TOPICS ----
    @commands.command(name="topiclist")
    async def list_topics(self, ctx):
        await ctx.send(f"📌 Available topics: {', '.join(label_map.values())}")

    # ---- CLEAR TOPIC ----
    @commands.command(name="topicclear")
    async def clear_topic(self, ctx):
        if ctx.channel.id in self.channel_topics:
            del self.channel_topics[ctx.channel.id]
            await ctx.send("🧹 Topic filter cleared for this channel. All messages are now allowed.")
        else:
            await ctx.send("ℹ️ No topic is currently set for this channel.")

    # ---- MESSAGE LISTENER ----
    @commands.Cog.listener()
    async def on_message(self, message):
        # ignore bot messages and commands
        if message.author.bot:
            return
        if message.content.startswith("!"):
            return

        channel_id = message.channel.id
        if channel_id not in self.channel_topics:
            return

        allowed_topic = self.channel_topics[channel_id]  # already canonical (e.g. "Sports")

        # ---- PREDICT TOPIC WITH CONFIDENCE ----
        inputs = tokenizer(message.content, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence_tensor, predicted_class = torch.max(probs, dim=1)

        predicted_topic = label_map[predicted_class.item()]  # canonical
        confidence_value = confidence_tensor.item()  # 0..1

        low_confidence = confidence_value < CONFIDENCE_THRESHOLD

        if predicted_topic != allowed_topic:
            will_delete = not low_confidence

            warning_text = (
                f"⚠️ Off-topic (Predicted: {predicted_topic}, "
                f"Confidence: {confidence_value*100:.2f}%). "
            )
            warning_text += "Message will be deleted in 10 seconds." if will_delete else "Low confidence — message will NOT be auto-deleted."

            warning_msg = await message.reply(warning_text)

            # ---- SAVE TO CSV ----
            with open(FILTERED_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    message.author.name,
                    message.content,
                    predicted_topic,
                    f"{confidence_value*100:.4f}",
                    allowed_topic
                ])

            if not will_delete:
                return

            # ---- SMOOTH COUNTDOWN ----
            for i in range(10, 0, -1):
                await warning_msg.edit(
                    content=(
                        f"⚠️ Off-topic (Predicted: {predicted_topic}, "
                        f"Confidence: {confidence_value*100:.2f}%). Deleting in {i} seconds..."
                    )
                )
                await asyncio.sleep(1)

            # ---- DELETE ----
            try:
                await message.delete()
                await warning_msg.edit(
                    content=f"✅ Message deleted (Predicted: {predicted_topic}, Confidence: {confidence_value*100:.2f}%)"
                )
            except discord.NotFound:
                pass

# setup
async def setup(bot):
    await bot.add_cog(TopicFilter(bot))
