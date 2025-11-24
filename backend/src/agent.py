import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 🛒 ORDER STATE
# ======================================================

@dataclass
class OrderState:
    drinkType: str | None = None
    size: str | None = None
    milk: str | None = None
    extras: list[str] = field(default_factory=list)
    name: str | None = None

    def is_complete(self):
        return all([
            self.drinkType,
            self.size,
            self.milk,
            self.extras is not None,
            self.name
        ])

    def to_dict(self):
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name,
        }

    def get_summary(self):
        if not self.is_complete():
            return "🔄 Order in progress..."
        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        return f"☕ {self.size.upper()} {self.drinkType.title()} with {self.milk.title()} milk{extras_text} for {self.name}"


@dataclass
class Userdata:
    order: OrderState
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# 🛠️ TOOLS
# ======================================================

@function_tool
async def set_drink_type(
    ctx: RunContext[Userdata],
    drink: Annotated[
        Literal["latte", "cappuccino", "americano", "espresso", "mocha", "coffee", "cold brew", "matcha"],
        Field(description="🎯 Drink type"),
    ],
):
    ctx.userdata.order.drinkType = drink
    print(f"✅ DRINK SET: {drink}")
    return f"☕ Great! One {drink} coming up!"


@function_tool
async def set_size(
    ctx: RunContext[Userdata],
    size: Annotated[
        Literal["small", "medium", "large", "extra large"],
        Field(description="📏 Drink size"),
    ],
):
    ctx.userdata.order.size = size
    print(f"✅ SIZE SET: {size}")
    return f"📏 {size.title()} size noted!"


@function_tool
async def set_milk(
    ctx: RunContext[Userdata],
    milk: Annotated[
        Literal["whole", "skim", "almond", "oat", "soy", "coconut", "none"],
        Field(description="🥛 Milk type"),
    ],
):
    ctx.userdata.order.milk = milk
    print(f"✅ MILK SET: {milk}")
    if milk == "none":
        return "🥛 Black coffee — bold choice!"
    return f"🥛 {milk.title()} milk added!"


@function_tool
async def set_extras(
    ctx: RunContext[Userdata],
    extras: Annotated[
        list[Literal["sugar", "whipped cream", "caramel", "extra shot", "vanilla", "cinnamon", "honey"]] | None,
        Field(description="🎯 Extras list"),
    ] = None,
):
    ctx.userdata.order.extras = extras if extras else []
    print(f"✅ EXTRAS SET: {ctx.userdata.order.extras}")

    if extras:
        return f"🎯 Added {', '.join(extras)}!"
    return "🎯 No extras added."


@function_tool
async def set_name(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="👤 Customer name")],
):
    ctx.userdata.order.name = name.strip().title()
    print(f"✅ NAME SET: {ctx.userdata.order.name}")
    return f"👤 Thanks {ctx.userdata.order.name}! Almost done."


@function_tool
async def complete_order(ctx: RunContext[Userdata]):
    order = ctx.userdata.order

    if not order.is_complete():
        missing = []
        if not order.drinkType: missing.append("☕ drink")
        if not order.size: missing.append("📏 size")
        if not order.milk: missing.append("🥛 milk")
        if order.extras is None: missing.append("🎯 extras")
        if not order.name: missing.append("👤 name")

        return f"🔄 Still need: {', '.join(missing)}"

    print(f"🎉 FINAL ORDER: {order.get_summary()}")

    try:
        save_order_to_json(order)
        extras_text = f" with {', '.join(order.extras)}" if order.extras else ""

        return (
            f"🎉 Perfect! Your {order.size} {order.drinkType} with "
            f"{order.mi
