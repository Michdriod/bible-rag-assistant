import asyncio
from sqlalchemy import select
from db.db import AsyncSessionLocal
from db.models import get_verse_model

async def debug():
    refs = [f"Matthew 5:{i}" for i in range(3,13)]
    async with AsyncSessionLocal() as s:
        Verse = get_verse_model("kjv")
        q = select(Verse).where(Verse.book == "Matthew", Verse.chapter == 5, Verse.verse.between(3, 12)).order_by(Verse.verse)
        res = await s.execute(q)
        verses = res.scalars().all()
        print("Queried (numeric) count:", len(verses))
        for v in verses:
            print(v.reference, v.book, v.chapter, v.verse)
asyncio.run(debug())