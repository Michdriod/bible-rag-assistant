import pytest
import asyncio
from app.rag_router import get_next_verse
from types import SimpleNamespace


class DummyRow:
    def __init__(self, reference, book, chapter, verse, text):
        self.reference = reference
        self.book = book
        self.chapter = chapter
        self.verse = verse
        self.text = text


class DummyResult:
    def __init__(self, rows):
        self._rows = rows
    def scalars(self):
        return self
    def first(self):
        return self._rows[0] if self._rows else None


class DummyDB:
    def __init__(self, mapping):
        # mapping: query_key -> DummyResult
        self.mapping = mapping
    async def execute(self, query):
        # very naive: inspect string version of query to decide
        q = str(query)
        if 'verse >' in q:
            return DummyResult(self.mapping.get('next_in_chapter', []))
        return DummyResult(self.mapping.get('next_chapter', []))


@pytest.mark.asyncio
async def test_get_next_verse_happy_path(monkeypatch):
    # Mock chunker and model
    from utils.chunker import BibleChunker
    monkeypatch.setattr('utils.chunker.BibleChunker.is_reference_format', lambda self, r: True)
    monkeypatch.setattr('utils.chunker.BibleChunker.parse_reference', lambda self, r: ('Matthew', 1, 1))

    # Mock get_verse_model to be irrelevant; db returns a row
    class FakeModel: pass

    db = DummyDB({'next_in_chapter': [DummyRow('Matthew 1:2', 'Matthew', 1, 2, 'Verse 2')]})

    res = await get_next_verse('Matthew 1:1', 'kjv', db)
    assert res.reference == 'Matthew 1:2'
    assert res.verse == 2


@pytest.mark.asyncio
async def test_get_next_verse_end_of_book(monkeypatch):
    monkeypatch.setattr('utils.chunker.BibleChunker.is_reference_format', lambda self, r: True)
    monkeypatch.setattr('utils.chunker.BibleChunker.parse_reference', lambda self, r: ('Revelation', 22, 21))

    db = DummyDB({'next_in_chapter': [], 'next_chapter': []})

    with pytest.raises(Exception):
        await get_next_verse('Revelation 22:21', 'kjv', db)
