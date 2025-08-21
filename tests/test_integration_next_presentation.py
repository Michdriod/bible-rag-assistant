import pytest
import httpx

BASE = "http://127.0.0.1:8000"

@pytest.mark.integration
def test_next_endpoint_returns_next_verse():
    r = httpx.get(f"{BASE}/api/bible/next", params={"reference":"John 3:16","version":"kjv"}, timeout=10)
    assert r.status_code == 200
    j = r.json()
    assert 'reference' in j
    assert j['reference'].startswith('John 3:')

@pytest.mark.integration
def test_presentation_page_served():
    r = httpx.get(f"{BASE}/static/presentation.html", timeout=10)
    assert r.status_code == 200
    assert '<title>Bible Presentation' in r.text
