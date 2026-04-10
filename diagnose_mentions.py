#!/usr/bin/env python3
"""Try every Gamma API approach to find mention markets."""

import sys, os
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests, json

BASE = "https://gamma-api.polymarket.com"
s = requests.Session()
s.headers["Accept"] = "application/json"

def try_request(label, path, params=None):
    try:
        resp = s.get(f"{BASE}{path}", params=params, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "markets" in data:
            # Single event response
            title = data.get("title", "???")
            n = len(data.get("markets", []))
            print(f"  OK {label}: event '{title}' with {n} markets")
            if n > 0:
                for m in data["markets"][:3]:
                    print(f"       groupItemTitle='{m.get('groupItemTitle','')}' q='{m.get('question','')[:50]}'")
            return data
        elif isinstance(data, list):
            print(f"  OK {label}: {len(data)} results")
            for item in data[:3]:
                title = item.get("title", item.get("question", "???"))[:60]
                n = len(item.get("markets", []))
                slug = item.get("slug", item.get("eventSlug", ""))[:40]
                git = item.get("groupItemTitle", "")
                extra = f" | git='{git}'" if git else ""
                extra += f" | {n} mkts" if n else ""
                print(f"       '{title}' slug={slug}{extra}")
            return data
        else:
            print(f"  WARN  {label}: unexpected type {type(data)}")
            return data
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        return None

print("=" * 70)
print("  APPROACH 1: /events/slug/{slug}  (slug in path)")
print("=" * 70)
try_request("weekly mentions", "/events/slug/what-will-trump-say-this-week-april-12")
try_request("Rutte event", "/events/slug/what-will-trump-say-during-bilateral-events-with-mark-rutte")

print("\n" + "=" * 70)
print("  APPROACH 2: /events?slug={slug}  (slug as query param)")
print("=" * 70)
try_request("weekly mentions", "/events", {"slug": "what-will-trump-say-this-week-april-12"})
try_request("Rutte event", "/events", {"slug": "what-will-trump-say-during-bilateral-events-with-mark-rutte"})

print("\n" + "=" * 70)
print("  APPROACH 3: /events?title=...  (text filter)")
print("=" * 70)
try_request("title=trump say", "/events", {"title": "trump say", "closed": "false", "limit": 5})
try_request("title_contains", "/events", {"title_contains": "trump say", "closed": "false", "limit": 5})

print("\n" + "=" * 70)
print("  APPROACH 4: /search  (Gamma search endpoint)")
print("=" * 70)
try_request("search trump mention", "/search", {"q": "trump mention"})
try_request("search trump say week", "/search", {"q": "trump say week"})

print("\n" + "=" * 70)
print("  APPROACH 5: /events?tag_id=...  (find Mentions tag first)")
print("=" * 70)
# First find the tag
tags_data = try_request("all tags", "/tags", {"limit": 100})
if tags_data and isinstance(tags_data, list):
    mention_tags = [t for t in tags_data if "mention" in t.get("label", "").lower()
                    or "mention" in t.get("slug", "").lower()]
    print(f"\n  Tags matching 'mention': {mention_tags[:5]}")
    for tag in mention_tags[:2]:
        tag_id = tag.get("id", "")
        try_request(f"events with tag_id={tag_id}", "/events",
                    {"tag_id": tag_id, "closed": "false", "limit": 5})

print("\n" + "=" * 70)
print("  APPROACH 6: /markets?event_slug=...  (markets filtered by event slug)")
print("=" * 70)
try_request("markets by event_slug", "/markets",
            {"event_slug": "what-will-trump-say-this-week-april-12",
             "active": "true", "limit": 5})

print("\n" + "=" * 70)
print("  APPROACH 7: /events ordered by volume (mention events have high vol)")
print("=" * 70)
data = try_request("events by volume", "/events",
                   {"closed": "false", "limit": 20, "order": "volume24hr", "ascending": "false"})
if data and isinstance(data, list):
    print("\n  Checking which are mention-like:")
    for ev in data:
        title = ev.get("title", "")
        if any(kw in title.lower() for kw in ["say", "mention", "tweet", "post"]):
            n = len(ev.get("markets", []))
            print(f"    >>> [{n} mkts] {title[:70]}")