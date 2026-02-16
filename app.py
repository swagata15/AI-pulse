"""
AI Pulse — Daily AI News Aggregator
Backend: FastAPI + SQLite cache + multiple data fetchers
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aipulse")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent / "cache.db"
STATIC_DIR = Path(__file__).parent / "static"
REFRESH_HOURS = 6  # how often we re-fetch
HYPE_DECAY_DAYS = 7  # items older than this with low score are pruned
HTTP_TIMEOUT = 20.0

AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "LLM", "GPT", "transformer", "diffusion", "generative AI", "computer vision",
    "NLP", "natural language", "reinforcement learning", "AI agent", "foundation model",
    "large language model", "text-to-image", "text-to-video", "speech synthesis",
    "AI coding", "copilot", "chatbot", "RAG", "retrieval augmented", "fine-tuning",
    "RLHF", "multimodal", "embedding", "vector database", "AI safety",
]

ITEMS_PER_CATEGORY = 12

CATEGORIES = {
    "repos": "Trending Repos",
    "models": "Models",
    "news": "News & Articles",
    "papers": "Research Papers",
    "tools": "Tools & Products",
    "community": "Community Buzz",
    "influencers": "AI Influencers",
}

# ---------------------------------------------------------------------------
# AI Influencers — add handles/URLs here
# ---------------------------------------------------------------------------
AI_INFLUENCERS = [
    {
        "name": "Andrej Karpathy",
        "handle": "@karpathy",
        "platform": "Twitter / YouTube",
        "url": "https://x.com/karpathy",
        "description": "Ex-Tesla AI lead & OpenAI founding member. Teaches deep learning from scratch on YouTube — the GOAT AI educator.",
        "topics": ["LLMs", "Neural Nets", "Education", "Deep Learning"],
    },
    {
        "name": "Andrew Ng",
        "handle": "@AndrewYNg",
        "platform": "Twitter / Coursera",
        "url": "https://x.com/AndrewYNg",
        "description": "Co-founder of Coursera, founder of DeepLearning.AI. Popularized ML for millions. Leads AI Fund investing in AI startups.",
        "topics": ["ML Education", "AI Strategy", "Startups", "Agentic AI"],
    },
    {
        "name": "Deedy Das",
        "handle": "@deaborysas",
        "platform": "Twitter",
        "url": "https://x.com/deedydas",
        "description": "Investor at Menlo Ventures. Sharp takes on AI infrastructure, foundation models, and the business side of AI.",
        "topics": ["AI Investing", "Infrastructure", "Foundation Models", "Startups"],
    },
    {
        "name": "Varun Mayya",
        "handle": "@VarunMayya",
        "platform": "Twitter / YouTube",
        "url": "https://x.com/VarunMayya",
        "description": "Founder of Avalon Labs & Scenes. Builds AI products and shares raw, practical takes on AI agents and productivity.",
        "topics": ["AI Agents", "Productivity", "Building in Public", "Startups"],
    },
    {
        "name": "Dharmesh Shah",
        "handle": "@dhaboresh",
        "platform": "Twitter / LinkedIn",
        "url": "https://x.com/dharmesh",
        "description": "CTO & co-founder of HubSpot. Built ChatSpot and Agent.ai. Deep thinker on AI-first products and SaaS.",
        "topics": ["AI Products", "SaaS", "Entrepreneurship", "AI Agents"],
    },
    {
        "name": "Andriy Burkov",
        "handle": "@buraborkov",
        "platform": "Twitter / LinkedIn",
        "url": "https://x.com/buraborkov",
        "description": "Author of 'The Hundred-Page Machine Learning Book'. Director of ML at Gartner. Clear, no-hype ML explanations.",
        "topics": ["Machine Learning", "Books", "ML Engineering", "Education"],
    },
    {
        "name": "Cassie Kozyrkov",
        "handle": "@quaesita",
        "platform": "Twitter / YouTube",
        "url": "https://x.com/quaesita",
        "description": "Ex-Chief Decision Scientist at Google. Makes AI and decision science approachable. Prolific writer and speaker.",
        "topics": ["Decision Science", "AI Strategy", "Statistics", "Leadership"],
    },
    {
        "name": "Sal Khan",
        "handle": "@saboralkhan",
        "platform": "Twitter / YouTube",
        "url": "https://x.com/saboralkhan",
        "description": "Founder of Khan Academy. Pioneering AI tutoring with Khanmigo (GPT-4 powered). Leading voice on AI in education.",
        "topics": ["AI in Education", "Khanmigo", "EdTech", "LLMs"],
    },
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            description TEXT,
            source TEXT,
            score REAL DEFAULT 0,
            extra TEXT,
            first_seen TEXT,
            last_updated TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON items(category)")
    conn.commit()
    conn.close()


def upsert_item(item: dict):
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()
    item_id = item.get("id") or hashlib.md5((item["title"] + item.get("url", "")).encode()).hexdigest()

    existing = conn.execute("SELECT score, first_seen FROM items WHERE id = ?", (item_id,)).fetchone()
    if existing:
        # Keep the higher score (hype persists)
        old_score = existing[0] or 0
        new_score = max(old_score, item.get("score", 0))
        conn.execute("""
            UPDATE items SET title=?, url=?, description=?, source=?, score=?, extra=?, last_updated=?
            WHERE id=?
        """, (item["title"], item.get("url"), item.get("description"),
              item.get("source"), new_score, json.dumps(item.get("extra", {})),
              now, item_id))
    else:
        conn.execute("""
            INSERT INTO items (id, category, title, url, description, source, score, extra, first_seen, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (item_id, item["category"], item["title"], item.get("url"),
              item.get("description"), item.get("source"), item.get("score", 0),
              json.dumps(item.get("extra", {})), now, now))
    conn.commit()
    conn.close()


def _normalize_score(item_dict):
    """Compute a normalized 0-100 score for cross-category comparison.

    Ranking logic:
    - GitHub repos/tools: stars are the primary signal (viral = 1000+ stars)
    - Models: weighted combo of downloads + likes
    - Community (HN/Reddit): upvotes + comment engagement
    - News: recency is king — newer articles score higher
    - Papers: baseline score (no engagement signal from ArXiv RSS)
    - Influencers: curated, always shown
    """
    raw = item_dict.get("score", 0)
    cat = item_dict.get("category", "")
    extra = item_dict.get("extra", {})

    if cat == "repos":
        stars = extra.get("stars", 0)
        return min(100, stars / 50)  # 5000 stars = 100
    elif cat == "tools":
        stars = extra.get("stars", 0)
        return min(100, stars / 30)  # 3000 stars = 100
    elif cat == "models":
        downloads = extra.get("downloads", 0)
        likes = extra.get("likes", 0)
        return min(100, (downloads / 50000) + (likes / 50))
    elif cat == "community":
        ups = extra.get("ups", 0) or extra.get("points", 0) or 0
        comments = extra.get("comments", 0)
        return min(100, (ups / 20) + (comments / 10))
    elif cat == "news":
        # Recency boost: newer = higher
        try:
            pub = extra.get("published", "")
            if pub:
                from dateutil import parser as dateparser
                pub_dt = dateparser.parse(pub)
                if pub_dt:
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    hours_old = max(0, (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600)
                    recency = max(0, 80 - hours_old * 2)  # loses 2 pts per hour
                    return recency
        except Exception:
            pass
        return 40  # fallback
    elif cat == "papers":
        return 30  # papers have no engagement metric from RSS
    elif cat == "influencers":
        return 90  # curated, always top
    return raw


def get_items_by_category():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM items
        ORDER BY score DESC, last_updated DESC
    """).fetchall()
    conn.close()

    result = {cat: [] for cat in CATEGORIES}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HYPE_DECAY_DAYS)).isoformat()

    for row in rows:
        d = dict(row)
        d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
        cat = d["category"]
        if cat not in result:
            continue
        # Prune old low-score items
        if d["first_seen"] < cutoff and d["score"] < 10:
            continue
        d["normalized_score"] = _normalize_score(d)
        result[cat].append(d)

    # Sort each category by normalized score, limit to ITEMS_PER_CATEGORY
    for cat in result:
        result[cat].sort(key=lambda x: x["normalized_score"], reverse=True)
        result[cat] = result[cat][:ITEMS_PER_CATEGORY]

    return result


def get_tldr_top5():
    """Return the top 5 items across ALL categories by normalized score.
    Ensures diversity — at most 2 items from the same category."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT * FROM items ORDER BY score DESC, last_updated DESC
    """).fetchall()
    conn.close()

    all_items = []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HYPE_DECAY_DAYS)).isoformat()
    for row in rows:
        d = dict(row)
        d["extra"] = json.loads(d["extra"]) if d["extra"] else {}
        if d["first_seen"] < cutoff and d["score"] < 10:
            continue
        if d["category"] == "influencers":
            continue  # influencers don't compete for TLDR
        d["normalized_score"] = _normalize_score(d)
        all_items.append(d)

    all_items.sort(key=lambda x: x["normalized_score"], reverse=True)

    # Pick top 5 with diversity constraint
    top5 = []
    cat_counts = {}
    for item in all_items:
        cat = item["category"]
        if cat_counts.get(cat, 0) >= 2:
            continue
        top5.append(item)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if len(top5) >= 5:
            break

    return top5


def set_meta(key: str, value: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def get_meta(key: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Data Fetchers
# ---------------------------------------------------------------------------

async def fetch_github_repos(client: httpx.AsyncClient):
    """Fetch trending AI repos from GitHub."""
    log.info("Fetching GitHub repos...")
    items = []
    queries = [
        "artificial+intelligence", "machine+learning", "LLM",
        "generative+AI", "AI+agent", "diffusion+model",
    ]
    for q in queries:
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
            resp = await client.get(
                f"https://api.github.com/search/repositories?q={q}+created:>{since}&sort=stars&order=desc&per_page=10",
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for repo in data.get("items", [])[:10]:
                    items.append({
                        "id": f"gh-{repo['full_name']}",
                        "category": "repos",
                        "title": repo["full_name"],
                        "url": repo["html_url"],
                        "description": (repo.get("description") or "")[:300],
                        "source": "GitHub",
                        "score": repo.get("stargazers_count", 0) / 100,
                        "extra": {
                            "stars": repo.get("stargazers_count", 0),
                            "language": repo.get("language"),
                            "forks": repo.get("forks_count", 0),
                            "topics": repo.get("topics", [])[:5],
                        },
                    })
            else:
                log.warning(f"GitHub search returned {resp.status_code} for query '{q}'")
            await asyncio.sleep(2)  # rate limit courtesy
        except Exception as e:
            log.error(f"GitHub fetch error for '{q}': {e}")
    return items


async def fetch_hackernews(client: httpx.AsyncClient):
    """Fetch AI-related stories from Hacker News."""
    log.info("Fetching Hacker News...")
    items = []
    try:
        # Use Algolia HN search API
        resp = await client.get(
            "https://hn.algolia.com/api/v1/search?query=AI+OR+LLM+OR+%22machine+learning%22+OR+%22artificial+intelligence%22&tags=story&hitsPerPage=30",
        )
        if resp.status_code == 200:
            data = resp.json()
            for hit in data.get("hits", []):
                url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}"
                items.append({
                    "id": f"hn-{hit['objectID']}",
                    "category": "community",
                    "title": hit.get("title", ""),
                    "url": url,
                    "description": "",
                    "source": "Hacker News",
                    "score": (hit.get("points", 0) or 0) / 50,
                    "extra": {
                        "points": hit.get("points", 0),
                        "comments": hit.get("num_comments", 0),
                        "author": hit.get("author", ""),
                        "hn_url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
                    },
                })
    except Exception as e:
        log.error(f"HN fetch error: {e}")
    return items


async def fetch_huggingface_models(client: httpx.AsyncClient):
    """Fetch trending models from Hugging Face."""
    log.info("Fetching Hugging Face models...")
    items = []
    tasks = [
        ("text-generation", "chat"),
        ("text-to-image", "image"),
        ("text-to-video", "video"),
        ("text2text-generation", "code"),
        ("automatic-speech-recognition", "audio"),
        ("text-to-speech", "audio"),
    ]
    for pipeline_tag, model_type in tasks:
        try:
            resp = await client.get(
                f"https://huggingface.co/api/models?pipeline_tag={pipeline_tag}&sort=downloads&direction=-1&limit=8",
            )
            if resp.status_code == 200:
                models = resp.json()
                for m in models:
                    downloads = m.get("downloads", 0) or 0
                    likes = m.get("likes", 0) or 0
                    items.append({
                        "id": f"hf-{m['id']}",
                        "category": "models",
                        "title": m["id"],
                        "url": f"https://huggingface.co/{m['id']}",
                        "description": (m.get("pipeline_tag", "") + " model").strip(),
                        "source": "Hugging Face",
                        "score": (downloads / 10000) + (likes / 100),
                        "extra": {
                            "downloads": downloads,
                            "likes": likes,
                            "pipeline": m.get("pipeline_tag", ""),
                            "model_type": model_type,
                            "tags": m.get("tags", [])[:5],
                        },
                    })
            await asyncio.sleep(1)
        except Exception as e:
            log.error(f"HF fetch error for {pipeline_tag}: {e}")
    return items


async def fetch_arxiv_papers(client: httpx.AsyncClient):
    """Fetch latest AI papers from ArXiv RSS."""
    log.info("Fetching ArXiv papers...")
    items = []
    feeds = [
        ("https://rss.arxiv.org/rss/cs.AI", "Artificial Intelligence"),
        ("https://rss.arxiv.org/rss/cs.LG", "Machine Learning"),
        ("https://rss.arxiv.org/rss/cs.CL", "Computation & Language"),
        ("https://rss.arxiv.org/rss/cs.CV", "Computer Vision"),
    ]
    for feed_url, area in feeds:
        try:
            resp = await client.get(feed_url)
            if resp.status_code == 200:
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:10]:
                    title = entry.get("title", "").replace("\n", " ").strip()
                    link = entry.get("link", "")
                    summary = entry.get("summary", "")[:300]
                    items.append({
                        "id": f"arxiv-{hashlib.md5(link.encode()).hexdigest()}",
                        "category": "papers",
                        "title": title,
                        "url": link,
                        "description": summary,
                        "source": f"ArXiv — {area}",
                        "score": 5,  # papers get a baseline score
                        "extra": {
                            "area": area,
                            "authors": entry.get("author", ""),
                        },
                    })
        except Exception as e:
            log.error(f"ArXiv fetch error for {area}: {e}")
    return items


async def fetch_reddit_ai(client: httpx.AsyncClient):
    """Fetch hot posts from AI subreddits via RSS."""
    log.info("Fetching Reddit AI posts...")
    items = []
    subreddits = [
        "MachineLearning", "artificial", "LocalLLaMA",
        "StableDiffusion", "ChatGPT", "singularity",
    ]
    for sub in subreddits:
        try:
            resp = await client.get(
                f"https://www.reddit.com/r/{sub}/hot.json?limit=10",
                headers={"User-Agent": "AIPulse/1.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    p = post["data"]
                    if p.get("stickied"):
                        continue
                    ups = p.get("ups", 0)
                    items.append({
                        "id": f"reddit-{p['id']}",
                        "category": "community",
                        "title": p.get("title", ""),
                        "url": f"https://reddit.com{p.get('permalink', '')}",
                        "description": (p.get("selftext", "") or "")[:300],
                        "source": f"r/{sub}",
                        "score": ups / 100,
                        "extra": {
                            "ups": ups,
                            "comments": p.get("num_comments", 0),
                            "subreddit": sub,
                        },
                    })
            await asyncio.sleep(2)  # Reddit rate limit
        except Exception as e:
            log.error(f"Reddit fetch error for r/{sub}: {e}")
    return items


async def fetch_ai_news(client: httpx.AsyncClient):
    """Fetch AI news from RSS feeds of major tech outlets."""
    log.info("Fetching AI news from RSS...")
    items = []
    feeds = [
        ("https://techcrunch.com/category/artificial-intelligence/feed/", "TechCrunch"),
        ("https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "The Verge"),
        ("https://feeds.arstechnica.com/arstechnica/technology-lab", "Ars Technica"),
        ("https://syncedreview.com/feed/", "Synced Review"),
        ("https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml", "MIT News"),
    ]
    ai_lower = [kw.lower() for kw in AI_KEYWORDS]

    for feed_url, source in feeds:
        try:
            resp = await client.get(feed_url)
            if resp.status_code == 200:
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:15]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")[:400]
                    link = entry.get("link", "")
                    combined = (title + " " + summary).lower()

                    # Filter for AI relevance (some feeds are broad)
                    is_ai = any(kw in combined for kw in ai_lower)
                    if source in ("TechCrunch", "The Verge", "Ars Technica") and not is_ai:
                        continue

                    items.append({
                        "id": f"news-{hashlib.md5(link.encode()).hexdigest()}",
                        "category": "news",
                        "title": title,
                        "url": link,
                        "description": summary,
                        "source": source,
                        "score": 8,  # news gets decent base score
                        "extra": {
                            "published": entry.get("published", ""),
                        },
                    })
        except Exception as e:
            log.error(f"News fetch error for {source}: {e}")
    return items


async def fetch_ai_tools(client: httpx.AsyncClient):
    """Fetch AI tools — using GitHub topics and curated sources."""
    log.info("Fetching AI tools...")
    items = []
    # Search for recently created AI tool repos
    tool_queries = [
        "AI+tool", "AI+app", "LLM+tool", "AI+assistant",
        "AI+workflow", "AI+productivity",
    ]
    for q in tool_queries:
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
            resp = await client.get(
                f"https://api.github.com/search/repositories?q={q}+created:>{since}&sort=stars&order=desc&per_page=5",
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for repo in data.get("items", [])[:5]:
                    items.append({
                        "id": f"tool-{repo['full_name']}",
                        "category": "tools",
                        "title": repo["name"],
                        "url": repo["html_url"],
                        "description": (repo.get("description") or "")[:300],
                        "source": "GitHub",
                        "score": repo.get("stargazers_count", 0) / 50,
                        "extra": {
                            "stars": repo.get("stargazers_count", 0),
                            "language": repo.get("language"),
                            "owner": repo["owner"]["login"],
                        },
                    })
            await asyncio.sleep(2)
        except Exception as e:
            log.error(f"Tools fetch error for '{q}': {e}")
    return items


# ---------------------------------------------------------------------------
# Master fetch orchestrator
# ---------------------------------------------------------------------------

def seed_influencers():
    """Insert curated influencer entries into the DB."""
    for inf in AI_INFLUENCERS:
        item = {
            "id": f"influencer-{hashlib.md5(inf['name'].encode()).hexdigest()}",
            "category": "influencers",
            "title": inf["name"],
            "url": inf.get("url", ""),
            "description": inf.get("description", ""),
            "source": inf.get("platform", ""),
            "score": 90,
            "extra": {
                "handle": inf.get("handle", ""),
                "platform": inf.get("platform", ""),
                "topics": inf.get("topics", []),
            },
        }
        upsert_item(item)
    if AI_INFLUENCERS:
        log.info(f"Seeded {len(AI_INFLUENCERS)} influencers")


async def refresh_all_data():
    """Run all fetchers and store results."""
    log.info("=== Starting full data refresh ===")
    start = time.time()

    # Seed influencers (curated list, no API needed)
    seed_influencers()

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        results = await asyncio.gather(
            fetch_github_repos(client),
            fetch_hackernews(client),
            fetch_huggingface_models(client),
            fetch_arxiv_papers(client),
            fetch_reddit_ai(client),
            fetch_ai_news(client),
            fetch_ai_tools(client),
            return_exceptions=True,
        )

    total = 0
    for result in results:
        if isinstance(result, Exception):
            log.error(f"Fetcher failed: {result}")
            continue
        for item in result:
            try:
                upsert_item(item)
                total += 1
            except Exception as e:
                log.error(f"Failed to upsert item: {e}")

    elapsed = time.time() - start
    set_meta("last_refresh", datetime.now(timezone.utc).isoformat())
    log.info(f"=== Refresh complete: {total} items in {elapsed:.1f}s ===")


# ---------------------------------------------------------------------------
# Cleanup old items
# ---------------------------------------------------------------------------

def cleanup_old_items():
    """Remove items older than HYPE_DECAY_DAYS with low scores."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HYPE_DECAY_DAYS)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    deleted = conn.execute(
        "DELETE FROM items WHERE first_seen < ? AND score < 10", (cutoff,)
    ).rowcount
    conn.commit()
    conn.close()
    if deleted:
        log.info(f"Cleaned up {deleted} old items")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()

    # Check if we need an initial fetch
    last_refresh = get_meta("last_refresh")
    needs_refresh = True
    if last_refresh:
        try:
            last_dt = datetime.fromisoformat(last_refresh)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last_dt < timedelta(hours=REFRESH_HOURS):
                needs_refresh = False
        except Exception:
            pass

    if needs_refresh:
        asyncio.create_task(refresh_all_data())

    # Schedule periodic refresh
    scheduler = AsyncIOScheduler()
    scheduler.add_job(refresh_all_data, "interval", hours=REFRESH_HOURS)
    scheduler.add_job(cleanup_old_items, "interval", hours=24)
    scheduler.start()

    yield

    scheduler.shutdown()


app = FastAPI(title="AI Pulse", lifespan=lifespan)

# Serve static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/feed")
async def get_feed():
    items = get_items_by_category()
    tldr = get_tldr_top5()
    last_refresh = get_meta("last_refresh")
    return {
        "categories": CATEGORIES,
        "items": items,
        "tldr": tldr,
        "last_refresh": last_refresh,
    }


@app.post("/api/refresh")
async def trigger_refresh():
    asyncio.create_task(refresh_all_data())
    return {"status": "refresh started"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
