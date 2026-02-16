# AI Pulse - Your Daily AI News Aggregator

A sleek web app that aggregates the latest happenings in AI into one place, refreshed daily.

## Categories

- **Trending Repos** — Hot GitHub repositories in AI/ML
- **Models** — Latest models from Hugging Face (chat, code, video, image, audio)
- **News & Articles** — AI news from top sources
- **Research Papers** — Latest papers from ArXiv
- **Tools & Products** — New AI tools and products
- **Community Buzz** — Hot discussions from Reddit & Hacker News

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open http://localhost:8000 in your browser.

## How It Works

- Data is fetched from GitHub, Hacker News, Hugging Face, ArXiv, and Reddit
- Results are cached in SQLite and refreshed every 6 hours
- Items with high engagement ("hype") persist longer on the feed
- The frontend auto-refreshes every 30 minutes

## Data Sources

| Source | What it provides |
|--------|-----------------|
| GitHub Search API | Trending AI/ML repositories |
| Hacker News API | AI-related stories and discussions |
| Hugging Face | Trending and new models |
| ArXiv RSS | Latest AI/ML research papers |
| Reddit RSS | Hot posts from AI subreddits |
