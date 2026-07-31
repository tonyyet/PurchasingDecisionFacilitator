# Purchasing Decision Facilitator

AI-powered **3-level purchasing decision analysis** that helps consumers cut through marketing hype before buying:

- **Level 1 — First-principles fact-check**: decomposes every verifiable product claim into physics/engineering, economics, and logic/evidence dimensions, with explicit confidence levels. Optional live web search reduces hallucination.
- **Level 2 — Intent matching**: decodes the seller's psychological triggers, what they hope you ignore, and whether the product actually fits *your* real need (match score 1–10).
- **Level 3 — Alternatives**: weighted multi-dimensional comparison of alternatives (same category, cross-category, and the "zero option" — buy nothing, change behavior).

Built as a plain-Python pipeline on [Pydantic AI](https://ai.pydantic.dev/) with typed structured output — fully controllable end-to-end.

## Features

- 📱 Mobile-first web UI (iPhone-friendly), Chinese interface
- 📸 Screenshot analysis: upload a product page screenshot → vision LLM / OCR extracts name, price, claims (bypasses anti-bot walls)
- 🔗 Smart link fetching: bare HTTP first, headless-browser fallback for Chinese e-commerce login walls (Taobao/JD/Meituan…)
- 🌐 Optional real-time web search (Firecrawl) for Level 1 fact-checking and Level 3 alternatives with source URLs
- ⚡ / 📖 **Detail level toggle**: choose concise or lengthy explanations for all three levels before submitting
- ⏳ Async task API: analysis takes 1–2 min; POST returns a `task_id`, frontend polls — works behind slow reverse proxies
- 🔒 Keys via environment variables only; nothing is hardcoded or logged

## Architecture

```
iPhone browser / curl
      │  HTTPS
      ▼
FastAPI (uvicorn)
   main.py           routes: GET /, POST /analyze, POST /analyze_image, GET /task/{id}, GET /health
   orchestrator.py   business logic: run_pipeline (L1 → L2 → L3)
   agents.py         3 typed agents (pydantic-ai + DeepSeek, thinking disabled)
   schemas.py        pydantic input/output contracts
   fetcher.py        smart link fetch (bare HTTP → Playwright fallback)
   image_extractor.py  screenshot → vision/OCR/heuristic extraction
   search_utils.py   Firecrawl web search (cached, fail-open)
   static/index.html mobile web UI
```

## Quick start

```bash
git clone <your-fork-url>
cd PurchasingDecisionFacilitator
python -m venv .venv && .venv\Scripts\activate   # Windows, or source .venv/bin/activate
pip install -r requirements.txt

# configure keys (never commit them)
copy .env.example .env   # then fill in DEEPSEEK_API_KEY

# run
cd app
python -m uvicorn main:app --host 127.0.0.1 --port 8501
```

Open http://127.0.0.1:8501

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | ✅ | – | DeepSeek API key (platform.deepseek.com) |
| `DEEPSEEK_API_BASE` | | `https://api.deepseek.com` | OpenAI-compatible base URL |
| `DEEPSEEK_MODEL` | | `deepseek-chat` | Model name |
| `FIRECRAWL_API_KEY` | | – | Enables live web search (firecrawl.dev) |
| `MIMO_CONFIG` | | – | Path to vision-provider JSON (screenshot analysis) |

> Note: the server reads environment variables directly. If you use a `.env` file, load it yourself (e.g. `python -c "from dotenv import load_dotenv; load_dotenv()"` or your shell tooling) — the app intentionally does not auto-load `.env`.

## API

### `POST /analyze` — text/link based analysis

```json
{
  "product": {"name": "...", "claims": "...", "price": "...", "link": "https://..."},
  "user": {"real_need": "...", "budget": "...", "scenario": "..."},
  "detail": "concise"
}
```

- `detail`: `"concise"` (default) | `"lengthy"`
- If `user` is all-empty → returns `status=need_user_input` (Level 1 first, then continues)
- Response: `{"task_id": "..."}` → poll `GET /task/{task_id}` until `status=done` (result contains `level1/level2/level3`)

### `POST /analyze_image` — screenshot analysis

```json
{"image_b64": "<base64 jpeg/heic>", "detail": "concise"}
```

### `GET /task/{task_id}`, `GET /health`

## Screenshot pipeline (optional)

1. Vision LLM (OpenAI-compatible, e.g. Xiaomi Mimo v2.5) via `MIMO_CONFIG`
2. Fallback: local OCR + DeepSeek text parse
3. Fallback: heuristic regex (price/name)

## Tech stack

FastAPI · Pydantic AI · DeepSeek · Firecrawl · Playwright · Pillow · Vanilla JS (no frontend framework)

## License

[MIT](LICENSE) — use it, fork it, learn from it. Pull requests welcome.
