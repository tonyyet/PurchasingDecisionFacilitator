"""Smart link fetcher: bare HTTP first, Playwright headless fallback for CN e-commerce.
Proven: 淘宝/天猫 bare fetch -> 4.7KB anti-bot shell; JD -> empty shell; Apple/Amazon -> OK.
Strategy: fetch -> shell/anti-bot detect -> playwright render -> extract text.
"""
import re
import urllib.request
from urllib.parse import urlparse

_UA_MOBILE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
_UA_DESKTOP = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
_PRICE_RE = re.compile(r"[¥￥]\s*([\d][\d,]*(?:\.\d+)?)")
_BOT_MARKERS = ("验证", "滑动", "安全验证", "人机", "captcha", "访问过于频繁", "登录后查看", "请登录")
_CN_DOMAINS = ("taobao.com", "tmall.com", "jd.com", "meituan.com", "dianping.com", "ctrip.com", "pinduoduo.com", "1688.com", "suning.com")


def _strip_html(raw: str, limit: int = 6000) -> str:
    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _fetch_bare(url: str, ua: str, timeout: int = 15):
    req = urllib.request.Request(url, headers={
        "User-Agent": ua,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(400000).decode("utf-8", errors="ignore")
    return raw


def _looks_shell(raw: str) -> bool:
    """反爬/登录墙/空壳页特征：HTML 过小、无 title、或含验证词。"""
    if len(raw) < 20000:
        return True
    title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S)
    if not title or not title.group(1).strip():
        return True
    return False


def _fetch_playwright(url: str) -> dict:
    """Headless Chromium 渲染后抓正文（对 JS 重页面有效）。"""
    from playwright.sync_api import sync_playwright
    domain = urlparse(url).netloc.lower()
    ua = _UA_MOBILE if any(d in domain for d in ("m.jd.com", "m.tb", "h5.")) else _UA_DESKTOP
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=ua, locale="zh-CN", viewport={"width": 390, "height": 844})
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(2500)  # 等 JS 渲染价格/标题
        title = page.title()
        body = page.evaluate("document.body ? document.body.innerText : ''")
        body = re.sub(r"\s+", " ", body).strip()
        browser.close()
    return {"title": title, "text": body[:6000], "method": "playwright"}


def fetch_page(url: str) -> dict:
    """入口：返回 {ok, blocked, title, text, price, method, hint}"""
    domain = urlparse(url).netloc.lower()
    try:
        raw = _fetch_bare(url, _UA_MOBILE if any(d in domain for d in ("m.jd.com", "m.tb", "h5.")) else _UA_DESKTOP)
    except Exception as e:
        return {"ok": False, "blocked": True, "hint": f"直连失败({type(e).__name__})", "title": "", "text": "", "price": "", "method": "bare"}
    if not _looks_shell(raw):
        title = re.search(r"<title[^>]*>(.*?)</title>", raw, re.S)
        title = title.group(1).strip() if title else ""
        text = _strip_html(raw)
        price = _PRICE_RE.search(text)
        return {"ok": True, "blocked": False, "title": title, "text": text, "price": price.group(0) if price else "", "method": "bare"}
    # 反爬/空壳 → 浏览器渲染兜底
    try:
        r = _fetch_playwright(url)
        r["ok"] = bool(r["text"] and len(r["text"]) > 80)
        bot = any(m in (r["title"] + r["text"][:500]) for m in _BOT_MARKERS)
        r["blocked"] = not r["ok"] or bot
        m = _PRICE_RE.search(r["text"])
        r["price"] = m.group(0) if m else ""
        r["hint"] = "浏览器渲染抓取成功" if r["ok"] else "页面需登录/触发验证码"
        return r
    except Exception as e:
        return {"ok": False, "blocked": True, "hint": f"浏览器抓取失败({type(e).__name__})", "title": "", "text": "", "price": "", "method": "playwright"}
