"""Smart link fetcher: bare HTTP first, Playwright headless fallback for CN e-commerce.
Proven: 淘宝/天猫 bare fetch -> 4.7KB anti-bot shell; JD -> empty shell; Apple/Amazon -> OK.
Strategy: fetch -> shell/anti-bot detect -> playwright render -> extract text.
"""
import ipaddress
import re
import socket
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


def _is_ssrf_blocked_ip(ip_str: str) -> bool:
    """SSRF 防护（CWE-918 fix）：拒绝私有/回环/链路本地/组播/保留地址。"""
    try:
        ip = ipaddress.ip_address(ip_str.split("%")[0])
    except ValueError:
        return True
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_unspecified or ip.is_reserved):
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        v4 = ip.ipv4_mapped
        if (v4.is_private or v4.is_loopback or v4.is_link_local
                or v4.is_multicast or v4.is_unspecified):
            return True
    return False


def validate_fetch_url(url: str) -> None:
    """任何网络操作前校验目标 URL：仅 http/https，拒绝解析到内网/回环地址。"""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(f"仅支持 http/https 链接: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError("链接缺少主机名")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"无法解析主机: {parsed.hostname}") from e
    for info in infos:
        if _is_ssrf_blocked_ip(info[4][0]):
            raise ValueError("已拦截：目标解析到私有/回环/链路本地地址")


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向每一跳都重新校验目标地址，禁止跳到内网。"""
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        newreq = super().redirect_request(req, fp, code, msg, headers, newurl)
        if newreq is not None:
            validate_fetch_url(newreq.full_url)
        return newreq


def _fetch_bare(url: str, ua: str, timeout: int = 15):
    validate_fetch_url(url)
    req = urllib.request.Request(url, headers={
        "User-Agent": ua,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    opener = urllib.request.build_opener(_ValidatingRedirectHandler())
    with opener.open(req, timeout=timeout) as r:
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
    except ValueError as e:
        return {"ok": False, "blocked": True, "hint": str(e), "title": "", "text": "", "price": "", "method": "blocked"}
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
