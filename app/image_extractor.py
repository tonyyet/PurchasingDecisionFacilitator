"""Screenshot -> product info extractor.
Primary: Xiaomi Mimo 2.5 vision LLM (OpenAI-compatible, structured JSON direct from image).
Fallback 1: local RapidOCR (Chinese) + DeepSeek text LLM parse.
Fallback 2: heuristic parse (price regex + line classification).
Mimo config file (reference-only, key read at runtime, never logged/committed):
  env MIMO_CONFIG -> path to JSON
  {"providers": {"Xiaomi": {"baseUrl": ".../v1", "apiKey": "...", "models": [{"id": "mimo-v2.5"}]}}}
Usage: extract_from_image(path_or_PIL) -> {name, price, claims, page_text, method}
"""
import base64
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

_CUR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CUR))

# ---------- image normalization (HEIC aware -> JPEG bytes) ----------
def _to_jpeg_bytes(src) -> bytes:
    try:
        from PIL import Image
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        Image = None
    if hasattr(src, "read"):
        data = src.read()
    elif isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        data = Path(src).read_bytes()
    if Image is None:
        return data  # no PIL: pass through, Mimo may still accept
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    # downscale huge photos (iPhone screenshots ~1200px+ are fine for VLM)
    max_side = 1600
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ---------- Mimo vision (primary) ----------
def _mimo_config():
    p = os.environ.get("MIMO_CONFIG")
    if not p:
        raise ValueError("MIMO_CONFIG env not set (path to OpenAI-compatible vision provider JSON)")
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    x = cfg["providers"]["Xiaomi"]
    return x["baseUrl"], x["apiKey"]


def _mimo_extract(jpeg: bytes, timeout=150):
    base_url, api_key = _mimo_config()
    b64 = base64.b64encode(jpeg).decode()
    payload = {
        "model": "mimo-v2.5",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": (
                "这是一张电商商品页截图。请提取：1) 商品名称 2) 价格（数字即可）"
                "3) 页面宣传的核心卖点/宣称（逐条列出，不遗漏）。"
                '只输出JSON，不要多余文字：{"name":"","price":"","claims":["..."]}'
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 1500,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode())
    content = resp["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError(f"Mimo 返回非 JSON: {content[:200]}")
    d = json.loads(m.group(0))
    claims = d.get("claims") or []
    if isinstance(claims, str):
        claims = [c for c in re.split(r"[;；\n]", claims) if c.strip()]
    return {
        "name": str(d.get("name", "")).strip(),
        "price": str(d.get("price", "")).strip(),
        "claims": "; ".join(str(c).strip() for c in claims if str(c).strip()),
        "page_text": "; ".join(str(c).strip() for c in claims if str(c).strip()),
    }


# ---------- fallback: OCR + DeepSeek text parse ----------
def _ocr_extract(jpeg: bytes):
    from ocr_utils import ocr_image  # optional local OCR helper (not required to run)
    from agents import _model, DEEPSEEK_SETTINGS  # reuse working DeepSeek text model

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.write(jpeg)
    tmp.close()
    try:
        text = ocr_image(tmp.name)
    finally:
        os.unlink(tmp.name)
    if not text.strip():
        raise ValueError("OCR 无有效文本")
    prompt = (
        "从以下电商商品页 OCR 文本中提取 商品名称/价格/宣传卖点。只输出JSON："
        '{"name":"","price":"","claims":["..."]}\n\n' + text[:6000]
    )
    import pydantic_ai
    from pydantic_ai import Agent
    from pydantic import BaseModel

    class _R(BaseModel):
        name: str = ""
        price: str = ""
        claims: list[str] = []

    a = Agent(model=_model, model_settings=DEEPSEEK_SETTINGS, output_type=_R)
    res = a.run_sync(prompt)
    d = res.output
    claims = d.claims if isinstance(d.claims, list) else [d.claims]
    return {
        "name": (d.name or "").strip(),
        "price": (d.price or "").strip(),
        "claims": "; ".join(str(c).strip() for c in claims if str(c).strip()),
        "page_text": text[:4000],
    }


# ---------- fallback: heuristic ----------
_PRICE_RE = re.compile(r"(?:¥|￥|RMB|rmb|价格[:：]?\s*)?(\d[\d,，]*\.?\d*)")
def _heuristic(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    price = ""
    for l in lines:
        m = _PRICE_RE.search(l)
        if m and len(m.group(1).replace(",", "").replace("，", "")) >= 2:
            price = m.group(1)
            break
    name = lines[0][:120] if lines else ""
    claims = "; ".join(lines[1:6])
    return {"name": name, "price": price, "claims": claims, "page_text": text[:4000]}


def extract_from_image(path_or_pil):
    """主入口。依次尝试: Mimo 视觉 -> RapidOCR+DeepSeek -> 启发式。"""
    try:
        jpeg = _to_jpeg_bytes(path_or_pil)
        try:
            d = _mimo_extract(jpeg)
            if not (d["name"] or d["claims"]):
                raise ValueError("Mimo 返回空结果")
            return {**d, "ok": True, "method": "mimo-vision",
                    "hint": "Mimo 2.5 视觉理解成功"}
        except Exception as e_mimo:
            try:
                d = _ocr_extract(jpeg)
                if not (d["name"] or d["claims"]):
                    raise ValueError("OCR 解析为空")
                return {**d, "ok": True, "method": "rapidocr+llm",
                        "hint": f"Mimo失败({str(e_mimo)[:60]})→RapidOCR+DeepSeek 成功"}
            except Exception as e_ocr:
                text = _ocr_text_soft(jpeg)
                h = _heuristic(text) if text else {"name": "", "price": "", "claims": "", "page_text": ""}
                ok = bool(h["name"] or h["price"])
                return {**h, "ok": ok,
                        "hint": (f"Mimo({str(e_mimo)[:60]}) / OCR({str(e_ocr)[:60]}) 均失败"
                                + ("→启发式抽取" if ok else f"，文本: {text[:60]}"))}
    except Exception as e:
        return {"ok": False, "hint": f"识别失败: {str(e)[:200]}",
                "name": "", "price": "", "claims": "", "page_text": "", "method": "none"}


def _ocr_text_soft(jpeg: bytes) -> str:
    try:
        from ocr_utils import ocr_image  # optional local OCR helper
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(jpeg)
        tmp.close()
        try:
            t = ocr_image(tmp.name)
        finally:
            os.unlink(tmp.name)
        return t or ""
    except Exception:
        return ""
