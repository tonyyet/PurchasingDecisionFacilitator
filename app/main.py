"""FastAPI web layer — serves the mobile page + JSON API.

异步任务设计：公网 trycloudflare 单请求约 100s 超时（524），而全链路分析耗时
120s+（图片识别 + 3 级 Agent）。因此 POST 立即返回 task_id，前端轮询 /task/{id}。
"""
import asyncio
import base64
import os
import re
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from schemas import AnalysisRequest, PipelineResult
from orchestrator import run_pipeline
from fetcher import fetch_page
from image_extractor import extract_from_image

APP_DIR = Path(__file__).parent
app = FastAPI(title="Purchasing Decision Facilitator", version="0.4.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


# 全局兜底：任何未捕获异常都返回 JSON（避免 FastAPI 默认纯文本 500 让前端 r.json() 崩溃）
@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    print(f"[500] {request.method} {request.url.path} -> {detail}", flush=True)
    lang = request.query_params.get("lang", "zh")
    msg = f"服务内部错误: {detail}" if lang != "en" else f"Internal server error: {detail}"
    return JSONResponse(status_code=500, content={"detail": msg})

import tempfile


def _need_input(msg: str) -> PipelineResult:
    """返回前端可展示的引导信息（非错误）。"""
    return PipelineResult(status="need_user_input", message=msg)


# ---- 异步任务存储（内存版）----
_TASKS: dict[str, dict] = {}


async def _task_run(task_id: str, fn, *args, lang: str = "zh"):
    """后台执行 fn，更新任务状态。"""
    _TASKS[task_id] = {"status": "running"}
    try:
        result = await asyncio.wait_for(fn(*args), timeout=360)
        _TASKS[task_id] = {"status": "done", "result": result}
    except asyncio.TimeoutError:
        _TASKS[task_id] = {
            "status": "error",
            "error": "分析超时，请稍后重试" if lang == "zh" else "Analysis timed out, please try again later",
        }
    except Exception as e:
        _TASKS[task_id] = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"[task {task_id}] FAIL: {_TASKS[task_id]['error']}", flush=True)


async def _task_full(req: AnalysisRequest) -> PipelineResult:
    """完整分析（可选抓链接 + 3-Level 报告）。"""
    if req.product.link and not req.product.page_text:
        f = await asyncio.to_thread(fetch_page, req.product.link)
        if f.get("blocked"):
            if req.language == "en":
                return _need_input(
                    f"Link fetching blocked: {f.get('hint','')}. Platforms like Taobao/JD use anti-bot "
                    "and login walls — upload a product-page screenshot instead for a smoother experience "
                    "(screenshots bypass anti-bot restrictions)."
                )
            return _need_input(
                f"链接抓取被拦截：{f.get('hint','')}。淘宝/京东等平台有反爬与登录墙，"
                "推荐直接「上传商品页截图」体验更稳（截图路径不受反爬影响）。"
            )
        req.product.page_text = f"标题: {f.get('title','')}\n价格: {f.get('price','')}\n内容: {f.get('text','')}"
        req.product.price = req.product.price or f.get("price", "")
    return await run_pipeline(req)


async def _task_image(img_path: str, detail: str = "concise", language: str = "zh") -> PipelineResult:
    """截图 → 视觉/OCR 抽取 → 3-Level 报告。"""
    ex = await asyncio.to_thread(extract_from_image, img_path)
    if not ex.get("ok"):
        if language == "en":
            return _need_input(
                f"Image recognition failed: {ex.get('hint','')} (please upload a clear, complete screenshot of the product page)"
            )
        return _need_input(f"图片识别失败：{ex.get('hint','')}（请上传清晰完整的商品页截图）")
    req = AnalysisRequest(
        product={"name": ex.get("name", ""), "claims": ex.get("claims", ""),
                 "price": ex.get("price", ""), "page_text": ex.get("page_text", "")},
        user={"real_need": "", "budget": "", "scenario": ""},
        detail=detail if detail in ("concise", "lengthy") else "concise",
        language=language if language in ("zh", "en") else "zh",
    )
    return await run_pipeline(req)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(req: AnalysisRequest):
    """异步全流程分析：立即返回 task_id，前端轮询 /task/{id}。"""
    task_id = uuid.uuid4().hex
    _TASKS[task_id] = {"status": "pending"}
    asyncio.create_task(_task_run(task_id, _task_full, req, lang=req.language))
    return {"task_id": task_id}


class ImageUpload(BaseModel):
    """前端 canvas 压缩后以 base64 上传（绕开 cloudflared 隧道对大 multipart 不可靠的问题）。"""
    image_b64: str
    detail: str = "concise"
    language: str = "zh"


@app.post("/analyze_image")
async def analyze_image(upload: ImageUpload):
    """异步截图分析（base64 JSON 版）：立即返回 task_id。"""
    try:
        data = base64.b64decode(upload.image_b64)
    except Exception:
        raise HTTPException(400, "图片数据解码失败，请重试")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "图片过大（>15MB），请压缩后重试")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(data)
    tmp.close()
    task_id = uuid.uuid4().hex
    _TASKS[task_id] = {"status": "pending"}
    asyncio.create_task(_task_image_cleanup(task_id, tmp.name, upload.detail, upload.language))
    return {"task_id": task_id}


async def _task_image_cleanup(task_id: str, img_path: str, detail: str = "concise", language: str = "zh"):
    try:
        await _task_run(task_id, _task_image, img_path, detail, language, lang=language)
    finally:
        try:
            os.unlink(img_path)
        except Exception:
            pass


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    t = _TASKS.get(task_id)
    if not t:
        raise HTTPException(404, "任务不存在或已过期")
    return {"status": t["status"], "result": t.get("result"), "error": t.get("error")}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "purchasing-decision-facilitator"}
