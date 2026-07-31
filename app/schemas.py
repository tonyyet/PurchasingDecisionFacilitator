"""Pydantic models = the input/output contract for the whole stack."""
from typing import Literal

from pydantic import BaseModel, Field

# ---------- Input ----------
class ProductInfo(BaseModel):
    name: str = ""
    claims: str = ""          # 宣传核心卖点/描述
    price: str = ""           # 价格
    link: str = ""            # 商品链接
    page_text: str = ""       # 从链接抓取的正文（可选）
    image_ocr: str = ""       # 图片 OCR 文本（可选）

class UserContext(BaseModel):
    real_need: str = ""       # 真实需求/痛点
    budget: str = ""          # 预算范围
    scenario: str = ""        # 使用场景/频率

class AnalysisRequest(BaseModel):
    product: ProductInfo
    user: UserContext = Field(default_factory=UserContext)
    detail: Literal["concise", "lengthy"] = "concise"  # 解释详细程度：简洁(默认)/详细
    language: Literal["zh", "en"] = "zh"  # 界面/输出语言：中文(默认)/英文

# ---------- Level 1 ----------
class ClaimCheck(BaseModel):
    claim: str
    verdict: str = ""
    confidence: str = ""      # 高/中/低
    source: str = ""          # 来源完整URL（必须以 http 开头）；无检索来源则留空，禁止填机构名/描述

class Level1Report(BaseModel):
    claims: list[ClaimCheck] = []
    physics: str = ""         # 物理/工程分析
    economics: str = ""       # 经济/商业分析
    logic: str = ""           # 逻辑/证据分析
    overall_confidence: str = ""
    likely_exaggeration: str = ""
    realistic_range: str = ""

# ---------- Level 2 ----------
class MatchItem(BaseModel):
    item: str
    detail: str = ""

class Level2Report(BaseModel):
    seller_target: str = ""          # 针对谁设计
    tactics: list[str] = []          # 心理技巧
    hidden_info: list[str] = []      # 有意忽略
    surface_need: str = ""
    deep_need: str = ""
    core_problem: str = ""
    match_score: int = 0             # 1-10
    matches: list[MatchItem] = []
    mismatches: list[MatchItem] = []
    overkill: str = ""

# ---------- Level 3 ----------
class NeedDim(BaseModel):
    name: str
    weight: str = "中"

class Alternative(BaseModel):
    name: str
    price: str = ""
    pros: list[str] = []
    cons: list[str] = []
    match: int = 0
    source: str = ""          # 来源完整URL（必须以 http 开头）；无检索来源则留空，禁止填机构名/描述

class Level3Report(BaseModel):
    key_needs: list[NeedDim] = []
    alternatives: list[Alternative] = []
    zero_plan: str = ""
    final_recommendation: str = ""

# ---------- Result ----------
class PipelineResult(BaseModel):
    status: str = "ok"               # ok | need_user_input
    message: str = ""
    product_summary: str = ""
    level1: Level1Report | None = None
    level2: Level2Report | None = None
    level3: Level3Report | None = None
