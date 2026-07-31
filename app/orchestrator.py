"""Business logic layer — the whole pipeline is plain Python, fully controllable."""
import asyncio
from schemas import AnalysisRequest, PipelineResult, ProductInfo, UserContext
from agents import level1_agent, level2_agent, level3_agent
from search_utils import web_search


def _fmt_results(results) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r.get('title','')}]({r.get('url','')}): {r.get('text','')}")
    return "\n".join(lines) if lines else ""


async def _search_parallel(queries):
    """Run web_search for each query (each in a thread, never blocking the loop)."""
    if not queries:
        return {}
    outs = await asyncio.gather(*(asyncio.to_thread(web_search, q) for q in queries))
    return dict(zip(queries, outs))


def _build_product_text(product: ProductInfo) -> str:
    parts = []
    if product.name:
        parts.append(f"产品名称: {product.name}")
    if product.claims:
        parts.append(f"宣传核心卖点: {product.claims}")
    if product.price:
        parts.append(f"价格: {product.price}")
    if product.link:
        parts.append(f"商品链接: {product.link}")
    if product.page_text:
        parts.append(f"页面内容:\n{product.page_text[:6000]}")
    if product.image_ocr:
        parts.append(f"图片OCR文本:\n{product.image_ocr[:3000]}")
    return "\n".join(parts) if parts else "(无产品信息)"


def _build_user_text(user: UserContext) -> str:
    parts = []
    if user.real_need:
        parts.append(f"我的真实需求/痛点: {user.real_need}")
    if user.budget:
        parts.append(f"我的预算范围: {user.budget}")
    if user.scenario:
        parts.append(f"我的使用场景/频率: {user.scenario}")
    return "\n".join(parts) if parts else "(未提供用户背景)"


def _detail_instruction(detail: str) -> str:
    if detail == "lengthy":
        return (
            "## 输出篇幅要求（详细模式）\n"
            "用户选择了详细解释：请在信息完整的前提下充分展开每个要点——说明推理依据、关键证据/来源、"
            "适用边界与潜在例外；可引用具体数字、机制原理与上下文关联；避免空话，但不要刻意压缩。"
        )
    return (
        "## 输出篇幅要求（简洁模式）\n"
        "用户选择了简洁解释：每个要点控制在 1-2 句话，使用短句与要点列表；"
        "不做长篇论证、不展开背景知识、不重复用户输入；在保持结论完整的前提下输出必要最短长度。"
    )


async def run_pipeline(req: AnalysisRequest) -> PipelineResult:
    """Level 1 -> Level 2 -> Level 3. 业务规则全部写在这里，显式可控。"""
    product_text = _build_product_text(req.product)
    user_text = _build_user_text(req.user)
    detail_inst = _detail_instruction(req.detail)

    # --- 联网检索①：Level 1 事实核查材料（搜索失败/无结果则静默降级，不阻塞主流程）---
    l1_queries = []
    if req.product.name:
        l1_queries.append(f"{req.product.name} 价格 参数 评价")
        l1_queries.append(f"{req.product.name} 缺点 争议")
    l1_search = await _search_parallel(l1_queries)
    l1_evidence = "\n\n".join(
        f"### 检索: {q}\n{_fmt_results(rs)}" for q, rs in l1_search.items() if rs
    )

    # --- Level 1: 第一性原理拆解（始终先跑）---
    l1_prompt = f"请对这个产品进行 Level 1 第一性原理分析:\n\n{product_text}\n\n{detail_inst}"
    if l1_evidence:
        l1_prompt += (
            "\n\n## 联网核查材料（实时搜索结果，可引用其中的 URL；如与页面宣称冲突，以真实来源为准）\n"
            + l1_evidence
            + "\n\n输出规则：凡基于上述检索材料得出的结论，必须在对应 claim 的 source 字段填来源的完整 URL（http 开头）；基于你自己知识的结论 source 留空。"
        )
    else:
        l1_prompt += "\n\n## 联网核查材料\n(本次检索不可用；请基于已有信息分析，并在不确定处明确标注)"
    l1_res = await level1_agent.run(l1_prompt)
    l1 = l1_res.output

    # --- Level 2: 意图匹配（用户背景缺失时用中性默认值继续，保证 iPhone 截图直传也能出完整报告）---
    if not (req.user.real_need or req.user.budget or req.user.scenario):
        user_text = "用户背景: 浏览者对该产品感兴趣，想了解它是否值得购买，希望获得客观拆解与替代方案建议（未提供更多背景）。"

    l2_prompt = (
        f"给定以下产品的 Level 1 分析结论和用户背景，进行 Level 2 意图匹配分析:\n\n"
        f"## 产品信息\n{product_text}\n\n"
        f"## Level 1 结论\n{l1.model_dump_json(indent=2)}\n\n"
        f"## 用户背景\n{user_text}\n\n{detail_inst}"
    )
    l2_res = await level2_agent.run(l2_prompt)
    l2 = l2_res.output

    # --- 联网检索②：Level 3 真实替代品候选（失败则降级为模型知识推荐）---
    l3_queries = []
    if req.product.name:
        l3_queries.append(f"{req.product.name} 同类 替代品 推荐")
        if req.product.price:
            l3_queries.append(f"{req.product.name} {req.product.price} 同价位 对比")
    l3_search = await _search_parallel(l3_queries)
    l3_evidence = "\n\n".join(
        f"### 检索: {q}\n{_fmt_results(rs)}" for q, rs in l3_search.items() if rs
    )

    # --- Level 3: 替代方案推荐 ---
    l3_prompt = (
        f"给定以下 Level 1/2 分析结论，进行 Level 3 替代方案推荐:\n\n"
        f"## 产品信息\n{product_text}\n\n"
        f"## Level 1 结论\n{l1.model_dump_json(indent=2)}\n\n"
        f"## Level 2 结论\n{l2.model_dump_json(indent=2)}\n\n"
        f"## 用户背景\n{user_text}\n\n{detail_inst}"
    )
    if l3_evidence:
        l3_prompt += (
            "\n\n## 实时搜索结果：替代品候选（优先从这些真实在售/有来源的商品中选取，注明价格与来源 URL；"
            "候选不足时再补充你自己的知识并明确标注「非检索来源」）\n" + l3_evidence
            + "\n\n输出规则：凡来自上述检索结果的替代品，必须在对应 alternative 的 source 字段填来源完整 URL（http 开头）；"
            "来自你自己知识的替代品 source 留空并标注「非检索来源」。"
        )
    else:
        l3_prompt += (
            "\n\n## 实时搜索结果：替代品候选\n(本次检索不可用；请基于已有知识推荐，并明确标注各商品价格来源为估算)"
        )
    l3_res = await level3_agent.run(l3_prompt)
    l3 = l3_res.output

    return PipelineResult(
        status="ok",
        message="分析完成。",
        product_summary=product_text,
        level1=l1,
        level2=l2,
        level3=l3,
    )
