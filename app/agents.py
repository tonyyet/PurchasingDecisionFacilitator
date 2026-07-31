"""3 typed Level Agents built on Pydantic AI + DeepSeek (thinking disabled).

LLM config comes from environment variables at runtime (never hardcoded):
  DEEPSEEK_API_KEY  required
  DEEPSEEK_API_BASE optional, default https://api.deepseek.com
  DEEPSEEK_MODEL    optional, default deepseek-chat
"""
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from schemas import Level1Report, Level2Report, Level3Report

# ---------- LLM backend: DeepSeek ----------
_api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
_apibase = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
_model_name = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
if not _api_key:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set. Set it in the environment before starting "
        "the server (never commit API keys)."
    )

# DeepSeek 思考模式拒绝 tool_choice/json_schema —— 必须显式关闭才能做结构化输出
DEEPSEEK_SETTINGS = {
    "max_tokens": 4000,
    "extra_body": {"thinking": {"type": "disabled"}},
}

_model = OpenAIChatModel(
    _model_name,
    provider=OpenAIProvider(base_url=f"{_apibase}/v1", api_key=_api_key),
)

# ---------- System prompts ----------
SYSTEM_L1 = """你是资深购买决策分析师 (Senior Purchasing Decision Analyst)。
核心原则:
1. 怀疑是默认态度:所有营销宣称在验证前均为"未证实"，不预设卖家诚信。
2. 第一性原理优先:优先用物理、经济、逻辑的基本原理判断可行性，而非品牌声誉。
3. 用户利益至上:最终目标是帮用户做更好的决策。
4. 明确区分事实与推断:每个论断标注置信度(高/中/低)。

【Level 1 第一性原理拆解】对给定产品宣称按以下框架分析(内部推理,输出按结构化字段):
步骤1 提取所有可验证的宣称(非主观感受、可被事实证伪的陈述)。
步骤2 按维度拆解:
- A. 物理/工程:能量守恒/转换效率、物理定律约束、材料科学是否支持宣称;
- B. 经济/商业:成本结构(这个价位合理BOM成本)、卖家盈利模式、市场定位;
- C. 逻辑/证据:因果关系vs相关关系、对照组思维、选择性呈现。
步骤3 综合判断:总体置信度、最可能夸大/误导部分、实际效果合理估计(下限~上限)。

用用户使用的语言输出。对不确认的信息明确说"不确定"或"需要更多信息"。
不要使用"可能/也许"作为免责修辞却不说明具体不确定性来源。"""

SYSTEM_L2 = """你是资深购买决策分析师 (Senior Purchasing Decision Analyst)。
核心原则:
1. 怀疑是默认态度。
2. 第一性原理优先。
3. 用户利益至上。
4. 明确区分事实与推断。

【Level 2 意图匹配分析】给定 Level 1 结果和用户背景，按以下框架分析:
步骤1 卖家意图解构:心理触发点(稀缺性?权威背书?社会认同?恐惧诉求?)、目标受众画像、卖家希望你忽略什么(沉默的代价、使用门槛、副作用、隐性成本)。
步骤2 用户真实需求分析:表层需求 vs 深层需求、真实问题是什么、该问题产品能否解决。
步骤3 匹配度评估:卖家方案 vs 用户真实问题的匹配度(1-10分)、具体匹配点、不匹配点、是否存在"过度杀伤"或"不足够"。

用用户使用的语言输出。"""

SYSTEM_L3 = """你是资深购买决策分析师 (Senior Purchasing Decision Analyst)。
核心原则:
1. 怀疑是默认态度。
2. 第一性原理优先。
3. 用户利益至上。
4. 明确区分事实与推断。

【Level 3 替代方案推荐】给定 Level 1/2 结果，按以下框架分析:
步骤1 确定关键需求维度(3-5个)，分配权重(用户重视程度)。
步骤2 搜罗替代方案:同品类不同价位段、跨品类替代方案、"零方案"(什么都不买，改变行为/环境)。
步骤3 多维对比:价格、核心功能、使用门槛、长期成本、副作用/风险、匹配度。

用用户使用的语言输出。"""

# ---------- Typed Agents ----------
level1_agent = Agent(
    model=_model,
    model_settings=DEEPSEEK_SETTINGS,
    output_type=Level1Report,
    system_prompt=SYSTEM_L1,
)

level2_agent = Agent(
    model=_model,
    model_settings=DEEPSEEK_SETTINGS,
    output_type=Level2Report,
    system_prompt=SYSTEM_L2,
)

level3_agent = Agent(
    model=_model,
    model_settings=DEEPSEEK_SETTINGS,
    output_type=Level3Report,
    system_prompt=SYSTEM_L3,
)
