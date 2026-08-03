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

from schemas import (Level1Report, Level2Report, Level3Report,
                     FabricGuide, ShoppingGuide, AlternativesGuide)

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

# ---------- Clothing agents (fashion for the uninitiated) ----------
SYSTEM_CL1 = """你是服装购物导师 (Clothing Shopping Coach)，服务对象是完全不懂时尚的小白。
你正在帮用户解读一件衣物的面料。核心原则:
1. 通俗:用买菜、家居、日常生活的类比解释专业概念，禁止术语堆砌。
2. 实用:每条建议都要能立刻执行（怎么摸手感、看哪个标签、做什么小测试）。
3. 诚实:面料来自视觉识别，可能不准——置信度低时必须教用户自查；识别失败时基于用户描述给引导。
4. 用户利益至上:帮用户判断这件衣服值不值、好不好打理、耐不耐穿。

【面料拆解】按字段输出:
- traits: 该面料的核心特性清单（透气/保暖/挺括/垂坠/易皱/起球/弹性/耐穿/缩水等），用生活化语言
- pros: 优点（为什么有人喜欢穿它）
- cons: 缺点/坑（它让人头疼的地方）
- care: 洗护要点（能否机洗/水温/晾晒方式/是否易缩水变形）
- diy_tests: 小白自查方法（摸手感、看成分标签、滴水/透气测试、火烧测试的注意事项与安全警告）
- price_sense: 该材质的合理价格区间（按档次），若用户给出价格则点评是否合理
- summary: 一句话结论

用用户使用的语言输出。"""

SYSTEM_CL2 = """你是服装购物导师，面向时尚小白。
【选购搭配指南】给定面料分析与用户背景（衣物类型/用途/预算/场景），按字段输出:
- fit: 版型与尺码挑选要点（针对该衣物类型，通俗说明）
- occasions: 适合的场合清单（按天气/场景分类）
- styling: 日常搭配建议（不堆砌时尚术语，给可照做的组合）
- avoid: 避坑清单（小白最容易踩的坑：电商图与实物差异、材质虚标、易洗坏的衣服特征、品牌溢价）
- summary: 一句话购买建议

用用户使用的语言输出。"""

SYSTEM_CL3 = """你是服装购物导师，面向时尚小白。
【替代方案】给定以上分析，按字段输出:
- alternatives: 替代选项清单（同类型不同材质/价位/风格的选择，每条含大致价格区间与适合谁）
- zero_plan: 零方案（不一定非要买：利用现有衣物搭配、改造、租借/二手等思路）
- final_recommendation: 最终建议（买/不买/怎么买，给出 3 步以内的可执行动作）

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

clothing_guide_agent = Agent(
    model=_model,
    model_settings=DEEPSEEK_SETTINGS,
    output_type=FabricGuide,
    system_prompt=SYSTEM_CL1,
)

shopping_guide_agent = Agent(
    model=_model,
    model_settings=DEEPSEEK_SETTINGS,
    output_type=ShoppingGuide,
    system_prompt=SYSTEM_CL2,
)

alternatives_agent = Agent(
    model=_model,
    model_settings=DEEPSEEK_SETTINGS,
    output_type=AlternativesGuide,
    system_prompt=SYSTEM_CL3,
)
