# my_agent.py
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU


class MockLLM:
    def __call__(self, messages, temperature=0.0, max_tokens=200):
        # Return fixed output for testing
        return (
            "stars: 4.5\n"
            "review: This is a mock review. The mouse performs well and feels comfortable."
        )

###########################################
# 1. Planner - 决定「做哪些步骤」
###########################################
class MyPlanner(PlanningBase):
    """
    A smarter planner that inherits PlanningBase
    and builds a multi-step execution plan.
    """

    def __init__(self, llm):
        super().__init__(llm=llm)

    def __call__(self, task_description):
        user_id = task_description["user_id"]
        item_id = task_description["item_id"]

        self.plan = [
            {
                "step": 1,
                "description": "Load the user profile.",
                "reasoning instruction": "Retrieve user metadata.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_user",
                    "args": {"user_id": user_id}
                }
            },
            {
                "step": 2,
                "description": "Load all historical reviews written by this user.",
                "reasoning instruction": "Gather user review history to infer habits.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_reviews",
                    "args": {"user_id": user_id}
                }
            },
            {
                "step": 3,
                "description": "Load item information.",
                "reasoning instruction": "Understand the item the user will review.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_item",
                    "args": {"item_id": item_id}
                }
            },
            {
                "step": 4,
                "description": "Load all reviews for this item.",
                "reasoning instruction": "Get background context and common opinions.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_reviews",
                    "args": {"item_id": item_id}
                }
            },
            {
                "step": 5,
                "description": "Analyze item rating distribution (mean, variance, skewness).",
                "reasoning instruction": "Get background context and common opinions.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_reviews",
                    "args": {"item_id": item_id}
                }
            },
            {
                "step": 6,
                "description": "Construct a persona for the user.",
                "reasoning instruction": "Analyze writing style and rating behavior.",
                "tool use instruction": {
                    "tool": "persona_builder.build",
                    "args": {}
                }
            },
            {
                "step": 7,
                "description": "Retrieve semantically relevant reviews and key item features.",
                "reasoning instruction": "Find similar past reviews & extract item characteristics.",
                "tool use instruction": {
                    "tool": "retriever.retrieve",
                    "args": {}
                }
            },
            {
                "step": 8,
                "description": "Generate final rating and review text.",
                "reasoning instruction": "Use reasoning to produce a realistic review.",
                "tool use instruction": {
                    "tool": "reasoner.generate",
                    "args": {}
                }
            }
        ]

        return self.plan



###########################################
# 2. Retriever - 检索用户记忆、相似评论（embedding等）
###########################################
from sentence_transformers import SentenceTransformer, util
import numpy as np

class MyRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
    
    def store_reviews(self, reviews):
        """存储 item reviews 到 retriever 和 memory"""
        for rv in reviews:
            text = rv.get("text", "")
            if text:
                self.cached_item_reviews.append(text)
                if self.memory:
                    self.memory(f"[ITEM_REVIEW] {text}")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True)

    def get_top_k_user_reviews(self, user_reviews, item_title, k=3):
        """从用户历史中找最相关的 k 条"""
        if len(user_reviews) == 0:
            return []

        # 提取 review 文本
        review_texts = [rv["text"] for rv in user_reviews]

        # embedding
        item_emb = self.embed(item_title)
        review_embs = self.embed(review_texts)

        # 计算相似度
        similarities = util.cos_sim(item_emb, review_embs)[0]

        # 选 topK
        topk_idx = np.argsort(-similarities)[:k]
        return [review_texts[i] for i in topk_idx]

    def get_item_key_features(self, item):
        """根据 item JSON 自动抽取关键内容（标题 + 特性 + 描述）"""
        fields = ["title", "brand", "feature", "description"]
        info_parts = []

        for f in fields:
            if f in item and item[f]:
                info_parts.append(f"{f}: {item[f]}")

        return "\n".join(info_parts)

    def retrieve(self, user_reviews, item):
        """总检索流程，返回给 Reasoner 使用"""
        item_title = item.get("title", "")

        similar_user_reviews = self.get_top_k_user_reviews(
            user_reviews, item_title, k=3
        )

        item_features = self.get_item_key_features(item)

        return {
            "similar_reviews": similar_user_reviews,
            "item_features": item_features
        }



###########################################
# 3. PersonaBuilder - 模拟用户风格
###########################################
class PersonaBuilder:
    def __init__(self, llm):
        self.llm = llm

    def build(self, user_profile, user_reviews):
        prompt = f"""
You are analyzing an Amazon user's behavior and writing style to construct a DETAILED persona.

USER PROFILE:
{user_profile}

USER REVIEW HISTORY (examples):
{user_reviews}

Extract the user's behavior in 4 dimensions:

1. WRITING STYLE:
- tone (casual/formal/blunt/emotional/humorous)
- punctuation style (exclamation marks, ellipsis, minimal punctuation)
- filler words (e.g., "honestly", "actually", "overall")
- sentence length (short/medium/long)
- detail level (low/medium/high)

2. RATING BEHAVIOR:
- rating tendency (generous/neutral/harsh)
- positivity_bias (positive/balanced/negative)
- complaint_vs_praise_ratio

3. CONTENT FOCUS:
Identify which aspects they care MOST about vs LEAST:
Possible focus aspects:
performance, value, durability, design, comfort, packaging, battery life, shipping

Return:
- priority: an ordered list (highest → lowest importance)
- ignore: aspects the user rarely discusses

4. LOGIC PATTERNS:
Analyze HOW the user structures arguments:
- contrast usage (often/sometimes/rare)
- evaluation_priority_reasoning (e.g., performance-first, value-first)
- heuristics (rules the user uses to judge items, e.g., "cheap = good")
- emotion_triggers (positive & negative triggers)
- summary_style (overall-summary / final-judgment / recommendation-ending)

IMPORTANT:
- Return ONLY a JSON dictionary.
- Do NOT include natural language outside JSON.
"""
        res = self.llm(messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return res



###########################################
# 4. Reasoner - 主推理模块
###########################################
import re

class MyReasoner:
    def __init__(self, llm):
        self.llm = llm

    # -----------------------------
    #   Parse LLM output
    # -----------------------------
    def _extract_results(self, text):
        stars_match = re.search(r"stars:\s*([0-9.]+)", text)
        review_match = re.search(r"review:\s*(.*)", text, re.DOTALL)

        if not stars_match or not review_match:
            return None

        stars = float(stars_match.group(1))
        stars = min(5.0, max(1.0, stars))
        stars = round(stars * 2) / 2.0

        review = review_match.group(1).strip()[:512]

        return {
            "stars": stars,
            "review": review
        }

    # -----------------------------
    #   NEW: Automatically infer domain
    # -----------------------------
    def infer_domain(self, item_info):
        """根据 item 信息推断 Domain"""
        text = str(item_info).lower()

        if any(k in text for k in ["game", "xbox", "ps5", "switch", "steam"]):
            return "video_games"

        if any(k in text for k in ["guitar", "piano", "violin", "instrument"]):
            return "musical_instruments"

        if any(k in text for k in ["scientific", "lab", "industrial", "measurement"]):
            return "industrial_scientific"

        return "generic"

    # -----------------------------
    #   NEW: Extract user expertise level from persona JSON
    # -----------------------------
    def infer_user_expertise(self, persona_json, domain):
        """根据 persona 判断用户是否在该领域专业"""

        text = str(persona_json).lower()

        # rules you can expand later
        domain_patterns = {
            "video_games": ["gamer", "gaming", "fps", "rpg", "switch", "console"],
            "musical_instruments": ["musician", "guitarist", "pianist", "practice", "tone quality"],
            "industrial_scientific": ["engineer", "lab", "mechanic", "precision", "measurement"]
        }

        if domain == "generic":
            return "none"

        for keyword in domain_patterns.get(domain, []):
            if keyword in text:
                return "expert"

        return "novice"

    # -----------------------------
    #   Final Review Generation
    # -----------------------------
    def generate_review(self, persona_json, user_profile, item_info, similar_reviews):
        domain = self.infer_domain(item_info)
        expertise = self.infer_user_expertise(persona_json, domain)

        prompt = f"""
You are simulating a human Amazon user writing a product review.

==== USER PERSONA (JSON) ====
{persona_json}

==== USER PROFILE ====
{user_profile}

==== PRODUCT INFORMATION ====
{item_info}

==== SIMILAR REVIEWS BY THE SAME USER ====
{similar_reviews}

==== DOMAIN DETECTED ====
{domain}

==== USER EXPERTISE LEVEL IN THIS DOMAIN ====
{expertise}

Write a review *fully consistent* with the user's persona AND their expertise level.

Rules based on expertise level:
- If expertise = "expert":
    • Use precise, domain-specific vocabulary
    • Be strict and analytical
    • Evaluate performance with technical criteria
- If expertise = "novice":
    • Use simple, consumer-friendly language
    • Focus on ease of use and general impressions
- If expertise = "none":
    • Write with everyday vocabulary
    • Avoid technical judgments

General Rules:
1. Write 2–4 sentences.
2. Follow persona's tone, punctuation habits, filler words, and logic.
3. Mention at least one specific detail about the item.
4. Rating must match persona’s rating-behavior tendency.
5. Format EXACTLY:

stars: <rating>
review: <text>

No extra commentary.
"""

        response = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=350
        )

        parsed = self._extract_results(response)
        if parsed:
            return parsed

        return {
            "stars": 4.0,
            "review": "The product performs reasonably well and offers acceptable value for its category."
        }



###########################################
# 5. Output Controller - 强制格式正确
###########################################
class OutputController:
    def parse(self, output):
        """支持 dict（Reasoner成功）和 str（fallback）"""

        if isinstance(output, dict):
            # Reasoner 已经给出结构化结果
            return output

        # 否则是 LLM 的文本输出 —— 做正则解析
        lines = output.split("\n")

        stars_line = next((l for l in lines if l.startswith("stars:")), None)
        review_line = next((l for l in lines if l.startswith("review:")), None)

        stars = float(stars_line.split(":", 1)[1].strip()) if stars_line else 4.0
        review = review_line.split(":", 1)[1].strip() if review_line else "Decent product."

        return {"stars": stars, "review": review[:512]}



###########################################
# 6. 整体 Agent（主要执行逻辑）
###########################################
class MySimulationAgent(SimulationAgent):
    """最终提交的 agent"""

    def __init__(self, llm: LLMBase):
        super().__init__(llm)

        # === Core Modules ===
        self.memory = MemoryDILU(llm=self.llm)
        self.planner = MyPlanner(llm=self.llm)
        self.retriever = MyRetriever(memory=self.memory)
        self.persona_builder = PersonaBuilder(llm=self.llm)
        self.reasoner = MyReasoner(llm=self.llm)
        self.controller = OutputController()

    def workflow(self):
        """执行 agent 的总流程"""

        task = self.task
        plan = self.planner.plan(task)

        # =========================
        # PHASE 1: TOOL USE
        # =========================
        user = self.interaction_tool.get_user(task["user_id"])
        item = self.interaction_tool.get_item(task["item_id"])
        item_reviews = self.interaction_tool.get_reviews(item_id=task["item_id"])
        user_reviews = self.interaction_tool.get_reviews(user_id=task["user_id"])

        # --- store item reviews into memory ---
        self.retriever.store_reviews(item_reviews)

        # --- retrieve similar reviews (based on user's past behavior) ---
        if user_reviews:
            similar_info = self.retriever.retrieve(
                user_reviews=user_reviews,
                item=item
            )
        else:
            similar_info = {"similar_reviews": [], "item_features": ""}

        # =========================
        # PHASE 2: PERSONA CONSTRUCTION
        # =========================
        user_review_texts = [rv["text"] for rv in user_reviews]
        persona_json = self.persona_builder.build(
            user_profile=user,
            user_reviews=user_review_texts
        )

        # =========================
        # PHASE 3: REASONING + REVIEW GENERATION
        # =========================
        raw_output = self.reasoner.generate_review(
            persona_json=persona_json,
            user_profile=user,
            item_info=item,
            similar_reviews=similar_info
        )

        # =========================
        # PHASE 4: Output Control
        # =========================
        return self.controller.parse(raw_output)
