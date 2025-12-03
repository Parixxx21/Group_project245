# websocietysimulator/agent/my_agent.py

from typing import Any, Dict, List, Optional, Union

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.memory_modules import MemoryBase
from sentence_transformers import SentenceTransformer, util
from langchain.docstore.document import Document

import numpy as np
import re
import json


###########################################################
# 0. (Optional) Mock LLM for quick local testing
###########################################################
class MockLLM:
    def __call__(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 200,
        **kwargs: Any,
    ) -> str:
        return (
            "stars: 4.5\n"
            "review: This is a mock review. The mouse performs well and feels comfortable."
        )


###########################################################
# 1. Simple helper functions for rating statistics
###########################################################
def _extract_rating_from_review(review: Dict[str, Any]) -> Optional[float]:
    """
    兼容多种字段名（Yelp/Amazon/Goodreads）:
    - stars / rating / overall / score
    """
    for key in ["stars", "rating", "overall", "score"]:
        if key in review and review[key] is not None:
            val = review[key]
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _collect_ratings(reviews: List[Dict[str, Any]]) -> List[float]:
    ratings: List[float] = []
    for rv in reviews:
        r = _extract_rating_from_review(rv)
        if r is not None:
            ratings.append(r)
    return ratings


def compute_rating_stats(
    item_reviews: List[Dict[str, Any]],
    user_reviews: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    统计 item 和 user 的打分行为，用来做：
    - prompt 中的“背景信息”
    - 后处理中的“星级校准”，帮助降低 star_mae
    """
    item_ratings = _collect_ratings(item_reviews)
    user_ratings = _collect_ratings(user_reviews)

    def _stats(rs: List[float]) -> Dict[str, Optional[float]]:
        if not rs:
            return {"mean": None, "median": None, "count": 0}
        return {
            "mean": float(np.mean(rs)),
            "median": float(np.median(rs)),
            "count": len(rs),
        }

    item_stat = _stats(item_ratings)
    user_stat = _stats(user_ratings)

    user_tendency = "unknown"
    if user_stat["mean"] is not None:
        m = user_stat["mean"]
        if m >= 4.3:
            user_tendency = "generous"
        elif m <= 3.2:
            user_tendency = "harsh"
        else:
            user_tendency = "neutral"

    return {
        "item_mean": item_stat["mean"],
        "item_median": item_stat["median"],
        "item_review_count": item_stat["count"],
        "user_mean": user_stat["mean"],
        "user_median": user_stat["median"],
        "user_review_count": user_stat["count"],
        "user_tendency": user_tendency,
    }


###########################################################
# 2. MemoryDILU wrapper with explicit retrive/add methods
###########################################################
class MemoryDILU(MemoryBase):
    """
    轻量封装一下 MemoryBase：
    - retriveMemory(query_scenario): 从 scenario_memory 里做相似搜索
    - addMemory(current_situation): 往 scenario_memory 里写入一条轨迹
    """

    def __init__(self, llm: LLMBase):
        super().__init__(memory_type="dilu", llm=llm)

    def retriveMemory(self, query_scenario: str) -> str:
        task_name = query_scenario

        if self.scenario_memory._collection.count() == 0:
            return ""

        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=20
        )

        task_trajectories = [
            result[0].metadata["task_trajectory"] for result in similarity_results
        ]
        return "\n".join(task_trajectories)

    def addMemory(self, current_situation: str) -> None:
        task_name = current_situation
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation,
            },
        )
        self.scenario_memory.add_documents([memory_doc])


###########################################################
# 3. Planner (hand-written multi-step plan, no LLM)
###########################################################
class MyPlanner(PlanningBase):
    """
    这里只返回一个静态 plan，主要是为了让整体结构清晰，
    方便以后扩展（不会额外消耗 LLM 调用）。
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)

    def __call__(self, task_description: Dict[str, Any]) -> List[Dict[str, Any]]:
        user_id = task_description["user_id"]
        item_id = task_description["item_id"]

        self.plan = [
            {
                "step": 1,
                "description": "Load the user profile.",
                "reasoning instruction": "Retrieve user metadata.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_user",
                    "args": {"user_id": user_id},
                },
            },
            {
                "step": 2,
                "description": "Load all historical reviews written by this user.",
                "reasoning instruction": "Gather user review history to infer habits.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_reviews",
                    "args": {"user_id": user_id},
                },
            },
            {
                "step": 3,
                "description": "Load item information.",
                "reasoning instruction": "Understand the item/business to be reviewed.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_item",
                    "args": {"item_id": item_id},
                },
            },
            {
                "step": 4,
                "description": "Load all reviews for this item.",
                "reasoning instruction": "Get background context and common opinions.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_reviews",
                    "args": {"item_id": item_id},
                },
            },
            {
                "step": 5,
                "description": "Analyze item and user rating distribution.",
                "reasoning instruction": "Summarize mean/median and rating tendency.",
                "tool use instruction": {
                    "tool": "compute_rating_stats",
                    "args": {},
                },
            },
            {
                "step": 6,
                "description": "Construct a persona for the user.",
                "reasoning instruction": "Analyze writing style and rating behavior.",
                "tool use instruction": {
                    "tool": "persona_builder.build",
                    "args": {},
                },
            },
            {
                "step": 7,
                "description": "Retrieve semantically relevant reviews and key item features.",
                "reasoning instruction": "Find similar past reviews & extract item characteristics.",
                "tool use instruction": {
                    "tool": "retriever.retrieve",
                    "args": {},
                },
            },
            {
                "step": 8,
                "description": "Generate final rating and review text.",
                "reasoning instruction": "Use reasoning to produce a realistic review.",
                "tool use instruction": {
                    "tool": "reasoner.generate_review",
                    "args": {},
                },
            },
        ]

        return self.plan


###########################################################
# 4. Retriever (user/item memories + sentence-transformers)
###########################################################
class MyRetriever:
    def __init__(
        self,
        user_mem: Optional[MemoryDILU] = None,
        item_mem: Optional[MemoryDILU] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(embedding_model)
        self.cached_item_reviews: List[str] = []
        self.cached_user_reviews: List[str] = []
        self.user_mem = user_mem
        self.item_mem = item_mem

    def store_reviews(self, reviews: List[Dict[str, Any]], type: str) -> None:
        """
        把 item / user 的 reviews 存进 retriever（以及可选的 memory）
        """
        if type == "items":
            for rv in reviews:
                text = rv.get("text", "")
                if text:
                    self.cached_item_reviews.append(text)
                    if self.item_mem is not None:
                        self.item_mem.addMemory(text)
        elif type == "user":
            for rv in reviews:
                text = rv.get("text", "")
                if text:
                    self.cached_user_reviews.append(text)
                    if self.user_mem is not None:
                        self.user_mem.addMemory(text)
        else:
            print("[DEBUG] Unknown review type:", type)

    def embed(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True)

    # ---------- 基于本地 embedding 的检索 ----------
    def get_top_k_user_reviews(
        self,
        user_reviews: List[Dict[str, Any]],
        item_title: str,
        k: int = 3,
    ) -> List[str]:
        if not user_reviews or not item_title:
            return []

        review_texts = [rv.get("text", "") for rv in user_reviews]
        if not any(review_texts):
            return []

        item_emb = self.embed(item_title)
        review_embs = self.embed(review_texts)

        similarities = util.cos_sim(item_emb, review_embs)[0]
        topk_idx = np.argsort(-similarities)[:k]

        return [review_texts[i] for i in topk_idx]

    # ---------- 从 MemoryDILU 里取用户评论 ----------
    def get_top_k_user_reviews_from_memory(self, item_title: str, k: int = 3) -> List[str]:
        if self.user_mem is None:
            return []

        query = f"user-style: {item_title}"
        raw = self.user_mem.retriveMemory(query)

        if not raw:
            return []

        results: List[str] = []
        if isinstance(raw, str):
            lines = [x.strip() for x in raw.split("\n") if x.strip()]
            results.extend(lines)

        return results[:k]

    # ---------- 从 MemoryDILU 里取 item 评论 ----------
    def get_top_k_item_reviews_from_memory(self, item_title: str, k: int = 3) -> List[str]:
        if self.item_mem is None:
            return []

        raw = self.item_mem.retriveMemory(f"item-related: {item_title}")
        if not raw:
            return []

        results: List[str] = []
        if isinstance(raw, str):
            lines = [x.strip() for x in raw.split("\n") if x.strip()]
            results.extend(lines)

        return results[:k]

    def get_item_key_features(self, item: Dict[str, Any]) -> str:
        """
        从 item/business JSON 中抽取关键信息
        兼容 Amazon / Yelp / Goodreads，不存在的字段会自动跳过
        """
        fields = [
            "title",
            "name",  # Yelp
            "brand",
            "feature",
            "categories",
            "attributes",
            "description",
            "main_category",
            "popular_shelves",
        ]
        info_parts: List[str] = []
        for f in fields:
            if f in item and item[f]:
                info_parts.append(f"{f}: {item[f]}")
        return "\n".join(info_parts)

    def _pick_item_anchor_reviews(
        self,
        item_reviews: List[Dict[str, Any]],
        item_title: str,
        k: int = 3,
    ) -> List[str]:
        """
        选几条 "other reviews for this business" 作为 topic anchor。
        优先用本地 reviews，memory 作为补充。
        """
        texts = [rv.get("text", "") for rv in item_reviews if rv.get("text")]
        anchors: List[str] = []

        if texts:
            # 简单拿前几条即可；也可以改成 embedding top-k
            anchors.extend(texts[:k])

        if len(anchors) < k and self.item_mem is not None:
            extra = self.get_top_k_item_reviews_from_memory(item_title, k=k - len(anchors))
            anchors.extend(extra)

        return anchors[:k]

    def retrieve(
        self,
        user_reviews: List[Dict[str, Any]],
        item: Dict[str, Any],
        item_reviews: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        汇总检索结果给 Reasoner 使用：
        - similar_user_reviews: 同一用户历史上和当前 item 相近的评论
        - similar_item_reviews: 该 business 上的其他评论（topic anchor）
        - item_features: item/business 结构化信息
        """
        item_title = item.get("title", "") or item.get("name", "")

        if self.user_mem is not None:
            similar_user_reviews = self.get_top_k_user_reviews_from_memory(
                item_title=item_title,
                k=3,
            )
        else:
            similar_user_reviews = self.get_top_k_user_reviews(
                user_reviews=user_reviews,
                item_title=item_title,
                k=3,
            )

        similar_item_reviews: List[str] = []
        if item_reviews:
            similar_item_reviews = self._pick_item_anchor_reviews(
                item_reviews=item_reviews,
                item_title=item_title,
                k=3,
            )

        item_features = self.get_item_key_features(item)
        return {
            "similar_user_reviews": similar_user_reviews,
            "similar_item_reviews": similar_item_reviews,
            "item_features": item_features,
        }


###########################################################
# 5. Persona Builder (LLM-based)
###########################################################
class PersonaBuilder:
    def __init__(self, llm: LLMBase):
        self.llm = llm

    def build(
        self,
        user_profile: Dict[str, Any],
        user_reviews: List[str],
    ) -> str:
        """
        调用 LLM，根据用户 profile + 历史评论，总结出 persona（JSON 字符串）
        """
        # 这里假定在外层已经截断过 user_reviews，不再做复杂控制
        prompt = f"""
You are analyzing an Amazon/Yelp/Goodreads user's behavior and writing style to construct a DETAILED persona.

USER PROFILE:
{user_profile}

USER REVIEW HISTORY (examples):
{user_reviews}

Extract the user's behavior in 5 dimensions and return ONLY a JSON dictionary:

1. writing_style:
  - tone (casual/formal/blunt/emotional/humorous, etc.)
  - punctuation_style
  - filler_words
  - sentence_length
  - detail_level

2. rating_behavior:
  - rating_tendency (generous/neutral/harsh)
  - positivity_bias
  - complaint_vs_praise_ratio

3. content_focus:
  - priority: ordered list of aspects they care MOST -> LEAST
    (performance, value, durability, design, comfort, packaging, battery_life, shipping, service, atmosphere, cleanliness, price)
  - ignore: aspects they almost never mention

4. logic_patterns:
  - contrast_usage
  - evaluation_priority_reasoning
  - heuristics
  - emotion_triggers
  - summary_style

5. emotional_style:
  - emotional_intensity (low/medium/high)
  - extreme_review_frequency (how often they write extremely positive or extremely negative reviews)
"""
        res = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res


###########################################################
# 6. Reasoner (rating + review generation)
###########################################################
class MyReasoner:
    MAX_PROMPT_CHARS = 15000  # 粗略限制，防止超过 16k token

    def __init__(self, llm: LLMBase):
        self.llm = llm

    # ---------- 解析 LLM 输出 ----------
    def _extract_results(self, text: str) -> Optional[Dict[str, Any]]:
        stars_match = re.search(r"stars:\s*([0-9.]+)", text)
        review_match = re.search(r"review:\s*(.*)", text, re.DOTALL)

        if not stars_match or not review_match:
            return None

        try:
            stars = float(stars_match.group(1))
        except ValueError:
            return None

        stars = min(5.0, max(1.0, stars))

        review = review_match.group(1).strip()[:512]
        return {"stars": stars, "review": review}

    def _calibrate_stars(self, stars: float, rating_stats: Dict[str, Any]) -> float:
        """
        - 先和 item 的历史平均做一个平滑
        - 再根据 user 的平均分加一点偏置
        """
        item_mean = rating_stats.get("item_mean")
        user_mean = rating_stats.get("user_mean")

        final = stars
        if item_mean is not None:
            final = 0.6 * stars + 0.4 * float(item_mean)

        if user_mean is not None:
            bias = float(user_mean) - 3.5
            final += 0.3 * bias

        final = min(5.0, max(1.0, final))
        final = round(final * 2) / 2.0
        return final

    # ---------- domain / expertise 推断 ----------
    def infer_domain(self, item_info: Dict[str, Any]) -> str:
        txt = str(item_info).lower()

        # Yelp 典型领域
        if any(k in txt for k in ["restaurant", "cafe", "coffee", "bar", "bistro", "diner", "pizza", "burger"]):
            return "restaurant"
        if any(k in txt for k in ["hotel", "inn", "resort", "motel"]):
            return "hotel"

        # Amazon 常见品类
        if any(k in txt for k in ["xbox", "ps5", "switch", "steam", "video game", "gaming"]):
            return "video_games"
        if any(k in txt for k in ["guitar", "piano", "violin", "instrument"]):
            return "musical_instruments"
        if any(k in txt for k in ["laptop", "phone", "tablet", "headphone", "camera", "ssd", "hard drive"]):
            return "electronics"
        if any(k in txt for k in ["book", "novel", "paperback", "hardcover", "author"]):
            return "books"
        if any(k in txt for k in ["scientific", "lab", "industrial", "measurement"]):
            return "industrial_scientific"

        return "generic"

    def infer_user_expertise(self, persona_json: Any, domain: str) -> str:
        txt = str(persona_json).lower()
        domain_patterns = {
            "video_games": ["gamer", "gaming", "fps", "rpg", "switch", "console"],
            "musical_instruments": ["musician", "guitarist", "pianist", "practice"],
            "industrial_scientific": ["engineer", "lab", "mechanic", "precision"],
            "electronics": ["engineer", "it", "programmer", "tech-savvy", "hardware"],
            "books": ["avid reader", "book reviewer", "literature"],
            "restaurant": ["foodie", "chef", "food blogger", "yelp elite"],
            "hotel": ["travel blogger", "frequent traveler"],
        }
        if domain == "generic":
            return "none"
        for kw in domain_patterns.get(domain, []):
            if kw in txt:
                return "expert"
        return "novice"

    def _domain_guidance(self, domain: str) -> str:
        if domain == "restaurant":
            return (
                "- For restaurants/cafes, mention food taste, portion size, service attitude, "
                "waiting time, atmosphere, cleanliness, and price/value.\n"
            )
        if domain == "hotel":
            return (
                "- For hotels, mention room cleanliness, noise level, staff friendliness, location convenience, "
                "check-in experience, and sleep quality.\n"
            )
        if domain == "electronics":
            return (
                "- For electronics, focus on performance, reliability, battery life, build quality, and ease of use.\n"
            )
        if domain == "video_games":
            return (
                "- For video games, focus on gameplay, graphics, performance, story, and replay value.\n"
            )
        if domain == "musical_instruments":
            return (
                "- For musical instruments, focus on sound quality, build quality, playability, and suitability for practice or performance.\n"
            )
        if domain == "books":
            return (
                "- For books, mention writing style, pacing, character development, and who might enjoy this book.\n"
            )
        return ""

    # ---------- 将“预期星级”映射到情绪强度 ----------
    def _sentiment_label(self, expected_star: float) -> str:
        if expected_star >= 4.5:
            return "strongly positive"
        if expected_star >= 3.8:
            return "moderately positive"
        if expected_star >= 3.0:
            return "mixed or neutral"
        if expected_star >= 2.0:
            return "moderately negative"
        return "strongly negative"

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len]

    def generate_review(
        self,
        persona_json: Any,
        user_profile: Dict[str, Any],
        item_info: Dict[str, Any],
        similar_reviews: Dict[str, Any],
        rating_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        domain = self.infer_domain(item_info)
        expertise = self.infer_user_expertise(persona_json, domain)
        domain_guidance = self._domain_guidance(domain)

        user_tendency = rating_stats.get("user_tendency", "unknown")
        item_mean = rating_stats.get("item_mean")
        item_review_count = rating_stats.get("item_review_count", 0)
        user_mean = rating_stats.get("user_mean")
        user_review_count = rating_stats.get("user_review_count", 0)

        rating_context_lines: List[str] = []
        if item_mean is not None and item_review_count:
            rating_context_lines.append(
                f"- This item/business has a historical average rating of {item_mean:.2f} / 5 "
                f"from about {item_review_count} reviews."
            )
        if user_mean is not None and user_review_count:
            rating_context_lines.append(
                f"- This user historically gives an average rating around {user_mean:.2f} / 5 "
                f"and is generally considered {user_tendency}."
            )
        rating_context = (
            "\n".join(rating_context_lines) if rating_context_lines else "No strong prior rating signals."
        )

        # 预估一个“期望星级”，用来控制情绪强度
        expected_star = item_mean if item_mean is not None else (user_mean or 3.8)
        sentiment_label = self._sentiment_label(expected_star)

        # 处理 similar_reviews 结构
        similar_user_reviews = similar_reviews.get("similar_user_reviews") or similar_reviews.get(
            "similar_reviews", []
        )
        similar_item_reviews = similar_reviews.get("similar_item_reviews", [])

        # 截断以避免 context 过长
        persona_str = self._truncate(str(persona_json), 6000)
        user_profile_str = self._truncate(str(user_profile), 2000)
        item_info_str = self._truncate(str(item_info), 2500)
        sim_user_str = self._truncate(str(similar_user_reviews), 2500)
        sim_item_str = self._truncate(str(similar_item_reviews), 2500)
        rating_context_str = self._truncate(rating_context, 1000)

        # 构造 prompt
        prompt = f"""
You are simulating a human user writing a product/business review.

==== USER PERSONA (JSON) ====
{persona_str}

==== USER PROFILE ====
{user_profile_str}

==== ITEM/BUSINESS INFORMATION ====
{item_info_str}

==== SIMILAR REVIEWS BY THE SAME USER ====
{sim_user_str}

==== OTHER REVIEWS FOR THIS BUSINESS ====
{sim_item_str}

==== RATING CONTEXT ====
{rating_context_str}

==== DOMAIN DETECTED ====
{domain}

==== USER EXPERTISE LEVEL IN THIS DOMAIN ====
{expertise}

Target overall sentiment: {sentiment_label}.
Make sure the tone of the review matches this sentiment:
- strongly positive: very enthusiastic, clear joy and recommendation.
- moderately positive: generally happy, maybe a few mild complaints.
- mixed or neutral: balanced tone, both pros and cons.
- moderately negative: clear dissatisfaction but still some positives.
- strongly negative: strong disappointment or frustration.

Domain-specific hints:
{domain_guidance}
General Rules:
1. Write 2–3 sentences.
2. Focus mainly on concrete aspects of THIS business (from the item information and other reviews),
   not on generic comments about life or long digressions.
3. At least half of the sentences should mention specific aspects
   (e.g., food taste, service, waiting time, atmosphere, cleanliness, price, performance, etc.).
4. Follow the persona's punctuation, filler words, and logic.
5. Rating must be consistent with the user's rating tendency AND the overall rating context.
6. Emotional intensity:
   - If you give 4.5 or 5.0 stars, express clear joy or enthusiasm.
   - If you give 2.0 stars or below, express obvious dissatisfaction or frustration.
   - Avoid writing a very neutral review when the rating is extremely high or extremely low.

Format EXACTLY:
stars: <rating>
review: <text>

No extra commentary.
"""
        response = self.llm(
            messages=[{"role": "user", "content": prompt[: self.MAX_PROMPT_CHARS]}],
            temperature=0.2,  # 略微降低温度，提升稳定性
            max_tokens=350,
        )

        parsed = self._extract_results(str(response))
        if parsed:
            raw_stars = parsed["stars"]
            calibrated_stars = self._calibrate_stars(raw_stars, rating_stats)
            parsed["stars"] = calibrated_stars
            return parsed

        # fallback
        fallback_stars = self._calibrate_stars(4.0, rating_stats)
        return {
            "stars": fallback_stars,
            "review": "The product or service performs reasonably well and offers acceptable value.",
        }


###########################################################
# 7. Output Controller
###########################################################
class OutputController:
    def parse(self, output: Any) -> Dict[str, Any]:
        if isinstance(output, dict):
            stars = float(output.get("stars", 4.0))
            stars = min(5.0, max(1.0, stars))
            stars = round(stars * 2) / 2.0
            review = str(output.get("review", "Decent product."))[:512]
            return {"stars": stars, "review": review}

        lines = str(output).split("\n")
        stars_line = next((l for l in lines if l.strip().lower().startswith("stars:")), None)
        review_lines = [l for l in lines if l.strip().lower().startswith("review:")]

        stars = 4.0
        review = "Decent product."

        if stars_line:
            try:
                stars = float(stars_line.split(":", 1)[1].strip())
            except Exception:
                pass

        if review_lines:
            first_idx = lines.index(review_lines[0])
            review = "\n".join(lines[first_idx:]).split(":", 1)[1].strip()

        stars = min(5.0, max(1.0, stars))
        stars = round(stars * 2) / 2.0
        return {"stars": stars, "review": review[:512]}


###########################################################
# 8. Final Simulation Agent
###########################################################
class MySimulationAgent(SimulationAgent):
    """
    Pipeline 概览（对应你画的图）：

    Planning → Context Gathering → Rating Stats Analyzing →
    Memory Retrieval + Domain-focus guidance →
    Persona Build → CoT Review Generation →
    Stats-based output adjustment → Generation
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm)
        self.enable_memory = True

        if hasattr(self.llm, "get_embedding_model") and self.enable_memory:
            self.user_mem = MemoryDILU(llm=self.llm)
            self.item_mem = MemoryDILU(llm=self.llm)
        else:
            self.user_mem = None
            self.item_mem = None

        self.planner = MyPlanner(llm=self.llm)
        self.retriever = MyRetriever(user_mem=self.user_mem, item_mem=self.item_mem)
        self.persona_builder = PersonaBuilder(llm=self.llm)
        self.reasoner = MyReasoner(llm=self.llm)
        self.controller = OutputController()

    def workflow(self) -> Dict[str, Any]:
        task = self.task  # dict (SimulationTask.to_dict())
        _plan = self.planner(task)  # 目前只是为了结构完整，不强制逐步执行

        # --------- 1. 通过 InteractionTool 拿数据 ---------
        user = self.interaction_tool.get_user(task["user_id"])
        item = self.interaction_tool.get_item(task["item_id"])
        item_reviews = self.interaction_tool.get_reviews(item_id=task["item_id"])
        user_reviews = self.interaction_tool.get_reviews(user_id=task["user_id"])

        # --------- 2. 统计 rating 行为 ---------
        rating_stats = compute_rating_stats(item_reviews, user_reviews)

        # --------- 3. 存入 retriever / memory ---------
        self.retriever.store_reviews(item_reviews, type="items")
        self.retriever.store_reviews(user_reviews, type="user")

        # --------- 4. 检索相似评论 & item 特征 ---------
        if user_reviews or item_reviews:
            similar_info = self.retriever.retrieve(
                user_reviews=user_reviews,
                item=item,
                item_reviews=item_reviews,
            )
        else:
            similar_info = {
                "similar_user_reviews": [],
                "similar_item_reviews": [],
                "item_features": "",
            }

        # --------- 5. 构建 persona（对 user_reviews 做截断，避免 context 溢出） ---------
        user_review_texts = [rv.get("text", "") for rv in user_reviews if rv.get("text")]
        # 最多 30 条，每条最多 300 字符
        compact_user_reviews = [txt[:300] for txt in user_review_texts[:30]]

        try:
            persona_json = self.persona_builder.build(
                user_reviews=compact_user_reviews,
                user_profile=user,
            )
        except Exception as e:
            print("\n[WARNING] Persona builder failed — likely context overflow.")
            print("Error:", e)
            print("→ Using empty persona and continuing...\n")
            persona_json = "{}"

        # --------- 6. Reasoner 生成评分 + 评论 ---------
        raw_output = self.reasoner.generate_review(
            persona_json=persona_json,
            user_profile=user,
            item_info=item,
            similar_reviews=similar_info,
            rating_stats=rating_stats,
        )

        # --------- 7. 输出控制 ---------
        result = self.controller.parse(raw_output)
        return result
