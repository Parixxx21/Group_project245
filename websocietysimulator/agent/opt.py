from typing import Any, Dict, List, Optional

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU, MemoryBase
from sentence_transformers import SentenceTransformer, util
from langchain.docstore.document import Document

import numpy as np
import re

# Modified
class MemoryDILU(MemoryBase):
    def __init__(self, llm):
        super().__init__(memory_type='dilu', llm=llm)

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=20)
            
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)

    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
        
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
        
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])

###########################################
# 0. MockLLM（本地快速测试用）
###########################################
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

def _extract_rating_from_review(review: Dict[str, Any]) -> Optional[float]:
    """
    兼容多种字段名（Yelp/Amazon/Goodreads）：
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


class MyPlanner(PlanningBase):
    """
    - 不调用 LLM，只返回一个手写的多步计划
    - 目前主要是为了结构清晰，方便以后扩展
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
                "reasoning instruction": "Understand the item the user will review.",
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
                "description": "Analyze item rating distribution (mean, variance, skewness).",
                "reasoning instruction": "Get background context and common opinions.",
                "tool use instruction": {
                    "tool": "interaction_tool.get_reviews",
                    "args": {"item_id": item_id},
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
                    "tool": "reasoner.generate",
                    "args": {},
                },
            },
        ]

        return self.plan


class MyRetriever:
    # Modified
    def __init__(
        self,
        user_mem: Optional[MemoryDILU] = None,
        item_mem: Optional[MemoryDILU] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        memory: MemoryDILU（可选），用于把 item review 写入长期记忆
        """
        self.model = SentenceTransformer(embedding_model)
        self.cached_item_reviews: List[str] = []
        self.cached_user_reviews: List[str] = []
        self.user_mem = user_mem
        self.item_mem = item_mem
    
    # Modified
    def store_reviews(self, reviews: List[Dict[str, Any]], type: str) -> None:
        """
        把 item 的 reviews 存进 retriever（和可选的 memory）
        """
        if (type == "items"):
            for rv in reviews:
                text = rv.get("text", "")
                if text:
                    self.cached_item_reviews.append(text)
                    if self.item_mem is not None:
                        # 往 MemoryDILU 里写入文本
                        self.item_mem.addMemory(f"{text}")
        elif (type == "user"):
            for rv in reviews:
                text = rv.get("text", "")
                if text:
                    self.cached_user_reviews.append(text)
                    if self.user_mem is not None:
                        # 往 MemoryDILU 里写入文本
                        self.user_mem.addMemory(f"{text}")
        else:
            print("[DEBUG]: Error storing with type:", type)
            print("[DEBUG REAL TYPE repr]:", repr(type))

    def embed(self, texts: Any):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_tensor=True)

    def get_top_k_user_reviews(
        self,
        user_reviews: List[Dict[str, Any]],
        item_title: str,
        k: int = 3,
    ) -> List[str]:
        """
        从用户历史评论中找出与当前 item 最相近的 k 条评论（基于 sentence-transformers）
        """
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
    
    # Modified
    def get_top_k_user_reviews_from_memory(self, item_title: str, k: int = 3):
        query = f"user-style: {item_title}"

        print("\n====================== [DEBUG] USER MEMORY RETRIEVAL ======================")
        print(f"[Query] {query}")

        raw = self.user_mem.retriveMemory(query)

        print(f"[RAW type] {type(raw)}")

        if raw is None:
            print("[RAW] None → return []")
            print("==========================================================================\n")
            return []

        if isinstance(raw, str):
            print(f"[RAW STRING (first 200 chars)] {repr(raw[:200])}")
        elif isinstance(raw, list):
            print(f"[RAW LIST len={len(raw)}]")
        else:
            print("[RAW] Unrecognized type")

        results = []

        # -------------------------
        # Case 1: result is string
        # -------------------------
        if isinstance(raw, str):
            lines = [x.strip() for x in raw.split("\n") if x.strip()]
            results.extend(lines)

        # -------------------------
        # Case 2: result list from Modified DILU, 这个应该不需要，是我之前改炸了的版本对应的
        # -------------------------
        elif isinstance(raw, list):
            for item in raw:

                # (Document, score)
                if isinstance(item, tuple):
                    doc = item[0]
                    text = getattr(doc, "page_content", None) or doc.metadata.get("content", "")
                # Document only
                elif hasattr(item, "page_content"):
                    text = item.page_content
                # string fallback
                elif isinstance(item, str):
                    text = item
                else:
                    continue

                if isinstance(text, str):
                    results.append(text)

        # -------------------------
        # Debug printing parsed results
        # -------------------------
        print(f"[Parsed USER_REVIEW count]: {len(results)}")
        for i, r in enumerate(results[:k]):
            print(f"  USER #{i+1}: {r[:120]}")

        print("==========================================================================\n")
        return results[:k]


    # Modified, you can use it to generate the similar item reviews
    def get_top_k_item_reviews_from_memory(self, item_title: str, k: int = 3):

        print("\n====================== [DEBUG] ITEM MEMORY RETRIEVAL ======================")
        print(f"[Query] item-related: {item_title}")

        raw = self.item_mem.retriveMemory(f"item-related: {item_title}")

        print(f"[RAW type] {type(raw)}")

        if raw is None:
            print("[RAW] None → return []")
            print("==========================================================================\n")
            return []

        results = []

        # Case 1: raw is a simple string
        if isinstance(raw, str):
            lines = [x.strip() for x in raw.split("\n") if x.strip()]
            results.extend(lines)

        # Case 2: raw is a tuple(Document, score), 同上
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, tuple):
                    doc = item[0]
                    text = getattr(doc, "page_content", None) or doc.metadata.get("content", "")
                elif hasattr(item, "page_content"):
                    text = item.page_content
                elif isinstance(item, str):
                    text = item
                else:
                    continue

                results.append(text)

        print(f"[Parsed ITEM_REVIEW count]: {len(results)}")
        for i, r in enumerate(results[:k]):
            print(f"  ITEM #{i+1}: {r[:120]}")

        print("==========================================================================\n")
        return results[:k]

    def get_item_key_features(self, item: Dict[str, Any]) -> str:
        """
        从 item/business JSON 中抽取关键信息
        兼容 Amazon / Yelp / Goodreads，不存在的字段会自动跳过
        """
        fields = [
            "title",
            "name",        # Yelp 常用字段
            "brand",
            "feature",
            "categories",  # Yelp / Amazon 都可能有
            "attributes",
            "description",
            "main_category", # **
            "popular_shelves" # **
        ]
        info_parts: List[str] = []
        for f in fields:
            if f in item and item[f]:
                info_parts.append(f"{f}: {item[f]}")
        return "\n".join(info_parts)

    def retrieve(
        self,
        user_reviews: List[Dict[str, Any]],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        汇总检索结果给 Reasoner 使用
        """
        item_title = item.get("title", "") or item.get("name", "")
        if (not self.user_mem):
            similar_user_reviews = self.get_top_k_user_reviews(
                user_reviews=user_reviews,
                item_title=item_title,
                k=3,
            )
        else:
            similar_user_reviews = self.get_top_k_user_reviews_from_memory(
                item_title=item_title,
                k=3,
            )
        item_features = self.get_item_key_features(item)
        return {
            "similar_reviews": similar_user_reviews,
            "item_features": item_features,
        }


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

        明确“情绪强度”和“是否容易给极端好评/差评”，
        帮助后面 Reasoner 写出更贴近真实 Yelp/Amazon 分布的情绪。
        """
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


class MyReasoner:
    def __init__(self, llm: LLMBase):
        self.llm = llm

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
            "restaurant": ["foodie", "chef", "food blogger"],
            "hotel": ["travel blogger", "frequent traveler"],
        }
        if domain == "generic":
            return "none"
        for kw in domain_patterns.get(domain, []):
            if kw in txt:
                return "expert"
        return "novice"

    def _domain_guidance(self, domain: str) -> str:
        """
        给 LLM 一点领域特定的写作提醒，有助于提高 topic match / review_generation。
        """
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

        rating_context_lines = []
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

        rating_context = "\n".join(rating_context_lines) if rating_context_lines else "No strong prior rating signals."

        prompt = f"""
You are simulating a human user writing a product/business review.

==== USER PERSONA (JSON) ====
{persona_json}

==== USER PROFILE ====
{user_profile}

==== ITEM/BUSINESS INFORMATION ====
{item_info}

==== SIMILAR REVIEWS BY THE SAME USER ====
{similar_reviews}

==== RATING CONTEXT ====
{rating_context}

==== DOMAIN DETECTED ====
{domain}

==== USER EXPERTISE LEVEL IN THIS DOMAIN ====
{expertise}

Guidelines:
- Follow the persona's tone, writing style, and rating tendency.
- Use the rating context as a prior: generally stay within ±1 star of the item's historical average,
  unless the persona and context clearly justify a strong deviation.
- If the persona is "harsh", they are more willing to give low ratings; if "generous", they rarely give very low ratings.

Domain-specific hints:
{domain_guidance}
General Rules:
1. Write 2–4 sentences.
2. Follow the persona's punctuation, filler words, and logic.
3. Mention at least one specific detail about the item/business.
4. Rating must be consistent with the user's rating tendency AND the overall rating context.
5. Emotional intensity:
   - If you give 4.5 or 5.0 stars, express clear joy or enthusiasm (e.g., "absolutely loved", "definitely recommend").
   - If you give 2.0 stars or below, express clear dissatisfaction or frustration.
   - Avoid writing a very neutral review when the rating is extremely high or extremely low.

Format EXACTLY:
stars: <rating>
review: <text>

No extra commentary.
"""
        response = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=350,
        )

        parsed = self._extract_results(response)
        if parsed:
            raw_stars = parsed["stars"]
            calibrated_stars = self._calibrate_stars(raw_stars, rating_stats)
            parsed["stars"] = calibrated_stars
            return parsed

        fallback_stars = self._calibrate_stars(4.0, rating_stats)
        return {
            "stars": fallback_stars,
            "review": "The product or service performs reasonably well and offers acceptable value.",
        }


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


class MySimulationAgent(SimulationAgent):
    # Modified, add enable_memory to enable ablation
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
        #self.retriever = MyRetriever(memory=self.memory)
        self.retriever = MyRetriever(user_mem=self.user_mem, item_mem=self.item_mem)
        self.persona_builder = PersonaBuilder(llm=self.llm)
        self.reasoner = MyReasoner(llm=self.llm)
        self.controller = OutputController()

    def workflow(self) -> Dict[str, Any]:
        task = self.task 
        _plan = self.planner(task)

        user = self.interaction_tool.get_user(task["user_id"])
        item = self.interaction_tool.get_item(task["item_id"])
        item_reviews = self.interaction_tool.get_reviews(item_id=task["item_id"])
        user_reviews = self.interaction_tool.get_reviews(user_id=task["user_id"])

        rating_stats = compute_rating_stats(item_reviews, user_reviews)

        self.retriever.store_reviews(item_reviews, type="items")
        self.retriever.store_reviews(user_reviews, type="user")

        if user_reviews:
            similar_info = self.retriever.retrieve(
                user_reviews=user_reviews,
                item=item,
            )
        else:
            similar_info = {"similar_reviews": [], "item_features": ""}

        user_review_texts = [rv.get("text", "") for rv in user_reviews]
        persona_json = self.persona_builder.build(
            user_profile=user,
            user_reviews=user_review_texts,
        )

        raw_output = self.reasoner.generate_review(
            persona_json=persona_json,
            user_profile=user,
            item_info=item,
            similar_reviews=similar_info,
            rating_stats=rating_stats,
        )

        result = self.controller.parse(raw_output)
        return result
