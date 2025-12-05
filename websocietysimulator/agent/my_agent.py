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
# 0. Mock LLM for quick local testing
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
    Extract a numeric rating from a review JSON.

    We support multiple field names across Yelp/Amazon/Goodreads:
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
    Compute rating statistics for the item and the user.

    These statistics are used both as background information in the prompt
    and later for numeric star calibration to reduce star_mae.
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
# 2. MemoryDILU wrapper with explicit retrieve/add methods
###########################################################
class MemoryDILU(MemoryBase):
    """
    Lightweight wrapper around MemoryBase.

    - retriveMemory(query_scenario): perform similarity search over scenario_memory
    - addMemory(current_situation): insert a trajectory into scenario_memory
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
    Planner module.

    Returns a static multi-step plan for clarity and extensibility.
    This makes the pipeline structure explicit without incurring
    extra LLM calls for planning.
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
        Store item/user reviews in the retriever (and optionally in memory).
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

    # ---------- Local embedding-based retrieval ----------
    def get_top_k_user_reviews(
        self,
        user_reviews: List[Dict[str, Any]],
        item_title: str,
        k: int = 3,
    ) -> List[str]:
        """
        Retrieve the top-k user reviews whose text is most similar to the item title
        using a local embedding model.
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

    # ---------- Retrieve user reviews from MemoryDILU ----------
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

    # ---------- Retrieve item reviews from MemoryDILU ----------
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
        Extract key fields from the item/business JSON.

        We handle Amazon / Yelp / Goodreads style JSON.
        Missing fields are simply skipped.
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
        Select a few "other reviews for this business" as topical anchors.

        We first use local item reviews, and then fall back to MemoryDILU
        if we need more examples.
        """
        texts = [rv.get("text", "") for rv in item_reviews if rv.get("text")]
        anchors: List[str] = []

        if texts:
            # For simplicity, just take the first few reviews.
            # (This could be replaced by embedding-based top-k if needed.)
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
        Gather retrieval results for the reasoner:

        - similar_user_reviews: historical reviews by the same user that are
          semantically close to the current item.
        - similar_item_reviews: other reviews for this item/business (topic anchors).
        - item_features: structured item/business information.
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
        Call the LLM to summarize a persona from user profile + historical reviews.

        The persona is returned as a JSON-formatted string.
        We assume user_reviews have already been truncated upstream.
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


###########################################################
# 6. Reasoner (rating + review generation)
###########################################################
class MyReasoner:
    """
    Reasoner module.

    Besides generating the final rating and review, this module also
    performs the *second stage* of context gathering:
    - aggregates rating statistics into a textual rating context;
    - injects domain hints and inferred user expertise;
    - merges persona JSON, retrieved user/item reviews, and item metadata
      into a single structured prompt (with truncation to fit context limits).

    High-level data loading happens in `MySimulationAgent.workflow`,
    while prompt-level context assembly and single-call CoT reasoning
    happen here in `generate_review`.
    """

    MAX_PROMPT_CHARS = 15000  # coarse limit, to avoid exceeding the model's context

    def __init__(self, llm: LLMBase):
        self.llm = llm

    # ---------- Parse LLM output ----------
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
        Calibrate the LLM-predicted rating using rating statistics.

        - Smooth toward the item's historical mean rating.
        - Apply a small bias based on the user's average rating.
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

    # ---------- Domain / expertise inference ----------
    def infer_domain(self, item_info: Dict[str, Any]) -> str:
        txt = str(item_info).lower()

        # Typical Yelp domains
        if any(k in txt for k in ["restaurant", "cafe", "coffee", "bar", "bistro", "diner", "pizza", "burger"]):
            return "restaurant"
        if any(k in txt for k in ["hotel", "inn", "resort", "motel"]):
            return "hotel"

        # Common Amazon categories
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

    # ---------- Map expected rating to sentiment band ----------
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
        """
        Single-call CoT reasoning for one (user, item) pair.

        This method:
        - gathers and formats all contextual signals (persona, domain, rating stats,
          retrieved reviews, item metadata);
        - constructs a structured prompt with explicit rules;
        - calls the LLM once to jointly produce rating + review;
        - calibrates the rating using rating_stats.
        """
        # ---- Context gathering (semantic/domain level) ----
        domain = self.infer_domain(item_info)
        expertise = self.infer_user_expertise(persona_json, domain)
        domain_guidance = self._domain_guidance(domain)

        # ---- Context gathering (numeric rating priors) ----
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

        # Estimate an "expected" star level to choose the target sentiment band.
        expected_star = item_mean if item_mean is not None else (user_mean or 3.8)
        sentiment_label = self._sentiment_label(expected_star)

        # ---- Context gathering (retrieved examples) ----
        similar_user_reviews = similar_reviews.get("similar_user_reviews") or similar_reviews.get(
            "similar_reviews", []
        )
        similar_item_reviews = similar_reviews.get("similar_item_reviews", [])

        # ---- Context gathering (prompt construction & truncation) ----
        persona_str = self._truncate(str(persona_json), 6000)
        user_profile_str = self._truncate(str(user_profile), 2000)
        item_info_str = self._truncate(str(item_info), 2500)
        sim_user_str = self._truncate(str(similar_user_reviews), 2500)
        sim_item_str = self._truncate(str(similar_item_reviews), 2500)
        rating_context_str = self._truncate(rating_context, 1000)

        # ---- Final prompt assembly (single-call CoT reasoning) ----
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
            temperature=0.2,  # slightly lower temperature for more stable outputs
            max_tokens=350,
        )

        parsed = self._extract_results(str(response))
        if parsed:
            raw_stars = parsed["stars"]
            calibrated_stars = self._calibrate_stars(raw_stars, rating_stats)
            parsed["stars"] = calibrated_stars
            return parsed

        # Fallback if parsing fails.
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
        """
        Normalize the final output into a {stars, review} dictionary.

        - Stars are clamped to [1, 5] and quantized to 0.5 increments.
        - Review text is truncated to 512 characters.
        """
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
    End-to-end simulation agent.

    Pipeline overview (corresponding to the workflow figure):

        Planning → High-level Context Gathering → Rating Stats Analysis →
        Memory Retrieval + Domain-focused guidance →
        Persona Build → Single-call CoT Review Generation →
        Stats-based output adjustment.

    High-level data loading and truncation happen in `workflow()`,
    while prompt-level context assembly and CoT reasoning happen
    inside `MyReasoner.generate_review()`.
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm)
        self.enable_memory = True

        # Initialize optional long-term memory modules if the LLM supports embeddings.
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
        """
        Single end-to-end call for one (user, item) pair.

        This function performs the *high-level* context gathering:
        - fetch user profile, item info, and historical reviews via InteractionTool;
        - compute rating statistics;
        - populate retriever/memory with raw reviews;
        - construct a truncated list of user reviews for persona building.

        It then delegates prompt-level context assembly and CoT reasoning
        to `MyReasoner.generate_review`, and finally normalizes the output
        with the OutputController.
        """
        task = self.task  # dict (SimulationTask.to_dict())
        _plan = self.planner(task)  # currently used only to keep the pipeline structure explicit

        # --------- 1. Fetch data via InteractionTool ---------
        user = self.interaction_tool.get_user(task["user_id"])
        item = self.interaction_tool.get_item(task["item_id"])
        item_reviews = self.interaction_tool.get_reviews(item_id=task["item_id"])
        user_reviews = self.interaction_tool.get_reviews(user_id=task["user_id"])

        # --------- 2. Compute rating statistics ---------
        rating_stats = compute_rating_stats(item_reviews, user_reviews)

        # --------- 3. Store reviews in retriever / memory ---------
        self.retriever.store_reviews(item_reviews, type="items")
        self.retriever.store_reviews(user_reviews, type="user")

        # --------- 4. Retrieve similar reviews & item features ---------
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

        # --------- 5. Build persona (truncating user_reviews to avoid overflow) ---------
        user_review_texts = [rv.get("text", "") for rv in user_reviews if rv.get("text")]
        # At most 30 reviews, each truncated to 300 characters.
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

        # --------- 6. Reasoner: generate rating + review ---------
        raw_output = self.reasoner.generate_review(
            persona_json=persona_json,
            user_profile=user,
            item_info=item,
            similar_reviews=similar_info,
            rating_stats=rating_stats,
        )

        # --------- 7. Output control & normalization ---------
        result = self.controller.parse(raw_output)
        return result
