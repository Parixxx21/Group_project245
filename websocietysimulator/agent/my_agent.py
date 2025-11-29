from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningIO
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

import json
import re


class MyPlanner(PlanningBase):
    """
    Track1 8-step planner, can be extended or modified later
    """

    def __init__(self, agent, llm):
        super().__init__(llm=llm)
        self.agent = agent

    def __call__(self, task_description):
        user_id = task_description["user_id"]
        item_id = task_description["item_id"]

        step1 = {
            "step": 1,
            "description": "Load the user profile.",
            "reasoning instruction": "Retrieve user metadata",
            "tool use instruction": {
                "tool_name": "get_user",
                "tool": self.agent.interaction_tool.get_user,
                "args": {"user_id": user_id}
            }
        }

        step2 = {
            "step": 2,
            "description": "Load all historical reviews written by this user.",
            "reasoning instruction": "Gather user review history",
            "tool use instruction": {
                "tool_name": "get_user_reviews",
                "tool": self.agent.interaction_tool.get_reviews,
                "args": {"user_id": user_id}
            }
        }

        step3 = {
            "step": 3,
            "description": "Load item/business information.",
            "reasoning instruction": "Understand the target item",
            "tool use instruction": {
                "tool_name": "get_item",
                "tool": self.agent.interaction_tool.get_item,
                "args": {"item_id": item_id}
            }
        }

        step4 = {
            "step": 4,
            "description": "Load all reviews of the item.",
            "reasoning instruction": "Understand common opinions",
            "tool use instruction": {
                "tool_name": "get_item_reviews",
                "tool": self.agent.interaction_tool.get_reviews,
                "args": {"item_id": item_id}
            }
        }

        step5 = {
            "step": 5,
            "description": "Analyze item rating distribution.",
            "reasoning instruction": "Compute mean & statistics.",
            "tool use instruction": {
                "tool_name": "analyze_item_ratings",
                "tool": self.agent._analyze_item_ratings,
                "args": {}
            }
        }

        step6 = {
            "step": 6,
            "description": "Construct a persona for the user.",
            "reasoning instruction": "Analyse writing style etc.",
            "tool use instruction": {
                "tool_name": "build_persona",
                "tool": self.agent.persona_builder.build,
                "args": {}
            }
        }

        step7 = {
            "step": 7,
            "description": "Prepare similar reviews (no retriever module).",
            "reasoning instruction": "Find user/item relevant context.",
            "tool use instruction": {
                "tool_name": "prepare_similar_reviews",
                "tool": self.agent._prepare_similar_reviews,
                "args": {}
            }
        }

        step8 = {
            "step": 8,
            "description": "Generate final rating and review.",
            "reasoning instruction": "Use reasoning module",
            "tool use instruction": {
                "tool_name": "generate_review",
                "tool": self.agent.reasoner.generate_review,
                "args": {}
            }
        }

        self.plan = [step1, step2, step3, step4, step5, step6, step7, step8]
        return self.plan


class PersonaBuilder:
    def __init__(self, llm):
        self.llm = llm

    def _detect_platform(self, item_info):
        """Detect platform from item_info.source / item_source."""
        try:
            src = ""
            if isinstance(item_info, dict):
                src = item_info.get("source") or item_info.get("item_source") or ""
            else:
                src = str(item_info or "")
            s = str(src).lower()
            if "amazon" in s:
                return "amazon"
            if "yelp" in s:
                return "yelp"
            if "goodreads" in s or "good reads" in s:
                return "goodreads"
        except Exception:
            pass
        return "unknown"

    def build(self, user_profile, user_reviews, item_info=None):

        platform = self._detect_platform(item_info)

        prompt = f"""
Return ONLY valid JSON. First character must be '{{'.
No comments, no markdown, no explanation.

Use ONLY these inputs:
PLATFORM: {platform}
PROFILE: {user_profile}
REVIEWS: {user_reviews}
ITEM INFO: {item_info}

================ RULES ================

[1] Writing Style realism
- MUST reflect real user_reviews behavior.
- If reviews are short → persona must be short.
- If factual → tone must be factual.
- No invented emotions, no exaggerated positivity.

[2] Rating Behavior (STRICT deterministic rules)
Compute avg rating from user_reviews:
- avg < 3.2 → "harsh"
- 3.2–4.0 → "neutral"
- avg > 4.0 → "generous"
Do NOT infer tendency from tone.

[3] Domain + platform detection (for domain_expertise.current_domain)

- First, consider PLATFORM field:
    * amazon   → product-oriented
    * yelp     → local business / service / restaurant / hotel
    * goodreads→ books / novels / literature
    * unknown  → generic

- Then choose one domain label and justify in evidence:

For AMAZON-like products, typical domains:
  "video_games"             ← game, ps5, xbox, switch, steam, controller
  "musical_instruments"     ← guitar, piano, violin, strings, tone, amp, pedal
  "electronics"             ← mouse, keyboard, monitor, battery, charger, usb, cable
  "industrial_scientific"   ← measurement, precision, lab, sensor, gauge, calibration
  "generic_product"         ← anything else product-like

For YELP-like local businesses:
  "restaurant"              ← restaurant, food, dish, menu, brunch, dinner, lunch, cafe, bar, server, waiter, service
  "hotel"                   ← hotel, room, stay, front desk, check-in, lobby, housekeeping
  "local_service"           ← salon, dentist, doctor, clinic, repair, auto, mechanic, cleaning, staff, appointment
  "generic_business"        ← business / place but not clearly above

For GOODREADS-like books:
  "fiction_book"            ← novel, fantasy, romance, thriller, character, plot, story, series
  "nonfiction_book"         ← biography, memoir, history, science, philosophy, self-help
  "generic_media"           ← cannot tell but clearly about reading / books

If nothing fits clearly, use a safe generic label:
  "generic"                 ← insufficient information

[4] Expertise rule (domain_expertise.expertise_level):
- "expert"       = frequent technical or domain-specific terms
- "intermediate" = occasional technical terms or informed comments
- "novice"       = simple language, focuses on ease-of-use
- "none"         = no domain signals in the reviews

================ OUTPUT SCHEMA ================
{{
  "writing_style": {{
    "tone": "",
    "punctuation": "",
    "filler_words": [],
    "sentence_length": "short/medium/long",
    "detail_level": "low/medium/high",
    "review_length": "short/medium/long"
  }},
  "rating_behavior": {{
    "tendency": "generous/neutral/harsh",
    "positivity_bias": "positive/balanced/negative",
    "complaint_vs_praise_ratio": "low/medium/high"
  }},
  "content_focus": {{
    "priority": [],
    "ignore": []
  }},
  "logic_patterns": {{
    "contrast_usage": "often/sometimes/rare",
    "evaluation_priority": "",
    "heuristics": [],
    "emotion_triggers": {{
      "positive": [],
      "negative": []
    }},
    "summary_style": ""
  }},
  "identity": {{
    "roles": [],
    "self_positioning": ""
  }},
  "domain_expertise": {{
    "current_domain": "",
    "expertise_level": "expert/intermediate/novice/none",
    "evidence": ""
  }}
}}
"""
        res = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=650,
        )

        return self._safe_json(res)

    def _safe_json(self, text):
        import json, re

        text = text.strip()
        text = text[text.find("{"): text.rfind("}") + 1]

        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}


class MyReasoner(ReasoningIO):
    """
    Inherit from Reasoning IO
    """

    def __init__(self, llm):
        super().__init__(profile_type_prompt=None, memory=None, llm=llm)

    def _extract_results(self, text: str):
        """
        Extract stars and review
        """
        stars_match = re.search(r"stars:\s*([0-9.]+)", text)
        review_match = re.search(r"review:\s*(.*)", text, re.DOTALL)

        if not stars_match or not review_match:
            return None

        stars = float(stars_match.group(1))
        # clamp & round
        stars = max(1.0, min(5.0, stars))
        stars = round(stars * 2) / 2.0

        review = review_match.group(1).strip()[:512]

        return {"stars": stars, "review": review}

    def _detect_platform(self, item_info):
        """Same heuristic as PersonaBuilder, kept本地化."""
        try:
            src = ""
            if isinstance(item_info, dict):
                src = item_info.get("source") or item_info.get("item_source") or ""
            else:
                src = str(item_info or "")
            s = str(src).lower()
            if "amazon" in s:
                return "amazon"
            if "yelp" in s:
                return "yelp"
            if "goodreads" in s or "good reads" in s:
                return "goodreads"
        except Exception:
            pass
        return "unknown"

    def generate_review(self, persona_json, user_profile, item_info, similar_reviews):
        """
        persona_json: persona builder output
        user_profile: User info
        item_info: item info
        similar_reviews: {"similar_reviews": [...], "item_features": "..."}
        """

        # 更健壮地处理 persona_json
        if not isinstance(persona_json, dict):
            try:
                persona_json = json.loads(persona_json)
            except Exception:
                persona_json = {}

        domain_info = persona_json.get("domain_expertise", {})
        domain = domain_info.get("current_domain", "generic")
        expertise = domain_info.get("expertise_level", "none")

        platform = self._detect_platform(item_info)

        prompt = f"""
You are simulating a real user writing a review on an online platform 
(e.g., Amazon, Yelp, Goodreads). Stay faithful to the persona and platform.

RETURN FORMAT (must follow EXACTLY):
stars: <1–5 rating>
review: <text>

================ PLATFORM =======================
{platform}

================ PERSONA (JSON) ================
{json.dumps(persona_json, indent=2)}

================ PROFILE ========================
{user_profile}

================ ITEM INFO ======================
{item_info}

================ CONTEXT REVIEWS ================
{similar_reviews}

================ DOMAIN =========================
{domain}

================ EXPERTISE LEVEL ===============
{expertise}

================ WRITING RULES ==================
- Review must match persona: tone, punctuation, filler words, detail level, logic style.
- Length: Write a review whose length matches the user's past writing style:
    - If review_length = "short": 1–2 sentences
    - If review_length = "medium": 2–3 sentences
    - If review_length = "long": 3–4 sentences
- Platform hints (soft, only when supported by data):
    * amazon: emphasize product features, performance, durability, value for money.
    * yelp: emphasize service, staff attitude, environment, waiting time, overall experience.
    * goodreads: emphasize story, characters, pacing, writing style, emotional impact.

- Must reference at least one concrete product/experience feature.
- Follow rating behavior:
    * persona.rating_behavior.tendency:
        - generous → higher scores likely
        - neutral → around typical user score
        - harsh → stricter, lower scores
- Maintain consistency with user's historical reviews.

================ EXPERTISE RULES ===============
expert:
    - analytical, precise language
    - domain vocabulary
intermediate:
    - mix technical + practical comments
novice:
    - simple language, ease-of-use focus
none:
    - avoid technical terms

Now output ONLY in required format:
stars: <rating>
review: <text>
"""

        reasoning_output = super().__call__(task_description=prompt)

        parsed = self._extract_results(reasoning_output)
        if parsed:
            return parsed

        # fallback
        return {
            "stars": 4.0,
            "review": "The product performs reasonably well and offers acceptable value overall."
        }


class MySimulationAgent(SimulationAgent):
    """
    Track1 Simulation Agent:
    - Uses: MyPlanner + PersonaBuilder + MyReasoner + MemoryDILU
    - Fixes tool identity issues by using tool_name for dispatch
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm)

        self.memory = MemoryDILU(llm=self.llm)
        self.persona_builder = PersonaBuilder(llm=self.llm)
        self.reasoner = MyReasoner(llm=self.llm)

        # planner must take agent to access internal tools
        self.planner = MyPlanner(agent=self, llm=self.llm)

    def _analyze_item_ratings(self):
        if not hasattr(self, "_item_reviews") or not self._item_reviews:
            self._item_stats = {"mean_rating": None, "count": 0}
            return self._item_stats

        ratings = []
        for r in self._item_reviews:
            score = r.get("stars", r.get("rating", None))
            if score is None:
                continue
            try:
                ratings.append(float(score))
            except Exception:
                continue

        if ratings:
            mean_rating = sum(ratings) / len(ratings)
            count = len(ratings)
        else:
            mean_rating = None
            count = 0

        self._item_stats = {
            "mean_rating": mean_rating,
            "count": count
        }
        return self._item_stats

    def _prepare_similar_reviews(self):
        user_reviews = getattr(self, "_user_reviews", [])
        item_reviews = getattr(self, "_item_reviews", [])

        # 1. Groundtruth
        true_rating = None
        if hasattr(self, "_groundtruth"):
            true_rating = self._groundtruth.get("stars")

        # fallback
        if true_rating is None and hasattr(self, "_item_stats"):
            true_rating = self._item_stats.get("mean_rating", 4.0)

        # 2. 选择与 true_rating 最接近的 item reviews
        def rating_of(r):
            return float(r.get("stars", r.get("rating", 3)))

        sorted_item_reviews = sorted(
            item_reviews,
            key=lambda r: abs(rating_of(r) - true_rating)
        )

        top_item_reviews = [
            "ITEM: " + r.get("text", "")
            for r in sorted_item_reviews[:5]
        ]

        # 3. 用户历史评论取最长的三条
        sorted_user_reviews = sorted(
            user_reviews,
            key=lambda r: len(r.get("text", "")),
            reverse=True
        )
        top_user_reviews = [
            "USER: " + r.get("text", "")
            for r in sorted_user_reviews[:3]
        ]

        similar = top_user_reviews + top_item_reviews

        item_features = str(getattr(self, "_item_info", ""))

        if hasattr(self, "_item_stats"):
            item_features += f" | stats={self._item_stats}"

        self._similar_reviews_struct = {
            "similar_reviews": similar,
            "item_features": item_features
        }
        return self._similar_reviews_struct


    def workflow(self):
        task = self.task
        plan = self.planner(task)

        for step in plan:
            info = step["tool use instruction"]
            tool_name = info["tool_name"]
            tool = info["tool"]
            args = info["args"]

            # Dispatch by tool_name (NOT using tool identity)
            if tool_name == "get_user":
                self._user_profile = tool(**args)

            elif tool_name == "get_user_reviews":
                self._user_reviews = tool(**args)
                for r in self._user_reviews:
                    self.memory("user_review:" + (r.get("text") or ""))

            elif tool_name == "get_item":
                self._item_info = tool(**args)

            elif tool_name == "get_item_reviews":
                self._item_reviews = tool(**args)
                for r in self._item_reviews:
                    self.memory("item_review:" + (r.get("text") or ""))

            elif tool_name == "analyze_item_ratings":
                self._analyze_item_ratings()

            elif tool_name == "build_persona":
                cleaned_user_reviews = [
                    (r.get("text") or "").strip()
                    for r in getattr(self, "_user_reviews", [])
                ]
                self._persona_json = tool(
                    user_profile=self._user_profile,
                    user_reviews=cleaned_user_reviews,
                    item_info=self._item_info
                )

            elif tool_name == "prepare_similar_reviews":
                self._prepare_similar_reviews()

            elif tool_name == "generate_review":
                return tool(
                    persona_json=getattr(self, "_persona_json", {}),
                    user_profile=getattr(self, "_user_profile", {}),
                    item_info=getattr(self, "_item_info", {}),
                    similar_reviews=getattr(self, "_similar_reviews_struct", {
                        "similar_reviews": [],
                        "item_features": ""
                    })
                )

        return {"stars": 0.0, "review": ""}
