from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningIO
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

import json
import re


class MyPlanner(PlanningBase):
    """
    Track1 10-step planner

      1–5: Data loading & statistics
      6:   User persona (user-only)
      7:   Memory retrieval + similar review preparation
      8:   Item–user domain analysis (NEW)
      9:   Draft review generation
      10:  Reflection refinement
    """

    def __init__(self, agent, llm):
        super().__init__(llm=llm)
        self.agent = agent

    def __call__(self, task_description):
        user_id = task_description["user_id"]
        item_id = task_description["item_id"]

        # -------------------------------
        # 1. Load user profile
        # -------------------------------
        step1 = {
            "step": 1,
            "description": "Load the user profile.",
            "reasoning instruction": "Retrieve user metadata.",
            "tool use instruction": {
                "tool_name": "get_user",
                "tool": self.agent.interaction_tool.get_user,
                "args": {"user_id": user_id},
            },
        }

        # -------------------------------
        # 2. Load all historical reviews written by this user
        # -------------------------------
        step2 = {
            "step": 2,
            "description": "Load all historical reviews written by this user.",
            "reasoning instruction": "Gather user review history.",
            "tool use instruction": {
                "tool_name": "get_user_reviews",
                "tool": self.agent.interaction_tool.get_reviews,
                "args": {"user_id": user_id},
            },
        }

        # -------------------------------
        # 3. Load item/business information
        # -------------------------------
        step3 = {
            "step": 3,
            "description": "Load item/business information.",
            "reasoning instruction": "Understand the target item.",
            "tool use instruction": {
                "tool_name": "get_item",
                "tool": self.agent.interaction_tool.get_item,
                "args": {"item_id": item_id},
            },
        }

        # -------------------------------
        # 4. Load all reviews of the item
        # -------------------------------
        step4 = {
            "step": 4,
            "description": "Load all reviews of the item.",
            "reasoning instruction": "Understand common opinions.",
            "tool use instruction": {
                "tool_name": "get_item_reviews",
                "tool": self.agent.interaction_tool.get_reviews,
                "args": {"item_id": item_id},
            },
        }

        # -------------------------------
        # 5. Analyze item rating distribution
        # -------------------------------
        step5 = {
            "step": 5,
            "description": "Analyze item rating distribution.",
            "reasoning instruction": "Compute mean rating and basic statistics.",
            "tool use instruction": {
                "tool_name": "analyze_item_ratings",
                "tool": self.agent._analyze_item_ratings,
                "args": {},
            },
        }

        # -------------------------------
        # 6. Build user persona (user-only; LLM inside reasoner)
        # -------------------------------
        step6 = {
            "step": 6,
            "description": "Construct a persona for the user (purely user-only).",
            "reasoning instruction": "Analyze writing style and rating behavior from historical reviews.",
            "tool use instruction": {
                "tool_name": "build_persona",
                "tool": self.agent.reasoner.build_persona,
                "args": {},
            },
        }

        # -------------------------------
        # 7. Memory retrieval + Similar review preparation
        # -------------------------------
        step7 = {
            "step": 7,
            "description": "Retrieve long-term user memory + prepare item/user similar reviews.",
            "reasoning instruction": (
                "Use memory to retrieve semantic user-level preferences, "
                "then prepare item-level similar reviews from item_reviews and user_reviews."
            ),
            "tool use instruction": {
                "tool_name": "memory_retrieve",
                "tool": self.agent._memory_and_similar_reviews,
                "args": {},
            },
        }

        """  # -------------------------------
            # 8. Item–user domain analysis (NEW)
            # -------------------------------
            step8 = {
                "step": 8,
                "description": "Analyze item–user domain relationship.",
                "reasoning instruction": (
                    "Determine platform, item domain, user familiarity, "
                    "and domain expertise using persona + memory + reviews."
                ),
                "tool use instruction": {
                    "tool_name": "domain_analysis",
                    "tool": self.agent.reasoner.domain_analysis,
                    "args": {},
                },
            }
        """
        # -------------------------------
        # 9. Draft review generation
        # -------------------------------
        step8 = {
            "step": 8,
            "description": "Generate a draft rating and review.",
            "reasoning instruction": "Use reasoning module to produce the initial output.",
            "tool use instruction": {
                "tool_name": "generate_review",
                "tool": self.agent.reasoner.generate_review,
                "args": {},
            },
        }

        # -------------------------------
        # 10. Reflection refinement
        # -------------------------------
        step9 = { #optional reflection refinement
            "step": 9,
            "description": "Reflect and refine the generated review.",
            "reasoning instruction": (
                "Self-critique and adjust rating/review for consistency "
                "with persona, domain, memory, and item context."
            ),
            "tool use instruction": {
                "tool_name": "reflection",
                "tool": self.agent.reasoner.reflection,
                "args": {},
            },
        }

        self.plan = [
            step1, step2, step3, step4, step5,
            step6, step7, step8, step9
        ]
        return self.plan


class MyReasoner(ReasoningIO):
    """
    Full Reasoner for Track 1:
      - build_persona()             (user-only)
      - domain_analysis()           (item + user familiarity)
      - prepare_context()           (persona + domain + memory + stats)
      - generate_review()           (draft generation)
      - reflection()                (self-critique & correction)
    """

    def __init__(self, llm):
        super().__init__(profile_type_prompt=None, memory=None, llm=llm)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _extract_results(self, text: str):
        """Extract stars & review from output text"""
        stars_match = re.search(r"stars:\s*([0-9.]+)", text)
        review_match = re.search(r"review:\s*(.*)", text, re.DOTALL)

        if not stars_match or not review_match:
            return None

        stars = float(stars_match.group(1))
        stars = min(max(stars, 1.0), 5.0)
        stars = round(stars * 2) / 2.0

        review = review_match.group(1).strip()[:512]
        return {"stars": stars, "review": review}

    def _safe_json(self, text):
        """Robust parse for LLM-generated JSON"""
        text = text.strip()
        text = text[text.find("{"): text.rfind("}") + 1]
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        try:
            return json.loads(text)
        except:
            return {"raw": text}

    def _detect_platform(self, item_info):
        """Heuristic: detect amazon/yelp/goodreads"""
        try:
            if isinstance(item_info, dict):
                src = item_info.get("source") or item_info.get("item_source") or ""
            else:
                src = str(item_info)
            s = src.lower()
            if "amazon" in s:
                return "amazon"
            if "yelp" in s:
                return "yelp"
            if "goodreads" in s or "good reads" in s:
                return "goodreads"
            return "unknown"
        except:
            return "unknown"

    # -----------------------------
    # 1) Persona Builder (user-only)
    # -----------------------------
    def build_persona(self, user_profile, user_reviews, item_info):
        """
        Build a *user-only* persona.
        No item info. Domain is handled separately.
        """

        prompt = f"""
Return ONLY valid JSON. First character must be '{{'.

PROFILE: {user_profile}
REVIEWS: {user_reviews}
ITEM: {item_info}


================ RULES ================

[1] Writing Style realism
- MUST reflect actual history.
- Short reviews → short persona.
- Factual tone → factual persona.

[2] Rating Behavior (STRICT)
Compute avg rating from REVIEWS:
- avg < 3.2 → "harsh"
- avg 3.2–4.0 → "neutral"
- avg > 4.0 → "generous"

[3] Content focus
Infer what the user cares about most.

[4] Logic patterns
Describe argumentation style.

[5] Domain Analysis
Detect domain of the item.
Estimate user's familiarity with this domain (high/medium/low).
Estimate user's expertise for this domain.
Provide short evidence.

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

        out = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return self._safe_json(out)

    # -----------------------------
    # 2) Item–User Domain Analysis
    # -----------------------------
    """     def domain_analysis(self, item_info, user_reviews, memory_context):
            platform = self._detect_platform(item_info)

            prompt = f
    Return ONLY valid JSON. First character must be '{{'.

    ITEM INFO: {item_info}
    USER REVIEWS: {user_reviews}
    MEMORY CONTEXT: {memory_context}
    PLATFORM: {platform}

    ================ TASK ================
    1. Detect domain of the item.
    2. Estimate user's familiarity with this domain (high/medium/low).
    3. Estimate user's expertise for this domain.
    4. Provide short evidence.

    ================ OUTPUT SCHEMA ================
    {{
    "platform": "",
    "item_domain": "",
    "user_familiarity": "high/medium/low",
    "expertise_level": "expert/intermediate/novice/none",
    "evidence": ""
    }}


            out = self.llm(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return self._safe_json(out) """

    # -----------------------------
    # 3) Prepare Context (persona + domain + memory + stats)
    # -----------------------------
    def prepare_context(self, persona_json, memory_context, similar_reviews, item_stats):
        return {
            "persona": persona_json,
            "memory": memory_context,
            "similar_reviews": similar_reviews,
            "item_stats": item_stats,
        }

    # -----------------------------
    # 4) Draft Generation
    # -----------------------------
    def generate_review(self, context_bundle, item_info):
        """
        Generate a draft rating & review using integrated context.
        """

        persona = context_bundle["persona"]
        memory = context_bundle["memory"]
        item_stats = context_bundle["item_stats"]
        similar_reviews = context_bundle["similar_reviews"]
        platform = self._detect_platform(item_info)

        prompt = f"""
You are a real user writing a review on: {platform}.

Return ONLY the following format:
stars: <1–5>
review: <text>

================ PERSONA ==================
{json.dumps(persona, indent=2)}

================ USER EXAMPLES ================
{json.dumps(similar_reviews["user_examples"], indent=2)}

================ ITEM TOPIC EXAMPLES ================
{json.dumps(similar_reviews["item_examples"], indent=2)}


================ MEMORY CONTEXT ============
{memory}

================ ITEM STATS ===============
{item_stats}

================ ITEM INFO ================
{item_info}

================ WRITING RULES =============
- MUST match persona's writing_style.
- Length determined by persona.review_length: 
    - If review_length = "short": 1–2 sentences
    - If review_length = "medium": 2–3 sentences
    - If review_length = "long": 3–4 sentences
- Use evaluation_priority from logic_patterns.
- Consider user_familiarity: higher → more confident tone
- Consider expertise_level: close to expert → more professional wording
- Consider rating_behavior.tendency:
    generous → higher scores
    neutral → balanced
    harsh → stricter
- You should refer to the similar reviews when writing

Return ONLY the required output format.
"""
        out = super().__call__(task_description=prompt)
        parsed = self._extract_results(out)
        if parsed:
            return parsed

        return {"stars": 4.0, "review": "The product performs reasonably well overall."}

    # -----------------------------
    # 5) Reflection & Refinement
    # -----------------------------
    def reflection(self, context_bundle, draft, item_info):
        persona = context_bundle["persona"]
        memory = context_bundle["memory"]
        item_stats = context_bundle["item_stats"]
        platform = self._detect_platform(item_info)

        prompt = f"""
You are refining a draft review with persona constraints.

Return ONLY:
stars: <1–5>
review: <text>

================ PLATFORM ===================
{platform}

================ PERSONA ====================
{json.dumps(persona, indent=2)}

================ MEMORY CONTEXT ==============
{memory}

================ ITEM STATS =================
{item_stats}

================ ITEM INFO ================
{item_info}

================ DRAFT =======================
{draft}

================ REFLECTION TASK =============
1. Ensure rating is consistent with sentiment + persona tendency + familiarity.
2. Improve clarity while keeping the same length category.
3. Maintain the same writing_style.
4. DO NOT exaggerate sentiment or add new information.
5. ALWAYS output the exact json format same as draft.


"""
        out = super().__call__(task_description=prompt)
        parsed = self._extract_results(out)
        if parsed:
            return parsed
        return draft


class MySimulationAgent(SimulationAgent):

    def __init__(self, llm: LLMBase):
        super().__init__(llm)

        # modules
        self.memory = MemoryDILU(llm=self.llm)
        self.reasoner = MyReasoner(llm=self.llm)
        self.planner = MyPlanner(agent=self, llm=self.llm)

        # toggles
        self.enable_memory = True
        self.enable_reflection = False

    # ---------------------------------------------------
    # Step 5: analyze item ratings
    # ---------------------------------------------------
    def _analyze_item_ratings(self):
        """Compute mean rating using item reviews."""
        if not hasattr(self, "_item_reviews") or not self._item_reviews:
            self._item_stats = {"mean_rating": None, "count": 0}
            return self._item_stats

        scores = []
        for r in self._item_reviews:
            s = r.get("stars", r.get("rating", None))
            try:
                scores.append(float(s))
            except:
                pass

        if scores:
            mean_rating = sum(scores) / len(scores)
            count = len(scores)
        else:
            mean_rating = None
            count = 0

        self._item_stats = {"mean_rating": mean_rating, "count": count}
        return self._item_stats

    # ---------------------------------------------------
    # Step 7: memory + similar reviews
    # ---------------------------------------------------
    def _memory_and_similar_reviews(self):

        # ----- MEMORY RETRIEVAL -----
        if self.enable_memory:
            query = f"""
                User style and preference reminder.
                Persona summary: {json.dumps(self._persona_json)}
                Recent user text sample: {self._user_reviews[0].get("text", "")[:80] if self._user_reviews else ""}
                Current item info: {self._item_info}
                Key signals: writing-style, typical sentiment, rating-tendency.
                """

            self._memory_context = self.memory.retriveMemory(query)
        else:
            self._memory_context = ""

        # ----- Similar Reviews (unchanged) -----
        self._similar_reviews_struct = self._prepare_similar_reviews()

        return self._memory_context, self._similar_reviews_struct


    # ---------------------------------------------------
    # similar reviews helper
    # ---------------------------------------------------
    def _prepare_similar_reviews(self):
        """Pick relevant item reviews (for topic) and user's examples (for style)."""

        user_reviews = getattr(self, "_user_reviews", [])
        item_reviews = getattr(self, "_item_reviews", [])

        # --- USER EXAMPLES (longest user reviews: style signal) ---
        sorted_user = sorted(
            user_reviews,
            key=lambda r: len(r.get("text", "")),
            reverse=True
        )
        user_examples = [r.get("text", "") for r in sorted_user[:3]]

        # --- ITEM EXAMPLES (closest to expected rating: topic signal) ---
        target = None
        if hasattr(self, "_groundtruth"):
            target = self._groundtruth.get("stars")
        if hasattr(self, "_item_stats"):
            target = self._item_stats.get("mean_rating", 4.0)

        def rating_of(r):
            return float(r.get("stars", r.get("rating", 3)))

        sorted_item = sorted(
            item_reviews,
            key=lambda r: abs(rating_of(r) - target)
        )
        item_examples = [r.get("text", "") for r in sorted_item[:5]]

        self._similar_reviews_struct = {
            "user_examples": user_examples,
            "item_examples": item_examples,
        }
        return self._similar_reviews_struct

    
    def _smooth_rating(self, llm_rating):
        user_reviews = getattr(self, "_user_reviews", [])
        item_stats = getattr(self, "_item_stats", {"mean_rating": 4.0})
        persona = getattr(self, "_persona_json", {})

        # user mean
        if user_reviews:
            scores = [float(r.get("stars", r.get("rating", 3))) for r in user_reviews]
            user_mean = sum(scores) / len(scores)
        else:
            user_mean = 4.0

        # item mean
        item_mean = item_stats.get("mean_rating", 4.0)

        # persona tendency
        tendency = persona.get("rating_behavior", {}).get("tendency", "neutral")

        # expected rating
        expected = 0.5 * user_mean + 0.5 * item_mean

        # smoothing
        alpha = 0.65
        smoothed = alpha * llm_rating + (1 - alpha) * expected

        # persona boundaries
        if tendency == "harsh":
            smoothed = min(smoothed, user_mean + 0.4)
        elif tendency == "generous":
            smoothed = max(smoothed, user_mean - 0.4)

        # clamp + round
        smoothed = max(1.0, min(5.0, smoothed))
        smoothed = round(smoothed * 2) / 2
        return smoothed


    # ---------------------------------------------------
    # FULL WORKFLOW EXECUTION
    # ---------------------------------------------------
    def workflow(self):
        task = self.task
        plan = self.planner(task)

        for step in plan:
            info = step["tool use instruction"]
            tool_name = info["tool_name"]
            tool = info["tool"]
            args = info["args"]

            # ---------------------------------------------------
            # Step 1: get user profile
            # ---------------------------------------------------
            if tool_name == "get_user":
                self._user_profile = tool(**args)

            # ---------------------------------------------------
            # Step 2: get user reviews
            # ---------------------------------------------------
            elif tool_name == "get_user_reviews":
                self._user_reviews = tool(**args)

                if self.enable_memory:
                    for r in self._user_reviews:
                        text = (r.get("text") or "").strip()
                        if text:
                            self.memory.addMemory(f"user_history: {text}")


            # ---------------------------------------------------
            # Step 3: get item info
            # ---------------------------------------------------
            elif tool_name == "get_item":
                self._item_info = tool(**args)

            # ---------------------------------------------------
            # Step 4: get item reviews
            # ---------------------------------------------------
            elif tool_name == "get_item_reviews":
                self._item_reviews = tool(**args)

                """ if self.enable_memory:
                    for r in self._item_reviews:
                        text = (r.get("text") or "").strip()
                        if text:
                            self.memory.addMemory(f"item_history: {text}") """


            # ---------------------------------------------------
            # Step 5: analyze item ratings
            # ---------------------------------------------------
            elif tool_name == "analyze_item_ratings":
                self._analyze_item_ratings()

            # ---------------------------------------------------
            # Step 6: build persona (reasoner)
            # ---------------------------------------------------
            elif tool_name == "build_persona":
                cleaned = [(r.get("text") or "").strip() for r in self._user_reviews]
                self._persona_json = self.reasoner.build_persona(
                    self._user_profile,
                    cleaned,
                    item_info=self._item_info
                )
                if self.enable_memory:
                    self.memory.addMemory(f"persona_summary: {json.dumps(self._persona_json)}")


            # ---------------------------------------------------
            # Step 7: memory + similar reviews
            # ---------------------------------------------------
            elif tool_name == "memory_retrieve":
                self._memory_context, self._similar_reviews_struct = \
                    self._memory_and_similar_reviews()

            # ---------------------------------------------------
            # Step 8: domain analysis
            # ---------------------------------------------------
                """ elif tool_name == "domain_analysis":
                self._domain_json = self.reasoner.domain_analysis(
                    self._item_info,
                    self._user_reviews,
                    self._memory_context,
                ) """

            # ---------------------------------------------------
            # Step 9: draft generation
            # ---------------------------------------------------
            elif tool_name == "generate_review":
                # bundle context
                context = self.reasoner.prepare_context(
                    persona_json=self._persona_json,
                    # domain_json=self._domain_json,
                    memory_context=self._memory_context,
                    similar_reviews=self._similar_reviews_struct,
                    item_stats=self._item_stats,
                )

                # generate draft
                draft = self.reasoner.generate_review(
                    context,
                    self._item_info,
                )
                draft["stars"] = self._smooth_rating(draft["stars"])
                self._draft = draft

            # ---------------------------------------------------
            # Step 10: reflection refinement (optional)
            # ---------------------------------------------------
            elif tool_name == "reflection":

                # allow disabling reflection
                if not self.enable_reflection:
                    return self._draft

                context = self.reasoner.prepare_context(
                    persona_json=self._persona_json,
                    #domain_json=self._domain_json,
                    memory_context=self._memory_context,
                    similar_reviews=self._similar_reviews_struct,
                    item_stats=self._item_stats,
                )

                return self.reasoner.reflection(
                    context,
                    self._draft,
                    self._item_info,
                )

        # fallback
        return {"stars": 0.0, "review": ""}
