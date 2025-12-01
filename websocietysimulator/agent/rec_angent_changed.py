from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from websocietysimulator.agent import RecommendationAgent
from collections import Counter
import json
import re
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")

def truncate_tokens(text, max_tokens=400):
    if not isinstance(text, str):
        text = str(text)
    toks = ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return ENC.decode(toks[:max_tokens])


class SafeIOUtils:

    @staticmethod
    def parse_json(text, fallback=None):
        """
        Parse text into JSON object (dict or list).
        Handles the following cases:
        - pure JSON string
        - JSON wrapped with extra text
        - markdown code block ```json
        - trailing commas
        """

        if not isinstance(text, str):
            return fallback

        raw = text.strip()

        # 1) remove markdown code block
        raw = re.sub(r"```json", "", raw, flags=re.IGNORECASE)
        raw = raw.replace("```", "")

        # 2) attempt direct load first (fast path)
        try:
            return json.loads(raw)
        except:
            pass

        # 3) try extracting the largest {...} or [...] block
        json_candidate = SafeIOUtils._extract_json_block(raw)

        if json_candidate is None:
            return fallback

        # 4) clean illegal trailing commas
        json_candidate = re.sub(r",\s*}", "}", json_candidate)
        json_candidate = re.sub(r",\s*]", "]", json_candidate)

        # 5) final attempt
        try:
            return json.loads(json_candidate)
        except:
            return fallback

    @staticmethod
    def parse_json_object(text, fallback=None):
        """Return a dict only."""
        res = SafeIOUtils.parse_json(text, fallback=fallback)
        return res if isinstance(res, dict) else fallback

    @staticmethod
    def parse_json_list(text, fallback=None):
        """Return a list only."""
        res = SafeIOUtils.parse_json(text, fallback=fallback)
        return res if isinstance(res, list) else fallback

    @staticmethod
    def _extract_json_block(text):
        """
        Extract the outermost {...} or [...] content.
        Picks the longest well-formed substring.
        """

        # find dict block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]

        # find list block
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]

        return None

class MyPlanner(PlanningBase):
    """
    Track1 8-step planner, can be extended or modified later
    """

    def __init__(self, agent, llm):
        super().__init__(llm=llm)
        self.agent = agent

    def __call__(self, task_description):
        user_id = task_description["user_id"]
        candidate_items = task_description["candidate_list"]

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
            "description": "Load item information for all candidate items",
            "reasoning instruction": "Retrieve product metadata",
            "tool use instruction": {
                "tool_name": "get_items_info",
                # 这个要自己wrap一个get_item的function我觉得
                "tool": self.agent._get_items_info,
                "args": {"item_ids": candidate_items}
            }
        }

        step4 = {
            "step": 4,
            "description": "Load item reviews for all candidate items",
            "reasoning instruction": "Get community opinion about each item",
            "tool use instruction": {
                "tool_name": "get_items_reviews",
                # 这个也是
                "tool": self.agent._get_items_reviews,
                "args": {"item_ids": candidate_items}
            }
        }

        step5 = {
            "step": 5,
            "description": "Build the user's preference profile.",
            "reasoning instruction": "Extract categories, liked attributes, disliked patterns.",
            "tool use instruction": {
                "tool_name": "build_preference_profile",
                "tool": self.agent.preference_builder.build,
                "args": {}
            }
        }

        step6 = {
            "step": 6,
            "description": "Generate final ranking using LLM with preference profile.",
            "reasoning instruction": "Use preference profile and item features to decide ranking.",
            "tool use instruction": {
                "tool_name": "generate_ranking",
                "tool": self.agent.reasoner.generate_ranking,
                "args": {}
            }
        }


        self.plan = [step1, step2, step3, step4, step5, step6]
        return self.plan


class PreferenceBuilder:
    def __init__(self, llm, memory=None, use_memory=True):
        self.llm = llm
        self.memory = memory
        self.use_memory = use_memory

    def _detect_platform(self, item_info):
        """
        item_info can be:
        - None
        - a single item_info dict
        - a dict of {item_id: item_info_dict}
        """
        try:
            if item_info is None:
                return "unknown"

            if isinstance(item_info, dict) and all(isinstance(v, dict) for v in item_info.values()):
                first = next(iter(item_info.values()))
                return self._detect_platform(first)

            if isinstance(item_info, dict):
                src = item_info.get("source") or item_info.get("item_source") or ""
            else:
                src = str(item_info or "")

            s = src.lower()
            if "amazon" in s:
                return "amazon"
            if "yelp" in s:
                return "yelp"
            if "goodreads" in s:
                return "goodreads"
        except:
            pass

        return "unknown"

    def _summarize_user_reviews(self, reviews):
        out = []
        for r in reviews or []:
            if not isinstance(r, dict):
                continue
            out.append({
                "item_id": r.get("item_id"),
                "stars": r.get("stars"),
                "text": (r.get("text") or "")[:200]
            })
        return out

    def _retrieve_memory_context(self, user_profile):
        """
        Query memory module for hidden long-term patterns.
        """
        if (not self.use_memory) or (self.memory is None):
            return ""

        try:
            q = (
                "Summarize this user's long-term stable preferences based on past memory data. "
                "Focus on categories, liked attributes, disliked aspects, writing style, common rating patterns. "
                f"User profile: {user_profile}"
            )
            raw = self.memory(q)
            return truncate_tokens(str(raw), max_tokens=350)
        except:
            return ""

    def build(self, user_profile, user_reviews, item_info=None):

        platform = self._detect_platform(item_info)
        summarized_reviews = self._summarize_user_reviews(user_reviews)

        # ====== NEW MEMORY CONTEXT ======
        memory_context = self._retrieve_memory_context(user_profile)

        # ====== FULL RULES INCLUDED ======
        prompt = f"""
Return ONLY valid JSON. The first non-whitespace character must be '{{'.
Do NOT output any commentary or explanation outside JSON.

Your task: extract a *user preference profile*.
All categories & attributes MUST come from:
- USER REVIEWS (text or implied keywords)
- ITEM INFO fields (platform-specific)
- MEMORY CONTEXT (if provided)

You MUST NOT hallucinate categories/attributes not grounded in the above.

================= INPUTS =================
PLATFORM:
{platform}

USER PROFILE:
{user_profile}

USER REVIEWS:
{summarized_reviews}

ITEM INFO:
{item_info}

MEMORY CONTEXT:
{memory_context}

================= RULES ==================

[1] semantic_categories (dict)
- Keys must come from text observed in:
    * user reviews, OR
    * item categories/fields, OR
    * memory context (if provided)
- Values must be EXACTLY one of:
    "high", "medium", "low"

Example:
{{
  "children's books": "high",
  "fantasy": "medium"
}}

[2] preferred_attributes (list)
- Attributes explicitly praised in review text.
- OR attributes from item_info ONLY IF those items were rated high.
- OR positive attributes implied in memory context.
- Must be short natural-language attribute labels.

[3] disliked_attributes (list)
- Attributes explicitly criticized.
- OR negative features implied in memory context.
- Should contain clear negative signals: e.g., "poor durability", "inaccurate description".

[4] attribute_preference_strength (dict)
- Combine information from:
    * preferred attributes
    * disliked attributes
    * memory context indicators
- Each key must be an attribute.
- Each value must be one of:
    "high", "medium", "low"

[5] price_sensitivity
Rules:
- If reviews or memory mention “expensive”, “overpriced” → "high"
- If reviews or memory say “good value”, “worth it” → "low"
- Otherwise → "medium"

[6] brand_loyalty (list)
For Amazon/Goodreads:
- If the user often praises the same brand/author, list them here.
- If memory context indicates repeated preferences, include them.

For Yelp:
- Typically leave empty unless user consistently praises a specific chain.

[7] topic_keywords (list of 5–10 strings)
- Extract ONLY from:
    * review text
    * item_info fields
    * memory context
- Must be real tokens or short phrases actually found or directly implied.
- No hallucinations.

[8] summary (1–2 sentences)
- Summarize key tastes:
    * preferred categories
    * favored attributes
    * disliked aspects
    * price sensitivity
- MUST NOT hallucinate new facts not in inputs.

================= OUTPUT JSON SCHEMA =================
{{
  "semantic_categories": {{}},
  "preferred_attributes": [],
  "disliked_attributes": [],
  "attribute_preference_strength": {{}},
  "price_sensitivity": "low/medium/high",
  "brand_loyalty": [],
  "topic_keywords": [],
  "summary": ""
}}
"""

        # ====== RUN LLM ======
        res = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )

        parsed = SafeIOUtils.parse_json(res, fallback={})

        # base schema
        schema = {
            "semantic_categories": {},
            "preferred_attributes": [],
            "disliked_attributes": [],
            "attribute_preference_strength": {},
            "price_sensitivity": "medium",
            "brand_loyalty": [],
            "topic_keywords": [],
            "summary": ""
        }

        if isinstance(parsed, dict):
            schema.update(parsed)

        return schema




class RankingReasoner(ReasoningBase):
    def __init__(self, llm, memory=None, use_memory=True):
        """
        llm: LLMBase 实例
        memory: MemoryDILU 实例（可为 None）
        use_memory: 是否在排序时使用记忆检索的信息
        """
        super().__init__(profile_type_prompt="", memory=memory, llm=llm)
        self.use_memory = use_memory

    def _summarize_candidate_reviews(self, reviews_dict):
        """
        reviews_dict: {item_id: [reviews]}
        每个 item 最多保留 2 条，每条 text 截断 200 字符
        """
        out = {}
        for iid, reviews in (reviews_dict or {}).items():
            if not isinstance(reviews, list):
                continue
            limited = []
            for r in reviews[:2]:   # 每个 item 只保留两条
                if not isinstance(r, dict):
                    continue
                limited.append({
                    "text": (r.get("text") or "")[:200],
                    "stars": r.get("stars")
                })
            out[iid] = limited
        return out

    def _build_memory_query(self, preference_profile, candidate_items):
        """
        构造一个用于 MemoryDILU 检索的 query 文本。
        目标：让 MemoryDILU 返回 “跟这个用户长期偏好最相关的历史片段”。
        """
        try:
            summary = preference_profile.get("summary", "")
            cats = list((preference_profile.get("semantic_categories") or {}).keys())
        except Exception:
            summary = ""
            cats = []

        # 取前几个候选 item 的 title/name，帮助 memory 选相似的历史内容
        item_titles = []
        for c in candidate_items[:5]:
            if not isinstance(c, dict):
                continue
            title = c.get("title") or c.get("name") or ""
            if title:
                item_titles.append(title)

        query = f"""
User long-term preference summary: {summary}
Main semantic categories: {cats}
Representative candidate items: {item_titles}

Please retrieve past user/item reviews that best reflect this user's stable tastes.
"""
        return query

    def _retrieve_memory_context(self, preference_profile, candidate_items):
        """
        调用 MemoryDILU，取出与当前用户偏好最相关的一些历史片段。
        如果 memory 不可用或调用失败，则返回空字符串。
        """
        if (not self.use_memory) or (self.memory is None):
            return ""

        try:
            query = self._build_memory_query(preference_profile, candidate_items)
            raw_memory = self.memory(query)  # MemoryDILU 在 baseline 里就是这样 __call__ 使用的
            # 控制长度，避免把 prompt 撑爆
            memory_context = truncate_tokens(str(raw_memory), max_tokens=400)
            return memory_context
        except Exception:
            return ""

    def generate_ranking(self, preference_profile, candidate_items, candidate_items_reviews):
        """
        Main ranking function (not using ReasoningBase.__call__()).
        """
        summarized_reviews = self._summarize_candidate_reviews(candidate_items_reviews)

        # ========= 新增：从 MemoryDILU 取出与本次排序相关的记忆 =========
        memory_context = self._retrieve_memory_context(preference_profile, candidate_items)

        prompt = f"""
You are ranking items for a user.

# USER MEMORY (retrieved by the agent)
The following texts are retrieved from the agent's long-term memory and
summarize the user's stable tastes and historically liked/disliked patterns:

{memory_context}

# USER PREFERENCE PROFILE (JSON)
This profile is built from the user's historical reviews and item information:

{json.dumps(preference_profile, indent=2)}

# CANDIDATE ITEMS
Each item has metadata fields such as title, categories, attributes, price, etc.:

{json.dumps(candidate_items, indent=2)}

# CANDIDATE REVIEWS
For each candidate item, you see up to 2 community reviews (truncated):

{json.dumps(summarized_reviews, indent=2)}

Your task:
Sort ALL candidate item_ids from most preferred to least preferred.

====================== RANKING RULES ======================

1. Use semantic_categories first  
   - Stronger category match → higher rank  
   - Use the strength weight ("high" > "medium" > "low")
   - This is the strongest indicator.

2. Use preferred_attributes  
   - If an item (its fields OR its reviews) contains or implies these attributes,
     increase its rank.

3. Use disliked_attributes  
   - If an item (its fields OR its reviews) contains or implies these attributes,
     decrease its rank.

4. Attribute Extraction and Alignment  
   - You MUST extract item attributes from BOTH:
        * candidate_items fields
        * candidate_items_reviews text
   - You MUST match extracted attributes against the user preference profile
     AND the retrieved memory context.
   - Items with more aligned attributes must rank higher.

5. Consider price_sensitivity  
   - If the item or its reviews mention “expensive”, “overpriced”, “worth it”,
     interpret them based on the user’s price_sensitivity.

6. Use brand_loyalty  
   - If the user prefers certain brands/authors and the item belongs to them,
     increase rank.

7. Consistency with Memory  
   - If memory_context shows strong historical preference for certain categories,
     authors, brands, or attributes, prefer items that match them.
   - If memory_context shows repeated complaints, down-rank similar items.

8. No Hallucinations  
   - Use only item_ids from candidate_items.
   - Do NOT create new ids.
   - Output must match the candidate list exactly.

9. Output Format  
   - Return ONLY a pure JSON list of item_ids.
   - No explanation.

================ OUTPUT EXAMPLE ================
["id1","id2","id3"]
"""
        raw_output = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )

        fallback = [item["item_id"] for item in candidate_items]
        ranked_ids = SafeIOUtils.parse_json_list(raw_output, fallback=fallback)

        # 过滤掉不在候选里的 id，并补齐缺失
        id_set = {item["item_id"] for item in candidate_items}
        ranked_ids = [i for i in ranked_ids if i in id_set]
        ranked_ids.extend([i for i in id_set if i not in ranked_ids])

        return ranked_ids




class MyRecommendationAgent(RecommendationAgent):
    """
    Track2 Recommendation Agent:
    Includes:
    - Planner
    - PreferenceBuilder (optional)
    - RankingReasoner (with memory)
    - MemoryDILU
    And supports ablation switches!
    """

    def __init__(
        self,
        llm: LLMBase,
        use_memory=True,
        use_preference_builder=True,
        use_candidate_reviews=True
    ):
        super().__init__(llm=llm)

        # ----- Ablation Flags -----
        self.use_memory = use_memory
        self.use_preference_builder = use_preference_builder
        self.use_candidate_reviews = use_candidate_reviews

        # ----- Tools / Modules -----
        self.memory = MemoryDILU(llm=self.llm)
        self.preference_builder = PreferenceBuilder(llm=self.llm)

        # pass memory into reasoner
        self.reasoner = RankingReasoner(
            llm=self.llm,
            memory=self.memory if self.use_memory else None,
            use_memory=self.use_memory
        )

        self.planner = MyPlanner(agent=self, llm=self.llm)

        # internal storage
        self._user_profile = None
        self._user_reviews = None
        self._candidate_items_info = {}
        self._candidate_items_reviews = {}
        self._history_item_info = {}

    # ============================================================
    #  Helpers for item info (unchanged)
    # ============================================================

    def _compress_item_info(self, item_dict):
        if not isinstance(item_dict, dict):
            return item_dict

        keys = [
            "item_id", "name", "title", "stars", "avg_rating",
            "average_rating", "review_count", "rating_number",
            "ratings_count", "price", "categories", "attributes",
            "description", "title_without_series",
        ]

        compressed = {k: item_dict.get(k) for k in keys if k in item_dict}

        desc = compressed.get("description")
        if isinstance(desc, str):
            compressed["description"] = desc[:300]

        return compressed

    def _get_items_info(self, item_ids):
        results = {}
        for iid in item_ids:
            try:
                item = self.interaction_tool.get_item(item_id=iid)
                results[iid] = self._compress_item_info(item)
            except Exception as e:
                results[iid] = {"item_id": iid, "error": str(e)}
        return results

    def _get_items_reviews(self, item_ids):
        if not self.use_candidate_reviews:
            return {}

        results = {}
        for iid in item_ids:
            try:
                reviews = self.interaction_tool.get_reviews(item_id=iid)
                results[iid] = reviews

                if self.use_memory:
                    for r in reviews:
                        self.memory("item_review:" + (r.get("text") or ""))
            except Exception:
                results[iid] = []
        return results

    def _extract_user_item_ids(self, reviews):
        ids = set()
        for r in reviews or []:
            if isinstance(r, dict) and "item_id" in r:
                ids.add(r["item_id"])
        return list(ids)

    # ============================================================
    #  Core workflow
    # ============================================================

    def workflow(self):
        task = self.task
        plan = self.planner(task)

        for step in plan:
            info = step["tool use instruction"]
            tool_name, tool, args = info["tool_name"], info["tool"], info["args"]

            if tool_name == "get_user":
                self._user_profile = tool(**args)

            elif tool_name == "get_user_reviews":
                reviews = tool(**args)
                self._user_reviews = reviews
                if self.use_memory:
                    for r in reviews:
                        self.memory("user_review:" + (r.get("text") or ""))

            elif tool_name == "get_items_info":
                self._candidate_items_info = tool(**args)

            elif tool_name == "get_items_reviews":
                self._candidate_items_reviews = tool(**args)

            elif tool_name == "build_preference_profile":
                if not self.use_preference_builder:
                    # Ablation: skip PB
                    self._preference_profile = {"semantic_categories": {}, "summary": ""}
                    continue

                # build PB
                history_item_ids = self._extract_user_item_ids(self._user_reviews)
                self._history_item_info = self._get_items_info(history_item_ids)

                cleaned_reviews = []
                for r in self._user_reviews or []:
                    if isinstance(r, dict):
                        cleaned_reviews.append({
                            "item_id": r.get("item_id"),
                            "text": (r.get("text") or "").strip(),
                            "stars": r.get("stars")
                        })

                self._preference_profile = tool(
                    user_profile=self._user_profile,
                    user_reviews=cleaned_reviews,
                    item_info=self._history_item_info
                )

            elif tool_name == "generate_ranking":
                items_list = []
                for iid, info_dict in self._candidate_items_info.items():
                    if isinstance(info_dict, dict):
                        info_dict["item_id"] = iid
                    items_list.append(info_dict)

                return tool(
                    preference_profile=self._preference_profile,
                    candidate_items=items_list,
                    candidate_items_reviews=self._candidate_items_reviews
                )

        return []



