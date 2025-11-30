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
    def __init__(self, llm):
        self.llm = llm

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

            # Dict-of-dicts: {item_id: item_info_dict}
            if isinstance(item_info, dict) and all(isinstance(v, dict) for v in item_info.values()):
                first = next(iter(item_info.values()))
                return self._detect_platform(first)

            # Single item_info dict
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
        except Exception:
            pass

        return "unknown"

    def _summarize_user_reviews(self, reviews):
        """
        压缩 user_reviews：只保留 item_id / stars / text 前 200 字符
        """
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

    def build(self, user_profile, user_reviews, item_info=None):

        platform = self._detect_platform(item_info)
        summarized_reviews = self._summarize_user_reviews(user_reviews)

        # ========= prompt 保留你原来的结构，只把 USER REVIEWS 换成 summarized_reviews =========
        prompt = f"""
Return ONLY valid JSON. The first non-whitespace character must be '{{'.

Your task: extract a *user preference profile*.
All categories & attributes MUST come from:
- USER REVIEWS (text or implied keywords)
- ITEM INFO fields (platform-specific)

Do NOT invent categories or attributes not grounded in the data.

================= INPUTS =================
PLATFORM: {platform}

USER PROFILE:
{user_profile}

USER REVIEWS:
{summarized_reviews}

ITEM INFO:
{item_info}

================= RULES ==================

[1] semantic_categories (dict)
Keys must appear in review text or item_info fields.
Values ∈ {{ "high","medium","low" }}

Example:
{{
  "children's books": "high",
  "fantasy": "medium"
}}

[2] preferred_attributes
List attributes praised in reviews or item_info (item_info only if user rate it high).

[3] disliked_attributes
List attributes criticized in reviews.

[4] attribute_preference_strength
Map each attribute to strength level:
{{ "attribute": "high"|"medium"|"low" }}

[5] price_sensitivity
Rules:
- mentions “expensive/overpriced” → "high"
- mentions “good value/worth it” → "low"
- otherwise → "medium"

[6] brand_loyalty
For Amazon/Goodreads: repeated brand/author praise.

[7] topic_keywords
5–10 keywords from review text or item_info.

[8] summary
1–2 sentence summary of preference tendencies.

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
        # ========= END prompt =========

        res = self.llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )

        parsed = SafeIOUtils.parse_json(res, fallback={})

        # 填默认 schema
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
    def __init__(self, llm, memory=None):
        super().__init__(profile_type_prompt="", memory=memory, llm=llm)

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

    def generate_ranking(self, preference_profile, candidate_items, candidate_items_reviews):
        """
        Main ranking function (not using ReasoningBase.__call__()).
        """
        summarized_reviews = self._summarize_candidate_reviews(candidate_items_reviews)

        # ========= 你的原 prompt，结构不动，只把 reviews 换成 summarized_reviews =========
        prompt = f"""
You are ranking items for a user.

USER PREFERENCE PROFILE (JSON):
{json.dumps(preference_profile, indent=2)}

CANDIDATE ITEMS:
{json.dumps(candidate_items, indent=2)}

CANDIDATE REVIEWS:
{json.dumps(summarized_reviews, indent=2)}

Your task:
Sort ALL candidate item_ids from most preferred to least preferred.

====================== RULES ======================

1. Use semantic_categories first  
   - Stronger category match → higher rank  
   - Use the strength weight ("high" > "medium" > "low")
   - This is the strongest indicater. 

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
   - You MUST match extracted attributes against the user preference profile.
   - Items with more aligned attributes must rank higher.

5. Consider price_sensitivity  
   - If the item or its reviews mention “expensive”, “overpriced”, “worth it”,
     interpret them based on the user’s price_sensitivity.

6. Use brand_loyalty  
   - If the user prefers certain brands/authors and the item belongs to them,
     increase rank.

7. No Hallucinations  
   - Use only item_ids from candidate_items.
   - Do NOT create new ids.
   - Output must match the candidate list exactly.

8. Output Format  
   - Return ONLY a pure JSON list of item_ids.
   - No explanation.

================ OUTPUT EXAMPLE ================
["id1","id2","id3"]
"""
        # ========= END prompt =========

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
    Uses:
    - MyPlanner
    - PreferenceBuilder
    - RankingReasoner
    - (optional) Memory module
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)

        # Tools
        self.memory = MemoryDILU(llm=self.llm)
        self.preference_builder = PreferenceBuilder(llm=self.llm)
        self.reasoner = RankingReasoner(llm=self.llm)

        # Planner (needs access to agent tools)
        self.planner = MyPlanner(agent=self, llm=self.llm)

        # Internal storage used during workflow
        self._user_profile = None
        self._user_reviews = None
        self._candidate_items_info = {}
        self._candidate_items_reviews = {}
        self._history_item_info = {}

    # ============================================================
    #  Helpers: wrappers for batch item info & reviews
    # ============================================================

    def _compress_item_info(self, item_dict):
        """
        对单个 item_info 做字段白名单过滤 + description 截断
        避免把巨量冗余字段塞进 prompt.
        """
        if not isinstance(item_dict, dict):
            return item_dict

        keys = [
            "item_id",
            "name",
            "title",
            "stars",
            "avg_rating",
            "average_rating",
            "review_count",
            "rating_number",
            "ratings_count",
            "price",
            "categories",
            "attributes",
            "description",
            "title_without_series",
        ]

        compressed = {k: item_dict.get(k) for k in keys if k in item_dict}

        desc = compressed.get("description")
        if isinstance(desc, str):
            compressed["description"] = desc[:300]  # 最多 300 字符

        return compressed

    def _get_items_info(self, item_ids):
        """
        item_ids: list of candidate item ids
        Returns a dict: {item_id: compressed_item_info_dict}
        """
        results = {}
        for iid in item_ids:
            try:
                item = self.interaction_tool.get_item(item_id=iid)
                results[iid] = self._compress_item_info(item)
            except Exception as e:
                results[iid] = {"item_id": iid, "error": str(e)}
        return results

    def _get_items_reviews(self, item_ids):
        """
        item_ids: list of candidate item ids
        Returns a dict: {item_id: [reviews]}
        """
        results = {}
        for iid in item_ids:
            try:
                reviews = self.interaction_tool.get_reviews(item_id=iid)
                results[iid] = reviews

                # Store into memory
                for r in reviews:
                    self.memory("item_review:" + (r.get("text") or ""))
            except Exception:
                results[iid] = []
        return results

    def _extract_user_item_ids(self, reviews):
        """
        从用户历史 review 里提取所有出现过的 item_id
        """
        ids = set()
        for r in reviews or []:
            if not isinstance(r, dict):
                continue
            if "item_id" in r:
                ids.add(r["item_id"])
        return list(ids)

    # ============================================================
    #  Core: workflow execution
    # ============================================================

    def workflow(self):
        """
        Execute planner-generated workflow to produce ranked item_ids.
        """
        task = self.task
        plan = self.planner(task)

        for step in plan:
            info = step["tool use instruction"]
            tool_name = info["tool_name"]
            tool = info["tool"]
            args = info["args"]

            # -------------------------
            # Dispatch by tool_name
            # -------------------------
            if tool_name == "get_user":
                self._user_profile = tool(**args)

            elif tool_name == "get_user_reviews":
                reviews = tool(**args)
                self._user_reviews = reviews
                for r in reviews:
                    self.memory("user_review:" + (r.get("text") or ""))

            elif tool_name == "get_items_info":
                # dict: {item_id: compressed_item_info}
                self._candidate_items_info = tool(**args)

            elif tool_name == "get_items_reviews":
                self._candidate_items_reviews = tool(**args)

            elif tool_name == "build_preference_profile":
                # 1) 历史 item_ids
                history_item_ids = self._extract_user_item_ids(self._user_reviews)
                # 2) 压缩后的历史 item info
                self._history_item_info = self._get_items_info(history_item_ids)
                # 3) 保留 item_id / stars / text 的 cleaned_reviews
                cleaned_reviews = []
                for r in self._user_reviews or []:
                    if not isinstance(r, dict):
                        continue
                    cleaned_reviews.append({
                        "item_id": r.get("item_id"),
                        "text": (r.get("text") or "").strip(),
                        "stars": r.get("stars")
                    })

                # 4) 调用 preference_builder
                self._preference_profile = tool(
                    user_profile=self._user_profile,
                    user_reviews=cleaned_reviews,
                    item_info=self._history_item_info
                )

            elif tool_name == "generate_ranking":
                # Convert dict → list for candidate_items
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

        # fallback
        return []



