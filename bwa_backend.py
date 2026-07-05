from __future__ import annotations

import operator
import os
import re
import json
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import time
import requests
from typing import Any

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM and Free API Wrappers
# -----------------------------
class HuggingFaceChatModel:
    def __init__(self, model_id: str = "meta-llama/Llama-3.3-70B-Instruct", api_key: str = None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "No Hugging Face API key found. Please set HUGGINGFACE_API_KEY or HF_TOKEN "
                "in your environment or .env file to run with free APIs."
            )
            
    def _convert_messages(self, messages: list) -> list:
        formatted = []
        for m in messages:
            if isinstance(m, SystemMessage):
                formatted.append({"role": "system", "content": m.content})
            elif isinstance(m, HumanMessage):
                formatted.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                formatted.append({"role": "assistant", "content": m.content})
            elif isinstance(m, dict):
                formatted.append(m)
            else:
                formatted.append({"role": "user", "content": str(m)})
        return formatted

    def invoke(self, messages: list, temperature: float = 0.7, max_tokens: int = 2048) -> AIMessage:
        formatted_messages = self._convert_messages(messages)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        url = f"https://api-inference.huggingface.co/models/{self.model_id}/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        max_retries = 5
        retry_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 503:
                    est_time = response.json().get("estimated_time", 15.0)
                    sleep_time = min(max(est_time, 5.0), 30.0)
                    print(f"Hugging Face model '{self.model_id}' is loading. Retrying in {sleep_time} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                if response.status_code == 429:
                    print(f"Hugging Face API rate limit reached. Retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2.0
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return AIMessage(content=content)
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Hugging Face API request failed after {max_retries} attempts: {e}")
                time.sleep(2.0)
                
        raise RuntimeError(f"Failed to get response from Hugging Face model {self.model_id}.")

    def with_structured_output(self, schema):
        return HuggingFaceStructuredOutputWrapper(self, schema)


class HuggingFaceStructuredOutputWrapper:
    def __init__(self, model: HuggingFaceChatModel, schema):
        self.model = model
        self.schema = schema

    def invoke(self, messages: list) -> Any:
        try:
            if hasattr(self.schema, "model_json_schema"):
                schema_json = json.dumps(self.schema.model_json_schema(), indent=2)
            else:
                schema_json = json.dumps(self.schema.schema(), indent=2)
        except Exception:
            schema_json = str(self.schema)

        json_instruction = (
            f"\n\nCRITICAL REQUIREMENT:\n"
            f"You MUST respond ONLY with a raw JSON object matching the JSON schema below.\n"
            f"Do not write any preamble, explanation, introduction, or notes. Do not include any HTML.\n"
            f"You MUST wrap your JSON output in a Markdown code block like this:\n"
            f"```json\n"
            f"{{\n"
            f"  ...\n"
            f"}}\n"
            f"```\n"
            f"Ensure your output is perfectly formatted JSON. Here is the JSON Schema:\n{schema_json}"
        )

        updated_messages = []
        has_system = False
        for m in messages:
            if isinstance(m, SystemMessage):
                updated_messages.append(SystemMessage(content=m.content + json_instruction))
                has_system = True
            elif isinstance(m, HumanMessage):
                updated_messages.append(HumanMessage(content=m.content))
            elif isinstance(m, AIMessage):
                updated_messages.append(AIMessage(content=m.content))
            else:
                updated_messages.append(m)

        if not has_system:
            updated_messages.insert(0, SystemMessage(content=json_instruction))

        response = self.model.invoke(updated_messages, temperature=0.1, max_tokens=4096)
        content = response.content.strip()
        parsed_json = self._extract_and_parse_json(content)

        try:
            if hasattr(self.schema, "model_validate"):
                return self.schema.model_validate(parsed_json)
            else:
                return self.schema.parse_obj(parsed_json)
        except Exception as e:
            raise ValueError(
                f"JSON validation failed for schema {self.schema.__name__}: {e}.\n"
                f"Model response was:\n{content}\n"
                f"Parsed JSON:\n{json.dumps(parsed_json, indent=2)}"
            )

    def _extract_and_parse_json(self, text: str) -> dict:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start:end+1].strip()
            else:
                json_str = text.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            cleaned = json_str
            cleaned = re.sub(r"\bTrue\b", "true", cleaned)
            cleaned = re.sub(r"\bFalse\b", "false", cleaned)
            cleaned = re.sub(r"\bNone\b", "null", cleaned)
            cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse text as JSON. Raw model text:\n{text}\n"
                    f"Attempted to parse clean text:\n{json_str}\nError: {e}"
                )


class OpenAICompatibleChatModel:
    def __init__(self, model_id: str, api_key: str, base_url: str):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model=model_id, api_key=api_key, base_url=base_url)
        
    def invoke(self, messages: list, **kwargs) -> Any:
        return self.llm.invoke(messages, **kwargs)
        
    def with_structured_output(self, schema):
        return HuggingFaceStructuredOutputWrapper(self, schema)


def _get_llm():
    # 1. Check Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        model_id = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return OpenAICompatibleChatModel(
            model_id=model_id,
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1"
        )
        
    # 2. Check Grok (xAI)
    grok_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if grok_key:
        model_id = os.getenv("GROK_MODEL", "grok-2-1212")
        return OpenAICompatibleChatModel(
            model_id=model_id,
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )

    # 3. Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
        
    # 4. Check Google/Gemini
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        from langchain_google_genai import ChatGoogleGenAI
        os.environ["GOOGLE_API_KEY"] = google_key
        return ChatGoogleGenAI(model="gemini-2.5-flash")
        
    # 5. Check Hugging Face (Free default)
    hf_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    if hf_key:
        model_id = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        return HuggingFaceChatModel(model_id=model_id, api_key=hf_key)
        
    raise ValueError(
        "No AI API keys configured. You must set GROQ_API_KEY, HUGGINGFACE_API_KEY, "
        "OPENAI_API_KEY, or GOOGLE_API_KEY in your .env file."
    )

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    llm = _get_llm()
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    # Check for Tavily API Key
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
            tool = TavilySearchResults(max_results=max_results)
            results = tool.invoke({"query": query})
            out: List[dict] = []
            for r in results or []:
                out.append(
                    {
                        "title": r.get("title") or "",
                        "url": r.get("url") or "",
                        "snippet": r.get("content") or r.get("snippet") or "",
                        "published_at": r.get("published_date") or r.get("published_at"),
                        "source": r.get("source") or "tavily",
                    }
                )
            return out
        except Exception as e:
            print(f"Tavily search failed, falling back to DuckDuckGo: {e}")
            
    # DuckDuckGo Search Fallback
    try:
        from duckduckgo_search import DDGS
        out: List[dict] = []
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            for r in results:
                out.append(
                    {
                        "title": r.get("title") or "",
                        "url": r.get("href") or "",
                        "snippet": r.get("body") or "",
                        "published_at": None,
                        "source": "duckduckgo",
                    }
                )
            return out
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    llm = _get_llm()
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    llm = _get_llm()
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    llm = _get_llm()
    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    llm = _get_llm()
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by a Hugging Face text-to-image model.
    Falls back to Google Gemini if HUGGINGFACE_API_KEY is not set.
    """
    hf_key = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if hf_key:
        model_id = os.environ.get("HUGGINGFACE_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
        headers = {
            "Authorization": f"Bearer {hf_key}",
            "Content-Type": "application/json"
        }
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        max_retries = 3
        retry_delay = 5.0
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=120)
                if response.status_code == 503:
                    est_time = response.json().get("estimated_time", 15.0)
                    sleep_time = min(max(est_time, 5.0), 30.0)
                    print(f"Hugging Face image model is loading. Retrying in {sleep_time} seconds (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                    continue
                if response.status_code == 429:
                    print(f"Hugging Face API rate limit reached. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                response.raise_for_status()
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Hugging Face image generation failed after {max_retries} attempts: {e}")
                time.sleep(retry_delay)
                
    # Fallback to Google Gemini
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        resp = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    )
                ],
            ),
        )

        parts = getattr(resp, "parts", None)
        if not parts and getattr(resp, "candidates", None):
            try:
                parts = resp.candidates[0].content.parts
            except Exception:
                parts = None

        if not parts:
            raise RuntimeError("No image content returned from Gemini (safety/quota/SDK change).")

        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return inline.data

    raise RuntimeError("No valid API keys (HUGGINGFACE_API_KEY or GOOGLE_API_KEY) found for image generation.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app

