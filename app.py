from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
try:
    from bwa_backend import app
except Exception as e:
    st.error(f"Failed to load backend graph: {e}")
    app = None


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception as e:
        st.error(f"Error during streaming node updates: {e}")

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception as e:
        st.error(f"Error during streaming node values: {e}")

    try:
        out = graph_app.invoke(inputs)
        yield ("final", out)
    except Exception as e:
        st.error(f"Graph execution failed: {e}")
        raise e


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    """
    Returns .md files in current working directory, newest first.
    Filters out files like requirements.txt, README.md, etc.
    """
    cwd = Path(".")
    files = [
        p for p in cwd.glob("*.md") 
        if p.is_file() and p.name.lower() not in ("readme.md", "implementation_plan.md", "walkthrough.md", "task.md")
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    """
    Use first '# ' heading as title if present.
    """
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


# -----------------------------
# Streamlit Configuration and Styling
# -----------------------------
st.set_page_config(
    page_title="Blog Writing Agent - Premium Console", 
    page_icon="✍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject CSS for stunning design aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }
    
    .description-text {
        color: #94a3b8;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    /* Modern sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Styling custom cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.1);
    }
    
    /* Dynamic tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9;
        background-color: rgba(30, 41, 59, 0.5);
    }
    
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        background-color: rgba(99, 102, 241, 0.1) !important;
        border-bottom: 2px solid #818cf8 !important;
    }
    
    /* Custom buttons with gradient */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.8rem !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-top: 0.5rem;
    }
    .status-active {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-inactive {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Design & Configurations
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/gradient/96/create-icon.png", width=64)
    st.markdown('<div class="gradient-text" style="font-size: 1.8rem;">Generator Console</div>', unsafe_allow_html=True)
    
    # Show active credentials
    st.markdown("### API Integration Status")
    hf_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if hf_key:
        st.markdown(f'<div class="status-badge status-active">🤖 Hugging Face API: Connected</div>', unsafe_allow_html=True)
        st.caption(f"LLM Model: `{os.getenv('HUGGINGFACE_MODEL', 'meta-llama/Llama-3.3-70B-Instruct')}`")
        st.caption(f"Image Model: `{os.getenv('HUGGINGFACE_IMAGE_MODEL', 'black-forest-labs/FLUX.1-schnell')}`")
    elif openai_key:
        st.markdown('<div class="status-badge status-info">🤖 OpenAI API: Connected (Paid)</div>', unsafe_allow_html=True)
        st.caption("LLM Model: `gpt-4o-mini`")
    elif google_key:
        st.markdown('<div class="status-badge status-info">🤖 Gemini API: Connected (Free/Paid)</div>', unsafe_allow_html=True)
        st.caption("LLM Model: `gemini-2.5-flash`")
    else:
        st.markdown('<div class="status-badge status-inactive">⚠️ No LLM Key Configured</div>', unsafe_allow_html=True)
        st.caption("Please add HUGGINGFACE_API_KEY in your .env file to run completely for free.")

    if tavily_key:
        st.markdown('<div class="status-badge status-active">🔎 Tavily Search: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-info">🔎 DDG Search: Active (Free fallback)</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("Configuration Options")
    topic = st.text_area(
        "Blog Topic / Prompt",
        placeholder="e.g., Explain the inner workings of Transformer models...",
        height=100
    )
    as_of = st.date_input("As-of Date", value=date.today())
    run_btn = st.button("🚀 Generate Article", type="primary")

    # Past blogs list
    st.divider()
    st.subheader("📚 Saved Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in workspace).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title[:35]}... ({p.name})"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox(
            "Load generated article",
            options=options,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load Selected Blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.success(f"Loaded {selected_md_file.name} successfully!")

# -----------------------------
# Main Panel Layout
# -----------------------------
st.markdown('<div class="gradient-text">Blog Writing Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description-text">'
    'An advanced agentic content creation workspace powered by LangGraph. It researches the web, '
    'outlines structure, drafts sections in parallel, generates dynamic diagrams using text-to-image AI, '
    'and outputs fully optimized Markdown publications.'
    '</div>', 
    unsafe_allow_html=True
)

# Tabs
tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
    ["📝 Article Preview", "🧩 Outlines & Plans", "🔎 Research & Evidence", "🖼️ Visual Assets", "🧾 Execution Logs"]
)

# Log persistence
if "logs" not in st.session_state:
    st.session_state["logs"] = []

def log(msg: str):
    st.session_state["logs"].append(msg)


# Storage for latest run
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# Validation / Key Safety Guard Check before running
if run_btn:
    if not topic.strip():
        st.warning("Please specify a topic or prompt for generation.")
        st.stop()
        
    if not hf_key and not openai_key and not google_key:
        st.error(
            "Configuration Error: No active API keys detected for the LLM. "
            "Please create a `.env` file in the project folder and define `HUGGINGFACE_API_KEY` "
            "to run using Hugging Face's free serverless inference API tier."
        )
        st.stop()

    if not app:
        st.error("Backend graph failed to compile or load. Check log files for details.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    # Set up progress containers
    status = st.status("Executing graph nodes...", expanded=True)
    progress_area = st.empty()

    current_state: Dict[str, Any] = {}
    last_node = None

    try:
        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != last_node:
                    status.write(f"⚙️ Running Node: **`{node_name}`**")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                summary = {
                    "Selected Mode": current_state.get("mode"),
                    "Needs Research": current_state.get("needs_research"),
                    "Search Queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                    "Evidence Count": len(current_state.get("evidence", []) or []),
                    "Tasks Planned": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                    "Drafted Sections": len(current_state.get("sections", []) or []),
                    "Images Proposed": len(current_state.get("image_specs", []) or []),
                }
                progress_area.json(summary)
                log(f"[{kind}] Node finished: {json.dumps(payload, default=str)[:1000]}")

            elif kind == "final":
                st.session_state["last_out"] = payload
                status.update(label="✨ Process Completed Successfully!", state="complete", expanded=False)
                log("[final] State compilation completed.")
    except Exception as e:
        status.update(label="❌ Generation Failed!", state="error", expanded=True)
        st.exception(e)

# Render last result (if any)
out = st.session_state.get("last_out")
if out:
    final_md = out.get("final") or ""

    # --- Preview Tab ---
    with tab_preview:
        if not final_md:
            st.info("No publication has been compiled yet. Start one using the Sidebar.")
        else:
            # Download actions container
            plan_obj = out.get("plan")
            if plan_obj:
                if hasattr(plan_obj, "blog_title"):
                    blog_title = plan_obj.blog_title
                elif isinstance(plan_obj, dict):
                    blog_title = plan_obj.get("blog_title", "blog")
                else:
                    blog_title = extract_title_from_md(final_md, "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                )
            with col_d2:
                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Zip Bundle (MD + Images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                )
            st.divider()

            render_markdown_with_local_images(final_md)

    # --- Plan Tab ---
    with tab_plan:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No planning metadata is available for this run.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"### Proposed Outline: *{plan_dict.get('blog_title')}*")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown(f'<div class="metric-card"><strong>🎯 Target Audience</strong><br/>{plan_dict.get("audience")}</div>', unsafe_allow_html=True)
            with col_p2:
                st.markdown(f'<div class="metric-card"><strong>🎭 Tone Style</strong><br/>{plan_dict.get("tone")}</div>', unsafe_allow_html=True)
            with col_p3:
                st.markdown(f'<div class="metric-card"><strong>📁 Content Category</strong><br/>{plan_dict.get("blog_kind", "explainer")}</div>', unsafe_allow_html=True)

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "Section ID": t.get("id"),
                            "Section Title": t.get("title"),
                            "Word Target": t.get("target_words"),
                            "Needs Search": t.get("requires_research"),
                            "Add Citations": t.get("requires_citations"),
                            "Write Code": t.get("requires_code"),
                            "Keywords / Tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("Section ID")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Expand Technical Node Specifications"):
                    st.json(tasks)

    # --- Evidence Tab ---
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No research evidence is listed (evergreen closed_book run, or no keywords generated).")
        else:
            st.markdown("### Web Research Sources & Citations")
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "Title": e.get("title"),
                        "Publish Date": e.get("published_at") or "Unknown",
                        "Provider Source": e.get("source") or "Search",
                        "Source URL": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Images Tab ---
    with tab_images:
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No visual image specifications have been defined for this run.")
        else:
            if specs:
                st.markdown("### Image Layout Blueprint")
                st.json(specs)

            if images_dir.exists():
                st.markdown("### Rendered Visual Media Assets")
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("No generated assets found in `images/` directory.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=f"Asset name: {p.name}", use_container_width=True)

                z = images_zip(images_dir)
                if z:
                    st.download_button(
                        "⬇️ Download All Visual Assets (Zip)",
                        data=z,
                        file_name="generated_assets.zip",
                        mime="application/zip",
                    )

    # --- Logs Tab ---
    with tab_logs:
        st.markdown("### Execution Console log")
        if st.session_state["logs"]:
            st.text_area(
                "Runtime execution steps (Newest at top)", 
                value="\n\n".join(st.session_state["logs"][::-1]), 
                height=500
            )
        else:
            st.info("No active logs recorded in the session.")
else:
    with tab_preview:
        st.info("Enter your parameters on the sidebar console and click 'Generate Article' to begin.")
