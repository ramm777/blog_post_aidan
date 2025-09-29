import streamlit as st
import json
from config import build_role_llm_config, build_image_request_url, _env
import autogen
# Removed unused direct pydantic imports (moved to utils)
import io
from contextlib import redirect_stdout
import threading, time
import base64
import re
try:
    from openai import AzureOpenAI  # openai>=1.x
except ImportError:
    AzureOpenAI = None
from utils import ReviewModel, build_summary_args, validate_review, _to_plain_text, DEFAULT_REVIEWERS
from agents import make_agents

# Supported Azure image sizes (previous 512/768 removed due to 400 errors)
SUPPORTED_IMAGE_SIZES = ["1024x1024", "1024x1536", "1536x1024", "auto"]

st.set_page_config(page_title="Multi-Agent Blogpost Generator", layout="wide")

st.title("Multi-Agent Blogpost Generator (AG2 + Azure OpenAI)")

with st.sidebar:
    st.header("Configuration")
    st.text_input("Azure Endpoint", value=_env.endpoint, key="base_url") 
    st.text_input("API Key", type="password", value=_env.api_key, key="api_key")
    st.text_input("API Version", value=_env.api_version, key="api_version", disabled=True)
    st.number_input("Timeout", value=60, key="timeout")
    st.number_input("Seed", value=42, key="seed")
    topic = st.text_area("Topic / Task", value="""Write a comprehensive and engaging blog post about using 'AutoGen – AG2' to create a multi-agent system for automated blog post generation. 

The blog post should:
- Be between 400-500 words
- Include a catchy title and a 2-sentence TL;DR summary
- Explain what AutoGen/AG2 is and its key capabilities for multi-agent orchestration
- Describe the architecture of a multi-agent blog writing system with these specific agents:
  * Writer Agent: Creates the initial blog content
  * Critic Agent: Orchestrates the review process and provides feedback
  * SEO_Reviewer: Optimizes for search engines and web discoverability
  * LegalReviewer: Ensures compliance and proper disclaimers
  * EthicsReviewer: Checks for bias and ethical considerations
  * ContentReviewer: Evaluates content quality and coherence
  * StyleReviewer: Ensures consistent tone and style
  * ImagePromptAgent: Generates relevant image prompts for illustrations
  * ImageCritic: Refines and improves image generation prompts
- Highlight 3-4 key benefits of using multi-agent systems for content creation
- Include a simple code example showing agent initialization or conversation flow
- Discuss real-world applications and potential use cases
- End with future possibilities and a call-to-action

Target audience: Technical professionals and developers interested in AI automation
Tone: Professional yet accessible, avoiding jargon where possible""")
    reviewer_max_turns = st.number_input("Reviewer Max Turns", min_value=1, max_value=10, value=2, step=1)
    main_max_turns = st.number_input("Main Chat Max Turns", min_value=1, max_value=20, value=5, step=1)
    capture_log = st.checkbox("Show Conversation Log", value=True, help="Capture and display the full agent conversation output below instead of only in terminal.")
    streaming_mode = st.selectbox("Streaming Mode", ["Off", "Per-Message"], help="Off: wait for full run. Per-Message: update UI as each agent message appears (not token-level).")
    auto_image = st.checkbox("Auto-generate illustrative image", value=False, help="Generate an image after the blog post using the Azure OpenAI image model.")
    image_size = st.selectbox("Image Size", SUPPORTED_IMAGE_SIZES, index=0, disabled=not auto_image, help="Azure supported sizes only.")
    st.markdown("**Enable Reviewers**")
    enabled_reviewers = {name: st.checkbox(name, value=True) for name in DEFAULT_REVIEWERS}
    debug_parsing = st.checkbox("Show Reviewer Debug", value=False, help="Show raw reviewer summaries and parsing errors for troubleshooting JSON extraction.")
    generate_btn = st.button("Generate Blogpost")

# --- Image generation helpers (unified) ------------------------------------
# Fix: previous implementation passed unsupported params (response_format, quality, output_compression)
# for certain Azure OpenAI image api versions causing 400 errors. We now use minimal supported args.
if AzureOpenAI:
    @st.cache_resource(show_spinner=False)
    def _get_azure_image_client():
        if not (_env.api_key and _env.endpoint and _env.api_version):
            return None
        return AzureOpenAI(api_key=_env.api_key, api_version=_env.api_version, azure_endpoint=_env.endpoint)
else:
    def _get_azure_image_client():  # type: ignore
        return None

def generate_azure_image(prompt: str, size: str):
    """Generate an image and return (bytes, error). Minimal parameter set for Azure compatibility.
    Falls back to first supported size if an invalid one is provided."""
    if size not in SUPPORTED_IMAGE_SIZES:
        size = SUPPORTED_IMAGE_SIZES[0]
    client = _get_azure_image_client()
    if client is None:
        return None, "Azure OpenAI client not initialized (missing dependency or credentials)."
    try:
        # Minimal parameter call; Azure returns base64 data in data[0].b64_json for image models.
        resp = client.images.generate(
            model=_env.model_t2i,
            prompt=prompt,
            size=size,
            n=1,
        )
        # Support both attribute and dict style access just in case.
        first = resp.data[0]
        b64_data = None
        if hasattr(first, 'b64_json') and first.b64_json:
            b64_data = first.b64_json
        elif isinstance(first, dict):
            b64_data = first.get('b64_json') or first.get('b64')
        if not b64_data and hasattr(first, 'url') and first.url:
            # Fallback: fetch URL (not implemented to keep offline friendly)
            return None, "Image URL returned instead of base64; direct download not implemented."
        if not b64_data:
            return None, "No base64 field found in image response."
        return base64.b64decode(b64_data), None
    except Exception as e:
        return None, str(e)

if generate_btn and topic:
    writer, critic = make_agents(int(reviewer_max_turns), enabled_reviewers=enabled_reviewers)
    task = topic
    # --- Streaming logic branch -------------------------------------------------
    if streaming_mode == "Per-Message":
        st.info("Per-Message streaming enabled (message-level, not token-level). Final structured JSON is parsed after completion.")
        buf = io.StringIO() if capture_log else None
        result_holder = {}
        stop_flag = threading.Event()
        # Placeholders
        msg_placeholder = st.empty()
        status_placeholder = st.empty()
        def run_chat():
            # Capture stdout if requested
            if capture_log:
                with redirect_stdout(buf):
                    result_holder['result'] = critic.initiate_chat(
                        recipient=writer,
                        message=task,
                        max_turns=int(main_max_turns),
                        summary_method="last_msg"
                    )
            else:
                result_holder['result'] = critic.initiate_chat(
                    recipient=writer,
                    message=task,
                    max_turns=int(main_max_turns),
                    summary_method="last_msg"
                )
            stop_flag.set()
        t = threading.Thread(target=run_chat, daemon=True)
        t.start()
        last_render_count = 0
        # Poll for new messages
        while not stop_flag.is_set():
            assembled_lines = []
            try:
                # critic.chat_messages: dict keyed by agent objects -> list[dict]
                for agent_obj, msgs in critic.chat_messages.items():
                    agent_name = getattr(agent_obj, 'name', 'Agent')
                    for m in msgs:
                        content = m.get('content') or ''
                        if content:
                            assembled_lines.append(f"**{agent_name}**: {content}")
            except Exception as e:  # Non-fatal; continue polling
                assembled_lines.append(f"_Polling error: {e}_")
            if len(assembled_lines) != last_render_count:
                msg_placeholder.markdown("\n\n".join(assembled_lines))
                last_render_count = len(assembled_lines)
            status_placeholder.info("Running... (Per-Message updates)")
            time.sleep(0.6)
        t.join()
        status_placeholder.success("Run complete.")
        result = result_holder.get('result')
        st.subheader("Final Blog Post")
        final_text = _to_plain_text(result.get("summary") if isinstance(result, dict) and result.get("summary") else result)
        st.write(final_text)
        if capture_log and buf is not None:
            st.subheader("Conversation Log")
            st.expander("Raw Conversation Output").code(buf.getvalue())
    else:
        # Original non-streaming path
        buf = io.StringIO() if capture_log else None
        with (redirect_stdout(buf) if capture_log else st.spinner("Running multi-agent collaboration...")):
            if not capture_log:
                result = critic.initiate_chat(
                    recipient=writer,
                    message=task,
                    max_turns=int(main_max_turns),
                    summary_method="last_msg"
                )
            else:
                with st.spinner("Running multi-agent collaboration..."):
                    result = critic.initiate_chat(
                        recipient=writer,
                        message=task,
                        max_turns=int(main_max_turns),
                        summary_method="last_msg"
                    )
        st.subheader("Final Blog Post")
        final_text = _to_plain_text(result["summary"] if isinstance(result, dict) and "summary" in result else result)
        st.write(final_text)
        if capture_log and buf is not None:
            st.subheader("Conversation Log")
            st.expander("Raw Conversation Output").code(buf.getvalue())
    # Persist state for post-processing across reruns
    st.session_state['critic'] = critic
    st.session_state['final_text'] = final_text
    st.session_state['enabled_reviewers_snapshot'] = enabled_reviewers

# --- Post-processing shown if we have a stored run --------------------------------
if 'critic' in st.session_state and 'final_text' in st.session_state:
    critic = st.session_state['critic']
    final_text = st.session_state['final_text']
    enabled_reviewers_snapshot = st.session_state.get('enabled_reviewers_snapshot', enabled_reviewers)

    # Display persisted final blog post on subsequent reruns (e.g. when generating images)
    if not generate_btn:
        st.subheader("Final Blog Post")
        st.write(final_text)

    # --- Common post-processing (structured feedback & images) ------------------
    st.subheader("Reviewer Structured Feedback")

    def collect_structured_reviews(critic_agent, enabled_map):
        collected_map = {}
        parse_errors = {}
        def _json_candidate(s: str) -> bool:
            s_strip = s.lstrip()
            if not s_strip.startswith('{'):
                return False
            return '"Reviewer"' in s_strip and '"Review"' in s_strip
        for agent_obj, msgs in critic_agent.chat_messages.items():
            name = getattr(agent_obj, 'name', '')
            if not enabled_map.get(name):
                continue
            for m in msgs:
                content = m.get('content') or ''
                if not content:
                    continue
                if not _json_candidate(content):
                    continue
                model_obj, err = validate_review(content)
                if model_obj:
                    collected_map[model_obj.Reviewer] = model_obj
                elif err:
                    parse_errors.setdefault(name, []).append({"fragment": content[:120], "err": err})
            summary_attr = getattr(agent_obj, 'summary', None)
            if summary_attr and isinstance(summary_attr, str) and _json_candidate(summary_attr):
                model_obj, err = validate_review(summary_attr)
                if model_obj:
                    collected_map[model_obj.Reviewer] = model_obj
                elif err:
                    parse_errors.setdefault(name + "#summary", []).append({"fragment": summary_attr[:120], "err": err})
        return list(collected_map.values()), parse_errors

    collected, parse_errors = collect_structured_reviews(critic, enabled_reviewers_snapshot)
    if collected:
        for r in collected:
            st.json(r.model_dump())
    else:
        st.info("No valid structured reviews parsed yet.")
    if debug_parsing and parse_errors:
        with st.expander("Reviewer Parsing Debug"):
            st.write(parse_errors)
    st.divider()
    st.subheader("Image Generation Curl Examples")
    gen_url = build_image_request_url("generations")
    edit_url = build_image_request_url("edits")
    st.code(f"""# Generation (minimal supported JSON body)
curl -X POST \"{gen_url}\" \\
  -H \"Content-Type: application/json\" \\
  -H \"Authorization: Bearer $AZURE_OPENAI_API_KEY\" \\
  -d '{{"prompt": "A photograph of a red fox in an autumn forest", "size": "1024x1024"}}' \
  | jq -r '.data[0].b64_json' | base64 --decode > generated_image.png

# Edit (supply original image + optional mask)
curl -X POST \"{edit_url}\" \\
  -H \"Authorization: Bearer $AZURE_OPENAI_API_KEY\" \\
  -F "image=@image_to_edit.png" \\
  -F "mask=@mask.png" \\
  -F "prompt=Make this black and white" \
  | jq -r '.data[0].b64_json' | base64 --decode > edited_image.png""", language="bash")

    # Inline Image Generation UI (unchanged)
    st.subheader("Try Image Generation Inline")
    with st.expander("Generate an Image (Azure OpenAI)"):
        img_prompt = st.text_input("Image Prompt", value="A photograph of a red fox in an autumn forest")
        size = st.selectbox("Size", SUPPORTED_IMAGE_SIZES, index=0)
        gen_btn = st.button("Generate Image Now")
        if gen_btn:
            if not AzureOpenAI:
                st.error("openai package not available for direct generation.")
            elif not (_env.api_key and _env.endpoint and _env.api_version):
                st.error("Missing Azure OpenAI configuration (endpoint/key/version).")
            else:
                with st.spinner("Generating image..."):
                    img_bytes, err = generate_azure_image(img_prompt, size)
                if err:
                    st.error(f"Image generation failed: {err}")
                else:
                    st.image(img_bytes, caption="Generated Image", width=800)
                    st.download_button("Download PNG", data=img_bytes, file_name="generated_image.png", mime="image/png")

    # Key Fixes list (unchanged)
    st.subheader("Key Fixes Before Publishing")
    st.markdown("""
- URL slug is blank – fill (e.g., `/autogen-ai-blog-writing`).
- Meta description is truncated; keep <= 155 chars, single primary keyword.
- Add TL;DR (1–2 sentences) immediately after the title.
- Verify heading hierarchy (promote major bullet groups to H2/H3).
- Remove anthropomorphic / ableist phrasing (e.g., “never sleeps”, “tiny, tireless”).
- Add a “Responsible Use & Disclaimer” (AI‑assisted, fact‑check, no sensitive data, trademarks, license).
- Insert internal links (AutoGen docs, Azure OpenAI docs) plus 1–2 authoritative external references.
- Provide alt text descriptions for any code blocks or diagrams.
- Ensure neutral, pronoun‑free agent role descriptions; replace “tough critic” with “rigorous reviewer.”
- Confirm model names are publicly available (e.g., `gpt-4o`) and remove experimental IDs.
- Fact‑check any statistics or claims; cite if used.
- Add accessibility note (WCAG contrast, screen‑reader friendly).
- Ensure JSON cost/debug metadata is not shown to end users.

After applying these, the post is suitable for human readers.
""")

    # Auto image generation (re-usable with persisted final_text)
    if 'final_text' in st.session_state and auto_image:
        st.subheader("Auto Generated Image")
        if not AzureOpenAI:
            st.info("openai package not installed; cannot generate image inline.")
        elif not (_env.api_key and _env.endpoint and _env.api_version):
            st.warning("Missing Azure OpenAI configuration; image not generated.")
        else:
            def extract_title(text: str) -> str:
                text = _to_plain_text(text)
                if not text:
                    return "AutoGen multi-agent blog"
                m = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
                if m:
                    return m.group(1).strip()[:120]
                for line in text.splitlines():
                    if line.strip().startswith('#'):
                        return line.lstrip('#').strip()[:120]
                for line in text.splitlines():
                    if line.strip():
                        return " ".join(line.strip().split()[:12])
                return "AutoGen multi-agent blog"
            title_fragment = extract_title(final_text)
            img_prompt = (
                "Illustrative, high-quality, modern diagram or hero image representing: "
                f"{title_fragment}. Minimal, professional, accessible, high contrast."
            )
            try:
                with st.spinner("Generating illustrative image..."):
                    img_bytes, err = generate_azure_image(img_prompt, image_size)
                if err:
                    raise RuntimeError(err)
                st.image(img_bytes, caption=f"Illustration: {title_fragment}", width=800)
                st.download_button("Download Image", data=img_bytes, file_name="blog_illustration.png", mime="image/png")
                st.caption(f"Prompt used: {img_prompt}")
            except Exception as e:
                st.error(f"Auto image generation failed: {e}")

    st.divider()
    st.subheader("AI-Derived Image Prompts")
    derived_prompts = []
    for agent_obj, msgs in critic.chat_messages.items():
        agent_name = getattr(agent_obj, 'name', '')
        if agent_name.startswith('ImagePromptAgent'):
            for m in msgs:
                content = m.get('content') or ''
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and 'prompts' in data and isinstance(data['prompts'], list):
                        derived_prompts.extend([p for p in data['prompts'] if isinstance(p, str)])
                except Exception:
                    pass
        if agent_name.startswith('ImageCritic'):
            for m in msgs:
                content = m.get('content') or ''
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and 'improved_prompts' in data:
                        improved = [p for p in data['improved_prompts'] if isinstance(p, str)]
                        if improved:
                            derived_prompts = improved
                except Exception:
                    pass
    if derived_prompts:
        for i, p in enumerate(derived_prompts, 1):
            st.markdown(f"**Prompt {i}:** {p}")
    else:
        st.info("No derived image prompts yet (may require more turns or content).")

    # Multi-image generation (unchanged logic but now outside generation rerun)
    if derived_prompts:
        with st.expander("Generate Images From Derived Prompts"):
            if 'derived_prompts_cache' not in st.session_state:
                st.session_state.derived_prompts_cache = derived_prompts
            else:
                if len(derived_prompts) != len(st.session_state.derived_prompts_cache):
                    st.session_state.derived_prompts_cache = derived_prompts
            cached_prompts = st.session_state.derived_prompts_cache
            selectable = {f"Prompt {i+1}": p for i, p in enumerate(cached_prompts)}
            default_keys = list(selectable.keys())[:1]
            chosen_keys = st.multiselect(
                "Prompts", list(selectable.keys()), default=default_keys, help="Choose one or more prompts to generate images for."
            )
            if not chosen_keys:
                chosen_prompts = [cached_prompts[0]]
            else:
                chosen_prompts = [selectable[k] for k in chosen_keys]
            safe_size = image_size if (auto_image and image_size != 'auto') else SUPPORTED_IMAGE_SIZES[0]
            st.caption(f"Using size: {safe_size} (auto mapped to {SUPPORTED_IMAGE_SIZES[0]} if selected).")
            gen_multi = st.button("Generate Selected Images", key="gen_derived_images")
            if gen_multi:
                for idx, raw_prompt in enumerate(chosen_prompts, 1):
                    cleaned_prompt = re.sub(r'\s+', ' ', raw_prompt).strip()
                    if not cleaned_prompt:
                        st.error(f"Prompt {idx} is empty after cleaning; skipped.")
                        continue
                    if len(cleaned_prompt) > 500:
                        cleaned_prompt = cleaned_prompt[:500]
                    with st.spinner(f"Generating image {idx}/{len(chosen_prompts)}..."):
                        try:
                            img_bytes, err = generate_azure_image(cleaned_prompt, safe_size)
                        except Exception as e:
                            img_bytes, err = None, str(e)
                    if err:
                        st.error(f"Prompt {idx} failed: {err}")
                        with st.expander(f"Debug Prompt {idx}"):
                            st.code(cleaned_prompt)
                    else:
                        st.image(img_bytes, caption=f"Derived Prompt {idx}", width=800)
                        st.caption(cleaned_prompt)
                        st.download_button(
                            f"Download Image {idx}", data=img_bytes, file_name=f"derived_prompt_{idx}.png", mime="image/png", key=f"dl_img_{idx}"
                        )
elif not generate_btn:
    st.info("Enter a topic and click Generate Blogpost")

# --- Publication Editor Section --------------------------------------------
if 'final_text' in st.session_state:
    with st.expander("Publication Editor (Refine & Polish)", expanded=False):
        st.markdown("Provide style preferences, then auto-refine or manually edit.")
        if 'editor_history' not in st.session_state:
            st.session_state.editor_history = []
        style_guidelines = st.text_area(
            "Style / Publication Guidelines",
            value=(
                "Audience: technical but non-expert; Tone: clear, concise, active voice; "
                "Keep paragraphs short; Ensure factual neutrality; Prefer present tense; "
                "Add TL;DR after title if missing; Ensure headings hierarchy (H2/H3)."
            ),
            height=120,
        )
        editable_text = st.text_area(
            "Editable Draft",
            value=st.session_state.get('edited_final_text', st.session_state['final_text']),
            height=400,
            key="editor_draft_text"
        )
        col_e1, col_e2, col_e3 = st.columns([1,1,1])
        with col_e1:
            apply_btn = st.button("Auto-Refine")
        with col_e2:
            accept_btn = st.button("Accept & Replace Main Final Text")
        with col_e3:
            revert_btn = st.button("Revert to Original")

        if apply_btn:
            # Initialize editor agent lazily
            if 'editor_agent' not in st.session_state:
                st.session_state.editor_agent = autogen.AssistantAgent(
                    name="Editor",
                    system_message=(
                        "You are a publication editor. You take a blog draft and publication guidelines and return a polished, publication-ready version. "
                        "Rules: Preserve technical accuracy; fix grammar, structure, heading hierarchy; insert a TL;DR (2 sentences) below the title if absent; "
                        "Ensure neutral inclusive language; remove filler; maintain markdown where useful; do NOT fabricate facts. Return ONLY the refined article markdown."
                    ),
                    llm_config=build_role_llm_config("editor"),
                )
            editor_agent = st.session_state.editor_agent
            try:
                result = editor_agent.generate_reply(
                    messages=[{"role": "user", "content": f"GUIDELINES:\n{style_guidelines}\n\nDRAFT TO EDIT:\n{editable_text}"}]
                )
                refined = result if isinstance(result, str) else result.get('content', '')
                if refined:
                    st.session_state.editor_history.append({
                        'input': editable_text,
                        'output': refined,
                        'guidelines': style_guidelines,
                    })
                    st.session_state.edited_final_text = refined
                    st.success("Refinement applied.")
                else:
                    st.warning("Editor returned empty output.")
            except Exception as e:
                st.error(f"Editor error: {e}")
        if accept_btn and st.session_state.get('edited_final_text'):
            st.session_state.final_text = st.session_state.edited_final_text
            st.success("Main final text updated.")
        if revert_btn:
            st.session_state.edited_final_text = st.session_state['final_text']
            st.info("Reverted editor draft to original final text.")
        if st.session_state.editor_history:
            with st.expander("Editor History", expanded=False):
                for i, h in enumerate(reversed(st.session_state.editor_history), 1):
                    st.markdown(f"**Revision {-i if False else i}**")
                    st.caption("Guidelines snapshot:")
                    st.code(h['guidelines'][:800])
                    st.caption("Refined Output:")
                    st.code(h['output'][:5000])
