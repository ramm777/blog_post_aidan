import autogen
from typing import Dict, Any
from utils import ReviewModel, build_summary_args
from config import build_role_llm_config

__all__ = ["make_agents"]

def make_agents(
    reviewer_max_turns: int,
    enable_image_agents: bool = True,
    num_image_critics: int = 1,
    enabled_reviewers: Dict | None = None,
):
    from utils import DEFAULT_REVIEWERS  # local import to avoid circulars
    enabled_reviewers = enabled_reviewers or {name: True for name in DEFAULT_REVIEWERS}

    writer = autogen.AssistantAgent(
        name="Writer",
        system_message=(
            "You are a writer. You write engaging and concise blogpost (with title) on given topics. "
            "You must polish your writing based on the feedback you receive and give a refined version. "
            "Only return your final work without additional comments. "
            "For images, you may instruct the user to run the provided curl commands with appropriate prompts."
        ),
        llm_config=build_role_llm_config("writer"),
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        system_message=(
            "You are a rigorous editorial reviewer (NOT a re-writer). Your job is to critically evaluate the Writer's draft and surface precise, high-impact improvements.\n"
            "Produce feedback that is: (1) Specific (quote or paraphrase exact weak spots); (2) Actionable (give clear fix instructions, not vague praise); (3) Evidence / rationale based (why it matters: clarity, accuracy, structure, audience fit, SEO, ethics, accessibility); (4) Scoped (do NOT rewrite entire sections—point to what to fix).\n"
            "ALWAYS flag: unsupported claims, vague abstractions, redundant sentences, structural gaps, missing transitions, inconsistent tone, bias/exclusionary phrasing, inaccessible wording (jargon without definition), missing TL;DR, absent sourcing where a fact/number appears, heading hierarchy issues.\n"
            "If a passage is acceptable, do NOT restate it—omit instead of filler praise. Avoid generic phrases (e.g., 'improve clarity', 'good flow')—be concrete. Do NOT invent facts or add new content. You ONLY return critique, not a rewritten draft."
        ),
        llm_config=build_role_llm_config("critic"),
    )

    def reviewer_msg(role_desc, role_name):
        return (
            f"You are {role_desc}. Respond ONLY with a single JSON object: {{\"Reviewer\": \"{role_name}\", \"Review\": \"- point 1; - point 2; - point 3\"}}. "
            "Rules: (1) Reviewer MUST equal your agent name exactly; (2) Review value is ONE string containing up to 3 semicolon-separated concise actionable bullet points; (3) No markdown fences, no lists/arrays, no extra keys, no commentary before or after JSON. Output nothing except that JSON object."
        )

    def _maybe(name, role_key, desc):
        if enabled_reviewers.get(name):
            return autogen.AssistantAgent(
                name=name,
                llm_config=build_role_llm_config(role_key),
                system_message=reviewer_msg(desc, name)
            )
        return None

    seo_reviewer = _maybe("SEOReviewer", "seo", "an SEO reviewer optimizing content for search engines")
    legal_reviewer = _maybe("LegalReviewer", "legal", "a legal reviewer ensuring legal compliance")
    ethics_reviewer = _maybe("EthicsReviewer", "ethics", "an ethics reviewer ensuring ethical soundness")
    fact_checker = _maybe("FactChecker", "critic", "a fact checker verifying factual accuracy and pointing out unsupported claims")
    bias_reviewer = _maybe("BiasReviewer", "critic", "a bias reviewer detecting biased or exclusionary language and suggesting neutral phrasing")
    accessibility_reviewer = _maybe("AccessibilityReviewer", "critic", "an accessibility reviewer improving clarity, inclusiveness, and suggesting alt-text opportunities")

    # Structured output requirement
    for _agent in [seo_reviewer, legal_reviewer, ethics_reviewer, fact_checker, bias_reviewer, accessibility_reviewer]:
        if _agent is not None and isinstance(_agent.llm_config, dict):
            _agent.llm_config["response_format"] = ReviewModel

    meta_reviewer = autogen.AssistantAgent(
        name="MetaReviewer",
        llm_config=build_role_llm_config("meta"),
        system_message="You are a meta reviewer aggregating other reviewers' feedback and giving final suggestions."
    )

    def reflection_message(recipient, messages, sender, config):
        last_content = ""
        if messages:
            for m in reversed(messages):
                if m.get("name") == "Writer" and m.get("content"):
                    last_content = m["content"]
                    break
            if not last_content:
                for m in reversed(messages):
                    c = m.get("content")
                    if c:
                        last_content = c
                        break
        agent_name = getattr(recipient, 'name', 'Reviewer')
        example_json = f"{{\"Reviewer\": \"{agent_name}\", \"Review\": \"- improve X; - clarify Y; - fix Z\"}}"
        return (
            "Return ONLY a valid JSON object with keys Reviewer and Review. Reviewer must be your exact agent name. "
            "The Review field is ONE string containing up to 3 bullet points separated by semicolons. No angle brackets, no placeholders, no additional keys, no markdown fences. Example: " + example_json + "\n\nCONTENT TO REVIEW:\n" + last_content
        )

    summary_args_structured = build_summary_args()

    review_chats = []
    for agent_obj in [seo_reviewer, legal_reviewer, ethics_reviewer, fact_checker, bias_reviewer, accessibility_reviewer]:
        if agent_obj is not None:
            review_chats.append({
                "recipient": agent_obj,
                "message": reflection_message,
                "summary_method": "reflection_with_llm",
                "summary_args": summary_args_structured,
                "max_turns": reviewer_max_turns,
            })
    review_chats.append({
        "recipient": meta_reviewer,
        "message": "Aggregate feedback from all reviewers and give final suggestions.",
        "max_turns": reviewer_max_turns,
    })

    image_prompt_agent = None
    image_critics = []
    if enable_image_agents:
        image_prompt_agent = autogen.AssistantAgent(
            name="ImagePromptAgent",
            system_message=(
                "You read the FINAL blog post content and produce DIVERSE image prompt JSON. "
                "Return ONLY JSON with key 'prompts' = list of 1-3 highly specific, stylistically varied, accessible prompts. "
                "Each prompt MUST: 1) reflect core themes & metaphors in the article, 2) avoid literal text rendering unless essential, 3) include accessibility/contrast considerations, 4) avoid branding or trademarked logos, 5) be under 140 words."
            ),
            llm_config=build_role_llm_config("image"),
        )
        for i in range(num_image_critics):
            image_critics.append(
                autogen.AssistantAgent(
                    name=f"ImageCritic{i+1}",
                    system_message=(
                        "You are an image prompt critic. Given a JSON list of prompts and the blog summary, you: "
                        "- Flag vagueness, bias, inaccessibility, brand/trademark risk, unsafe content. "
                        "- Suggest concise improvements. Return ONLY JSON with keys: 'critic':'ImageCritic', 'issues':[], 'improved_prompts':[]."
                    ),
                    llm_config=build_role_llm_config("imagecritic"),
                )
            )
    if image_prompt_agent:
        def _latest_main_content(msgs):
            for m in reversed(msgs or []):
                c = m.get('content') if isinstance(m, dict) else None
                if c:
                    return c
            return ""
        def _latest_image_prompts():
            for agent_ref, msg_list in image_prompt_agent.chat_messages.items():
                if msg_list:
                    last = msg_list[-1]
                    if isinstance(last, dict) and last.get('content'):
                        return last['content']
            return "{\"prompts\": []}"
        review_chats.extend([
            {
                "recipient": image_prompt_agent,
                "message": lambda recipient, messages, sender, config: (
                    "Here is the latest draft blog post. Produce image prompt JSON now.\n\n" + _latest_main_content(messages)
                ),
                "summary_method": "last_msg",
                "max_turns": 1,
            },
            *[
                {
                    "recipient": ic,
                    "message": lambda recipient, messages, sender, config: (
                        "Critique & improve these prompts based on the blog post context.\nPrompts JSON: " + _latest_image_prompts()
                    ),
                    "summary_method": "last_msg",
                    "max_turns": 1,
                } for ic in image_critics
            ],
        ])

    critic.register_nested_chats(review_chats, trigger=writer)
    return writer, critic
