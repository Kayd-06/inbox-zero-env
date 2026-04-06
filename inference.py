#!/usr/bin/env python3
"""
InboxZeroEnv – Baseline Inference Script  (v3)
===============================================
Runs an OpenAI-compatible LLM agent through all 3 tasks of InboxZeroEnv
and outputs deterministic, reproducible scores.

Environment variables (required):
    API_BASE_URL   – Base URL of the OpenAI-compatible API (e.g. https://api.openai.com/v1)
    MODEL_NAME     – Model identifier (e.g. gpt-4o, meta-llama/Llama-3-8b-instruct)
    HF_TOKEN       – Bearer token / API key (used as the OpenAI API key)

Usage:
    API_BASE_URL=https://api.openai.com/v1 \\
    MODEL_NAME=gpt-4o \\
    HF_TOKEN=sk-... \\
    python inference.py

Output:
    JSON summary written to stdout and saved as results.json.
    Must complete under 20 minutes.

New in v3:
    - Chain-of-thought (CoT) reasoning before JSON output
    - 3 few-shot gold examples in system prompt
    - Per-category accuracy breakdown in results.json
    - Validation retry: if reply action has no response, retry once
    - Structured penalty_breakdown logged per step
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Import the environment
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import InboxZeroEnv, Action

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
MAX_TOTAL_MINUTES = 18  # hard stop before the 20-minute limit


# ---------------------------------------------------------------------------
# Prompt templates  (v3: Chain-of-Thought + 3 few-shot gold examples)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant. You will be shown one email at a time.
    Your job is to choose exactly one triage action for each email.

    Available actions:
    - "delete"         : For spam, phishing, irrelevant promotions, or junk.
    - "archive"        : For legitimate but low-priority emails (newsletters, notifications, billing receipts).
    - "mark_important" : For important emails that need attention but NO reply (calendar invites, reports, announcements).
    - "reply"          : For emails that require a human response. You MUST also provide a "response" field.

    Rules:
    1. Never delete a legitimate email — only delete clear spam or phishing.
    2. Never leave a spam email as anything other than deleted.
    3. If the action is "reply", your response must be professional, acknowledge the email, and be at least 40 words.
    4. Think step-by-step inside <thinking>...</thinking> tags before outputting your JSON.
    5. After your thinking, respond ONLY with valid JSON matching this schema:
       {"action_type": "<action>", "response": "<reply text or null>"}
    6. No markdown, no explanation outside the JSON (after the thinking block).

    --- FEW-SHOT EXAMPLES ---

    Example 1 — Spam phishing email:
    Input: Subject: "Your PayPal account has been suspended" | From: security@paypal-alerts.net | is_spam: true
    <thinking>
    The domain "paypal-alerts.net" is not PayPal's official domain. This is a phishing attempt.
    The correct action is "delete".
    </thinking>
    {"action_type": "delete", "response": null}

    Example 2 — Meeting request requiring a reply:
    Input: Subject: "Q3 Planning Session — Friday 2pm" | From: sarah.johnson@acmecorp.com | requires_response: true | priority: high
    <thinking>
    This is a legitimate meeting request from a colleague. It explicitly requires a response to confirm attendance.
    The correct action is "reply" with a professional confirmation.
    </thinking>
    {"action_type": "reply", "response": "Hi Sarah, thank you for the meeting invite. I will confirm my attendance for the Q3 planning session this Friday at 2:00 PM. I'll review the agenda items in advance and come prepared to discuss the budget and OKR alignment. Please let me know if anything changes. Best regards."}

    Example 3 — Low-priority newsletter:
    Input: Subject: "Monthly Notion Updates" | From: updates@notion.so | is_spam: false | requires_response: false | priority: low
    <thinking>
    This is a legitimate product newsletter from Notion. It's informational, low priority, and requires no action.
    The correct action is "archive".
    </thinking>
    {"action_type": "archive", "response": null}
    --- END EXAMPLES ---
""").strip()


def make_user_prompt(obs: Any) -> str:
    """Build the per-step user message from an Observation."""
    email = obs.current_email
    deadline_str = f"Deadline: {email.deadline} steps" if email.deadline is not None else "No hard deadline"

    history_str = ""
    if obs.recent_action_history:
        history_lines = []
        for h in obs.recent_action_history[-3:]:  # last 3 only
            mark = "✓" if h.was_correct else "✗"
            history_lines.append(
                f"  {mark} Step {h.step + 1}: Email #{h.email_id} "
                f"'{h.email_subject_snippet[:35]}' → {h.action_type} (score: {h.score:.2f})"
            )
        history_str = "\nRecent decisions:\n" + "\n".join(history_lines)

    return textwrap.dedent(f"""
        === EMAIL #{email.id} ===
        Subject   : {email.subject}
        From      : {email.sender}
        Priority  : {email.priority}
        Requires response: {"Yes" if email.requires_response else "No"}
        {deadline_str}

        Body:
        {email.body}

        Inbox remaining: {obs.inbox_remaining}
        Step: {obs.step_count + 1}
        Task: {obs.task_name} ({obs.task_difficulty.upper()})
        {history_str}

        Think step-by-step in <thinking> tags, then output JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, model: str, user_prompt: str) -> Optional[str]:
    """Call the LLM and return the raw text response, with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,          # Deterministic
                max_tokens=768,           # v3: bumped for CoT
                timeout=60,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] LLM call failed: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    return None


def parse_action(raw: Optional[str]) -> Action:
    """
    Parse LLM output into an Action.

    v3: Strips <thinking>...</thinking> CoT block before parsing JSON.
    Falls back to 'archive' on parse failure.
    """
    if not raw:
        return Action(action_type="archive", response=None)
    try:
        # Strip chain-of-thought thinking block
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL).strip()

        # Strip any accidental markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]).strip()

        # Find the JSON object
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        data = json.loads(cleaned)
        action_type = data.get("action_type", "archive")
        response = data.get("response") or None
        if isinstance(response, str) and response.strip() == "":
            response = None
        return Action(action_type=action_type, response=response)
    except Exception as e:
        print(f"  [parse_action] Failed to parse: {raw!r} — {e}", file=sys.stderr)
        return Action(action_type="archive", response=None)


def validate_and_maybe_retry(
    client: OpenAI,
    model: str,
    obs: Any,
    action: Action,
) -> Action:
    """
    v3: If action_type is 'reply' but response is missing/empty, retry once.
    This catches cases where the model forgets the response field.
    """
    if action.action_type == "reply" and not action.response:
        print("  [retry] Reply action has no response — retrying with explicit reminder...",
              file=sys.stderr)
        retry_prompt = make_user_prompt(obs) + (
            "\n\nIMPORTANT: You chose 'reply' but did not include a 'response' field. "
            "You MUST include a 'response' field with at least 40 words of professional reply text."
        )
        raw = call_llm(client, model, retry_prompt)
        retried = parse_action(raw)
        if retried.action_type == "reply" and retried.response:
            return retried
        # Still no response — fall back to mark_important to avoid 0.0 score
        print("  [retry] Still no response after retry — falling back to mark_important.",
              file=sys.stderr)
        return Action(action_type="mark_important", response=None)
    return action


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    model: str,
    task_name: str,
    deadline: float,
) -> Dict[str, Any]:
    """Run the agent through a single task and return a result dict."""
    print(f"\n--- START TASK: {task_name.upper()} ---")
    print(f"{'=' * 60}")
    print(f"  Task: {task_name.upper()}")
    print(f"{'=' * 60}")

    env = InboxZeroEnv(task_name=task_name)
    obs = env.reset()

    step_results: List[Dict[str, Any]] = []
    category_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "cumulative_score": 0.0}
    )
    done = False

    while not done:
        if time.monotonic() > deadline:
            print("  [TIMEOUT] Hard stop reached — terminating task early.", file=sys.stderr)
            break

        user_prompt = make_user_prompt(obs)
        email = obs.current_email

        print(f"--- STEP {obs.step_count + 1} ---")
        print(f"  Step {obs.step_count + 1} | Email #{email.id}: {email.subject[:50]!r}")

        raw = call_llm(client, model, user_prompt)
        action = parse_action(raw)

        # v3: Validate and retry if needed
        action = validate_and_maybe_retry(client, model, obs, action)

        print(f"    → action: {action.action_type}", end="")
        if action.response:
            print(f" | reply ({len(action.response)} chars)", end="")
        print()

        obs, reward, done, info = env.step(action)

        penalty_note = ""
        if info.get("penalty_breakdown"):
            tags = [f"[{k}:{v:+.2f}]" for k, v in info["penalty_breakdown"].items()]
            penalty_note = "  " + " ".join(tags)

        print(f"    ✓ score: {reward.score:.4f} | {reward.reason[:75]}{penalty_note}")

        # Update per-category stats
        cat = info["email_category"]
        was_correct = info["chosen_action"] == info["correct_action"]
        category_stats[cat]["total"] += 1
        category_stats[cat]["cumulative_score"] += reward.score
        if was_correct:
            category_stats[cat]["correct"] += 1

        step_results.append({
            "step": info["step_count"],
            "email_id": info["email_id"],
            "category": info["email_category"],
            "priority": info["email_priority"],
            "correct_action": info["correct_action"],
            "chosen_action": info["chosen_action"],
            "score": reward.score,
            "action_was_valid": reward.action_was_valid,
            "penalty_breakdown": info.get("penalty_breakdown", {}),
        })

        if obs is None:
            break

    summary = env.summary()
    print(f"\n  Final score: {summary['final_score']:.4f} ({summary['steps_taken']} steps)")
    env.render()
    print(f"--- END TASK: {task_name.upper()} ---")

    # Build per-category accuracy table
    per_category = {
        cat: {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy_pct": round(100 * s["correct"] / s["total"], 1) if s["total"] else 0.0,
            "avg_score": round(s["cumulative_score"] / s["total"], 4) if s["total"] else 0.0,
        }
        for cat, s in sorted(category_stats.items())
    }

    return {
        "task": task_name,
        "difficulty": summary["task_difficulty"],
        "task_display_name": summary["task_name"],
        "total_emails": summary["total_emails"],
        "steps_taken": summary["steps_taken"],
        "cumulative_score": summary["cumulative_score"],
        "final_score": summary["final_score"],
        "avg_step_score": summary["avg_step_score"],
        "efficiency": summary["efficiency"],
        "correct_actions": summary["correct_actions"],
        "accuracy_pct": summary["accuracy_pct"],
        "action_counts": summary["action_counts"],
        "per_category": per_category,
        "step_results": step_results,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

    # --- v3: Typo-resilient token fix ---
    if token:
        # User common typo: 'h£_' instead of 'hf_'
        if "h£_" in token:
            print("  [FIX] Automatically corrected 'h£_' typo to 'hf_' in token.", file=sys.stderr)
            token = token.replace("h£_", "hf_")
        
        # Strip any remaining non-ASCII characters that cause UnicodeEncodeErrors
        original_len = len(token)
        token = "".join(c for c in token if ord(c) < 128)
        if len(token) < original_len:
            print("  [FIX] Automatically stripped invalid non-ASCII characters from token.", file=sys.stderr)
    # ------------------------------------

    if not token:
        print(
            "ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"InboxZeroEnv Inference  (v3 — CoT + few-shot)")
    print(f"  API base : {api_base}")
    print(f"  Model    : {model}")
    print(f"  Token    : {'*' * (len(token) - 4) + token[-4:]}")

    client = OpenAI(api_key=token, base_url=api_base)

    start_time = time.monotonic()
    hard_deadline = start_time + MAX_TOTAL_MINUTES * 60

    results: List[Dict[str, Any]] = []

    for task_name in ["easy", "medium", "hard"]:
        if time.monotonic() > hard_deadline:
            print(f"  [TIMEOUT] Skipping task '{task_name}' — approaching time limit.")
            break
        result = run_task(client, model, task_name, hard_deadline)
        results.append(result)

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------
    elapsed = time.monotonic() - start_time
    if results:
        avg_score = sum(r["final_score"] for r in results) / len(results)
        avg_accuracy = sum(r["accuracy_pct"] for r in results) / len(results)
    else:
        avg_score = 0.0
        avg_accuracy = 0.0

    # Aggregate per-category stats across all tasks
    global_category: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "cumulative_score": 0.0}
    )
    for r in results:
        for cat, stats in r.get("per_category", {}).items():
            global_category[cat]["correct"] += stats["correct"]
            global_category[cat]["total"] += stats["total"]
            global_category[cat]["cumulative_score"] += stats["avg_score"] * stats["total"]

    global_per_category = {
        cat: {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy_pct": round(100 * s["correct"] / s["total"], 1) if s["total"] else 0.0,
            "avg_score": round(s["cumulative_score"] / s["total"], 4) if s["total"] else 0.0,
        }
        for cat, s in sorted(global_category.items())
    }

    output = {
        "env_version": "3.0.0",
        "model": model,
        "api_base": api_base,
        "elapsed_seconds": round(elapsed, 2),
        "average_score": round(avg_score, 6),
        "average_accuracy_pct": round(avg_accuracy, 2),
        "global_per_category": global_per_category,
        "tasks": results,
    }

    print(f"\n{'=' * 60}")
    print(f"  OVERALL AVERAGE SCORE: {avg_score:.4f}")
    print(f"  OVERALL ACCURACY:      {avg_accuracy:.1f}%")
    print(f"  Elapsed time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Per-category summary table
    print(f"\n  {'Category':<22} {'Acc%':>6}  {'AvgScore':>9}  {'n':>4}")
    print(f"  {'-' * 46}")
    for cat, s in global_per_category.items():
        print(f"  {cat:<22} {s['accuracy_pct']:>5.1f}%  {s['avg_score']:>9.4f}  {s['total']:>4}")

    output_json = json.dumps(output, indent=2)
    print("\n" + output_json)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(output_json)
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
