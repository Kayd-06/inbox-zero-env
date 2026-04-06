"""
InboxZeroEnv – Deterministic Grader  (v3 — adversarial + semantic scoring)
===========================================================================
Input:  (email: Email, action: Action, task_difficulty: str)
Output: Reward (score in [0.0, 1.0], reason, action_was_valid)

Key changes in v3
-----------------
* Keyword groups expanded from 7 → 10 (added: empathy, resolution, timeline)
* Semantic proximity bonus: replies that reference the email subject/sender
  receive a +0.05 quality boost (applied within tier scoring)
* customer_complaint category gets a dedicated partial-credit path (was 0.05)
* HARD task strict-zero rules unchanged from v2
* All grading remains 100% deterministic — no LLM, no randomness
"""

from __future__ import annotations

import re
from typing import Optional

from .models import Action, Email, Reward

# ---------------------------------------------------------------------------
# Reply quality — keyword groups and thresholds
# ---------------------------------------------------------------------------

_MIN_REPLY_LENGTH: int = 40          # v3: 40 chars minimum

# --- Original 7 keyword groups ---
_ACCEPTANCE_KEYWORDS      = {"confirmed", "confirm", "attending", "will attend", "i'll attend"}
_ACKNOWLEDGEMENT_KEYWORDS = {"acknowledged", "received", "understood", "noted", "i understand"}
_COMMITMENT_KEYWORDS      = {"will", "shall", "i'll", "asap", "immediately", "right away"}
_GRATITUDE_KEYWORDS       = {"thank", "thanks", "appreciate", "grateful"}
_MEETING_KEYWORDS         = {"meeting", "schedule", "calendar", "invite", "attend", "reschedule"}
_URGENCY_KEYWORDS         = {"urgent", "immediately", "right away", "emergency", "incident", "authorize"}
_APPROVAL_KEYWORDS        = {"approved", "approve", "sign-off", "signed off", "lgtm", "looks good"}

# --- New in v3: 3 additional semantic groups ---
_EMPATHY_KEYWORDS         = {"sorry", "apologize", "apologies", "understand your frustration",
                              "i understand", "we understand", "empathize", "sincerely apologize"}
_RESOLUTION_KEYWORDS      = {"resolve", "fix", "refund", "replacement", "escalate", "solution",
                              "address", "rectify", "remediate", "reissue", "compensate"}
_TIMELINE_KEYWORDS        = {"by eod", "within 24 hours", "by tomorrow", "before the deadline",
                              "before end of day", "today", "immediately", "right away", "asap"}

_ALL_KEYWORD_GROUPS = [
    _ACCEPTANCE_KEYWORDS,
    _ACKNOWLEDGEMENT_KEYWORDS,
    _COMMITMENT_KEYWORDS,
    _GRATITUDE_KEYWORDS,
    _MEETING_KEYWORDS,
    _URGENCY_KEYWORDS,
    _APPROVAL_KEYWORDS,
    _EMPATHY_KEYWORDS,
    _RESOLUTION_KEYWORDS,
    _TIMELINE_KEYWORDS,
]

# ---------------------------------------------------------------------------
# Shared utility helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and collapse punctuation to spaces for keyword matching."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def _count_keyword_groups_matched(text: str) -> int:
    """Return the number of semantic keyword groups matched in *text*."""
    normalised = _normalise(text)
    tokens = set(normalised.split())
    count = 0
    for group in _ALL_KEYWORD_GROUPS:
        matched = False
        for kw in group:
            if " " in kw:
                if kw in normalised:
                    matched = True
                    break
            else:
                if kw in tokens:
                    matched = True
                    break
        if matched:
            count += 1
    return count


def _is_critical_missed_reply(email: Email, chosen: str) -> bool:
    """True when a required high-priority reply was not sent under tight deadline."""
    return (
        email.requires_response
        and email.priority == "high"
        and email.deadline is not None
        and email.deadline <= 2
        and chosen != "reply"
    )


# ---------------------------------------------------------------------------
# Reply quality scoring — deterministic, shared by medium + hard
# ---------------------------------------------------------------------------

def _semantic_proximity_bonus(email: Email, response: str) -> float:
    """
    Award a small bonus when the reply explicitly references the email's
    subject snippet or sender name — indicating the agent read the email.
    Returns 0.05 if matched, else 0.0.
    """
    resp_lower = response.lower()
    subject_words = set(
        w for w in re.split(r"[^\w]+", email.subject.lower())
        if len(w) >= 4  # skip short filler words
    )
    sender_name = email.sender.split("@")[0].replace(".", " ").replace("-", " ").lower()
    sender_words = set(w for w in sender_name.split() if len(w) >= 4)
    meaningful = subject_words | sender_words
    if any(word in resp_lower for word in meaningful):
        return 0.05
    return 0.0


def _score_reply(email: Email, response: Optional[str]) -> tuple[float, str]:
    """
    Score a reply action deterministically.

    Tiers (v3 — 10 keyword groups):
      No response text                         → 0.0
      Response < _MIN_REPLY_LENGTH chars       → 0.1
      Adequate length, 0 keyword groups        → 0.25
      Adequate length, 1–2 groups              → 0.55
      Adequate length, 3–5 groups              → 0.80
      Adequate length, 6+ groups               → 1.0
      +0.05 bonus for semantic proximity        (capped at 1.0)
    """
    if not response:
        return 0.0, "Reply action submitted with no response text — score: 0.0."

    stripped = response.strip()
    if len(stripped) < _MIN_REPLY_LENGTH:
        return 0.1, (
            f"Reply body is too short ({len(stripped)}/{_MIN_REPLY_LENGTH} chars minimum). "
            "Minimal partial credit awarded."
        )

    groups = _count_keyword_groups_matched(stripped)
    proximity_bonus = _semantic_proximity_bonus(email, stripped)
    proximity_note = " +0.05 semantic proximity bonus." if proximity_bonus > 0 else ""

    if groups == 0:
        raw = 0.25
        score = min(raw + proximity_bonus, 1.0)
        return score, (
            f"Reply meets length requirement but contains no recognisable intent keywords "
            f"({groups}/10 groups). Score: {score:.2f}.{proximity_note}"
        )
    if groups <= 2:
        raw = 0.55
        score = min(raw + proximity_bonus, 1.0)
        return score, (
            f"Reply matches {groups}/10 semantic keyword group(s). "
            f"Partial credit — response lacks depth or specificity. Score: {score:.2f}.{proximity_note}"
        )
    if groups <= 5:
        raw = 0.80
        score = min(raw + proximity_bonus, 1.0)
        return score, (
            f"Reply matches {groups}/10 semantic keyword group(s). "
            f"Good response with clear intent and acknowledgement. Score: {score:.2f}.{proximity_note}"
        )
    raw = 1.0
    score = min(raw + proximity_bonus, 1.0)
    return score, (
        f"Reply matches {groups}/10 semantic keyword group(s). "
        f"Excellent response — professional, specific, and actionable. Score: {score:.2f}.{proximity_note}"
    )


# ---------------------------------------------------------------------------
# Main grading entry point
# ---------------------------------------------------------------------------

def grade(email: Email, action: Action, task_difficulty: str = "hard") -> Reward:
    """
    Score the agent's action against the email's ground-truth correct_action.

    Parameters
    ----------
    email            : The email being triaged.
    action           : The agent's chosen action.
    task_difficulty  : "easy" | "medium" | "hard"

    Returns
    -------
    Reward
    """
    correct: str = email.correct_action
    chosen: str  = action.action_type

    if task_difficulty == "easy":
        return _grade_easy(email, action, correct, chosen)
    if task_difficulty == "medium":
        return _grade_medium(email, action, correct, chosen)
    return _grade_hard(email, action, correct, chosen)


# ---------------------------------------------------------------------------
# EASY grader — binary spam classification
# ---------------------------------------------------------------------------

def _grade_easy(email: Email, action: Action, correct: str, chosen: str) -> Reward:
    """
    EASY: Delete spam, keep legitimate emails.  Binary 1.0 / 0.0.
    """
    if email.is_spam:
        if chosen == "delete":
            return Reward(
                score=1.0,
                reason=(
                    f"✓ Correctly identified spam from '{email.sender}' and deleted it. "
                    "Category: spam/phishing."
                ),
                action_was_valid=True,
            )
        return Reward(
            score=0.0,
            reason=(
                f"✗ Spam email from '{email.sender}' should be deleted, "
                f"but agent chose '{chosen}'. False negative — spam escapes inbox."
            ),
            action_was_valid=True,
        )
    # Legitimate email
    if chosen == "delete":
        return Reward(
            score=0.0,
            reason=(
                f"✗ False positive: deleted a legitimate email from '{email.sender}'. "
                "Destructive action on non-spam is always incorrect."
            ),
            action_was_valid=True,
        )
    return Reward(
        score=1.0,
        reason=(
            f"✓ Correctly preserved a legitimate email from '{email.sender}' "
            f"(action: '{chosen}'). Not spam — correct to keep."
        ),
        action_was_valid=True,
    )


# ---------------------------------------------------------------------------
# MEDIUM grader — priority-aware triage with partial credit
# ---------------------------------------------------------------------------

def _grade_medium(email: Email, action: Action, correct: str, chosen: str) -> Reward:
    """
    MEDIUM: Priority-aware triage.

    Scoring rules:
    - Spam binary:  delete → 1.0 | anything else → 0.0
    - Delete legit  → always 0.0
    - Exact match   → 1.0 (reply: blended with quality/2 + 0.5)
    - Close miss    → 0.2–0.7 partial credit
    """
    if email.is_spam:
        if chosen == "delete":
            return Reward(
                score=1.0,
                reason=f"✓ Spam from '{email.sender}' correctly deleted.",
                action_was_valid=True,
            )
        return Reward(
            score=0.0,
            reason=(
                f"✗ Spam from '{email.sender}' must be deleted — "
                f"agent chose '{chosen}' instead."
            ),
            action_was_valid=True,
        )

    if chosen == "delete":
        return Reward(
            score=0.0,
            reason=(
                f"✗ Destructive error: deleted a legitimate email from '{email.sender}'. "
                "Deleting non-spam is always incorrect in the medium task."
            ),
            action_was_valid=True,
        )

    if chosen == correct:
        if chosen == "reply":
            q_score, q_reason = _score_reply(email, action.response)
            blended = round(0.5 + q_score * 0.5, 4)
            return Reward(
                score=blended,
                reason=(
                    f"✓ Correct action 'reply' for email from '{email.sender}'. "
                    f"Reply quality: {q_reason} Blended score: {blended}."
                ),
                action_was_valid=True,
            )
        return Reward(
            score=1.0,
            reason=(
                f"✓ Correct action '{chosen}' for email '{email.subject[:50]}' "
                f"(priority: {email.priority}, category: {email.category})."
            ),
            action_was_valid=True,
        )

    score, reason = _partial_credit_medium(email, correct, chosen)
    return Reward(score=score, reason=reason, action_was_valid=True)


def _partial_credit_medium(email: Email, correct: str, chosen: str) -> tuple[float, str]:
    """Partial credit matrix for medium task — non-exact, non-destructive mismatches."""
    priority = email.priority
    deadline = email.deadline

    if correct == "mark_important" and chosen == "archive":
        if priority == "high":
            return 0.2, (
                f"✗ High-priority email (deadline={deadline}) should be flagged important, "
                "not archived. Significant miss."
            )
        return 0.5, (
            "~ Archiving an email that should be marked important — "
            "acceptable for medium/low priority items."
        )

    if correct == "archive" and chosen == "mark_important":
        if priority == "low":
            return 0.5, (
                "~ Over-flagged a low-priority email as important. "
                "Archive was the correct action; minor overreaction."
            )
        return 0.7, (
            "~ Marked important instead of archiving — "
            "acceptable for non-low-priority items with no follow-up needed."
        )

    if correct == "reply" and chosen in ("mark_important", "archive"):
        if deadline is not None and deadline <= 3:
            return 0.1, (
                f"✗ Time-sensitive reply missed (deadline={deadline} steps). "
                f"Used '{chosen}' — sender is waiting for a response."
            )
        return 0.3, (
            f"~ Reply was expected but agent chose '{chosen}'. "
            "Partial credit for not deleting; response still needed."
        )

    return 0.1, (
        f"✗ Action '{chosen}' is incorrect for this email. "
        f"Expected '{correct}' (priority: {priority}, category: {email.category})."
    )


# ---------------------------------------------------------------------------
# HARD grader — strict full pipeline
# ---------------------------------------------------------------------------

def _grade_hard(email: Email, action: Action, correct: str, chosen: str) -> Reward:
    """
    HARD: Full pipeline with strict scoring.

    Strict rules (v2):
    ① Spam: delete → 1.0 | anything else → 0.0
    ② Delete legitimate email → 0.0  (destructive, always wrong)
    ③ High-priority required reply missed with deadline ≤ 2 → 0.0  (critical failure)
    ④ Correct action with reply → reply quality score [0.0, 1.0]
    ⑤ Correct non-reply action → 1.0
    ⑥ Close mismatches → reduced partial credit (see below)
    """
    # ① Spam handling
    if email.is_spam:
        if chosen == "delete":
            return Reward(
                score=1.0,
                reason=(
                    f"✓ Spam/phishing from '{email.sender}' correctly deleted. "
                    f"Category: {email.category}."
                ),
                action_was_valid=True,
            )
        return Reward(
            score=0.0,
            reason=(
                f"✗ Critical failure: spam from '{email.sender}' (category: {email.category}) "
                f"must be deleted — agent chose '{chosen}'. Score: 0.0."
            ),
            action_was_valid=True,
        )

    # ② Destructive error on legitimate email
    if chosen == "delete":
        return Reward(
            score=0.0,
            reason=(
                f"✗ Destructive error: '{email.subject[:50]}' is a legitimate "
                f"{email.category} email from '{email.sender}' — it must not be deleted. "
                "Score: 0.0."
            ),
            action_was_valid=True,
        )

    # ③ Critical missed reply (strict zero)
    if _is_critical_missed_reply(email, chosen):
        return Reward(
            score=0.0,
            reason=(
                f"✗ Critical failure: high-priority '{email.subject[:50]}' from "
                f"'{email.sender}' requires an immediate reply (deadline={email.deadline} steps). "
                f"Agent chose '{chosen}' instead. Score: 0.0."
            ),
            action_was_valid=True,
        )

    # ④ + ⑤ Correct action
    if chosen == correct:
        if chosen == "reply":
            q_score, q_reason = _score_reply(email, action.response)
            return Reward(
                score=round(q_score, 4),
                reason=(
                    f"✓ Correct action 'reply' for '{email.subject[:50]}'. "
                    f"{q_reason}"
                ),
                action_was_valid=True,
            )
        return Reward(
            score=1.0,
            reason=(
                f"✓ Correct action '{chosen}' for '{email.subject[:50]}' "
                f"(priority: {email.priority}, category: {email.category})."
            ),
            action_was_valid=True,
        )

    # ⑥ Partial credit
    score, reason = _partial_credit_hard(email, correct, chosen)
    return Reward(score=score, reason=reason, action_was_valid=True)


def _partial_credit_hard(email: Email, correct: str, chosen: str) -> tuple[float, str]:
    """
    Strict partial-credit matrix for the hard task (v3).

    Guiding principle: the hard task should be genuinely challenging.
    Partial credits are intentionally lower than medium.
    v3 adds a dedicated customer_complaint path for better signal.
    """
    priority = email.priority
    deadline = email.deadline
    category = email.category

    # Reply expected → archive/mark_important (non-critical)
    if correct == "reply" and chosen in ("mark_important", "archive"):
        # deadline > 2 (otherwise caught by ③ above and returned 0.0)
        if priority == "high":
            return 0.1, (
                f"✗ High-priority reply missed for '{email.subject[:50]}'. "
                f"Agent chose '{chosen}' — sender expects a response. Score: 0.1."
            )
        return 0.2, (
            f"~ Reply was expected for '{email.subject[:50]}' (priority: {priority}). "
            f"Agent chose '{chosen}'; no response was sent. Score: 0.2."
        )

    # mark_important expected → archive
    if correct == "mark_important" and chosen == "archive":
        if priority == "high":
            return 0.25, (
                f"✗ High-priority email '{email.subject[:50]}' should be flagged important, "
                "not archived. Important items may be missed. Score: 0.25."
            )
        return 0.45, (
            f"~ Archiving instead of marking important for '{email.subject[:50]}' "
            f"(priority: {priority}). Acceptable but suboptimal. Score: 0.45."
        )

    # archive expected → mark_important
    if correct == "archive" and chosen == "mark_important":
        if priority == "low":
            return 0.4, (
                f"~ Over-flagged low-priority '{email.subject[:50]}' as important. "
                "Archive was correct; creates noise. Score: 0.4."
            )
        return 0.5, (
            f"~ Marked '{email.subject[:50]}' important instead of archiving — "
            "tolerable for non-low-priority items. Score: 0.5."
        )

    # Spurious reply (reply when not needed) — tighter than v1
    if correct in ("mark_important", "archive") and chosen == "reply":
        return 0.1, (
            f"✗ Unsolicited reply for '{email.subject[:50]}' which only needed '{correct}'. "
            "Excessive action wastes effort. Score: 0.1."
        )

    # v3: Customer complaint — acknowledge any action that isn't destructive
    if category == "customer_complaint" and chosen != "delete":
        return 0.15, (
            f"~ Customer complaint '{email.subject[:50]}' needs a reply, but agent chose "
            f"'{chosen}'. At least the email was not deleted — minimal acknowledgement credit. "
            "Score: 0.15."
        )

    # Fallback
    return 0.05, (
        f"✗ Action '{chosen}' is incorrect for '{email.subject[:50]}'. "
        f"Expected '{correct}' (priority: {priority}, category: {category}). Score: 0.05."
    )


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_action(action: Action) -> tuple[bool, str]:
    """
    Check structural validity of an action before grading.

    Returns (is_valid, reason).
    """
    if action.action_type == "reply" and not action.response:
        return False, (
            "Structural error: action_type='reply' requires a non-empty 'response' field. "
            "Score: 0.0."
        )
    if action.action_type != "reply" and action.response:
        # Non-fatal — just note it; the response field will be ignored
        return True, (
            f"Note: 'response' field is ignored for action_type='{action.action_type}' "
            "and will not affect scoring."
        )
    return True, "Action is structurally valid."
