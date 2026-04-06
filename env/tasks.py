"""
InboxZeroEnv – Task Definitions
=================================
Exactly 3 tasks: EASY, MEDIUM, HARD.
Each task defines which emails it uses, how the grader is configured,
and includes a self-contained deterministic grader entry point.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Action, Email, Reward
from .grader import grade, validate_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_emails(path: Optional[str] = None) -> List[Email]:
    """Load the canonical email dataset from disk."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "emails.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Email(**item) for item in raw]


# ---------------------------------------------------------------------------
# Task descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskConfig:
    """Immutable task configuration."""

    name: str
    description: str
    difficulty: str                   # "easy" | "medium" | "hard"
    email_filter_categories: List[str] = field(default_factory=list)
    max_steps: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "email_filter_categories": self.email_filter_categories,
            "max_steps": self.max_steps,
        }


# ---------------------------------------------------------------------------
# Grader entry points — deterministic, standalone, testable
# ---------------------------------------------------------------------------

class EasyGrader:
    """
    EASY task grader: spam-vs-not-spam binary classification.

    Score is strictly 0.0 or 1.0.
    """

    difficulty = "easy"

    @staticmethod
    def grade(email: Email, action: Action) -> Reward:
        is_valid, validation_msg = validate_action(action)
        if not is_valid:
            return Reward(score=0.0, reason=validation_msg, action_was_valid=False)
        return grade(email, action, task_difficulty="easy")


class MediumGrader:
    """
    MEDIUM task grader: priority-aware triage with archiving.

    Score ranges continuously in [0.0, 1.0].
    Correct action → 1.0
    Semantically close → 0.3–0.7
    Completely wrong   → 0.0
    """

    difficulty = "medium"

    @staticmethod
    def grade(email: Email, action: Action) -> Reward:
        is_valid, validation_msg = validate_action(action)
        if not is_valid:
            return Reward(score=0.0, reason=validation_msg, action_was_valid=False)
        return grade(email, action, task_difficulty="medium")


class HardGrader:
    """
    HARD task grader: full pipeline including reply generation.

    Reply quality is scored deterministically by keyword coverage and length.
    Score ranges continuously in [0.0, 1.0].
    """

    difficulty = "hard"

    @staticmethod
    def grade(email: Email, action: Action) -> Reward:
        is_valid, validation_msg = validate_action(action)
        if not is_valid:
            return Reward(score=0.0, reason=validation_msg, action_was_valid=False)
        return grade(email, action, task_difficulty="hard")


# ---------------------------------------------------------------------------
# Canonical task registry
# ---------------------------------------------------------------------------

EASY_TASK = TaskConfig(
    name="SpamSentinel",
    description=(
        "Classify each email as spam (delete) or legitimate (any non-delete action). "
        "Only spam/phishing/promotion emails are included. "
        "Binary scoring: 1.0 for correct classification, 0.0 for incorrect."
    ),
    difficulty="easy",
    email_filter_categories=["spam", "phishing", "promotion", "newsletter", "notification", "billing"],
    max_steps=15,
)

MEDIUM_TASK = TaskConfig(
    name="PriorityTriageDesk",
    description=(
        "Handle a realistic inbox mix of work emails, meeting requests, billing notices, "
        "and newsletters. Apply correct priority-aware triage: delete spam, archive low-priority "
        "items, mark high-priority items as important, and reply when required. "
        "Partial credit awarded for semantically close decisions."
    ),
    difficulty="medium",
    email_filter_categories=[
        "spam", "phishing", "newsletter", "notification",
        "billing", "work", "meeting_request",
    ],
    max_steps=20,
)

HARD_TASK = TaskConfig(
    name="FullInboxZero",
    description=(
        "Complete end-to-end email triage: classify spam, handle priorities, archive routine items, "
        "flag important items, and compose high-quality replies for emails requiring responses. "
        "Reply quality is scored deterministically on length and keyword coverage. "
        "Strict scoring with partial credit on reply quality."
    ),
    difficulty="hard",
    email_filter_categories=[
        "spam", "phishing", "newsletter", "notification",
        "billing", "work", "meeting_request", "urgent_work", "customer_complaint",
    ],
    max_steps=30,
)

ALL_TASKS: Dict[str, TaskConfig] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}

GRADERS: Dict[str, type] = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}


# ---------------------------------------------------------------------------
# Email filtering per task
# ---------------------------------------------------------------------------

def get_emails_for_task(task: TaskConfig, email_path: Optional[str] = None) -> List[Email]:
    """
    Return the ordered list of emails relevant to the given task.

    Emails are deterministically ordered by their ID (ascending) after
    filtering by category.
    """
    all_emails = _load_emails(email_path)
    if task.email_filter_categories:
        filtered = [
            e for e in all_emails
            if e.category in task.email_filter_categories
        ]
    else:
        filtered = list(all_emails)

    # Deterministic ordering by ID
    filtered.sort(key=lambda e: e.id)
    return filtered
