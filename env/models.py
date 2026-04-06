"""
InboxZeroEnv – Data Models
All Pydantic models used across the environment, grader, and tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Core domain object
# ---------------------------------------------------------------------------

class Email(BaseModel):
    """A single email in the inbox."""

    id: int = Field(..., description="Unique email identifier (monotonically increasing).")
    subject: str = Field(..., description="Email subject line.")
    sender: str = Field(..., description="Sender email address.")
    body: str = Field(..., description="Full email body text.")
    priority: Literal["low", "medium", "high"] = Field(
        ..., description="Triage priority level."
    )
    is_spam: bool = Field(..., description="True if the email is spam / phishing.")
    requires_response: bool = Field(
        ..., description="True if the email explicitly requires a human reply."
    )
    deadline: Optional[int] = Field(
        None,
        description=(
            "Steps remaining before the deadline becomes critical. "
            "None means no hard deadline."
        ),
    )
    # Metadata used by grader — not normally exposed to agents
    correct_action: Literal["delete", "archive", "mark_important", "reply"] = Field(
        ..., description="The single best action for this email."
    )
    category: str = Field(
        ...,
        description=(
            "Human-readable category tag, e.g. 'spam', 'newsletter', "
            "'meeting_request', 'urgent_work'."
        ),
    )

    class Config:
        frozen = True  # emails are immutable once loaded


# ---------------------------------------------------------------------------
# Agent interface types
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """An action emitted by the agent for the current email."""

    action_type: Literal["delete", "archive", "mark_important", "reply"] = Field(
        ..., description="Which triage operation to perform."
    )
    response: Optional[str] = Field(
        None,
        description=(
            "Required when action_type == 'reply'. "
            "Contains the drafted reply text the agent wishes to send."
        ),
    )

    @field_validator("response", mode="before")
    @classmethod
    def validate_response(cls, v: Optional[str], info: Any) -> Optional[str]:
        # We allow None for non-reply actions; enforcement is in the grader.
        if v is not None:
            v = str(v).strip()
        return v


# ---------------------------------------------------------------------------
# Context-aware recent action summary (for Observation history)
# ---------------------------------------------------------------------------

class RecentActionSummary(BaseModel):
    """A compact record of one past triage decision, included in Observation."""

    step: int = Field(..., description="Step index (0-based).")
    email_id: int = Field(..., description="ID of the email that was triaged.")
    email_subject_snippet: str = Field(
        ..., description="First 40 chars of the subject, for quick context."
    )
    action_type: str = Field(..., description="The action that was taken.")
    score: float = Field(..., description="Score awarded for this action.")
    was_correct: bool = Field(
        ..., description="True if the action matched the ground-truth correct_action."
    )


class Observation(BaseModel):
    """The observation returned to the agent after each step."""

    current_email: Email = Field(..., description="The email the agent must triage now.")
    inbox_remaining: int = Field(
        ..., description="Number of emails not yet processed (including current one)."
    )
    step_count: int = Field(..., description="Zero-indexed step number.")
    task_name: str = Field(..., description="Name of the active task.")
    task_difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty tier of the active task."
    )
    # Context window: last N triage decisions (empty at episode start)
    recent_action_history: List[RecentActionSummary] = Field(
        default_factory=list,
        description=(
            "Structured summary of the last 1–5 triage decisions. "
            "Enables context-aware reasoning about past mistakes."
        ),
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Per-step score strictly in [0.0, 1.0].",
    )
    reason: str = Field(..., description="Human-readable explanation of the score.")
    action_was_valid: bool = Field(
        ..., description="False when the action itself was structurally invalid."
    )


# ---------------------------------------------------------------------------
# Full internal state (for state() method)
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """Complete serialisable snapshot of the environment's internal state."""

    task_name: str
    task_difficulty: Literal["easy", "medium", "hard"]
    email_ids: List[int] = Field(
        ..., description="Ordered list of all email IDs in this episode."
    )
    current_index: int = Field(
        ..., description="Index into email_ids pointing at the current email."
    )
    step_count: int
    max_steps: int
    done: bool
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of {email_id, action_type, response, score} dicts.",
    )
    # Context window exposed to state() — last 5 decisions
    last_actions: List[RecentActionSummary] = Field(
        default_factory=list,
        description="Structured last-N action context window (up to 5 entries).",
    )
    # Human-readable decision summary
    decision_summary: str = Field(
        default="",
        description=(
            "Short narrative summary of agent decisions so far: "
            "correct_count, mistake_count, dominant_action."
        ),
    )
    cumulative_score: float = Field(0.0, description="Sum of per-step scores so far.")
    total_emails: int
    # Overuse / pattern tracking
    action_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of each action_type used so far in this episode.",
    )
    consecutive_wrong: int = Field(
        0,
        description="Number of consecutive steps with score == 0.0.",
    )

    @property
    def inbox_remaining(self) -> int:
        return max(0, self.total_emails - self.current_index)
