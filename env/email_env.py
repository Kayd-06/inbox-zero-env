"""
InboxZeroEnv – Core RL Environment  (v3)
=========================================
OpenEnv-compatible reinforcement learning environment for email triage.

Public API
----------
    reset()            → Observation
    step(action)       → (Observation | None, Reward, done: bool, info: dict)
    state()            → dict  (full serialisable internal state)
    render()           → str   (human-readable current state table)
    seed(n)            → None  (no-op; documents the RL contract)
    from_config(path)  → InboxZeroEnv  (classmethod; loads from openenv.yaml)

New in v3
---------
* render(mode="human") — formatted state table for debugging and demos
* seed(n) — documents gymnasium RL contract (env is always deterministic)
* from_config(yaml_path) — clean instantiation from openenv.yaml
* penalty_breakdown field added to info dict (per-step penalty log)
* All v2 features retained unchanged
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    Email,
    EnvironmentState,
    Observation,
    RecentActionSummary,
    Reward,
)
from .tasks import ALL_TASKS, GRADERS, TaskConfig, get_emails_for_task
from .grader import validate_action

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Penalty constants  (all deterministic, no randomness)
# ---------------------------------------------------------------------------

_STEP_OVERHEAD_RATE: float = 0.02       # per excess step past midpoint
_REPEATED_MISTAKE_PENALTY: float = 0.10 # triggered when consecutive_wrong >= 3
_OVERUSE_THRESHOLD: float = 0.70        # fraction to trigger overuse penalty
_OVERUSE_PENALTY: float = 0.05          # applied to the current step
_SCORE_FLOOR: float = 0.0
_CONTEXT_WINDOW: int = 5                # number of recent actions in Observation

# ---------------------------------------------------------------------------
# Efficiency weight
# ---------------------------------------------------------------------------

_EFFICIENCY_WEIGHT: float = 0.20
_STEP_SCORE_WEIGHT: float = 0.80


class InboxZeroEnv:
    """
    OpenEnv-compatible email triage environment (v2).

    Parameters
    ----------
    task_name  : "easy" | "medium" | "hard"
    email_path : Optional path to a custom emails.json.
    max_steps  : Override the task's default max_steps.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        task_name: str = "hard",
        email_path: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        if task_name not in ALL_TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid options: {list(ALL_TASKS.keys())}"
            )

        self._task_config: TaskConfig = ALL_TASKS[task_name]
        self._grader = GRADERS[task_name]
        self._email_path = email_path
        self._max_steps: int = (
            max_steps if max_steps is not None else self._task_config.max_steps
        )

        # Fixed email sequence (loaded once, immutable)
        self._emails: List[Email] = get_emails_for_task(self._task_config, email_path)

        # Mutable episode state (initialised by reset())
        self._current_index: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._action_history: List[Dict[str, Any]] = []
        self._recent_actions: List[RecentActionSummary] = []    # context window
        self._cumulative_score: float = 0.0
        self._action_counts: Counter = Counter()
        self._consecutive_wrong: int = 0
        self._episode_start_time: float = 0.0

        # Auto-reset
        self.reset()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset to start of episode. Returns the first Observation."""
        self._current_index = 0
        self._step_count = 0
        self._done = False
        self._action_history = []
        self._recent_actions = []
        self._cumulative_score = 0.0
        self._action_counts = Counter()
        self._consecutive_wrong = 0
        self._episode_start_time = time.monotonic()
        return self._make_observation()

    def step(
        self, action: Action
    ) -> Tuple[Optional[Observation], Reward, bool, Dict[str, Any]]:
        """
        Process one action for the current email.

        Returns
        -------
        observation : Observation | None — next email, or None if done.
        reward      : Reward — per-step scored signal.
        done        : bool — True when episode ends.
        info        : dict — rich diagnostics.
        """
        if self._done:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )

        current_email = self._emails[self._current_index]

        # ── Structural validation ─────────────────────────────────────────
        is_valid, validation_msg = validate_action(action)
        if not is_valid:
            reward = Reward(score=0.0, reason=validation_msg, action_was_valid=False)
            self._update_tracking(current_email, action, reward)
            info = self._build_info(current_email, action, reward)
            self._advance()
            obs = self._make_observation() if not self._done else None
            return obs, reward, self._done, info

        # ── Base grading ──────────────────────────────────────────────────
        reward = self._grader.grade(current_email, action)

        # ── Advanced penalty system ───────────────────────────────────────
        reward = self._apply_penalties(current_email, action, reward)

        # ── Record ────────────────────────────────────────────────────────
        self._update_tracking(current_email, action, reward)
        self._cumulative_score += reward.score
        info = self._build_info(current_email, action, reward)

        # ── Advance ───────────────────────────────────────────────────────
        self._advance()
        obs = self._make_observation() if not self._done else None
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """
        Full, serialisable snapshot of internal state (OpenEnv state() contract).
        Includes context window, decision summary, and pattern counters.
        """
        snap = EnvironmentState(
            task_name=self._task_config.name,
            task_difficulty=self._task_config.difficulty,   # type: ignore[arg-type]
            email_ids=[e.id for e in self._emails],
            current_index=self._current_index,
            step_count=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            action_history=self._action_history,
            last_actions=list(self._recent_actions),
            decision_summary=self._build_decision_summary(),
            cumulative_score=round(self._cumulative_score, 6),
            total_emails=len(self._emails),
            action_counts=dict(self._action_counts),
            consecutive_wrong=self._consecutive_wrong,
        )
        return snap.model_dump()

    # ------------------------------------------------------------------
    # Gymnasium-compatible extension methods (v3)
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> str:
        """
        Return a formatted human-readable summary of the current episode state.
        Useful for debugging and demo outputs.
        """
        if self._done or self._current_index >= len(self._emails):
            email_info = "[Episode complete]"
            subject = ""
            sender = ""
            priority = ""
            category = ""
        else:
            email = self._emails[self._current_index]
            subject = email.subject[:55] + ("..." if len(email.subject) > 55 else "")
            sender = email.sender
            priority = email.priority.upper()
            category = email.category
            email_info = f"Email #{email.id}"

        correct = sum(1 for h in self._action_history if h.get("was_correct", False))
        total_so_far = len(self._action_history)
        accuracy = f"{100 * correct / total_so_far:.1f}%" if total_so_far else "N/A"
        avg = (
            f"{self._cumulative_score / total_so_far:.4f}" if total_so_far else "N/A"
        )
        lines = [
            f"{'=' * 64}",
            f"  InboxZeroEnv v3 | Task: {self._task_config.name} "
            f"({self._task_config.difficulty.upper()})",
            f"{'=' * 64}",
            f"  Step          : {self._step_count} / {self._max_steps}",
            f"  {email_info:<16}: {subject}",
        ]
        if not self._done and self._current_index < len(self._emails):
            lines += [
                f"  Sender        : {sender}",
                f"  Priority      : {priority}   Category: {category}",
            ]
        lines += [
            f"  Inbox left    : {max(0, len(self._emails) - self._current_index)}",
            f"  Cumul. score  : {self._cumulative_score:.4f}",
            f"  Avg step score: {avg}",
            f"  Accuracy      : {accuracy}  ({correct}/{total_so_far} correct)",
            f"  Consec. wrong : {self._consecutive_wrong}",
            f"  Action counts : {dict(self._action_counts)}",
            f"{'=' * 64}",
        ]
        rendered = "\n".join(lines)
        if mode == "human":
            print(rendered)
        return rendered

    def seed(self, seed: Optional[int] = None) -> None:
        """
        No-op for API compatibility (gymnasium contract).
        InboxZeroEnv is always fully deterministic regardless of seed.
        """
        pass  # deterministic by design

    @classmethod
    def from_config(cls, yaml_path: str, **overrides: Any) -> "InboxZeroEnv":
        """
        Instantiate InboxZeroEnv from an openenv.yaml configuration file.

        Parameters
        ----------
        yaml_path : str
            Path to the openenv.yaml file.
        **overrides
            Any keyword arguments to override the config (task_name, max_steps).

        Example
        -------
            env = InboxZeroEnv.from_config("openenv.yaml", task_name="hard")
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for from_config(). Run: pip install pyyaml"
            )
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ep = cfg.get("entry_point", {})
        init_params = ep.get("init_params", {})
        task_name = overrides.pop("task_name", init_params.get("task_name", "hard"))
        max_steps = overrides.pop("max_steps", None)
        return cls(task_name=task_name, max_steps=max_steps, **overrides)

    def __repr__(self) -> str:
        return (
            f"InboxZeroEnv(task='{self._task_config.name}', "
            f"difficulty='{self._task_config.difficulty}', "
            f"step={self._step_count}/{self._max_steps}, "
            f"emails={len(self._emails)}, "
            f"done={self._done})"
        )

    # ------------------------------------------------------------------
    # Scoring and reporting
    # ------------------------------------------------------------------

    def final_score(self) -> float:
        """
        Efficiency-weighted episode score, strictly in [0.0, 1.0].

        Formula:
            avg_step_score = cumulative_score / total_emails
            efficiency     = 1 - (steps_taken / max_steps)
            final_score    = avg_step_score × 0.8 + efficiency × 0.2
        """
        total = len(self._emails)
        if total == 0:
            return 0.0
        avg_step_score = min(self._cumulative_score / total, 1.0)
        steps_taken = self._step_count
        efficiency = max(0.0, 1.0 - steps_taken / self._max_steps)
        raw = _STEP_SCORE_WEIGHT * avg_step_score + _EFFICIENCY_WEIGHT * efficiency
        return round(min(max(raw, 0.0), 1.0), 6)

    def summary(self) -> Dict[str, Any]:
        """Clean, human-readable episode summary."""
        total = len(self._emails)
        avg_step = round(self._cumulative_score / total, 6) if total else 0.0
        efficiency = round(
            max(0.0, 1.0 - self._step_count / self._max_steps), 4
        )
        correct_count = sum(
            1 for h in self._action_history
            if h.get("was_correct", False)
        )
        return {
            "task_name": self._task_config.name,
            "task_difficulty": self._task_config.difficulty,
            "total_emails": total,
            "steps_taken": self._step_count,
            "max_steps": self._max_steps,
            "cumulative_score": round(self._cumulative_score, 6),
            "avg_step_score": avg_step,
            "efficiency": efficiency,
            "final_score": self.final_score(),
            "correct_actions": correct_count,
            "accuracy_pct": round(100 * correct_count / total, 1) if total else 0.0,
            "action_counts": dict(self._action_counts),
            "decision_summary": self._build_decision_summary(),
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Observation:
        if self._current_index >= len(self._emails):
            raise RuntimeError("No emails left — episode should be marked done.")
        email = self._emails[self._current_index]
        return Observation(
            current_email=email,
            inbox_remaining=len(self._emails) - self._current_index,
            step_count=self._step_count,
            task_name=self._task_config.name,
            task_difficulty=self._task_config.difficulty,   # type: ignore[arg-type]
            recent_action_history=list(self._recent_actions),
        )

    def _advance(self) -> None:
        """Move to the next email; mark done if inbox empty or steps exhausted."""
        self._current_index += 1
        self._step_count += 1
        if (
            self._current_index >= len(self._emails)
            or self._step_count >= self._max_steps
        ):
            self._done = True

    # ------------------------------------------------------------------
    # Advanced penalty system
    # ------------------------------------------------------------------

    def _apply_penalties(
        self, email: Email, action: Action, reward: Reward
    ) -> Reward:
        """
        Apply deterministic, additive penalties on top of the base reward.

        Penalties (all described in Reward.reason):
        P1 — Step overhead:   past midpoint, -0.02 per excess step
        P2 — Repeated mistake: 3+ consecutive score-0 steps → -0.10
        P3 — Overuse:         single action > 70% of all actions → -0.05
        """
        score = reward.score
        penalty_notes: List[str] = []

        # P1: Step overhead
        midpoint = self._max_steps // 2
        if self._step_count > midpoint:
            excess = self._step_count - midpoint
            overhead = round(excess * _STEP_OVERHEAD_RATE, 4)
            if overhead > 0.0:
                prev_score = score
                score = max(score - overhead, _SCORE_FLOOR)
                penalty_notes.append(
                    f"[P1] Step overhead: -{overhead:.3f} "
                    f"(step {self._step_count + 1}/{self._max_steps})."
                )

        # P2: Repeated-mistake penalty
        # (uses _consecutive_wrong BEFORE this step's update)
        if self._consecutive_wrong >= 3 and reward.score == 0.0:
            score = max(score - _REPEATED_MISTAKE_PENALTY, _SCORE_FLOOR)
            penalty_notes.append(
                f"[P2] Repeated mistake penalty: -{_REPEATED_MISTAKE_PENALTY} "
                f"({self._consecutive_wrong} consecutive zero-score steps)."
            )

        # P3: Overuse penalty
        total_actions_so_far = sum(self._action_counts.values())
        if total_actions_so_far >= 4:  # only meaningful after a few steps
            chosen_count = self._action_counts.get(action.action_type, 0)
            # Note: action_counts updated AFTER penalties, so use current count + 1
            fraction = (chosen_count + 1) / (total_actions_so_far + 1)
            if fraction > _OVERUSE_THRESHOLD:
                score = max(score - _OVERUSE_PENALTY, _SCORE_FLOOR)
                penalty_notes.append(
                    f"[P3] Action overuse: '{action.action_type}' used "
                    f"{chosen_count + 1}/{total_actions_so_far + 1} times "
                    f"({fraction:.0%} > {_OVERUSE_THRESHOLD:.0%} threshold). "
                    f"-{_OVERUSE_PENALTY}."
                )

        if not penalty_notes:
            return reward

        score = round(min(max(score, 0.0), 1.0), 6)
        combined_reason = reward.reason + " | Penalties: " + " ".join(penalty_notes)
        return Reward(
            score=score,
            reason=combined_reason,
            action_was_valid=reward.action_was_valid,
        )

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------

    def _update_tracking(
        self, email: Email, action: Action, reward: Reward
    ) -> None:
        """Update all mutable counters and context window."""
        was_correct = action.action_type == email.correct_action

        # Full history record
        self._action_history.append(
            {
                "step": self._step_count,
                "email_id": email.id,
                "email_subject": email.subject[:60],
                "email_category": email.category,
                "email_priority": email.priority,
                "correct_action": email.correct_action,
                "action_type": action.action_type,
                "response_length": len(action.response) if action.response else 0,
                "score": round(reward.score, 4),
                "reason": reward.reason,
                "action_was_valid": reward.action_was_valid,
                "was_correct": was_correct,
            }
        )

        # Context window (last N summaries, shown in Observation)
        summary_entry = RecentActionSummary(
            step=self._step_count,
            email_id=email.id,
            email_subject_snippet=email.subject[:40],
            action_type=action.action_type,
            score=round(reward.score, 4),
            was_correct=was_correct,
        )
        self._recent_actions.append(summary_entry)
        if len(self._recent_actions) > _CONTEXT_WINDOW:
            self._recent_actions.pop(0)

        # Action usage counter
        self._action_counts[action.action_type] += 1

        # Consecutive wrong tracking
        if reward.score == 0.0:
            self._consecutive_wrong += 1
        else:
            self._consecutive_wrong = 0

    def _build_info(
        self, email: Email, action: Action, reward: Reward
    ) -> Dict[str, Any]:
        """Rich diagnostics returned alongside each step (v3: added penalty_breakdown)."""
        # Extract penalty annotations from the reason string for structured logging
        reason = reward.reason
        penalty_breakdown: Dict[str, float] = {}
        import re as _re
        for tag, label in (("P1", "step_overhead"), ("P2", "repeated_mistake"), ("P3", "action_overuse")):
            match = _re.search(rf"\[{tag}\][^\[]*?(-\d+\.\d+)", reason)
            if match:
                penalty_breakdown[label] = float(match.group(1))

        return {
            "step_count": self._step_count,
            "email_id": email.id,
            "email_subject": email.subject[:60],
            "email_category": email.category,
            "email_priority": email.priority,
            "correct_action": email.correct_action,
            "chosen_action": action.action_type,
            "score": round(reward.score, 4),
            "action_was_valid": reward.action_was_valid,
            "reason": reward.reason,
            "penalty_breakdown": penalty_breakdown,
            "cumulative_score": round(self._cumulative_score, 6),
            "emails_remaining": max(0, len(self._emails) - self._current_index - 1),
            "consecutive_wrong": self._consecutive_wrong,
            "action_counts": dict(self._action_counts),
            "done": self._done,
            # Preview of efficiency metric
            "current_efficiency": round(
                max(0.0, 1.0 - (self._step_count + 1) / self._max_steps), 4
            ),
        }

    def _build_decision_summary(self) -> str:
        """Generate a short narrative summary of agent decisions so far."""
        if not self._action_history:
            return "No actions taken yet."
        total = len(self._action_history)
        correct = sum(1 for h in self._action_history if h.get("was_correct", False))
        avg_score = (
            sum(h["score"] for h in self._action_history) / total
        ) if total else 0.0
        dominant = (
            self._action_counts.most_common(1)[0][0]
            if self._action_counts
            else "none"
        )
        return (
            f"{correct}/{total} correct decisions | "
            f"avg score: {avg_score:.3f} | "
            f"dominant action: '{dominant}' ({self._action_counts.get(dominant, 0)} uses) | "
            f"consecutive wrong: {self._consecutive_wrong}"
        )
