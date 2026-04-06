---
title: InboxZeroEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# InboxZeroEnv

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)
![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)
![Version](https://img.shields.io/badge/Version-3.0.0-purple)

> **Production-grade, OpenEnv-compatible reinforcement learning environment for email triage.**

InboxZeroEnv simulates realistic human email triage workflows (Gmail/Outlook-style). An LLM agent processes a sequence of emails and must make deterministic, high-quality triage decisions across three progressively difficult tasks. Designed to rigorously evaluate LLM decision-making on real-world asymmetric classification problems.

---

## Overview

| Property | Value |
|---|---|
| API Contract | OpenEnv (`reset`, `step`, `state`) + Gymnasium (`render`, `seed`) |
| Determinism | ✅ 100% reproducible — no randomness anywhere |
| Data Models | Pydantic v2 (immutable, fully validated) |
| Grading | Dense partial-credit, 10-group keyword scoring, semantic proximity |
| Tasks | 3 (EASY / MEDIUM / HARD) |
| Dataset | **35 emails** across 10 categories (incl. 5 adversarial edge cases) |
| Python | 3.11+ |
| Docker | ✅ Included |
| HuggingFace Spaces | ✅ Compatible |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent (LLM)                          │
│         CoT Reasoning → JSON Output → parse_action()        │
└──────────────────────────┬──────────────────────────────────┘
                           │  Action(action_type, response)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   InboxZeroEnv  (email_env.py)              │
│  reset() ──► step() ──► state() ──► render() ──► summary() │
│                           │                                  │
│           ┌───────────────┼──────────────────┐              │
│           │     Penalty System (P1/P2/P3)     │              │
│           └───────────────┬──────────────────┘              │
│                           ▼                                  │
│                    Grader (grader.py)                        │
│    EasyGrader / MediumGrader / HardGrader                   │
│    10 keyword groups + semantic proximity bonus             │
└──────────────────────────┬──────────────────────────────────┘
                           │  (Reward, info)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Dataset (data/emails.json)                     │
│   35 emails: 10 categories, 5 adversarial edge cases        │
│   Deterministically ordered by ID — no shuffling           │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
email-env/
├── env/
│   ├── __init__.py       # Package exports
│   ├── models.py         # Pydantic data models (Email, Action, Observation, Reward, …)
│   ├── email_env.py      # Core InboxZeroEnv class — reset/step/state/render/seed
│   ├── grader.py         # Deterministic grader — 10 keyword groups, semantic proximity
│   └── tasks.py          # Task configs + grader classes (Easy / Medium / Hard)
├── data/
│   └── emails.json       # 35 realistic emails (5 adversarial edge cases)
├── inference.py          # Baseline inference: CoT + few-shot, per-category stats
├── openenv.yaml          # OpenEnv v3 configuration
├── Dockerfile            # Production Docker image
├── requirements.txt      # Python dependencies (pydantic, openai, pyyaml)
└── README.md             # This file
```

---

## Quick Start

```bash
pip install -r requirements.txt

export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."
python inference.py
```

Results are printed to stdout and saved to `results.json` (includes per-category accuracy).

---

## Environment API

```python
from env import InboxZeroEnv, Action

# Create with task name
env = InboxZeroEnv(task_name="hard")   # "easy" | "medium" | "hard"

# Load directly from config file (v3)
env = InboxZeroEnv.from_config("openenv.yaml", task_name="hard")

# Reset — returns first Observation
obs = env.reset()

# Step — returns (Observation | None, Reward, done, info)
action = Action(
    action_type="reply",
    response="Thank you for your email. I confirm I will attend the Q3 planning session this Friday at 2pm."
)
obs, reward, done, info = env.step(action)

print(f"Score: {reward.score:.4f} | {reward.reason}")
print(f"Penalties: {info['penalty_breakdown']}")   # v3: structured penalty log
print(f"Done: {done}")

# Human-readable state table (v3: gymnasium-compatible render)
env.render()

# Full internal state (serialisable)
state = env.state()

# Episode summary with per-category accuracy
summary = env.summary()
print(f"Final score: {summary['final_score']:.4f}")

# Gymnasium compatibility (no-op — env is always deterministic)
env.seed(42)
```

---

## Data Models

### Email
```python
class Email(BaseModel):
    id: int
    subject: str
    sender: str
    body: str
    priority: Literal["low", "medium", "high"]
    is_spam: bool
    requires_response: bool
    deadline: Optional[int]       # steps until deadline becomes critical
    correct_action: Literal["delete", "archive", "mark_important", "reply"]
    category: str                 # e.g. "spam", "urgent_work", "newsletter"
```

### Action
```python
class Action(BaseModel):
    action_type: Literal["delete", "archive", "mark_important", "reply"]
    response: Optional[str]       # Required when action_type == "reply"
```

### Observation
```python
class Observation(BaseModel):
    current_email: Email
    inbox_remaining: int
    step_count: int
    task_name: str
    task_difficulty: Literal["easy", "medium", "hard"]
    recent_action_history: List[RecentActionSummary]   # last 5 decisions
```

### Reward
```python
class Reward(BaseModel):
    score: float              # strictly in [0.0, 1.0]
    reason: str               # human-readable with ✓/✗/~ prefix
    action_was_valid: bool
```

---

## Tasks

### 🟢 EASY — SpamSentinel

**Objective:** Binary spam classification. Delete spam, preserve everything else.

| Property | Value |
|---|---|
| Emails | 20 (spam + phishing + promotion + newsletter + notification + billing) |
| Max steps | 15 |
| Scoring | Binary: 1.0 (correct) / 0.0 (wrong) |
| Challenge | Promotional emails from legitimate senders vs. spam |

### 🟡 MEDIUM — PriorityTriageDesk

**Objective:** Priority-aware inbox triage with partial credit.

| Property | Value |
|---|---|
| Emails | 26 (+ work + meeting_request) |
| Max steps | 20 |
| Scoring | 1.0 (correct), 0.3–0.7 (semantically close), 0.0–0.2 (wrong) |
| Challenge | Distinguishing archive vs. mark_important vs. reply based on priority |

### 🔴 HARD — FullInboxZero

**Objective:** Full pipeline with strict scoring, adversarial emails, and reply generation.

| Property | Value |
|---|---|
| Emails | 33 (+ urgent_work + customer_complaint) |
| Max steps | 30 |
| Scoring | Strict zeros for critical failures; dense partial credit elsewhere |
| Challenge | 5 adversarial edge-case emails; reply quality scored on 10 keyword groups |

---

## Reward System

### Per-Step Scoring

| Outcome | Score |
|---|---|
| Correct action | 1.0 |
| Close (mark_important ↔ archive) | 0.4–0.5 |
| Missed reply, non-critical | 0.1–0.2 |
| Incorrect action | 0.05–0.2 |
| Critical failure (spam kept, legit deleted, urgent reply missed) | **0.0** |
| Invalid action (structural error) | **0.0** |

### Final Score Formula

```
avg_step_score = Σ step_scores / total_emails
efficiency     = 1 − (steps_taken / max_steps)
final_score    = avg_step_score × 0.80 + efficiency × 0.20
```

### Penalty System

| ID | Penalty | Trigger | Amount |
|---|---|---|---|
| P1 | Step overhead | After midpoint | −0.02 per excess step |
| P2 | Repeated mistake | 3+ consecutive zero-score steps | −0.10 |
| P3 | Action overuse | Single action_type > 70% of all actions | −0.05 |

Penalties are logged in `info["penalty_breakdown"]` for full transparency.

---

## Grader (v3)

The grader (`env/grader.py`) is **100% deterministic** — pure keyword matching, length checks, and rule-based logic. No LLM involved.

### 10 Semantic Keyword Groups

| Group | Example Keywords |
|---|---|
| Acceptance | confirmed, attending, will attend |
| Acknowledgement | received, understood, noted |
| Commitment | will, shall, asap, right away |
| Gratitude | thank, appreciate, grateful |
| Meeting | meeting, schedule, calendar, invite |
| Urgency | urgent, emergency, incident |
| Approval | approved, sign-off, lgtm, looks good |
| **Empathy** *(new in v3)* | sorry, apologize, understand your frustration |
| **Resolution** *(new in v3)* | resolve, refund, replacement, escalate |
| **Timeline** *(new in v3)* | by EOD, within 24 hours, by tomorrow |

### Reply Scoring Tiers

| Length | Keyword Groups Matched | Base Score | + Proximity Bonus |
|---|---|---|---|
| < 40 chars | — | 0.10 | — |
| ≥ 40 chars | 0 | 0.25 | +0.05 if references subject/sender |
| ≥ 40 chars | 1–2 | 0.55 | +0.05 if references subject/sender |
| ≥ 40 chars | 3–5 | 0.80 | +0.05 if references subject/sender |
| ≥ 40 chars | 6+ | 1.00 | (already capped) |

**Semantic Proximity Bonus (v3):** An additional +0.05 is awarded when the reply references words from the original email's subject or sender name, confirming the agent actually read the email.

---

## Adversarial Edge Cases (v3)

Five emails were carefully designed to challenge naive classification:

| ID | Description | Tricky Because |
|---|---|---|
| 31 | Phishing posing as internal IT (Microsoft 365 verification) | Sender domain resembles corporate IT |
| 32 | Urgent API design decision via reply-chain follow-up | Reads like FYI but requires a decision |
| 33 | Legitimate newsletter from The Pragmatic Engineer | Known brand, not spam — must archive not delete |
| 34 | Overdue invoice escalating to collections | Needs a dispute reply, not just archive |
| 35 | Successful CI/CD pipeline notification | Automated but legitimate — archive not delete |

---

## Inference Script (v3)

The baseline agent uses chain-of-thought reasoning with few-shot examples:

### Chain-of-Thought Prompting
```
<thinking>
The domain "paypal-alerts.net" is not PayPal's official domain. This is phishing.
The correct action is "delete".
</thinking>
{"action_type": "delete", "response": null}
```

### Validation Retry
If the agent selects `reply` but produces no response text, the script automatically retries with an explicit reminder before falling back to `mark_important`.

### Per-Category Output
`results.json` includes accuracy and average score broken down by email category:
```json
"global_per_category": {
  "spam":              {"accuracy_pct": 100.0, "avg_score": 1.0, "total": 8},
  "urgent_work":       {"accuracy_pct":  80.0, "avg_score": 0.87, "total": 5},
  "customer_complaint":{"accuracy_pct": 100.0, "avg_score": 0.72, "total": 2}
}
```

---

## Docker

```bash
# Build
docker build -t inboxzeroenv:latest .

# Run
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  inboxzeroenv:latest
```

---

## Dataset

35 realistic emails across 10 categories:

| Category | Count | Correct Action |
|---|---|---|
| spam | 4 | delete |
| phishing | 4 | delete |
| promotion | 2 | delete |
| newsletter | 4 | archive |
| notification | 4 | archive |
| billing | 3 | archive / mark_important |
| work | 5 | reply / mark_important |
| meeting_request | 3 | reply / mark_important |
| urgent_work | 8 | reply |
| customer_complaint | 2 | reply |
| **Total** | **35** | |

---

## HuggingFace Spaces Deployment

1. Create a new HuggingFace Space (Docker SDK).
2. Upload all project files (or push via `git`).
3. Add secrets in Space Settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
4. The Space will auto-build the Docker image and run `inference.py`.
5. Logs and `results.json` will be available in the Space output.

---

## Determinism Guarantees

- Email ordering is fixed (sorted by ID — no shuffling).
- `temperature=0.0` in all LLM calls.
- Grader uses only deterministic rule matching (no LLM, no embeddings).
- No random seeds, no stochastic components.
- Same model + same inputs → identical scores across all runs and environments.

---

## License

MIT
