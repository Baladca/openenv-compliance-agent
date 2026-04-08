---
title: Openenv Compliance Agent
emoji: 🌖
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
license: apache-2.0
tags:
- openenv
---


This Space implements a Content Moderation Benchmark using the OpenEnv framework. It evaluates an LLM's ability to handle three distinct safety tiers in a deterministic moderation environment.

Benchmark Performance
- Success Rate: 100% (3/3 tasks solved)
- Trajectory: `1.00, 1.00, 1.00`
- Total Reward: 3.00
- Efficiency: 3 steps total (1.0 steps per task)

Technical Implementation
- Framework: OpenEnv (Python)
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Inference: OpenAI-compatible client via Hugging Face Router.
- Resilience: Features a heuristic fallback layer to ensure environment completion even during high latency or credit depletion.

Task Specification
1. Task 1 (Easy): Spam/Scam detection -> Action: `reject`
2. Task 2 (Medium): PII Redaction -> Action: `redact` (Target: Email string)
3. Task 3 (Hard): Security Threat -> Action: `escalate` (Target: Firewall bypass)

---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference