# Enterprise Agent: On-Premise Function-Calling AI Assistant

An on-premise AI agent that handles everyday enterprise tasks from natural-language requests. A small LLM (Qwen3-4B-Instruct-2507) is fine-tuned for tool calling with Llama-Factory, quantized to GGUF, served by Ollama, and connected to real backend actions through FastAPI, n8n, and a SQL database.

> **Status:** Work in progress. See [Roadmap](#roadmap).

## Problem

Employees spend time every day on small administrative tasks: checking leave balances, applying for leave, booking meeting rooms, opening IT tickets, looking up orders, and searching SOPs. Each task lives in a different system with its own login and workflow.

The usual LLM approaches don't fit:

| Approach | Problem |
|---|---|
| Cloud LLMs (ChatGPT, etc.) | Employee data, orders, and internal SOPs cannot leave the company network for security and compliance reasons |
| Large on-prem models (70B class) | GPU cost is too high for a typical department |
| Off-the-shelf small models on-prem | Unreliable tool calling: malformed JSON, wrong tool, missing or invented arguments, and tools triggered during small talk |

## Goal

Show that a **4B model, after fine-tuning, can run on-prem at low cost and call tools reliably**, and that it is wired to a real backend that completes the task instead of only chatting.

The claim is backed by numbers: base vs fine-tuned on JSON validity, tool accuracy, argument accuracy, false trigger rate, and latency, with only a small accuracy drop after Q4_K_M quantization.

## Features

Users describe a task in plain language. The agent picks the right tool, generates valid JSON arguments, and the backend executes the action.

| Tool | Description | Backend action |
|---|---|---|
| `get_leave_balance` | Check remaining leave | Read database |
| `apply_leave` | Submit a leave request | Write database + notify manager via n8n |
| `book_meeting_room` | Book a meeting room | Check conflicts, write database |
| `create_it_ticket` | Open an IT ticket | Write database + send email via n8n |
| `query_order_status` | Look up order status | Read database |
| `search_sop` | Search company SOPs and policies | RAG over Qdrant |

Example:

```
User:  Book the 3F meeting room tomorrow from 2 to 3 PM for the project weekly meeting
Agent → book_meeting_room({"room": "3F", "date": "2026-10-07", "start": "14:00", "end": "15:00", "title": "Project weekly meeting"})
```

## Pipeline

```text
0. Define tools  →  1. Build dataset    →  2. Baseline eval
                                               ↓
5. Export/quantize ← 4. Fine-tuned eval  ←  3. QLoRA fine-tune
   ↓
6. Backend       →  7. Deployment       →  8. CI
```

### 0. Define tools

- `data/tools_schema.json` defines all six tools: name, description, argument types, and required fields.
- It is the single source of truth for data synthesis, training, evaluation, and the backend implementation.

### 1. Build the dataset

- **Synthetic data (main source):** Traditional Chinese conversations for the six tools, covering:
  - Normal requests: "明天下午兩點訂 3F 會議室"
  - Missing arguments → the model asks a clarifying question instead of guessing
  - **No-tool cases:** small talk and unrelated questions, to keep the false trigger rate low
  - Multi-turn: after a tool result comes back, the model replies in natural language
- **glaive-function-calling-v2 (supporting):** a subset that keeps general function-calling ability and prevents overfitting to the six tools. It is English, so its share stays small.
- Everything is converted to Llama-Factory sharegpt format with a `tools` field.
- A held-out **test set** is split off first and never used in training.

### 2. Baseline evaluation

- Run the un-tuned Qwen3-4B-Instruct-2507 on the test set and record the numbers.
- Fine-tuning then targets where the base model is actually weak (e.g. Chinese relative dates, false triggers).

### 3. QLoRA fine-tuning

- Llama-Factory with `template: qwen3_nothink`; configs live in `configs/`.
- Loss curves and hyperparameters are recorded.
- Trained locally on an 8 GB laptop GPU: 4-bit quantization, batch size 1, gradient accumulation 8, `cutoff_len` 2048.

### 4. Fine-tuned evaluation

- Same test set, same script (`eval/run_eval.py`), results reported side by side with the baseline.

### 5. Export and quantize

- Merge LoRA → convert to GGUF with llama.cpp → quantize to Q4_K_M.
- Write a Modelfile whose tool template matches the training template → `ollama create`.
- Evaluate again to measure the accuracy cost of quantization.

### 6. Backend

- **FastAPI + LangGraph:** user message → Ollama → parse tool call → execute tool → return result to the model → reply to the user.
- **Database:** SQLAlchemy, using SQLite for development and MariaDB in production (only the connection string changes).
- **n8n:** email and manager notifications.
- **`search_sop`:** RAG with Qdrant in local mode (no separate server).

### 7. Deployment

- All services run natively on Windows without Docker (see [Quick Start](#quick-start)).
- A Docker Compose file is planned as an optional one-command deployment.

### 8. CI

- GitHub Actions runs ruff and pytest.

## Architecture

```text
            ┌──────────────┐
User ─────▶ │   FastAPI    │ ── LangGraph agent workflow
            └──────┬───────┘
                   │
     ┌─────────────┼──────────────┬─────────────────┐
     ▼             ▼              ▼                 ▼
  Ollama      SQLite / MariaDB   n8n           Qdrant (local)
  (fine-tuned   leave, rooms,    email,        SOP documents
   Qwen3-4B     tickets, orders  notifications  for RAG
   Q4_K_M)
```

## Tech Stack

- **LLM serving:** Ollama, Open WebUI (optional chat UI)
- **Base model:** Qwen3-4B-Instruct-2507
- **Fine-tuning:** Llama-Factory (QLoRA), llama.cpp (GGUF quantization)
- **Backend:** Python, FastAPI, LangGraph, SQLAlchemy
- **Automation:** n8n
- **Data:** SQLite (dev) / MariaDB (prod), Qdrant
- **Infra:** GitHub Actions (ruff + pytest), Docker Compose (planned)

## Evaluation

Base and fine-tuned models are compared on the same held-out test set.

| Model | JSON validity | Tool accuracy | Argument accuracy | False trigger rate | Latency |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 base | | | | | |
| Qwen3-4B-Instruct-2507 fine-tuned | | | | | |
| Qwen3-4B-Instruct-2507 fine-tuned Q4_K_M | | | | | |

## Roadmap

**Phase 1: end-to-end MVP.** Prove the whole chain works with a small dataset; most integration issues (data format, chat template, GGUF conversion, Ollama tool template) show up here.

- [ ] Define `tools_schema.json`
- [ ] Build ~50 samples and convert to sharegpt format
- [ ] Baseline evaluation
- [ ] QLoRA fine-tune
- [ ] Fine-tuned evaluation
- [ ] Export to GGUF Q4_K_M and run in Ollama

**Phase 2: scale up and build the app.**

- [ ] Expand the dataset to 1,000+ samples and tune hyperparameters
- [ ] FastAPI + LangGraph backend with database, n8n, and RAG
- [ ] GitHub Actions CI
- [ ] Optional Docker Compose deployment

## Project Structure

```text
enterprise-agent/
├── .github/workflows/ci.yml
├── app/                    # FastAPI agent service
│   ├── main.py
│   ├── agent.py            # LangGraph workflow
│   ├── tools/              # Tool implementations
│   └── db.py               # SQLAlchemy (SQLite / MariaDB)
├── db/
│   ├── schema.sql
│   └── seed.sql
├── n8n/workflows/          # Exported n8n workflows
├── data/
│   ├── tools_schema.json
│   ├── synth/              # Synthetic data scripts
│   ├── sop/                # Sample SOP documents for RAG
│   └── dataset_info.json
├── configs/                # Llama-Factory configs
├── eval/run_eval.py
├── export/Modelfile
└── tests/
```

## Quick Start

Runs natively on Windows; no Docker required.

**Requirements:** Python 3.11+, Node.js (for n8n), [Ollama](https://ollama.com), and an NVIDIA GPU for fine-tuning (RTX 50-series needs a PyTorch build with CUDA 12.8+).

```powershell
git clone https://github.com/<your-username>/enterprise-agent.git
cd enterprise-agent

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# Load the fine-tuned model into Ollama
ollama create enterprise-agent -f export/Modelfile

# Start the API
uvicorn app.main:app --reload

# Optional: start n8n in another terminal
npx n8n
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| n8n | http://localhost:5678 |
| Ollama | http://localhost:11434 |

## Data Notice

All data in this repository is synthetic. No real employee, order, or company SOP data is used.
