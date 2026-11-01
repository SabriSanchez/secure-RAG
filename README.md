<div align="center">

# 🐉 SecureRAG

### A local, privacy-first RAG system that won't leak your secrets to the LLM
### Like a dragon that guards your secrets. The LLM can see the cave, but never the gold.

**Hybrid BM25 + vector retrieval · Secret masking before generation · AST-based SAST on ingestion**

<br>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Qwen3-0.6B-FFD21E?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

![OWASP](https://img.shields.io/badge/OWASP_LLM_Top_10-LLM02-000000?style=flat-square&logo=owasp&logoColor=white)
![CWE](https://img.shields.io/badge/SAST-CWE_mapped-blue?style=flat-square)
![Tests](https://img.shields.io/badge/tests-33_passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)
![mypy](https://img.shields.io/badge/mypy-strict-2A6DB2?style=flat-square)
![flake8](https://img.shields.io/badge/lint-flake8-informational?style=flat-square)

[Why it matters](#-why-it-matters) •
[Security](#-security-layer) •
[Architecture](#%EF%B8%8F-architecture) •
[Quick start](#-quick-start) •
[Usage](#-usage) •
[Testing](#-testing--code-quality)

</div>

---

## 🎯 Why it matters

*Every codebase has treasure hidden inside: API keys, passwords, tokens. SecureRAG is the dragon sleeping on top of it.*

Retrieval-Augmented Generation pulls real files into an LLM's prompt. Real codebases contain **hardcoded API keys, private keys, database URLs and tokens**, so a naive RAG pipeline can send them straight to the model, which may then repeat them in its answer.

SecureRAG answers technical questions about the **[vLLM](https://github.com/vllm-project/vllm) codebase** fully offline on CPU, and treats every retrieved chunk as **untrusted** until it has been sanitized.

| | |
|---|---|
| 🔐 **Secret masking** | 13 detectors catch and mask credentials in retrieved context *before* it reaches the LLM ([OWASP LLM02: Sensitive Information Disclosure](https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/)) |
| 🧪 **Static analysis** | Every Python file is parsed into an AST at ingestion and flagged for dangerous sinks, mapped to **CWE IDs** |
| 🔎 **Hybrid search** | BM25 (keywords) + MiniLM embeddings (meaning), fused with **Reciprocal Rank Fusion** |
| 💻 **100% local** | No API calls, no data leaves the machine. Runs on CPU with Qwen3-0.6B |
| 📏 **Measured** | Built-in Recall@k evaluator against a ground-truth question set |

---

## 🔐 Security layer

### 1. `SecuritySanitizer`: credential masking (OWASP LLM02)

**The Hoard Keeper**.*Only the gold is hidden. The rest of the cave stays visible, so the LLM still understands the code.*

Every chunk passes through the sanitizer right before it goes into the prompt. Only the **secret itself** is replaced, so the LLM still sees useful context (variable names, URL host, quotes):

```diff
- password = "Sup3rS3cret!"
+ password = "[MASKED_CREDENTIAL]"

- postgres://admin:hunter2@db.local:5432/app
+ postgres://admin:[MASKED_URL_PASSWORD]@db.local:5432/app

- Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIi...
+ Authorization: Bearer [MASKED_JWT_TOKEN]
```

<details>
<summary><b>📋 Supported detectors (click to expand)</b></summary>
<br>

| Detector | Example of what it catches |
|---|---|
| `PRIVATE_KEY` | RSA / OpenSSH / PKCS#8 / encrypted / PGP private key blocks |
| `PRIVATE_KEY_TAIL` | The **second half** of a key that the chunker split across two chunks |
| `URL_PASSWORD` | `scheme://user:password@host` |
| `ANTHROPIC_KEY` | `sk-ant-...` |
| `OPENAI_KEY` | `sk-...`, `sk-proj-...`, `sk-svcacct-...` |
| `STRIPE_KEY` | `sk_live_...`, `rk_test_...` |
| `GITHUB_TOKEN` | `ghp_...`, `gho_...`, `github_pat_...` |
| `SLACK_TOKEN` | `xoxb-...`, `xoxp-...` |
| `GOOGLE_API_KEY` | `AIza...` |
| `AWS_KEY` | `AKIA...` / `ASIA...` access key IDs |
| `JWT_TOKEN` | `eyJ...`.`eyJ...`.`...` |
| `CREDENTIAL` | `password = "..."`, `"client_secret": "..."`, `aws_secret_access_key = '...'` |
| `ENV_SECRET` | `export DB_PASSWORD=...`, `.env`-style `*_TOKEN=...` lines |

</details>

**Design decisions worth noting:**

- 🧩 **Chunk-boundary aware.** A PEM key cut in half by the chunker is still caught: the head is matched up to the end of the chunk, and a dedicated `_TAIL` pattern catches the orphaned second half. *Cut a key in two, and the dragon still finds both halves.*
- 🎯 **Ordered from specific to generic.** Specific token formats run before the generic `key = "value"` rules, so each secret gets the most precise label and is masked exactly once.
- 🚫 **Low false positives.** Code that only *looks* like a secret is left alone: `tokenizer = "Qwen/Qwen3-0.6B"`, `eos_token = "<|endoftext|>"`, `api_key = os.environ[...]`, `MAX_TOKENS=100000`. These are covered by negative tests. *A good dragon doesn't burn every visitor. tokenizer = "Qwen..." is a friend, not a thief.*

### 2. `SASTScanner`: static analysis on ingestion

*Dragon Eyes. "eval() and shell=True are fire. The dragon knows where the fire is.*

While indexing, every file is scanned and findings are logged as `[SAST Alert]`. Python files are parsed with `ast` and walked node by node, so it detects real **calls**, not just strings. Non-Python files, or Python that fails to parse, fall back to regex rules.

| Finding | CWE |
|---|---|
| `eval()`, `exec()`, `compile()`, `__import__()` | [CWE-95](https://cwe.mitre.org/data/definitions/95.html): Eval Injection |
| `os.system()`, `os.popen()`, `shell=True` | [CWE-78](https://cwe.mitre.org/data/definitions/78.html): OS Command Injection |
| `pickle.load()` / `pickle.loads()` | [CWE-502](https://cwe.mitre.org/data/definitions/502.html): Deserialization of Untrusted Data |
| `yaml.load()` without `SafeLoader` | [CWE-20](https://cwe.mitre.org/data/definitions/20.html): Improper Input Validation |
| `subprocess.run/call/Popen` | Flagged for manual review |

### 3. Defensive prompt design

- The system prompt tells the model to answer **only from the provided context** and to say it doesn't know otherwise, which reduces hallucination.
- Context is packed into a **strict token budget**: highest-ranked chunks first, lowest-ranked dropped, so the input can't overflow the model.
- Pydantic models validate every input and output (questions, sources, answers).

---

## 🗺️ Map of the Lair (Architecture)

```mermaid
flowchart LR
    subgraph Ingestion["📥 Ingestion · index"]
        A[vLLM source<br/>.py .md .rst .txt] --> S{{🧪 SAST Scanner<br/>AST + regex}}
        S -->|CWE alerts| L[(logs)]
        S --> C[Chunker<br/>Python- and Markdown-aware<br/>with overlap]
        C --> B[(BM25 index)]
        C --> V[(Vector index<br/>MiniLM-L6-v2)]
    end

    subgraph Query["🔎 Retrieval · search"]
        Q[User question] --> B2[BM25]
        Q --> V2[Vector search]
        B2 --> R[Reciprocal Rank<br/>Fusion]
        V2 --> R
        R -.optional.-> X[Cross-encoder<br/>re-ranker]
    end

    subgraph Gen["🤖 Generation · answer"]
        R --> SAN{{🔐 Security<br/>Sanitizer}}
        X -.-> SAN
        SAN --> P[Token-budgeted<br/>prompt]
        P --> LLM[Qwen3-0.6B<br/>local CPU]
        LLM --> OUT[JSON answer<br/>+ sources]
    end

    B -.-> B2
    V -.-> V2
```

**Retrieval details**

- **Structure-aware chunking.** Python is split on `class` / `def` boundaries and Markdown on headings, then recursively down to a max size with overlap. Exact character offsets are kept so every answer cites its source precisely.
- **Weighted RRF.** Both rankings are fused with `score = Σ w / (60 + rank)`, with BM25 weighted higher because exact identifiers matter a lot in code search.
- **Optional re-ranking** with `cross-encoder/ms-marco-MiniLM-L-6-v2` (disabled by default to keep CPU latency low).

**Results:** benchmarked with the built-in `evaluate` command against a private local test set of questions with ground-truth sources (not included in this repo). It meets the Recall@5 targets of **≥ 80 % on documentation** and **≥ 50 % on code** questions.

---

## 🚀 Quick start

### *🥚 Hatch the Dragon (make install = hatching the egg, index = "the dragon learns its cave")*

**Requirements:** Python 3.12 and [`uv`](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/SabriSanchez/secure-RAG.git
cd secure-RAG
make install                 # uv sync --extra dev
```

Put the corpus you want to index (e.g. a clone of vLLM) in `data/raw/`, then:

```bash
make run ARGS="index"        # chunk + SAST scan + build BM25 and vector indexes
make run ARGS="search 'How does PagedAttention allocate KV cache blocks?'"
```

> 💡 Models (MiniLM, Qwen3-0.6B) are downloaded from Hugging Face on first use. After that everything runs offline.

---

## 💻 Usage

The CLI is built with [Fire](https://github.com/google/python-fire). Each method of `RAGCLI` is a command:

| Command | What it does |
|---|---|
| `index [--max_chunk_size 2000]` | Ingest `data/raw/`, run SAST, build both indexes in `data/processed/` |
| `search "<query>" [--k 5]` | Show the top-k chunks with file path, offsets and a snippet |
| `answer "<query>" [--k 3]` | Retrieve → **sanitize** → generate an answer with Qwen3 (JSON output) |
| `search_dataset <questions.json> [--k 5]` | Batch retrieval for a question set |
| `answer_dataset <search_results.json>` | Batch answer generation from saved retrieval results |
| `evaluate <search_results.json> <ground_truth.json>` | Compute Recall@k against ground truth |

<details>
<summary><b>Example: <code>answer</code> output</b></summary>

```json
{
  "question_id": "…",
  "question": "How does vLLM schedule requests?",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm/core/scheduler.py",
      "first_character_index": 10240,
      "last_character_index": 12187
    }
  ],
  "answer": "…"
}
```

</details>

---

## 🔥 Trial by Fire: Testing & code quality

```bash
make test          # pytest
make lint          # flake8 + mypy
make lint-strict   # flake8 + mypy --strict
```

The security test suite (`tests/test_security.py`, **33 tests**) checks both sides:

- ✅ **21 positive cases.** Every secret type is masked **and** the raw secret is gone from the output, including private keys split by the chunker.
- ❌ **9 negative cases.** Normal code and docs (tokenizer names, env lookups, numeric config, prose about passwords) stay **exactly unchanged**.
- 🎯 **Precision checks.** Surrounding context is kept, each secret is masked once, and the mask count is accurate.

---

## 📁 Project structure

```
src/
├── main.py                  # Fire CLI entrypoint (RAGCLI)
├── cli/                     # index · search · answer · evaluate commands
├── indexing/
│   ├── security.py          # 🔐 SecuritySanitizer + 🧪 SASTScanner
│   ├── chunker.py           # Python- and Markdown-aware recursive chunking
│   ├── bm25_indexer.py
│   └── vector_indexer.py
├── retrieval/
│   ├── bm25_retriever.py
│   ├── vector_retriever.py
│   └── hybrid_retriever.py  # RRF fusion + optional cross-encoder
└── general_utils/           # Pydantic models, parsing, text helpers
tests/
└── test_security.py
```

---

## 🧠 What I learned

- **LLM security is a data-flow problem.** The safest place to stop a leak is the boundary between retrieval and the prompt, not the model's output.
- **Detection is a precision/recall trade-off.** Catching secrets is easy; *not* masking `tokenizer = "..."` in a codebase full of tokenizers takes careful patterns and negative tests.
- **Chunking creates security edge cases.** Splitting text for retrieval can split a private key in half, so the sanitizer has to handle fragments.
- **AST beats grep for SAST.** Parsing code finds real calls and ignores comments and strings that only mention `eval`.

---

<div align="center">

### 👋 About me

I'm a **cybersecurity student** interested in **AI/LLM security, application security and secure software development**, and I'm looking for my first role in the field.

[![GitHub](https://img.shields.io/badge/GitHub-SabriSanchez-181717?style=for-the-badge&logo=github)](https://github.com/SabriSanchez)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sabri-sanchez/)

<sub>⭐ Leave a star. Dragons love shiny things!</sub>

*Here be dragons. 🐉 (old maps wrote this on dangerous, unknown places)*

</div>
