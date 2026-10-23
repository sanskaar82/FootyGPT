# FootyGPT — Technical Design Document

| Field | Value |
|---|---|
| **Document status** | Draft — awaiting approval |
| **Implementation status** | Not started. Nothing in this document has been built or tested. |
| **Date** | 2026-10-23 |
| **Scope** | Initial version (v1) of FootyGPT |

> **Important:** Everything described below is a **proposal**. No code, configuration, database, or tests exist yet. Every feature, endpoint, table, and command is **planned**, not implemented. Example requests and responses are illustrative and have not been produced by a running system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [User Request Flow](#4-user-request-flow)
5. [Frontend Design](#5-frontend-design)
6. [Backend API Design](#6-backend-api-design)
7. [Database Design](#7-database-design)
8. [AI Integration](#8-ai-integration)
9. [Data Validation](#9-data-validation)
10. [Error Handling and Security](#10-error-handling-and-security)
11. [Proposed Project Structure](#11-proposed-project-structure)
12. [Testing Strategy](#12-testing-strategy)
13. [Design Decisions and Trade-offs](#13-design-decisions-and-trade-offs)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Assumptions and Open Questions](#15-assumptions-and-open-questions)

---

## 1. Project Overview

### What is FootyGPT?

FootyGPT is a **planned** AI-powered football (soccer) companion. Users type questions in plain English and receive clear, conversational answers about:

- **Rules** — e.g. "When is a player offside?"
- **Tactics** — e.g. "What is a high press?"
- **Formations** — e.g. "How does a 4-3-3 differ from a 4-4-2?"
- **Players** — e.g. "Why is Johan Cruyff considered so influential?"
- **History** — e.g. "Who won the first World Cup?"

### What problem does it solve?

Football knowledge is spread across rulebooks, articles, videos, and forums. Newcomers often find the rules and tactical vocabulary confusing, and general search results can be cluttered or too technical. FootyGPT aims to provide **one simple place** to ask football questions and get beginner-friendly explanations, with the ability to ask follow-up questions in the same conversation.

### Who is it for?

- New fans learning the game.
- Casual fans who want to understand tactics and terminology.
- Anyone curious about football history and famous players.

### What is out of scope for v1?

- Live scores, fixtures, or real-time match data.
- User accounts and login.
- Betting advice or predictions.
- Mobile apps.

---

## 2. Technology Stack

| Technology | Role | Why it is used (plain-language explanation) |
|---|---|---|
| **Python** | Programming language | One language for the whole project. Readable, beginner-friendly, and has excellent libraries for web and AI work. |
| **Streamlit** | Frontend / user interface | Builds a web interface using only Python — no HTML, CSS, or JavaScript required. Has built-in chat components. |
| **FastAPI** | Backend API | A web framework that receives requests from the frontend, runs the application logic, and returns responses. Generates interactive API documentation automatically. |
| **SQLite** | Database | Stores conversations and messages in a single file on disk. No separate database server to install or run. |
| **SQLAlchemy** | Database interaction (ORM) | Lets us work with database rows as Python objects instead of writing raw SQL. Makes switching to another database later easier. |
| **Pydantic** | Request and response validation | Defines the exact shape of data the API accepts and returns, and automatically rejects invalid data with clear error messages. FastAPI uses it natively. |
| **OpenRouter API** | Access to AI models | A single API that gives access to many large language models (LLMs) from different providers. Lets us change the model via configuration without rewriting code. |
| **pytest** | Automated testing | The most widely used Python testing tool. Simple syntax for writing tests that check the application behaves correctly. |

**Supporting libraries (proposed, to be confirmed):**

| Library | Purpose |
|---|---|
| `uvicorn` | Server that runs the FastAPI application. |
| `httpx` | HTTP client used by the backend to call OpenRouter and by the frontend to call the backend. Also used in tests. |
| `pydantic-settings` | Loads configuration (API key, model name, etc.) from environment variables. |
| `python-dotenv` | Loads environment variables from a local `.env` file during development. |

---

## 3. System Architecture

### Architecture Diagram

```
                        ┌──────────────────────┐
                        │         User         │
                        │    (web browser)     │
                        └──────────┬───────────┘
                                   │  types question / reads answer
                                   ▼
                        ┌──────────────────────┐
                        │  Streamlit Frontend  │
                        │  (user interface)    │
                        └──────────┬───────────┘
                                   │  HTTP + JSON  (e.g. POST /chat)
                                   ▼
                        ┌──────────────────────┐
                        │   FastAPI Backend    │
                        │  - validation        │
                        │  - business logic    │
                        │  - AI service        │
                        └─────┬──────────┬─────┘
            SQLAlchemy (ORM)  │          │  HTTPS + JSON
                              ▼          ▼
               ┌──────────────────┐   ┌──────────────────────┐
               │  SQLite Database │   │    OpenRouter API    │
               │  (footygpt.db)   │   │          │           │
               │  - conversations │   │          ▼           │
               │  - messages      │   │  Selected AI Model   │
               └──────────────────┘   └──────────────────────┘
```

### Component Responsibilities

| Component | Responsibilities | Why it is needed |
|---|---|---|
| **User** | Asks football questions and reads answers in a web browser. | The person the application serves. |
| **Streamlit Frontend** | Displays the chat interface, collects input, shows answers, loading states, and errors. Talks **only** to the backend. | Gives users a friendly interface without needing web-development skills to build it. |
| **FastAPI Backend** | Validates requests, coordinates database reads/writes, builds the prompt, calls OpenRouter, handles errors, and returns structured responses. | Keeps logic and secrets (API key) **out of the frontend**. Creates a clear, testable API that a different frontend could reuse later. |
| **SQLite Database** | Persists conversations and messages. | Lets users revisit past conversations and lets the AI see earlier messages for follow-up questions. |
| **OpenRouter API** | Forwards the prompt to the configured AI model and returns its answer. | Provides the AI capability. Using OpenRouter means one integration supports many models. |

### Key Communication Rules

1. The frontend **never** calls OpenRouter or the database directly.
2. The OpenRouter API key exists **only** in the backend's environment.
3. All frontend ↔ backend communication uses JSON over HTTP.
4. In local development, both run on the same machine (proposed: frontend on port `8501`, backend on port `8000`).

---

## 4. User Request Flow

### Step-by-step

1. **The user enters a question** in the Streamlit chat input and presses Enter or Send.
2. **Streamlit sends the request** to FastAPI as `POST /chat`, including the question and (if continuing a conversation) the `conversation_id`.
3. **FastAPI validates the request** using a Pydantic schema. Empty or overly long questions are rejected with a `422` error.
4. **The backend prepares the AI prompt.** If a `conversation_id` is provided, it loads recent messages from SQLite. It then combines the football system instructions, previous messages, and the new question.
5. **The backend sends the prompt to OpenRouter**, which forwards it to the configured AI model.
6. **The AI generates an answer**, and OpenRouter returns it to the backend.
7. **The backend stores** the user's question and the AI's answer in SQLite (if history is enabled).
8. **The backend returns the answer** as a validated JSON response.
9. **Streamlit displays the answer** in the chat window.

### Sequence Diagram

```
User        Streamlit           FastAPI              SQLite          OpenRouter / AI Model
 │              │                   │                    │                    │
 │ type question│                   │                    │                    │
 │─────────────>│                   │                    │                    │
 │              │ POST /chat        │                    │                    │
 │              │──────────────────>│                    │                    │
 │              │                   │ validate (Pydantic)│                    │
 │              │                   │───┐                │                    │
 │              │                   │<──┘                │                    │
 │              │                   │ load history       │                    │
 │              │                   │ (if conversation_id)                    │
 │              │                   │───────────────────>│                    │
 │              │                   │<───────────────────│                    │
 │              │                   │ send prompt (system + history + question)
 │              │                   │────────────────────────────────────────>│
 │              │                   │                    │      generate answer
 │              │                   │<────────────────────────────────────────│
 │              │                   │ save question + answer                  │
 │              │                   │───────────────────>│                    │
 │              │                   │<───────────────────│                    │
 │              │ 200 OK (answer)   │                    │                    │
 │              │<──────────────────│                    │                    │
 │ show answer  │                   │                    │                    │
 │<─────────────│                   │                    │                    │
```

**Design note — when is the question saved?** The proposal is to save the user message and the AI reply **together, after** a successful AI response, inside a single database transaction. This avoids conversations containing unanswered questions when the AI call fails. (See [Open Questions](#15-assumptions-and-open-questions).)

---

## 5. Frontend Design

### Layout Sketch (planned)

```
┌──────────────────────────────────────────────────────────────────────┐
│ ┌──────────────────────┐ ┌─────────────────────────────────────────┐ │
│ │  SIDEBAR             │ │  ⚽ FootyGPT                             │ │
│ │                      │ │  Your AI football companion             │ │
│ │  [+ New conversation]│ │─────────────────────────────────────────│ │
│ │                      │ │                                         │ │
│ │  Past conversations  │ │  🧑 What is the offside rule?           │ │
│ │  • Offside rule      │ │                                         │ │
│ │  • 4-3-3 explained   │ │  ⚽ A player is in an offside position   │ │
│ │  • 1966 World Cup    │ │     if any part of their head, body...  │ │
│ │                      │ │                                         │ │
│ │  [🗑 Delete current] │ │  🧑 Can you give an example?            │ │
│ │                      │ │                                         │ │
│ │  Backend: 🟢 Online  │ │  ⏳ Thinking...                          │ │
│ │                      │ │                                         │ │
│ │                      │ │─────────────────────────────────────────│ │
│ │                      │ │ [ Ask a football question...     ] [➤]  │ │
│ └──────────────────────┘ └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Planned behaviour | Streamlit feature (proposed) |
|---|---|---|
| **Chat interface** | Shows the conversation as alternating user and assistant message bubbles, oldest at top. | `st.chat_message` |
| **Question input** | Text box fixed at the bottom of the page with placeholder text "Ask a football question…". | `st.chat_input` |
| **Submit button** | Built into the chat input (send arrow); pressing Enter also submits. Empty input is not sent. | `st.chat_input` |
| **AI response display** | Assistant answers rendered as Markdown so lists, bold text, and headings display nicely. | `st.markdown` inside `st.chat_message` |
| **Conversation history** | Sidebar lists past conversations by title (newest first). Clicking one loads its messages. | `st.sidebar`, `st.button` |
| **Loading indicators** | A spinner ("Thinking…") while waiting for the backend; input is effectively paused until the response arrives. | `st.spinner` |
| **Error messages** | Friendly messages for failures, e.g. "FootyGPT is unavailable right now. Please try again." Technical details are not shown to users. | `st.error`, `st.warning` |
| **Clear conversation option** | "New conversation" starts a fresh chat. "Delete conversation" removes the current one (with a confirmation step). | `st.button` |
| **Backend status (optional)** | Small indicator showing whether `GET /health` succeeds. | `st.sidebar` |

### Frontend State

Streamlit reruns the script on every interaction, so the frontend will keep a small amount of state in `st.session_state`:

- `conversation_id` — the currently open conversation (or `None` for a new one).
- `messages` — the messages currently displayed.

The **database remains the source of truth**; the session state is just a local copy for display.

---

## 6. Backend API Design

**Base URL (local development, proposed):** `http://localhost:8000`
All request and response bodies use JSON. FastAPI will also expose auto-generated interactive docs at `/docs`.

### Endpoint Summary

| Method | Path | Purpose | Required for v1? |
|---|---|---|---|
| `GET` | `/health` | Check backend availability | **Required** |
| `POST` | `/chat` | Submit a question and receive an AI response | **Required** |
| `GET` | `/conversations` | List conversation history | **Required** (if persistence approved) |
| `GET` | `/conversations/{conversation_id}` | Retrieve a specific conversation with messages | **Required** (if persistence approved) |
| `DELETE` | `/conversations/{conversation_id}` | Delete a conversation | **Required** (if persistence approved) |
| `PATCH` | `/conversations/{conversation_id}` | Rename a conversation | *Optional — future* |
| `GET` | `/models` | Show which AI model is configured | *Optional — future* |

### Standard Error Format (proposed)

All application errors return a consistent shape:

```json
{
  "detail": "Human-readable description of the problem."
}
```

Validation errors (`422`) use FastAPI's default format, which lists each invalid field.

---

### 6.1 `GET /health`

**Purpose:** Lets the frontend (or a developer) confirm the backend is running.

**Request:** No body.

**Response — `200 OK`:**
```json
{
  "status": "ok",
  "database": "ok",
  "ai_configured": true
}
```

- `ai_configured` indicates whether an API key and model name are set — it does **not** call OpenRouter (to avoid cost and latency).

**Possible errors:**

| Status | When |
|---|---|
| `503 Service Unavailable` | Database cannot be reached. Body includes `"database": "error"`. |

---

### 6.2 `POST /chat`

**Purpose:** Send a football question and receive an AI answer. Creates a new conversation if no `conversation_id` is supplied.

**Request — new conversation:**
```json
{
  "message": "What is the offside rule?"
}
```

**Request — continue existing conversation:**
```json
{
  "message": "Can you give an example?",
  "conversation_id": 1
}
```

**Response — `200 OK`:**
```json
{
  "conversation_id": 1,
  "user_message": {
    "id": 1,
    "role": "user",
    "content": "What is the offside rule?",
    "created_at": "2026-10-23T14:30:00Z"
  },
  "assistant_message": {
    "id": 2,
    "role": "assistant",
    "content": "A player is in an offside position if...",
    "created_at": "2026-10-23T14:30:04Z"
  },
  "model": "<configured-model-id>"
}
```

**Possible errors:**

| Status | When | Example `detail` |
|---|---|---|
| `422 Unprocessable Entity` | Empty/whitespace message, message too long, wrong types. | (FastAPI field-level validation output) |
| `404 Not Found` | `conversation_id` does not exist. | `"Conversation 99 not found."` |
| `429 Too Many Requests` | OpenRouter rate limit reached. | `"The AI service is busy. Please wait a moment and try again."` |
| `502 Bad Gateway` | OpenRouter or the model returned an error or an empty/invalid response. | `"The AI service returned an error. Please try again."` |
| `503 Service Unavailable` | API key or model not configured, or database unavailable. | `"AI service is not configured."` |
| `504 Gateway Timeout` | AI did not respond within the timeout. | `"The AI took too long to respond. Please try again."` |

---

### 6.3 `GET /conversations`

**Purpose:** List all conversations (newest first) for the sidebar. Does not include full message content.

**Request:** No body. Optional query parameters (proposed): `limit` (default 50), `offset` (default 0).

**Response — `200 OK`:**
```json
[
  {
    "id": 2,
    "title": "How does a 4-3-3 work?",
    "created_at": "2026-10-23T15:00:00Z",
    "updated_at": "2026-10-23T15:05:00Z",
    "message_count": 4
  },
  {
    "id": 1,
    "title": "What is the offside rule?",
    "created_at": "2026-10-23T14:30:00Z",
    "updated_at": "2026-10-23T14:32:00Z",
    "message_count": 4
  }
]
```

An empty list `[]` is returned if there are no conversations.

**Possible errors:**

| Status | When |
|---|---|
| `422` | Invalid `limit`/`offset` values. |
| `503` | Database unavailable. |

---

### 6.4 `GET /conversations/{conversation_id}`

**Purpose:** Retrieve one conversation with all of its messages in chronological order.

**Request:** No body. Example: `GET /conversations/1`

**Response — `200 OK`:**
```json
{
  "id": 1,
  "title": "What is the offside rule?",
  "created_at": "2026-10-23T14:30:00Z",
  "updated_at": "2026-10-23T14:32:00Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What is the offside rule?",
      "created_at": "2026-10-23T14:30:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "A player is in an offside position if...",
      "created_at": "2026-10-23T14:30:04Z"
    }
  ]
}
```

**Possible errors:**

| Status | When |
|---|---|
| `404` | Conversation does not exist. |
| `422` | `conversation_id` is not an integer. |
| `503` | Database unavailable. |

---

### 6.5 `DELETE /conversations/{conversation_id}`

**Purpose:** Permanently delete a conversation and all its messages.

**Request:** No body. Example: `DELETE /conversations/1`

**Response — `204 No Content`:** Empty body.

**Possible errors:**

| Status | When |
|---|---|
| `404` | Conversation does not exist. |
| `422` | `conversation_id` is not an integer. |
| `503` | Database unavailable. |

---

### 6.6 Optional Endpoints (not part of v1 unless approved)

| Endpoint | Purpose | Example |
|---|---|---|
| `PATCH /conversations/{conversation_id}` | Rename a conversation. | Request: `{"title": "Offside explained"}` → `200 OK` with updated conversation. |
| `GET /models` | Show the configured model name (useful for display/debugging). | Response: `{"model": "<configured-model-id>"}` |

---

## 7. Database Design

### Overview

The initial design uses **two tables** with a **one-to-many** relationship: one conversation contains many messages.

### Entity Relationship Diagram

```
┌──────────────────────────────┐            ┌──────────────────────────────────┐
│        conversations         │            │             messages             │
├──────────────────────────────┤            ├──────────────────────────────────┤
│ PK  id          INTEGER      │──┐         │ PK  id               INTEGER     │
│     title       VARCHAR(200) │  │  1 : N  │ FK  conversation_id  INTEGER     │
│     created_at  DATETIME     │  └────────<│     role             VARCHAR(20) │
│     updated_at  DATETIME     │            │     content          TEXT        │
└──────────────────────────────┘            │     model            VARCHAR(100)│
                                            │     created_at       DATETIME    │
                                            └──────────────────────────────────┘
```

### Table: `conversations`

Stores one row per chat session.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | **Primary key**, auto-increment | Unique conversation identifier. |
| `title` | `VARCHAR(200)` | `NOT NULL` | Short label shown in the sidebar. Proposed: first ~60 characters of the first question. |
| `created_at` | `DATETIME` | `NOT NULL`, default = current UTC time | When the conversation started. |
| `updated_at` | `DATETIME` | `NOT NULL`, default = current UTC time, updated on new message | When the last message was added (used for sorting). |

### Table: `messages`

Stores each individual message — both user questions and AI answers.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | `INTEGER` | **Primary key**, auto-increment | Unique message identifier. |
| `conversation_id` | `INTEGER` | **Foreign key** → `conversations.id`, `NOT NULL`, `ON DELETE CASCADE`, indexed | Which conversation this message belongs to. |
| `role` | `VARCHAR(20)` | `NOT NULL`, must be `'user'` or `'assistant'` | Who wrote the message. |
| `content` | `TEXT` | `NOT NULL` | The message text. |
| `model` | `VARCHAR(100)` | Nullable | The AI model that generated the answer (only for `assistant` messages). Useful when the configured model changes. |
| `created_at` | `DATETIME` | `NOT NULL`, default = current UTC time | When the message was created. Used to order messages. |

### Relationships

- **One conversation → many messages.** Each message belongs to exactly one conversation.
- **Cascade delete:** deleting a conversation deletes all its messages.
  - *Note:* SQLite only enforces foreign keys when `PRAGMA foreign_keys = ON` is set per connection. The plan is to enable this via a SQLAlchemy connection event **and** configure the SQLAlchemy relationship with `cascade="all, delete-orphan"` as a second safeguard.

### How questions and answers are stored

For a successful `POST /chat` request:

1. If no `conversation_id` was given, insert a new `conversations` row (title from the question).
2. Insert a `messages` row with `role = 'user'` and the question.
3. Insert a `messages` row with `role = 'assistant'`, the AI answer, and the model name.
4. Update `conversations.updated_at`.
5. Commit all of the above as **one transaction** — either everything is saved or nothing is.

**Example data:**

`conversations`

| id | title | created_at | updated_at |
|---|---|---|---|
| 1 | What is the offside rule? | 2026-10-23 14:30:00 | 2026-10-23 14:32:00 |

`messages`

| id | conversation_id | role | content | model | created_at |
|---|---|---|---|---|---|
| 1 | 1 | user | What is the offside rule? | NULL | 2026-10-23 14:30:00 |
| 2 | 1 | assistant | A player is in an offside position if... | `<model-id>` | 2026-10-23 14:30:04 |

### Schema creation

For v1, tables will be created at backend startup using SQLAlchemy's `Base.metadata.create_all()`. A migration tool (e.g. Alembic) is deferred until the schema needs to change (see [Section 13](#13-design-decisions-and-trade-offs)).

---

## 8. AI Integration

### How OpenRouter fits in

OpenRouter is a service that provides **one API** to reach many AI models. Its chat endpoint follows the widely used "chat completions" format: you send a list of messages, and it returns the model's reply.

```
FastAPI backend ──HTTPS POST──> https://openrouter.ai/api/v1/chat/completions
                                      │
                                      ▼
                              Configured AI model
```

The backend will contain a small, dedicated **AI service module** (`ai_service.py`) responsible for all communication with OpenRouter. The rest of the code calls this module and never talks to OpenRouter directly — which also makes it easy to replace with a fake in tests.

### Configuration (environment variables)

| Variable | Required | Purpose | Example value |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Secret key for OpenRouter. | *(never committed — set locally)* |
| `OPENROUTER_MODEL` | Yes | Model identifier to use. | *(chosen during setup — see note below)* |
| `OPENROUTER_BASE_URL` | No | API base URL. | `https://openrouter.ai/api/v1` |
| `AI_TIMEOUT_SECONDS` | No | Max time to wait for a reply. | `30` |
| `AI_MAX_HISTORY_MESSAGES` | No | How many previous messages to include as context. | `10` |
| `AI_MAX_TOKENS` | No | Upper limit on answer length. | `800` |
| `AI_TEMPERATURE` | No | Creativity of answers (lower = more factual). | `0.3` |
| `DATABASE_URL` | No | Database location. | `sqlite:///./footygpt.db` |
| `BACKEND_URL` | No (frontend) | Where the frontend finds the backend. | `http://localhost:8000` |

> **Model selection note:** This design deliberately does **not** hardcode or assume any particular model, and does **not** assume any model is free. Model availability, pricing, and rate limits on OpenRouter change over time. The model will be chosen and verified at setup time by checking OpenRouter's current model list, and set via `OPENROUTER_MODEL`. Changing models requires only changing this value.

### Sending a question

The backend builds a list of messages in this order:

1. **System message** — football-focused instructions (below).
2. **Previous messages** — the most recent `AI_MAX_HISTORY_MESSAGES` messages from this conversation, oldest first (only when continuing a conversation).
3. **New user message** — the current question.

**Illustrative request body sent to OpenRouter:**

```json
{
  "model": "<value of OPENROUTER_MODEL>",
  "messages": [
    { "role": "system", "content": "You are FootyGPT, a friendly football expert..." },
    { "role": "user", "content": "What is the offside rule?" },
    { "role": "assistant", "content": "A player is in an offside position if..." },
    { "role": "user", "content": "Can you give an example?" }
  ],
  "max_tokens": 800,
  "temperature": 0.3
}
```

**Headers:**

```
Authorization: Bearer <OPENROUTER_API_KEY loaded from environment>
Content-Type: application/json
```

(OpenRouter also supports optional headers identifying the calling app; these may be added but are not required.)

### Receiving and returning the response

1. Check the HTTP status code (see error handling below).
2. Read the answer text from `choices[0].message.content` in the response JSON.
3. If the answer is missing or empty, treat it as a provider error (`502`).
4. Trim whitespace and return the text to the chat route, which saves it and returns it to the frontend.

### Football-focused system instructions (draft)

```
You are FootyGPT, a friendly and knowledgeable football (soccer) assistant.

- Answer questions about football rules, tactics, formations, players, clubs,
  competitions, and football history.
- Explain concepts clearly for beginners. Avoid jargon, or explain it when used.
- Keep answers concise and well-structured; use short lists where helpful.
- If you are not sure about a fact (e.g. exact dates or statistics), say so
  rather than guessing.
- You do not have access to live scores or real-time information. If asked
  about very recent events, explain that your knowledge may be out of date.
- If a question is not related to football, politely explain that you focus on
  football and invite the user to ask a football question.
- Do not provide betting advice.
```

This text will live in one place in the code (e.g. `prompts.py`) so it can be reviewed and adjusted easily.

### Conversation context

- Models do not "remember" past messages on their own — the backend must re-send relevant history each time.
- Only the last **N** messages are sent (default proposal: 10) to control cost, speed, and the model's context-length limit.
- A brand-new conversation sends only the system message and the question.

### Handling timeouts, rate limits, and provider failures

| Situation | How it is detected | Backend behaviour | Status returned to frontend |
|---|---|---|---|
| **Timeout** | No response within `AI_TIMEOUT_SECONDS`. | Log the event; nothing saved. | `504` |
| **Rate limit** | OpenRouter returns HTTP `429`. | Log; nothing saved. Optionally one short retry (see Open Questions). | `429` |
| **Insufficient credits / payment issue** | OpenRouter returns HTTP `402`. | Log clearly for the developer. | `503` with a generic user message |
| **Invalid key** | OpenRouter returns HTTP `401`. | Log clearly for the developer. | `503` with a generic user message |
| **Provider/model error** | HTTP `5xx`, or other unexpected status. | Log status code (not the key). | `502` |
| **Malformed/empty response** | Missing `choices` or empty content. | Log. | `502` |
| **Network failure** | Connection error. | Log. | `502` |

The frontend translates each of these into a friendly message and lets the user retry.

### Protecting the API key

- The key is read **only** from the `OPENROUTER_API_KEY` environment variable (optionally loaded from a local `.env` file during development).
- `.env` will be listed in `.gitignore` so it is never committed.
- A `.env.example` file will show the variable **names** with placeholder values only.
- The key is never sent to the frontend, never returned in responses, and never written to logs.
- If the key is missing, the backend still starts (so `/health` works), but `POST /chat` returns `503 "AI service is not configured."`

---

## 9. Data Validation

### Why Pydantic?

Pydantic lets us describe the data we expect as Python classes. FastAPI uses these classes to automatically:

- Reject requests with missing or wrong-type fields (returning a `422` with details).
- Convert JSON into Python objects and back.
- Document the API in the auto-generated `/docs` page.

### Proposed Validation Rules

| Field | Rule |
|---|---|
| `message` | Required string. Leading/trailing whitespace stripped. Must not be empty after stripping. Max 2,000 characters (proposed). |
| `conversation_id` | Optional. If present, must be a positive integer. |
| `role` (responses) | Must be `"user"` or `"assistant"`. |

### Example Schemas (illustrative — not yet written)

```python
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: int | None = Field(default=None, gt=0)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be empty.")
        return value


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # allows building from SQLAlchemy objects

    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: MessageOut
    assistant_message: MessageOut
    model: str


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    ai_configured: bool
```

### Example validation failure

Request:
```json
{ "message": "   " }
```

Response — `422 Unprocessable Entity` (FastAPI default format, abbreviated):
```json
{
  "detail": [
    {
      "loc": ["body", "message"],
      "msg": "Value error, Message must not be empty.",
      "type": "value_error"
    }
  ]
}
```

**Important separation:** Pydantic schemas (API data shapes) and SQLAlchemy models (database tables) will be kept in **separate files**. They look similar but serve different purposes.

---

## 10. Error Handling and Security

### Error Handling

| Scenario | Where it is caught | Response | What the user sees (frontend) |
|---|---|---|---|
| **Empty question** | Frontend (doesn't send) **and** Pydantic validation (backend) | `422` | Nothing sent, or "Please enter a question." |
| **Invalid request** (wrong types, missing fields, too long) | Pydantic validation | `422` with field details | "Your question couldn't be processed. Please check it and try again." |
| **Unknown conversation** | Chat/conversation routes | `404` | "That conversation no longer exists." |
| **AI provider error** | AI service | `502` | "The AI service returned an error. Please try again." |
| **AI timeout** | AI service | `504` | "The AI took too long to respond. Please try again." |
| **Rate limit** | AI service (OpenRouter `429`) | `429` | "FootyGPT is busy. Please wait a moment and try again." |
| **Database failure** | Database layer / route | `503`; transaction rolled back | "Something went wrong saving your conversation. Please try again." |
| **Missing API configuration** | AI service checks config before calling | `503` | "FootyGPT isn't fully set up yet." |
| **Backend unreachable** | Frontend HTTP client | (no response) | "Can't reach the FootyGPT server. Is the backend running?" |
| **Unexpected error** | Global exception handler | `500` with generic message | "Something unexpected happened. Please try again." |

**Principles:**

- Users see **friendly, non-technical** messages. Developers see details in **server logs**.
- Stack traces and internal details are **never** returned in API responses.
- Database writes for a chat exchange happen in **one transaction**, rolled back on any failure.
- Custom exception classes (e.g. `AIServiceError`, `AITimeoutError`, `AIRateLimitError`, `AIConfigError`) will be mapped to HTTP status codes in one place.

### Security

| Concern | Planned measure |
|---|---|
| **API key exposure** | Key stored only in environment variables / local `.env`; `.env` git-ignored; `.env.example` contains placeholders only. Never logged or returned. |
| **Secrets in code** | No keys, tokens, or passwords anywhere in source code or documentation. |
| **Logging sensitive data** | Logs record status codes, timings, and error types — not API keys. Full message content logging off by default. |
| **Input abuse** | Maximum message length enforced; history length capped to limit cost. |
| **SQL injection** | All database access through SQLAlchemy ORM (parameterised queries); no raw SQL built from user input. |
| **Rendering AI output** | Display AI output as Markdown with Streamlit's default (HTML not enabled — `unsafe_allow_html` stays off). |
| **Network exposure** | Backend binds to `localhost` (`127.0.0.1`) by default for local development. |
| **CORS** | Not required for Streamlit (its server-side Python makes the HTTP calls, not the browser). CORS stays disabled unless a browser-based frontend is added. |
| **Authentication** | **Not included in v1** — the app is intended for local, single-user use. All conversations are visible to anyone with access to the app. Must be added before any public deployment. |
| **Prompt injection** | System instructions keep the model on-topic, but this cannot be fully guaranteed. Risk is low because the model has no tools or access to private data. |
| **User privacy** | Conversations stored locally in SQLite; users can delete conversations. Questions are sent to OpenRouter and the model provider — this will be stated in the README. |

---

## 11. Proposed Project Structure

> **Proposed only.** None of these directories or files exist yet.

```
footygpt/
├── backend/
│   ├── __init__.py
│   ├── main.py               # Creates the FastAPI app, registers routes, startup tasks
│   ├── config.py             # Loads settings from environment variables (pydantic-settings)
│   ├── database.py           # SQLAlchemy engine, session factory, SQLite foreign-key setup
│   ├── models.py             # SQLAlchemy table models: Conversation, Message
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── crud.py               # Database operations (create/read/delete conversations & messages)
│   ├── exceptions.py         # Custom error classes and exception-to-HTTP mapping
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py         # GET /health
│   │   ├── chat.py           # POST /chat
│   │   └── conversations.py  # GET/DELETE /conversations...
│   └── services/
│       ├── __init__.py
│       ├── ai_service.py     # Calls OpenRouter, handles AI errors
│       └── prompts.py        # Football system prompt and prompt-building helpers
│
├── frontend/
│   ├── app.py                # Streamlit application (chat UI, sidebar)
│   └── api_client.py         # Small helper functions for calling the backend
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures: test DB, test client, fake AI service
│   ├── test_health.py
│   ├── test_chat.py
│   ├── test_conversations.py
│   ├── test_schemas.py
│   ├── test_crud.py
│   └── test_ai_service.py
│
├── .env.example              # Variable names with placeholder values (no real secrets)
├── .gitignore                # Ignores .env, *.db, virtual env, caches
├── requirements.txt          # Pinned Python dependencies
├── pytest.ini                # pytest configuration (or [tool.pytest] in pyproject.toml)
├── design.md                 # This document
└── README.md                 # Project introduction and setup guide (to be written later)
```

### Purpose of Major Components

| Component | Purpose |
|---|---|
| `backend/main.py` | Entry point. Builds the app, includes routers, creates tables at startup. |
| `backend/config.py` | Single place for all settings; reads environment variables and provides defaults. |
| `backend/database.py` | Database connection setup and a `get_db` dependency that gives each request its own session. |
| `backend/models.py` | Defines the `conversations` and `messages` tables as Python classes. |
| `backend/schemas.py` | Defines the shape of API inputs and outputs (validation). |
| `backend/crud.py` | Keeps database queries out of route functions, so they're reusable and testable. |
| `backend/routes/` | Thin HTTP layer: receive request → call services/CRUD → return response. |
| `backend/services/` | Business logic that isn't about HTTP or the database — mainly the AI integration. |
| `frontend/app.py` | Everything the user sees. |
| `frontend/api_client.py` | Wraps HTTP calls to the backend and converts errors into friendly messages. |
| `tests/` | Automated tests (see Section 12). |

**Why this structure?** Each file has one clear job. A beginner can find "where is the database table?" (`models.py`) or "where do we call the AI?" (`ai_service.py`) quickly, without the project being split into too many layers.

---

## 12. Testing Strategy

### Tools (proposed)

- **pytest** — test runner.
- **FastAPI `TestClient`** — sends fake HTTP requests to the backend without starting a server.
- **In-memory / temporary SQLite database** — each test gets a clean database; the real `footygpt.db` is never touched.
- **Dependency overrides / mocking** — replace the real AI service with a fake one so tests **never call OpenRouter**, cost nothing, and give predictable results.

### Unit Tests vs Integration Tests

| | Unit tests | Integration tests |
|---|---|---|
| **What** | Test one small piece in isolation. | Test several pieces working together through the API. |
| **Example** | "Does `ChatRequest` reject a blank message?" | "Does `POST /chat` create a conversation and save two messages?" |
| **Dependencies** | None or mocked. | Real FastAPI app + test SQLite DB; AI still mocked. |
| **Speed** | Very fast. | Fast (still local). |

### Planned Test Coverage

| Area | Type | Example test cases |
|---|---|---|
| **Validation** (`test_schemas.py`) | Unit | Blank message rejected; whitespace stripped; message over 2,000 chars rejected; negative `conversation_id` rejected. |
| **Database operations** (`test_crud.py`) | Unit (with test DB) | Create conversation; add messages; list newest first; get by ID; delete cascades to messages; missing ID returns `None`. |
| **AI service** (`test_ai_service.py`) | Unit (HTTP mocked) | Builds correct message list (system + history + question); history limited to N; parses answer; maps `429` → rate-limit error; timeout → timeout error; empty content → provider error; missing key → config error; API key never appears in raised error messages. |
| **Health endpoint** (`test_health.py`) | Integration | Returns `200` and `status: ok`; reports `ai_configured: false` when key is missing. |
| **Chat endpoint** (`test_chat.py`) | Integration (AI mocked) | New conversation created; follow-up uses existing conversation; history passed to AI; `404` for unknown conversation; `422` for blank message; AI failure returns correct status **and saves nothing**. |
| **Conversation endpoints** (`test_conversations.py`) | Integration | List returns summaries; get returns ordered messages; delete returns `204` then `404` on second call. |
| **Frontend** | Manual (v1) | Checklist-based manual testing of the Streamlit UI (see Phase 8). Automated UI tests deferred. |

### Principles

- Tests must pass **without** an internet connection or API key.
- An optional, manually-run "live" test against OpenRouter may be added later, clearly marked and skipped by default.

---

## 13. Design Decisions and Trade-offs

### Why this stack fits

The stack is **all-Python**, uses well-documented, popular tools, and requires **no external servers** besides OpenRouter. A beginner can run the whole system on a laptop with two terminal commands, while the backend/frontend split still teaches real-world architecture.

### Benefits and Limitations

| Technology | Benefits | Limitations | Mitigation |
|---|---|---|---|
| **Streamlit** | Pure Python; built-in chat widgets; very fast to build. | Limited layout/styling control; reruns the whole script on each interaction; not ideal for large multi-user production apps. | Acceptable for v1. Because logic lives in the backend, the frontend can be replaced later (e.g. with React) without backend changes. |
| **FastAPI** | Fast, modern, automatic validation and docs; great test support. | Adds a second process to run; slightly more setup than putting everything in Streamlit. | Worth it: keeps the API key server-side, enables testing, and separates concerns. |
| **SQLite** | Zero setup; single file; included with Python. | Limited concurrent writes; not suited to multiple servers. | Fine for single-user local use. SQLAlchemy makes moving to PostgreSQL mostly a configuration change. |
| **SQLAlchemy** | Pythonic database access; protects against SQL injection; DB-agnostic. | Learning curve for beginners. | Use simple, explicit patterns in `crud.py`. |
| **OpenRouter** | One API for many models; easy to switch models; no vendor lock-in to one provider. | Requires internet and an account; costs and rate limits depend on chosen model; adds a third-party in the data path; model availability can change. | Model is configurable; robust error handling; tests use mocks. |

### Architectural Decisions Requiring Approval

| # | Decision | Proposal | Alternative |
|---|---|---|---|
| **D1** | Separate frontend and backend | Streamlit + FastAPI as **two processes** communicating over HTTP. | Single Streamlit app calling OpenRouter directly (simpler, but less testable and mixes concerns). |
| **D2** | Conversation persistence in v1 | **Include** SQLite persistence and the conversation endpoints in v1. | Keep history only in the browser session for v1; add the DB later. |
| **D3** | Response delivery | **Non-streaming**: wait for the complete answer, show a spinner. | Streaming word-by-word (better UX, but more complex in both backend and frontend). |
| **D4** | Authentication | **None** in v1; local single-user only. | Add simple user accounts (adds a `users` table and auth logic). |
| **D5** | Context window | Send the last **10** messages as context. | Different limit, or summarisation of older messages. |
| **D6** | Off-topic questions | Model **politely declines** non-football questions via system prompt. | Allow general questions, or add a separate classification step. |
| **D7** | Schema management | `create_all()` at startup; **no migrations** in v1. | Use Alembic from the start. |
| **D8** | Sync vs async | Use **synchronous** SQLAlchemy and a sync HTTP client for simplicity. | Fully async stack (more scalable, harder for beginners). |
| **D9** | Save behaviour on AI failure | Save **nothing** if the AI call fails. | Save the user question regardless and mark the reply as failed. |
| **D10** | AI model | Chosen at setup via `OPENROUTER_MODEL`; **no model assumed**. | Decide a specific model now. |

---

## 14. Implementation Roadmap

> **No phase has been started.** Each phase should be reviewed before moving to the next.

### Phase 1 — Project Setup

**Outcome:** An empty but runnable project skeleton.

- Create folder structure, virtual environment, `requirements.txt`, `.gitignore`, `.env.example`.
- Minimal FastAPI app with `GET /health`; minimal Streamlit page.
- Configuration loading via `config.py`.

**Acceptance criteria:**
- [ ] Backend starts locally and `GET /health` returns `200`.
- [ ] Streamlit page opens in the browser and displays the FootyGPT title.
- [ ] `.env` is git-ignored; `.env.example` contains no real secrets.
- [ ] `pytest` runs (with at least one passing health test).

### Phase 2 — Database Design and Implementation

**Outcome:** Database tables and CRUD functions exist.

- `database.py`, `models.py` (Conversation, Message), `crud.py`.
- Foreign keys enabled; cascade delete configured.

**Acceptance criteria:**
- [ ] Tables are created automatically at backend startup.
- [ ] CRUD functions create, list, retrieve, and delete conversations and messages.
- [ ] Deleting a conversation removes its messages.
- [ ] Unit tests for CRUD pass using a test database.

### Phase 3 — Backend API Development

**Outcome:** All endpoints work, using a **placeholder** AI reply.

- Pydantic schemas; routes for `/chat` and `/conversations`.
- Custom exceptions and consistent error responses.

**Acceptance criteria:**
- [ ] All five endpoints behave as documented in Section 6 (with a placeholder AI answer).
- [ ] Invalid input returns `422`; unknown IDs return `404`.
- [ ] Endpoints are visible and usable in `/docs`.
- [ ] Integration tests for endpoints pass.

### Phase 4 — OpenRouter Integration

**Outcome:** The backend returns real AI answers.

- `ai_service.py` and `prompts.py`; model selected via configuration.
- Timeouts, rate limits, and provider errors mapped to HTTP responses.

**Acceptance criteria:**
- [ ] With a valid key and model configured, `POST /chat` returns a relevant football answer (verified manually).
- [ ] Missing key returns `503` without crashing the server.
- [ ] Timeout, `429`, and `5xx` cases return the documented status codes (verified with mocked tests).
- [ ] API key never appears in logs or responses.

### Phase 5 — Frontend Development

**Outcome:** A usable chat interface connected to the backend.

- Chat display, chat input, spinner, friendly error messages, `api_client.py`.

**Acceptance criteria:**
- [ ] User can type a question and see the AI answer in the chat.
- [ ] A spinner is shown while waiting.
- [ ] If the backend is stopped, a friendly error is shown (no crash or stack trace).
- [ ] Empty input is not submitted.

### Phase 6 — Conversation Persistence

**Outcome:** Users can continue, revisit, and delete conversations.

- Follow-up questions include previous context.
- Sidebar with conversation list, "New conversation", and "Delete conversation".

**Acceptance criteria:**
- [ ] A follow-up question ("Can you give an example?") is answered in context.
- [ ] Past conversations appear in the sidebar and can be reopened after restarting the app.
- [ ] Deleting a conversation removes it from the sidebar and the database.
- [ ] Only the configured number of history messages is sent to the AI.

### Phase 7 — Testing and Security

**Outcome:** Test coverage and security measures reviewed and completed.

- Fill test gaps from Section 12; review error handling and secrets handling.

**Acceptance criteria:**
- [ ] All tests pass offline without an API key.
- [ ] Every error scenario in Section 10 has either an automated test or a documented manual check.
- [ ] Search of the codebase finds no hardcoded secrets.
- [ ] Logs reviewed: no API keys present.

### Phase 8 — Final Integration

**Outcome:** v1 complete, documented, and demonstrated end-to-end.

- Write `README.md` with **verified** setup steps.
- Manual end-to-end test checklist executed.

**Acceptance criteria (definition of done for v1):**
- [ ] A new developer can follow the README to run the app locally.
- [ ] Rules, tactics, formations, players, and history questions all return sensible answers (manual check with sample questions).
- [ ] Conversation history works across app restarts.
- [ ] All automated tests pass.
- [ ] Known limitations are documented.

### Planned local run commands (NOT yet tested)

> These commands are **planned** and will only be verified during Phase 1. They are included for reviewers to understand the intended developer workflow.

```bash
# Create and activate a virtual environment (Windows, Git Bash)
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Configure secrets: copy the example and add your own key and model
cp .env.example .env

# Terminal 1 — start the backend
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — start the frontend
streamlit run frontend/app.py

# Run tests
pytest
```

---

## 15. Assumptions and Open Questions

### Assumptions

1. FootyGPT v1 runs **locally** for a **single user**; no public deployment.
2. The developer has Python **3.11 or newer** installed (proposed minimum — to be confirmed).
3. The developer will create an OpenRouter account and supply their **own** API key.
4. Any AI usage costs are acceptable to the developer; costs depend on the chosen model.
5. Answers come from the AI model's general knowledge; there is **no live data** and no external football database.
6. English is the only supported language in v1 (the model may still respond in other languages if asked).
7. An internet connection is available when chatting (not required for running tests).

### Open Questions (to settle before coding)

| # | Question | Proposed answer |
|---|---|---|
| Q1 | Which OpenRouter model should be used initially? | Decide at Phase 4 after checking current availability and pricing. |
| Q2 | Should conversation persistence be in v1, or deferred? | Include in v1 (D2). |
| Q3 | Should responses stream word-by-word? | No for v1 (D3). |
| Q4 | Should non-football questions be refused, or answered briefly? | Politely redirect to football (D6). |
| Q5 | How many history messages should be sent as context? | 10 (D5). |
| Q6 | Should the backend retry once on a rate limit or timeout? | No automatic retry in v1; user retries manually. Revisit if needed. |
| Q7 | Should a failed AI call still save the user's question? | No (D9). |
| Q8 | Maximum question length? | 2,000 characters. |
| Q9 | Minimum Python version? | 3.11. |
| Q10 | Should the frontend's backend URL be configurable? | Yes, via `BACKEND_URL` env variable. |
| Q11 | Use `requirements.txt` or `pyproject.toml`? | `requirements.txt` for simplicity. |
| Q12 | Is version control (git) to be set up in Phase 1? | Recommended yes — the project folder is not yet a git repository. |

### Risks

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| **AI gives incorrect facts** ("hallucinations"), especially dates and statistics. | Users may learn wrong information. | Medium–High | System prompt asks the model to admit uncertainty; UI disclaimer that answers may contain errors. |
| **Outdated knowledge** about recent transfers, results, or rule changes. | Answers about recent events may be wrong. | High | Prompt instructs the model to mention its knowledge may be out of date; live data is out of scope for v1. |
| **Model availability or pricing changes** on OpenRouter. | App stops working or costs rise. | Medium | Model is configurable; clear error handling for `402`/`404`/`5xx`. |
| **Rate limits** (especially on low-cost tiers). | Users see "busy" errors. | Medium | Friendly error messages; limited history size reduces token usage. |
| **API key leak** via commit or logs. | Unauthorized usage and charges. | Low (with measures) | `.gitignore`, `.env.example`, no key logging, secret search in Phase 7. |
| **SQLite concurrency limits.** | Errors if used by many users simultaneously. | Low (single-user v1) | Document limitation; PostgreSQL migration path via SQLAlchemy. |
| **Scope creep** (live scores, accounts, etc.). | Delays v1. | Medium | Out-of-scope list in Section 1; changes go through approval. |

### Future Enhancements (out of scope for v1)

- Streaming responses.
- User accounts and authentication.
- Live football data (scores, fixtures, standings) via a sports data API.
- Retrieval from trusted sources (e.g. official Laws of the Game) to reduce incorrect answers.
- Conversation renaming and search.
- Formation diagrams and visual tactics boards.
- Docker setup and cloud deployment.
- Migration to PostgreSQL and Alembic.
- Multi-language support.

---

*End of document — awaiting review and approval. No implementation will begin until approval is given.*
