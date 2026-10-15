# Keystone Chatbot

Keystone is an enterprise-grade AI sales and support chatbot backend powered by **LangGraph**, **FastAPI**, and **PostgreSQL**. It enables businesses to register their operational details, services, and policies, providing an autonomous conversational agent that answers customer queries, executes domain-specific workflows, looks up customer profiles, and negotiates orders with deterministic human-in-the-loop approval guardrails before committing directly to PostgreSQL.

> [!NOTE]
> **Automated Demo Startup Seeding**: For seamless demonstration and testing, demo customer profiles (Vivek: `9876543210`, Alice: `1234567890`) and a default business (`default`: Keystone Cafe & Pizza) are automatically seeded into PostgreSQL at application startup. This allows the Order Specialist Agent to autonomously fetch registered delivery addresses and usual items during chat sessions.

---

## 1. System Architecture & Flow

The system employs a **Hierarchical Domain-Agent Architecture** with layered security ingress, stateful LangGraph orchestration, scoped tool chains, rolling conversational summarization, and PostgreSQL checkpointer persistence.

The complete architecture flow diagram is detailed below (source file: [architecture_flow.mermaid](file:///Users/vivekchole/workspace/interview-project/keystone/docs/architecture_flow.mermaid)):

```mermaid
flowchart TD
    %% Styling & Classes
    classDef client fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef security fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#e65100;
    classDef router fill:#ede7f6,stroke:#5e35b1,stroke-width:2px,color:#311b92;
    classDef agent fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#1b5e20;
    classDef tool fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;
    classDef interrupt fill:#ffebee,stroke:#d32f2f,stroke-width:3px,color:#b71c1c;
    classDef summary fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c;
    classDef storage fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#263238;

    subgraph INGRESS ["1. Ingress & Security Guardrails"]
        ClientReq["Client HTTP Request<br/>(SSE Chat / Approve)"]:::client
        LockMgr{"ThreadLockManager<br/>(Per-Thread Lock)"}:::security
        ConflictErr["409 Conflict Response<br/>(Thread busy)"]:::security
        Sanitize["Prompt Injection & Length Sanitizer<br/>(Delimiters, &lt;customer_message&gt;, 1k limit)"]:::security
    end

    subgraph LANGGRAPH_CORE ["2. LangGraph Stateful Orchestrator"]
        StartNode(["START"]):::storage
        AuthGate["auth_gatekeeper<br/>(Phone & OTP Verification / 3 Tries)"]:::agent
        AuthCheck{"is_authenticated?"}:::router
        AuthChallenge["Return Phone/OTP Challenge<br/>(Yield to User)"]:::client
        
        IntentRouter["intent_router<br/>(Domain Intent Classifier)"]:::router
        DomainRoute{"Domain Route"}:::router

        subgraph DOMAIN_AGENTS ["Domain Specialist Nodes"]
            QueryAgent["query<br/>(Business Q&A from Context)"]:::agent
            FeedbackAgent["feedback<br/>(Feedback Collector)"]:::agent
            ComplaintAgent["complaint<br/>(Customer Escalations)"]:::agent
            OrderAgent["order<br/>(Autonomous Order Specialist)"]:::agent
        end

        subgraph ORDER_TOOL_LOOP ["Order Specialist Tool Loop"]
            OrderToolCheck{"Tool Invocation?"}:::router
            OrderToolsNode["order_tools<br/>(ToolNode Execution)"]:::tool
            ProfileTool["get_customer_profile<br/>(Fetch name, address, usual items)"]:::tool
            GetOrderTool["get_order<br/>(Fetch order status & details)"]:::tool
            TakeOrderTool["take_order<br/>(Validate items, idempotency check)"]:::tool
            ApprovalInterrupt{{"interrupt()<br/>Human-in-the-Loop Approval"}}:::interrupt
        end

        subgraph CONVERSATION_MANAGEMENT ["Memory & Rolling Summarization"]
            SummarizeCheck{"Message count ><br/>SUMMARIZE_THRESHOLD (10)?"}:::router
            SummarizeNode["summarize_conversation<br/>(LLM rolls history into summary)"]:::summary
            EvictMsgs["RemoveMessage<br/>(Evict raw history from state)"]:::summary
        end

        EndNode(["END"]):::storage
    end

    subgraph PERSISTENCE ["3. Persistence Layer (PostgreSQL)"]
        Checkpointer[("AsyncPostgresSaver<br/>Thread State Checkpointer")]:::storage
        DBTables[("PostgreSQL Database<br/>(businesses, customers, orders)")]:::storage
    end

    %% Ingress Flow
    ClientReq --> LockMgr
    LockMgr -- "Lock Failed (Timeout)" --> ConflictErr
    LockMgr -- "Lock Acquired" --> Sanitize
    Sanitize --> StartNode

    %% Auth Flow
    StartNode --> AuthGate
    AuthGate --> AuthCheck
    AuthCheck -- "No" --> AuthChallenge
    AuthChallenge --> EndNode
    AuthCheck -- "Yes" --> IntentRouter

    %% Intent Routing
    IntentRouter --> DomainRoute
    DomainRoute -- "query" --> QueryAgent
    DomainRoute -- "feedback" --> FeedbackAgent
    DomainRoute -- "complaint" --> ComplaintAgent
    DomainRoute -- "order" --> OrderAgent

    %% Order Agent & Tool Loop
    OrderAgent --> OrderToolCheck
    OrderToolCheck -- "Yes (tool_calls)" --> OrderToolsNode
    OrderToolsNode --> ProfileTool
    OrderToolsNode --> GetOrderTool
    OrderToolsNode --> TakeOrderTool

    ProfileTool --> DBTables
    GetOrderTool --> DBTables
    TakeOrderTool --> ApprovalInterrupt

    ApprovalInterrupt -- "Halt & Stream SSE Event" --> ClientReq
    ClientReq -- "POST /approve (Approved)" --> TakeOrderTool
    TakeOrderTool -- "Commit with Idempotency Key" --> DBTables
    OrderToolsNode -- "Tool Output Message" --> OrderAgent

    %% Summarization Flow
    QueryAgent --> SummarizeCheck
    FeedbackAgent --> SummarizeCheck
    ComplaintAgent --> SummarizeCheck
    OrderToolCheck -- "No (Natural language reply)" --> SummarizeCheck

    SummarizeCheck -- "Yes (> 10 msgs)" --> SummarizeNode
    SummarizeNode --> EvictMsgs
    EvictMsgs --> EndNode
    SummarizeCheck -- "No (<= 10 msgs)" --> EndNode

    %% Checkpointer Linking
    LANGGRAPH_CORE -. "Save/Load State" .- Checkpointer
```

---

## 2. Core Architectural Pillars

### 1. Rolling Conversation Summarization & Message Trimming
In multi-turn dialogues, sending full historical message arrays causes token bloat, high latency, and eventual context window overflow. Keystone solves this using a two-tier memory strategy:
* **Automatic Summarization Node (`summarize_conversation_node`)**:
  * Managed via conditional edge `should_summarize` which inspects thread state after agent execution.
  * When cumulative messages exceed `SUMMARIZE_THRESHOLD` (default: 10 messages), older messages are passed to a lightweight summarizer LLM.
  * The resulting summary is stored in `state["summary"]`, and older messages are pruned from state via LangGraph `RemoveMessage` operations while keeping the last `SUMMARY_BUFFER_SIZE` (default: 4) messages intact.
  * The accumulated summary is automatically injected into downstream agent system prompts (`Summary of earlier conversation:`).
* **Sliding Window Message Trimming (`trim_conversation_messages`)**:
  * Pre-invocation token guardrail using LangChain's `trim_messages` bounded by `MAX_CONTEXT_MESSAGES` (default: 20 messages).
  * Enforces that message context cleanly begins on a `HumanMessage`, eliminating orphaned `ToolMessage` execution errors.

### 2. Thread Safety & Per-Thread Concurrency Control
Concurrent chat requests or simultaneous user approval clicks on the same session thread can corrupt the PostgreSQL checkpointer state or trigger duplicate orders.
* **`ThreadLockManager` ([`app/core/security.py`](file:///Users/vivekchole/workspace/interview-project/keystone/app/core/security.py))**:
  * Implements per-thread asynchronous locks (`asyncio.Lock` mapped by `thread_id`).
  * Enforces a configurable acquisition timeout `THREAD_LOCK_TIMEOUT` (default: 5.0 seconds).
  * Returns an immediate HTTP **`409 Conflict`** (`"A request for this chat session is currently being processed. Please wait a moment and try again."`) if a lock cannot be obtained, preventing state overwrite collisions and ensuring serial consistency.

### 3. Enterprise Guardrails & Resiliency
* **Prompt Injection & Delimiter Neutralization**:
  * Incoming customer messages are sanitized through `detect_prompt_injection` and `sanitize_delimiters`.
  * User input is wrapped within `<customer_message>...</customer_message>` boundary tags.
  * Adversarial closing tags (`</customer_message>`), system role overrides, and jailbreak commands are detected and neutralized before invoking the LLM.
* **Message Length & Context Bounding**:
  * User inputs are strictly capped at `MAX_MESSAGE_LENGTH` (1,000 characters) to mitigate token-exhaustion denial-of-service attempts.
* **LangGraph Recursion Guardrail**:
  * Every graph invocation enforces `LANGGRAPH_RECURSION_LIMIT` (default: 15 steps) to prevent cyclic or infinite tool loops.
* **Deterministic Order Idempotency**:
  * The `take_order` tool derives a SHA-256 idempotency key from `business_id:thread_id:customer_name:items:special_instructions`.
  * Backed by a PostgreSQL database-level `UniqueConstraint("business_id", "idempotency_key")`.
  * Pre-checks and post-resume handlers detect duplicate order submissions and return the existing order details rather than creating duplicate orders.
* **Tool Error Isolation**:
  * All tools (`get_customer_profile`, `take_order`, `get_order`) encapsulate database operations in robust error handling, sanitizing raw database exceptions and returning graceful user-facing error messages without crashing the graph.

### 4. Deterministic Human-in-the-Loop (`interrupt`) Breakpoints
* When the Order Specialist Agent decides to commit an order via `take_order`, graph execution reaches a native LangGraph `interrupt({"action": "order_approval", ...})`.
* Execution pauses deterministically before database insertion, streaming an SSE event indicating approval is required.
* The order is committed only when the client or supervisor sends `POST /keystone/business/{id}/chat/{thread_id}/approve` with `{"approved": true}` or confirms via chat. If rejected, the transaction is cleanly aborted.

---

## 3. Evaluation Suite (DeepEval & Deterministic Evals)

Keystone maintains a strict separation between **offline unit tests** and **live evaluation suites**:

```
keystone/
├── tests/                           # Fast, offline unit tests (46 tests, ~1.2s, no live API required)
│   ├── test_auth_gatekeeper.py
│   ├── test_concurrency_lock.py
│   ├── test_conversation_summary.py
│   ├── test_customer_profile.py
│   ├── test_message_trimming.py
│   ├── test_order_approval.py
│   ├── test_order_idempotency.py
│   ├── test_prompt_injection_guardrail.py
│   ├── test_recursion_guardrail.py
│   └── test_tool_error_resiliency.py
│
└── evals/                           # Live DeepEval evaluation suite (eval marker, LLM-as-a-judge)
    ├── conftest.py                  # GeminiModel judge fixture with LLM_MODEL_NAME & LLM_MODEL_KEY
    ├── test_faithfulness_eval.py    # FaithfulnessMetric (business context grounding & refusal)
    ├── test_deterministic_order_flow.py # Multi-turn deterministic end-to-end integration test
    └── test_agent_metrics_eval.py   # Agentic metrics: ToolCorrectness, ArgumentCorrectness,
                                     # TaskCompletion, PlanQuality, PlanAdherence, StepEfficiency
```

### Evaluated Agentic Metrics

| Metric | Target | How It Evaluates Keystone |
| :--- | :--- | :--- |
| **`FaithfulnessMetric`** | Business Q&A | Assesses factual grounding against PostgreSQL business context. Verifies that the agent answers services/timings truthfully and refuses unsupported services. |
| **`ToolCorrectnessMetric`** | Tool Selection | Evaluates whether the agent selected the appropriate tool (`get_customer_profile`) from available domain tools. |
| **`ArgumentCorrectnessMetric`**| Parameter Accuracy | Evaluates whether tool arguments (`phone_number: "9876543210"`) match the conversational context. |
| **`TaskCompletionMetric`** | Goal Attainment | Evaluates whether the requested objective (e.g. order lookup or order placement) was completely fulfilled. |
| **`PlanQualityMetric`** | Strategic Planning | Assesses whether the formulated SOP plan is logically sound and comprehensive. |
| **`PlanAdherenceMetric`** | Execution Fidelity | Measures whether live graph execution matched each planned step without skipping or hallucinating steps. |
| **`StepEfficiencyMetric`** | Trajectory Economy | Confirms the agent executed the minimal necessary steps without redundant loops, repeated routing, or errors. |

> [!TIP]
> **Dynamic Runtime Trace Generation**: In [`evals/test_agent_metrics_eval.py`](file:///Users/vivekchole/workspace/interview-project/keystone/evals/test_agent_metrics_eval.py), the `execution_trace` for `PlanQualityMetric`, `PlanAdherenceMetric`, and `StepEfficiencyMetric` is **constructed dynamically at test time from the live graph stream** (`graph.astream(...)`). It captures real node transitions, tool invocations, customer confirmation turns, and human-in-the-loop interrupts as they occur.

---

## 4. Getting Started

### 1. Prerequisites
* **Python**: 3.11+
* **Package Manager**: [`uv`](https://docs.astral.sh/uv/)
* **Docker & Docker Compose** (for PostgreSQL)

### 2. Start PostgreSQL
```bash
docker-compose up -d
```
*(Runs PostgreSQL container `keystone_postgres` on port `5432` with database `keystone`.)*

### 3. Environment Variables
Create or verify your `.env` file in the project root:

```ini
# Database configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/keystone

# LangChain / LLM Configuration
LLM_MODEL_NAME=gemini-3.5-flash-lite
LLM_MODEL_KEY=your-gemini-api-key
LLM_MODEL_PROVIDER=google_genai

# LangSmith Tracing (Optional - runs normally if omitted or false)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=keystone-chatbot

# Guardrails & Memory Limits
LANGGRAPH_RECURSION_LIMIT=15
THREAD_LOCK_TIMEOUT=5.0
MAX_CONTEXT_MESSAGES=20
SUMMARIZE_THRESHOLD=10
SUMMARY_BUFFER_SIZE=4
```

### 4. Start the Application Server
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
API documentation is interactively available at: `http://localhost:8000/docs`.

---

## 5. Running Tests & Evaluations

### Run Fast Unit Tests (Offline)
Unit tests run locally with mock checkpoints and isolated databases without consuming LLM API quota:
```bash
uv run pytest tests/ -v
```
*(46 passed in ~1.2s)*

### Run Live DeepEval Evaluations
Run the production evaluation suite against the live graph:
```bash
# Run all evaluations
uv run pytest evals/ -v

# Run specific evaluation modules
uv run pytest evals/test_faithfulness_eval.py -v
uv run pytest evals/test_deterministic_order_flow.py -v
uv run pytest evals/test_agent_metrics_eval.py -v -s
```

---

## 6. End-to-End API Walkthrough (cURL)

### 1. Register a Business
```bash
curl -X POST "http://localhost:8000/keystone/business/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Super Fast Laundry",
    "services": ["dry clean", "wash and fold", "iron press"],
    "timings": "8:00 AM to 10:00 PM Daily",
    "description": "Premium express laundry and dry cleaning services"
  }'
```
*(Copy the returned `"id"` as `BUSINESS_ID`).*

### 2. Session Auth Gatekeeper (Phone & OTP Verification)
Every new chat thread requires one-time phone verification before routing to specialist agents:

**Step 2.1: Start Conversation**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```
*Response: Prompts for 10-digit mobile number.*

**Step 2.2: Provide Mobile Number**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "9876543210"}'
```
*Response: Dispatches OTP (Demo OTP: `1234`).*

**Step 2.3: Verify OTP**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "1234"}'
```
*Response: `OTP verified successfully! How can I assist you today?`*

---

### 3. Business Q&A (Context-Grounded)
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "What services do you provide and what are your hours?"}'
```

---

### 4. Order Flow (Autonomous Profile Lookup & Human-in-the-Loop Interrupt)

**Step 4.1: Request Order (Autonomous Profile Lookup)**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "I would like to order 2 Margherita pizzas to my registered address please."}'
```
*The agent autonomously calls `get_customer_profile(phone_number="9876543210")`, retrieves Vivek's saved address (`42 Tech Boulevard, Suite 5`), presents the summary, and asks for confirmation.*

**Step 4.2: Customer Confirmation (Triggers `interrupt` Breakpoint)**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "Yes, details are correct. Please place my order."}'
```
*The agent invokes `take_order`. Execution halts at the **LangGraph interrupt breakpoint** before writing to PostgreSQL, streaming an SSE event:*
```json
data: {"node": "__interrupt__", "approval_required": true, "action": "order_approval", "content": "Approval required to place order for Vivek: ['2 Margherita pizzas']", "data": {"customer_name": "Vivek", "items": ["2 Margherita pizzas"], "special_instructions": "Deliver to 42 Tech Boulevard, Suite 5", ...}}
```

**Step 4.3: Supervisor / Customer Approval (`POST /approve`)**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01/approve" \
  -H "Content-Type: application/json" \
  -d '{"approved": true}'
```
*The graph resumes with `Command(resume={"approved": True})`, commits the order into PostgreSQL with a deterministic idempotency key, and confirms:*
```text
data: {"node": "order_tools", "content": "Success: Order placed with ID: f74b210a-3a... for Vivek. Items: ['2 Margherita pizzas']."}
data: {"node": "order", "content": "Your order for 2 Margherita pizzas has been successfully placed! Order ID: f74b210a-3a..."}
```

---

### 5. Check Order Status (`get_order` Tool)
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01" \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you check status of order ORDER_ID?"}'
```

---

### 6. Inspect Session State & History
```bash
curl -X GET "http://localhost:8000/keystone/business/BUSINESS_ID/chat/thread_demo_01"
```
*Returns the complete message history, rolling summary, authentication state, and any pending approval details.*
