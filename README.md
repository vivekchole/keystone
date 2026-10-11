# Keystone Chatbot

## 1. About the Project
Keystone is an AI-powered sales and marketing chatbot backend. It allows businesses to register their details and services, and provides an intelligent LangGraph-powered AI assistant that customers can chat with. The chatbot can answer questions about the business using its context, intelligently route intents, and interactively handle orders with approval guardrails before saving directly into a PostgreSQL database.

## 2. Architecture & Design Changes

### Domain Specialist Agents with Scoped Tool Chains
Rather than using a single monolithic LLM with all tools or splitting conversation stages into brittle hardcoded micro-nodes (e.g. `order_intake` vs `order_approval`), the system uses a **Hierarchical Domain-Agent Architecture**:

```
                       ┌─────────────────┐
                       │  intent_router  │
                       └────────┬────────┘
                                │
        ┌───────────────┬───────┴────────┬───────────────┐
        │ "query"       │ "order"        │ "complaint"   │ "feedback"
        ▼               ▼                ▼               ▼
   ┌─────────┐   ┌─────────────┐  ┌─────────────┐   ┌──────────┐
   │  query  │   │ order_node  │  │complaint_nod│   │ feedback │
   └────┬────┘   └──────┬──────┘  └──────┬──────┘   └────┬─────┘
        │               │                │               │
        │        (tool_call?)     (tool_call?)           │
        │         NO ───┴─── YES   NO ───┴─── YES        │
        │         │          │     │          │          │
        │         │          ▼     │          ▼          │
        │         │   ┌──────────┐ │   ┌──────────┐      │
        │         │   │order_tool│ │   │complt_tol│      │
        │         │   │(take/get)│ │   │(log/call)│      │
        │         │   └────┬─────┘ │   └────┬─────┘      │
        │         │        │       │        │            │
        │         │   (loops back) │   (loops back)      │
        │         │        └───────┼────────┘            │
        │         ▼                ▼                     ▼
        └───────────────────────► END ◄──────────────────┘
```

### Why This Approach Over Binding All Tools to a Single LLM?

1. **Prevents Tool Confusion & Hallucinations**: 
   When dozens of tools are bound to a single LLM, the model frequently invokes the wrong tool, invents missing parameters, or triggers irreversible actions prematurely. Scoping tools per domain (`order_tools` only has `take_order` and `get_order`) keeps the model's decision space clean and reliable.
2. **Reduced Token Overhead & Cost**: 
   Binding all company tools to every request bloats prompt tokens on every turn. Domain routing ensures only relevant tool schemas and system instructions are loaded.
3. **Specialized Tools Per Node**: 
   Each node can evolve independently with its own domain tools:
   * **`order` node**: Specialized tools such as `take_order`, `get_order`, `cancel_order`, `apply_discount`.
   * **`complaint` node**: Specialized tools such as `log_complaint`, `escalate_to_support`, `check_ticket_status`.
   * **`query` node**: Retrieval tools such as `search_faq`, `check_business_hours`.
4. **Natural "To-and-Fro" Conversational Negotiation**:
   Instead of rigidly separating intake and approval across separate nodes, the unified `order_node` manages the full dialog loop. If the user changes an item during the approval stage (e.g., *"Change coffee to burger before confirming"*), the agent updates the summary, asks for approval again, and **only executes `take_order` when final approval is given**.
5. **Approval Guardrails for Read/Write Operations**:
   Both `take_order` (write) and `get_order` (read) have conversational approval guardrails where the agent gathers details, confirms IDs/summaries with the customer, and executes only upon confirmation.

## 3. How to Run This Project

### 1. Start the PostgreSQL Service
The database runs via Docker Compose. From the root directory, run:
```bash
docker-compose up -d
```
*(This starts the `keystone_postgres` container on port 5432.)*

### 2. Start the Python Server
We use `uv` for package management and `uvicorn` as our ASGI server. 
Make sure you have an `.env` file with your `LLM_MODEL_KEY` (e.g., Gemini or OpenAI).

Run the server with:
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Happy Flow Test (cURL)

Once the server is running, you can test the complete system flow from your terminal using these `curl` commands.

### 1. Create a Business
```bash
curl -X POST "http://localhost:8000/keystone/business/" \
-H "Content-Type: application/json" \
-d '{
  "name": "ABC Dry cleaner",
  "services": ["dry clean", "iron press"],
  "timings": "9am to 9pm",
  "description": "ABC Dry cleaner is famous dry cleaner"
}'
```
*(Copy the `"id"` from the response and replace `BUSINESS_ID` in the commands below.)*

### 2. Get Business Information
```bash
curl -X GET "http://localhost:8000/keystone/business/BUSINESS_ID"
```

### 3. Ask About Business (Chat API)
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_123" \
-H "Content-Type: application/json" \
-d '{"message": "Hi, what services do you offer?"}'
```

### 4. Create an Order (Chat API)
Placing an order requires a conversation flow:

**Initiate Order:**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_123" \
-H "Content-Type: application/json" \
-d '{"message": "I would like to place an order for dry cleaning"}'
```

**Provide Details:**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_123" \
-H "Content-Type: application/json" \
-d '{"message": "My name is Vivek and I want to order 2 shirts for dry clean"}'
```

**Approve Order:**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_123" \
-H "Content-Type: application/json" \
-d '{"message": "Yes, I approve the order"}'
```

### 5. Get Chat History
```bash
curl -X GET "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_123"
```
