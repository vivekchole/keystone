# Keystone Chatbot

## 1. About the Project
Keystone is an AI-powered sales and marketing chatbot backend. It allows businesses to register their details and services, and provides an intelligent LangGraph-powered AI assistant that customers can chat with. The chatbot can answer questions about the business using its context, intelligently route intents, and interactively handle orders with approval guardrails before saving directly into a PostgreSQL database.

## 2. Architecture & Design Changes

### 1. Session Auth Gatekeeper (OTP Verification for New Chats)
For new chat sessions, users are unauthenticated by default. The `auth_gatekeeper` node guards entry into the graph:
* **Mobile Number Intake**: Prompts the user for a 10-digit phone number.
* **OTP Verification**: Sends an OTP (currently demo/hardcoded to `1234`), allowing **up to 3 attempts**.
* **Session Persistence**: Once verified, `is_authenticated=True` is recorded in state via PostgreSQL checkpointer. All subsequent turns in the thread skip the gatekeeper and flow straight to `intent_router`.

### 2. Domain Specialist Agents with Scoped Tool Chains
Rather than using a single monolithic LLM with all tools or splitting conversation stages into brittle hardcoded micro-nodes (e.g. `order_intake` vs `order_approval`), the system uses a **Hierarchical Domain-Agent Architecture**:

```
                               ┌─────────────────┐
                               │      START      │
                               └────────┬────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      auth_gatekeeper        │
                         │ (Phone & OTP check / 3 tries│
                         └──────────────┬──────────────┘
                                        │
                        Is user authenticated (True)?
                                        │
                       NO ──────────────┴────────────── YES
                       │                                 │
                       ▼                                 ▼
             (Ask for phone / OTP)              ┌─────────────────┐
                 Return to User                 │  intent_router  │
                 (up to 3 tries)                └────────┬────────┘
                                                         │
        ┌───────────────┬────────────────┬───────────────┴───────────────┐
        │ "query"       │ "order"        │ "complaint"                   │ "feedback"
        ▼               ▼                ▼                               ▼
   ┌─────────┐   ┌─────────────┐  ┌─────────────┐                  ┌──────────┐
   │  query  │   │ order_node  │  │complaint_nod│                  │ feedback │
   └────┬────┘   └──────┬──────┘  └──────┬──────┘                  └────┬─────┘
        │               │                │                              │
        │        (tool_call?)     (tool_call?)                          │
        │         NO ───┴─── YES   NO ───┴─── YES                       │
        │         │          │     │          │                         │
        │         │          ▼     │          ▼                         │
        │         │   ┌──────────┐ │   ┌──────────┐                     │
        │         │   │order_tool│ │   │complt_tol│                     │
        │         │   │(take/get)│ │   │(log/call)│                     │
        │         │   └────┬─────┘ │   └────┬─────┘                     │
        │         │        │       │        │                           │
        │         │   (loops back) │   (loops back)                     │
        │         │        └───────┼────────┘                           │
        │         ▼                ▼                                    ▼
        └───────────────────────► END ◄─────────────────────────────────┘
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
  "name": "Super Fast Laundry",
  "services": ["dry clean", "wash and fold", "iron press"],
  "timings": "8am to 10pm",
  "description": "Best laundry service in town"
}'
```
*(Copy the `"id"` from the response and replace `BUSINESS_ID` in the commands below.)*

### 2. Get Business Information
```bash
curl -X GET "http://localhost:8000/keystone/business/BUSINESS_ID"
```

### 3. Session Authentication Gatekeeper (New Chat Session)
Every new chat thread requires a one-time session authentication via phone and OTP before interacting with the assistant.

**Step 3.1: Start Chat (Prompt for Mobile Number)**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "Hello!"}'
```
*Response: Prompt to provide 10-digit mobile number.*

**Step 3.2: Send Mobile Number**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "9876543210"}'
```
*Response: OTP dispatched (Demo OTP: `1234`).*

**Step 3.3: Verify OTP**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "1234"}'
```
*Response: `OTP verified successfully! How can I assist you today?`*

---

### 4. Query Business Information (Post-Authentication)
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "What services do you offer?"}'
```

---

### 5. Create and Modify an Order (To-and-Fro Flow)

**Step 5.1: Initiate Order**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "I would like to place an order for dry cleaning"}'
```

**Step 5.2: Provide Order Details**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "My name is Vivek and I want to order 2 shirts for dry clean"}'
```
*Agent provides summary and asks for confirmation.*

**Step 5.3: Modify Order Mid-Conversation**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "Wait, change the order to 3 shirts and 1 pant"}'
```
*Agent updates the items to 3 shirts and 1 pant, re-summarizes, and asks for approval on the updated order.*

**Step 5.4: Confirm and Approve Order**
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "Yes, I approve the order"}'
```
*Agent executes `take_order` tool, writes to PostgreSQL, and returns the order confirmation with Order ID.*

---

### 6. Check Order Status (`get_order` Tool)
```bash
curl -N -X POST "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101" \
-H "Content-Type: application/json" \
-d '{"message": "Can you check status of order ORDER_ID?"}'
```
*(Replace `ORDER_ID` with the ID returned in Step 5.4).*

---

### 7. Get Complete Chat History
```bash
curl -X GET "http://localhost:8000/keystone/business/BUSINESS_ID/chat/test_chat_101"
```

