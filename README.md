# Keystone Chatbot

## 1. About the Project
Keystone is an AI-powered sales and marketing chatbot backend. It allows businesses to register their details and services, and provides an intelligent LangGraph-powered AI assistant that customers can chat with. The chatbot can answer questions about the business using its context, intelligently route intents, and seamlessly transition into an "order intake" flow to collect customer details and place orders directly into a PostgreSQL database.

## 2. How to Run This Project

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

## 3. Happy Flow Test (cURL)

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
