import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.langgraph.state import ChatbotState
from app.models.order import Order
from app.core.database import SessionLocal


def initialize_chat_model(
    model_name: str, model_key: str, model_provider: str = None
):
    """
    Initializes and returns the LangChain chat model.
    Accepts an optional model_provider to explicitly set the provider
    (e.g. 'openai', 'anthropic').
    Sets environment variables to ensure compatibility with providers
    that don't accept api_key via kwargs.
    """
    if not model_name or not model_key:
        raise ValueError(
            "Missing LLM_MODEL_NAME or LLM_MODEL_KEY in environment"
        )

    # Fallback env vars for broad compatibility
    if "gpt" in model_name or model_provider == "openai":
        os.environ["OPENAI_API_KEY"] = model_key
    elif "claude" in model_name or model_provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = model_key
    elif "gemini" in model_name or model_provider == "google_genai":
        os.environ["GOOGLE_API_KEY"] = model_key
    elif "cohere" in model_name or model_provider == "cohere":
        os.environ["COHERE_API_KEY"] = model_key

    return init_chat_model(
        model=model_name, model_provider=model_provider, api_key=model_key
    )


# Global LLM instance initialized from environment variables if present
llm = initialize_chat_model(
    model_name=os.getenv("LLM_MODEL_NAME", ""),
    model_key=os.getenv("LLM_MODEL_KEY", ""),
    model_provider=os.getenv("LLM_MODEL_PROVIDER"),
)


# =====================================================================
# ORDER DOMAIN TOOLS
# =====================================================================

@tool
def take_order(
    customer_name: str,
    items: list[str],
    special_instructions: str = "",
    business_id: str = "default",
) -> str:
    """
    Official tool to save an approved order to the database.
    Call this ONLY when customer has confirmed their order details.
    """
    db = SessionLocal()
    try:
        new_order = Order(
            business_id=business_id,
            customer_name=customer_name,
            items=items,
            special_instructions=special_instructions,
        )
        db.add(new_order)
        db.commit()
        db.refresh(new_order)
        return (
            f"Success: Order placed with ID: {new_order.id} for "
            f"{customer_name}. Items: {items}."
        )
    except Exception as e:
        return f"Error placing order: {str(e)}"
    finally:
        db.close()


@tool
def get_order(order_id: str) -> str:
    """
    Retrieve details and current status of an existing order by order ID.
    Call this when the user asks for order status and has confirmed the order ID.
    """
    db = SessionLocal()
    try:
        order = db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return f"Order not found with ID: {order_id}."
        return (
            f"Order ID: {order.id}, Customer: {order.customer_name}, "
            f"Items: {order.items}, Status: {order.status}, "
            f"Special Instructions: {order.special_instructions or 'None'}."
        )
    except Exception as e:
        return f"Error retrieving order: {str(e)}"
    finally:
        db.close()


order_tools = [take_order, get_order]
order_llm = llm.bind_tools(order_tools)


# =====================================================================
# NODES
# =====================================================================

def intent_router_node(state: ChatbotState):
    """
    Determines high-level domain intent based on user's message.
    Options: query, order, feedback, complaint.
    """
    last_message = state["messages"][-1].content
    sys_prompt = (
        "Determine the domain intent of the user. Options: query, order, "
        "feedback, complaint.\n"
        "Note: Placing, cancelling, inquiring about, checking status of, or "
        "modifying orders all fall under 'order'.\n"
        "Respond ONLY with one of those words."
    )
    response = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=last_message)]
    )

    content = response.content
    if isinstance(content, list):
        text = " ".join([b.get("text", "") for b in content if isinstance(b, dict)])
    else:
        text = str(content)

    intent = text.strip().lower()
    return {"intent": intent}


def query_node(state: ChatbotState):
    business_ctx = state.get("business_context", "No context provided")
    sys_prompt = (
        "You are a helpful assistant for a business. Answer the user's query "
        f"based ONLY on this context: {business_ctx}"
    )
    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )
    return {"messages": [response]}


def order_node(state: ChatbotState):
    """
    Autonomous Order Specialist Agent.
    Manages conversational to-and-fro with the user for:
    1. Placing an order (intake -> summarize -> wait for approval -> call take_order)
    2. Getting an order (ask for order ID -> confirm ID with user -> call get_order)
    """
    business_id = state.get("business_id", "default")
    sys_prompt = (
        f"You are the Order Specialist for this business (business_id: '{business_id}').\n"
        "You have tools: `take_order` and `get_order`.\n\n"
        "RULES FOR PLACING AN ORDER (`take_order`):\n"
        "1. Collect: Customer Name and Items ordered.\n"
        "2. If customer changes or updates items at any point, acknowledge the change "
        "and present the updated summary.\n"
        "3. ALWAYS summarize the order and ask for user confirmation/approval before placing.\n"
        "4. STRICT RULE: DO NOT call `take_order` until the user has explicitly confirmed/approved "
        "the summary. If not confirmed, reply in natural language asking for confirmation.\n"
        f"5. When approved, invoke `take_order` passing business_id='{business_id}'.\n\n"
        "RULES FOR FETCHING AN ORDER (`get_order`):\n"
        "1. Ask the customer for their Order ID if not already provided.\n"
        "2. Once the customer provides the Order ID, confirm it with them or verify that they "
        "want you to look it up.\n"
        "3. Once approved/confirmed by the customer, invoke `get_order(order_id=...)`.\n\n"
        "Never invent details. Speak politely."
    )

    response = order_llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )
    return {"messages": [response]}


def feedback_node(state: ChatbotState):
    sys_prompt = "Thank the user for their feedback."
    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )
    return {"messages": [response]}


def complaint_node(state: ChatbotState):
    sys_prompt = (
        "Apologize to the user and record their complaint regarding a "
        "previous order."
    )
    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )
    return {"messages": [response]}
