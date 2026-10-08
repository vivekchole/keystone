import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.langgraph.state import ChatbotState
from pydantic import BaseModel, Field
from app.models.order import Order
from app.core.database import SessionLocal


class TakeOrderTool(BaseModel):
    """Call this tool to officially submit an approved order."""

    customer_name: str = Field(description="The name of the customer")
    items: list[str] = Field(description="List of items or services ordered")
    special_instructions: str = Field(
        description="Any special instructions or notes", default=""
    )


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


def intent_router_node(state: ChatbotState):
    # Determine the intent based on user's message
    last_message = state["messages"][-1].content
    sys_prompt = (
        "Determine the intent of the user. Options: query, order, "
        "order_approval, feedback, complaint. Respond ONLY with one "
        "of those words."
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
    # Answer questions using business context
    business_ctx = state.get("business_context", "No context provided")
    sys_prompt = (
        "You are a helpful assistant for a business. Answer the user's query "
        f"based ONLY on this context: {business_ctx}"
    )

    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )
    return {"messages": [response]}


def order_intake_node(state: ChatbotState):
    # Collect order details or confirm order
    sys_prompt = (
        "You are taking an order for the business. You MUST collect the "
        "following required details before confirming:\n"
        "1. Customer Name\n"
        "2. Items to be ordered\n"
        "If any of these are missing, politely ask the user for them. "
        "Once you have all the required details, summarize the order and "
        "explicitly ask the user for confirmation/approval."
    )
    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )

    # We could do some logic to check if LLM asked for confirmation
    # For now, just return the response
    return {"messages": [response]}


def take_order(business_id: str, order_data: dict):
    db = SessionLocal()
    try:
        new_order = Order(
            business_id=business_id,
            customer_name=order_data.get("customer_name", "Unknown"),
            items=order_data.get("items", []),
            special_instructions=order_data.get("special_instructions", ""),
        )
        db.add(new_order)
        db.commit()
        print(f"[*] Order saved to DB! ID: {new_order.id}")
    except Exception as e:
        print(f"[*] Error saving order: {e}")
    finally:
        db.close()
    return True


def order_approval_node(state: ChatbotState):
    # Bind the tool to the LLM
    llm_with_tools = llm.bind_tools([TakeOrderTool])

    sys_prompt = (
        "The user was asked to approve an order. Determine if they "
        "approved or "
        "rejected it. If they approved, call the TakeOrderTool with the "
        "extracted details from the conversation. If rejected, just respond "
        "nicely and ask what they want to change."
    )

    response = llm_with_tools.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"]
    )

    # Check if the LLM decided to call the tool (meaning it was approved)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        args = tool_call["args"]

        business_id = state.get("business_id", "unknown")
        take_order(business_id, args)

        # Override the LLM's raw response for the user
        response = AIMessage(
            content="Order confirmed successfully! We have submitted your "
            "request."
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
