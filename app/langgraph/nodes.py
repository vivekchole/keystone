import os
import re
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.langgraph.state import ChatbotState
from app.models.order import Order
from app.core.database import SessionLocal
from app.langgraph.prompts import get_prompt_config


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

def auth_gatekeeper_node(state: ChatbotState):
    """
    Session Gatekeeper for new threads:
    1. If already authenticated -> pass through immediately.
    2. If no phone number yet -> prompt for mobile number.
    3. If phone received -> send hardcoded OTP ('1234') and prompt for OTP.
    4. If waiting for OTP -> verify OTP (allow up to 3 attempts).
    """
    if state.get("is_authenticated"):
        return {}

    last_user_msg = state["messages"][-1].content.strip()
    phone = state.get("phone_number")
    otp = state.get("otp")
    attempts = state.get("otp_attempts") or 0

    # Step 1: No phone number registered yet
    if not phone:
        # Check if user provided a phone number (e.g. 10 digits)
        digits_only = re.sub(r"\D", "", last_user_msg)
        if len(digits_only) < 10:
            reply = (
                "Welcome! To start chatting, please provide your 10-digit "
                "mobile number for verification."
            )
            return {"messages": [AIMessage(content=reply)]}

        detected_phone = digits_only[-10:]
        hardcoded_otp = "1234"

        reply = (
            f"We have sent an OTP to {detected_phone}. "
            f"(Demo OTP: {hardcoded_otp})\n"
            "Please enter the OTP to verify your session:"
        )
        return {
            "phone_number": detected_phone,
            "otp": hardcoded_otp,
            "otp_attempts": 0,
            "messages": [AIMessage(content=reply)],
        }

    # Step 2: Phone is registered, verify OTP
    entered_otp = re.sub(r"\D", "", last_user_msg)
    if entered_otp == otp or last_user_msg == otp:
        reply = "OTP verified successfully! How can I assist you today?"
        return {
            "is_authenticated": True,
            "messages": [AIMessage(content=reply)],
        }
    else:
        attempts += 1
        if attempts >= 3:
            reply = (
                "You have exceeded the maximum of 3 OTP attempts. "
                "Please provide your mobile number again to restart verification."
            )
            return {
                "phone_number": None,
                "otp": None,
                "otp_attempts": 0,
                "messages": [AIMessage(content=reply)],
            }
        else:
            remaining = 3 - attempts
            reply = (
                f"Incorrect OTP. Please re-enter the OTP "
                f"({remaining} attempt(s) remaining):"
            )
            return {
                "otp_attempts": attempts,
                "messages": [AIMessage(content=reply)],
            }


def intent_router_node(state: ChatbotState):
    """
    Determines high-level domain intent based on user's message.
    Options: query, order, feedback, complaint.
    """
    cfg = get_prompt_config("intent_router")
    sys_prompt = cfg["template"]
    version = cfg["version"]

    last_message = state["messages"][-1].content
    response = llm.invoke(
        [SystemMessage(content=sys_prompt), HumanMessage(content=last_message)],
        config={
            "metadata": {
                "prompt_name": "intent_router",
                "prompt_version": version,
            },
            "tags": ["prompt:intent_router", f"version:{version}"],
        },
    )

    content = response.content
    if isinstance(content, list):
        text = " ".join([b.get("text", "") for b in content if isinstance(b, dict)])
    else:
        text = str(content)

    intent = text.strip().lower()
    return {"intent": intent}


def query_node(state: ChatbotState):
    cfg = get_prompt_config("query")
    version = cfg["version"]
    business_ctx = state.get("business_context", "No context provided")
    sys_prompt = cfg["template"].format(business_context=business_ctx)

    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"],
        config={
            "metadata": {
                "prompt_name": "query",
                "prompt_version": version,
            },
            "tags": ["prompt:query", f"version:{version}"],
        },
    )
    return {"messages": [response]}


def order_node(state: ChatbotState):
    """
    Autonomous Order Specialist Agent.
    Manages conversational to-and-fro with the user for:
    1. Placing an order (intake -> summarize -> wait for approval -> call take_order)
    2. Getting an order (ask for order ID -> confirm ID with user -> call get_order)
    """
    cfg = get_prompt_config("order")
    version = cfg["version"]
    business_id = state.get("business_id", "default")
    sys_prompt = cfg["template"].format(business_id=business_id)

    response = order_llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"],
        config={
            "metadata": {
                "prompt_name": "order",
                "prompt_version": version,
            },
            "tags": ["prompt:order", f"version:{version}"],
        },
    )
    return {"messages": [response]}


def feedback_node(state: ChatbotState):
    cfg = get_prompt_config("feedback")
    version = cfg["version"]
    sys_prompt = cfg["template"]

    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"],
        config={
            "metadata": {
                "prompt_name": "feedback",
                "prompt_version": version,
            },
            "tags": ["prompt:feedback", f"version:{version}"],
        },
    )
    return {"messages": [response]}


def complaint_node(state: ChatbotState):
    cfg = get_prompt_config("complaint")
    version = cfg["version"]
    sys_prompt = cfg["template"]

    response = llm.invoke(
        [SystemMessage(content=sys_prompt)] + state["messages"],
        config={
            "metadata": {
                "prompt_name": "complaint",
                "prompt_version": version,
            },
            "tags": ["prompt:complaint", f"version:{version}"],
        },
    )
    return {"messages": [response]}

