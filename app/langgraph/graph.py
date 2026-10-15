from app.core.database import checkpointer
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.langgraph.state import ChatbotState
from app.langgraph.nodes import (
    auth_gatekeeper_node,
    intent_router_node,
    query_node,
    order_node,
    order_tools,
    feedback_node,
    complaint_node,
    summarize_conversation_node,
    should_summarize,
)


def route_auth(state: ChatbotState):
    """
    Route after auth_gatekeeper:
    - If user just got authenticated in this turn or is unverified -> stop at END (yield to user)
    - If user was already authenticated from previous turn -> proceed to intent_router
    """
    if state.get("is_authenticated"):
        # Check if the gatekeeper just produced the verification greeting in this turn
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "content") and "OTP verified successfully!" in last_msg.content:
            return END
        return "intent_router"
    return END


def route_intent(state: ChatbotState):
    intent = state.get("intent")
    if intent == "order":
        return "order"
    elif intent == "feedback":
        return "feedback"
    elif intent == "complaint":
        return "complaint"
    else:
        return "query"  # Default fallback


def route_order(state: ChatbotState):
    """
    Route after order_node:
    - If the agent invoked tools -> proceed to order_tools
    - Otherwise -> check if rolling summarization is needed
    """
    route = tools_condition(state)
    if route == "tools":
        return "tools"
    return should_summarize(state)


builder = StateGraph(ChatbotState)

# Nodes
builder.add_node("auth_gatekeeper", auth_gatekeeper_node)
builder.add_node("intent_router", intent_router_node)
builder.add_node("query", query_node)
builder.add_node("order", order_node)
builder.add_node("order_tools", ToolNode(order_tools))
builder.add_node("feedback", feedback_node)
builder.add_node("complaint", complaint_node)
builder.add_node("summarize_conversation", summarize_conversation_node)

# Flow: START -> auth_gatekeeper
builder.add_edge(START, "auth_gatekeeper")
builder.add_conditional_edges(
    "auth_gatekeeper",
    route_auth,
    {
        "intent_router": "intent_router",
        END: END,
    },
)

# Route intent to domain nodes
builder.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "query": "query",
        "order": "order",
        "feedback": "feedback",
        "complaint": "complaint",
    },
)

# Standard nodes complete through rolling summarizer check
builder.add_conditional_edges(
    "query",
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        "end": END,
    },
)
builder.add_conditional_edges(
    "feedback",
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        "end": END,
    },
)
builder.add_conditional_edges(
    "complaint",
    should_summarize,
    {
        "summarize_conversation": "summarize_conversation",
        "end": END,
    },
)

# Order tool loop
builder.add_conditional_edges(
    "order",
    route_order,
    {
        "tools": "order_tools",
        "summarize_conversation": "summarize_conversation",
        "end": END,
    },
)
builder.add_edge("order_tools", "order")

# Summarize conversation completes to END
builder.add_edge("summarize_conversation", END)

graph = builder.compile(checkpointer=checkpointer)


