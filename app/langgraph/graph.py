from app.core.database import checkpointer
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from app.langgraph.state import ChatbotState
from app.langgraph.nodes import (
    intent_router_node,
    query_node,
    order_node,
    order_tools,
    feedback_node,
    complaint_node,
)


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


builder = StateGraph(ChatbotState)

# Nodes
builder.add_node("intent_router", intent_router_node)
builder.add_node("query", query_node)
builder.add_node("order", order_node)
builder.add_node("order_tools", ToolNode(order_tools))
builder.add_node("feedback", feedback_node)
builder.add_node("complaint", complaint_node)

# Flow
builder.add_edge(START, "intent_router")
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

# Standard nodes complete to END
builder.add_edge("query", END)
builder.add_edge("feedback", END)
builder.add_edge("complaint", END)

# Order tool loop
builder.add_conditional_edges(
    "order",
    tools_condition,
    {
        "tools": "order_tools",
        END: END,
    },
)
builder.add_edge("order_tools", "order")


graph = builder.compile(checkpointer=checkpointer)

