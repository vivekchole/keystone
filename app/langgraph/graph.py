from app.core.database import checkpointer
from langgraph.graph import StateGraph, START, END
from app.langgraph.state import ChatbotState
from app.langgraph.nodes import (
    intent_router_node,
    query_node,
    order_intake_node,
    order_approval_node,
    feedback_node,
    complaint_node,
)


def route_intent(state: ChatbotState):
    intent = state.get("intent")
    if intent == "order":
        return "order_intake"
    elif intent == "order_approval":
        return "order_approval"
    elif intent == "feedback":
        return "feedback"
    elif intent == "complaint":
        return "complaint"
    else:
        return "query"  # Default fallback


builder = StateGraph(ChatbotState)

builder.add_node("intent_router", intent_router_node)
builder.add_node("query", query_node)
builder.add_node("order_intake", order_intake_node)
builder.add_node("order_approval", order_approval_node)
builder.add_node("feedback", feedback_node)
builder.add_node("complaint", complaint_node)

builder.add_edge(START, "intent_router")
builder.add_conditional_edges("intent_router", route_intent)
builder.add_edge("query", END)
builder.add_edge("order_intake", END)
builder.add_edge("order_approval", END)
builder.add_edge("feedback", END)
builder.add_edge("complaint", END)


graph = builder.compile(checkpointer=checkpointer)
