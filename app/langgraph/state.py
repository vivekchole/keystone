from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatbotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    business_context: str
    business_id: Optional[str]
    intent: Optional[str]
    order_details: Optional[dict]
    approval_status: Optional[str]
