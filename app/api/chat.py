import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.langgraph.graph import graph
from app.core.database import get_db
from app.models.business import Business

router = APIRouter(prefix="/keystone/business")


class ChatRequest(BaseModel):
    message: str


# Kept for non-streaming endpoints if needed
class ChatResponse(BaseModel):
    reply: str


@router.post("/{business_id}/chat/{chat_id}")
async def chat_with_business(
    business_id: str,
    chat_id: str,
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    # Retrieve business context from DB
    business_info = (
        db.query(Business).filter(Business.id == business_id).first()
    )
    if not business_info:
        return {"error": "Business not found."}

    desc_str = (
        f" Description: {business_info.description}."
        if business_info.description
        else ""
    )
    ctx_str = (
        f"Business Name: {business_info.name}.{desc_str} "
        f"Services: {', '.join(business_info.services)}. "
        f"Timings: {business_info.timings}"
    )

    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "business_context": ctx_str,
        "business_id": business_id,
    }

    async def event_generator():
        # Stream updates from the LangGraph execution
        config = {"configurable": {"thread_id": chat_id}}

        try:
            async for output in graph.astream(inputs, config=config):
                for node_name, state_update in output.items():
                    if "messages" in state_update:
                        messages = state_update["messages"]
                        if messages:
                            last_msg = messages[-1]
                            content = (
                                last_msg.content
                                if hasattr(last_msg, "content")
                                else str(last_msg)
                            )

                            # Format as SSE
                            payload = {"node": node_name, "content": content}
                            yield f"data: {json.dumps(payload)}\n\n"

            # Send completion event
            yield "data: [DONE]\n\n"
        except Exception as e:
            import logging
            logging.error(f"Error in chat stream: {e}", exc_info=True)
            error_msg = str(e) if str(e) else repr(e)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{business_id}/chat/{chat_id}")
async def get_chat_history(business_id: str, chat_id: str):
    config = {"configurable": {"thread_id": chat_id}}
    state_snapshot = await graph.aget_state(config)
    
    if not state_snapshot or not state_snapshot.values:
        return {"messages": []}

    messages = state_snapshot.values.get("messages", [])
    
    # Format messages for the frontend
    formatted_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content if hasattr(msg, "content") else str(msg)
        formatted_messages.append({
            "role": role,
            "content": content
        })

    return {"messages": formatted_messages}
