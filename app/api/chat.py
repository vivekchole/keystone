import json
from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from app.langgraph.graph import graph
from app.core.database import get_db
from app.models.business import Business

router = APIRouter(prefix="/keystone/business")


class ChatRequest(BaseModel):
    message: str


# Kept for non-streaming endpoints if needed
class ChatResponse(BaseModel):
    reply: str


class ApprovalRequest(BaseModel):
    approved: bool = True
    reason: Optional[str] = None


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

    config = {
        "configurable": {"thread_id": chat_id},
        "metadata": {
            "business_id": business_id,
            "chat_id": chat_id,
        },
        "tags": ["keystone-chat", f"business:{business_id}"],
    }

    # Check if thread has an active interrupt waiting for human approval
    state_snapshot = await graph.aget_state(config)
    has_pending_interrupt = bool(
        state_snapshot and state_snapshot.tasks and any(t.interrupts for t in state_snapshot.tasks)
    )

    if has_pending_interrupt:
        clean_msg = request.message.strip().lower()
        if clean_msg in [
            "yes", "approve", "approved", "confirm", "confirmed",
            "proceed", "ok", "yep", "sure", "place order"
        ]:
            inputs = Command(resume={"approved": True})
        elif clean_msg in [
            "no", "cancel", "cancelled", "reject", "rejected", "stop", "don't"
        ]:
            inputs = Command(resume={"approved": False, "reason": "Customer cancelled."})
        else:
            inputs = Command(resume={"approved": False, "reason": f"Customer revised: {request.message}"})
    else:
        inputs = {
            "messages": [HumanMessage(content=request.message)],
            "business_context": ctx_str,
            "business_id": business_id,
        }

    async def event_generator():
        try:
            async for output in graph.astream(inputs, config=config):
                for node_name, state_update in output.items():
                    if node_name == "__interrupt__":
                        intr_item = (
                            state_update[0]
                            if (isinstance(state_update, (list, tuple)) and state_update)
                            else None
                        )
                        intr_data = (
                            intr_item.value
                            if (intr_item and hasattr(intr_item, "value"))
                            else {}
                        )
                        payload = {
                            "node": "__interrupt__",
                            "approval_required": True,
                            "action": intr_data.get("action", "order_approval"),
                            "content": intr_data.get(
                                "message", "Order pending human approval."
                            ),
                            "data": intr_data,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif isinstance(state_update, dict) and "messages" in state_update:
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


@router.post("/{business_id}/chat/{chat_id}/approve")
async def approve_order(
    business_id: str,
    chat_id: str,
    request: ApprovalRequest,
):
    """
    Explicit endpoint for frontend UI buttons or human supervisors to approve or reject
    an order currently halted at a LangGraph interrupt.
    """
    config = {
        "configurable": {"thread_id": chat_id},
        "metadata": {
            "business_id": business_id,
            "chat_id": chat_id,
        },
        "tags": ["keystone-chat", f"business:{business_id}", "order-approval"],
    }

    state_snapshot = await graph.aget_state(config)
    has_pending_interrupt = bool(
        state_snapshot and state_snapshot.tasks and any(t.interrupts for t in state_snapshot.tasks)
    )
    if not has_pending_interrupt:
        return {"status": "error", "message": "No order pending approval for this chat session."}

    inputs = Command(resume={"approved": request.approved, "reason": request.reason})

    async def event_generator():
        try:
            async for output in graph.astream(inputs, config=config):
                for node_name, state_update in output.items():
                    if node_name == "__interrupt__":
                        intr_item = (
                            state_update[0]
                            if (isinstance(state_update, (list, tuple)) and state_update)
                            else None
                        )
                        intr_data = (
                            intr_item.value
                            if (intr_item and hasattr(intr_item, "value"))
                            else {}
                        )
                        payload = {
                            "node": "__interrupt__",
                            "approval_required": True,
                            "action": intr_data.get("action", "order_approval"),
                            "content": intr_data.get(
                                "message", "Order pending human approval."
                            ),
                            "data": intr_data,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif isinstance(state_update, dict) and "messages" in state_update:
                        messages = state_update["messages"]

                        if messages:
                            last_msg = messages[-1]
                            content = (
                                last_msg.content
                                if hasattr(last_msg, "content")
                                else str(last_msg)
                            )
                            payload = {"node": node_name, "content": content}
                            yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            import logging
            logging.error(f"Error in approval stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{business_id}/chat/{chat_id}")
async def get_chat_history(business_id: str, chat_id: str):
    config = {"configurable": {"thread_id": chat_id}}
    state_snapshot = await graph.aget_state(config)
    
    if not state_snapshot or not state_snapshot.values:
        return {
            "messages": [],
            "is_pending_approval": False,
            "pending_approval_details": [],
        }

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

    pending_interrupts = []
    if state_snapshot.tasks:
        for t in state_snapshot.tasks:
            if t.interrupts:
                for intr in t.interrupts:
                    pending_interrupts.append(
                        intr.value if hasattr(intr, "value") else str(intr)
                    )

    return {
        "messages": formatted_messages,
        "is_pending_approval": len(pending_interrupts) > 0,
        "pending_approval_details": pending_interrupts,
    }

