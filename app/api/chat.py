import json
import logging
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from app.langgraph.graph import graph
from app.core.database import get_db
from app.core.lock_manager import thread_lock_manager
from app.models.business import Business

logger = logging.getLogger(__name__)

# Configurable recursion limit to prevent runaway LLM agent loops
DEFAULT_RECURSION_LIMIT = 15
RECURSION_LIMIT = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT)))

router = APIRouter(prefix="/keystone/business")


class ChatRequest(BaseModel):
    message: str


# Kept for non-streaming endpoints if needed
class ChatResponse(BaseModel):
    reply: str


class ApprovalRequest(BaseModel):
    approved: bool = True
    reason: Optional[str] = None


def extract_text_content(msg) -> str:
    """Extract clean string text from message content, supporting string or list of dict blocks."""
    if not hasattr(msg, "content"):
        return str(msg)
    content = msg.content
    if isinstance(content, list):
        return " ".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]) or str(content)
    return str(content)


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

    # Acquire per-thread lock to prevent concurrent checkpoint mutations & race conditions
    acquired = await thread_lock_manager.acquire(chat_id)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A request is already being processed for this chat session. Please wait a moment.",
        )

    try:
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
            "recursion_limit": RECURSION_LIMIT,
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
    except Exception:
        await thread_lock_manager.release(chat_id)
        raise

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
                    elif (
                        node_name != "summarize_conversation"
                        and isinstance(state_update, dict)
                        and "messages" in state_update
                    ):
                        messages = state_update["messages"]

                        if messages:
                            last_msg = messages[-1]
                            content = extract_text_content(last_msg)

                            # Format as SSE
                            payload = {"node": node_name, "content": content}
                            yield f"data: {json.dumps(payload)}\n\n"

            # Send completion event
            yield "data: [DONE]\n\n"
        except GraphRecursionError as e:
            logger.warning(
                "Recursion limit (%s) exceeded for chat_id=%s, business_id=%s: %s",
                RECURSION_LIMIT,
                chat_id,
                business_id,
                e,
            )
            payload = {
                "node": "guardrail",
                "error_type": "recursion_limit",
                "content": "I apologize, but this request took too many steps to resolve. Could you please clarify or rephrase your request?",
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in chat stream: {e}", exc_info=True)
            error_msg = str(e) if str(e) else repr(e)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await thread_lock_manager.release(chat_id)

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
    # Acquire per-thread lock to prevent concurrent checkpoint mutations & race conditions
    acquired = await thread_lock_manager.acquire(chat_id)
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A request is already being processed for this chat session. Please wait a moment.",
        )

    try:
        config = {
            "configurable": {"thread_id": chat_id},
            "metadata": {
                "business_id": business_id,
                "chat_id": chat_id,
            },
            "tags": ["keystone-chat", f"business:{business_id}", "order-approval"],
            "recursion_limit": RECURSION_LIMIT,
        }

        state_snapshot = await graph.aget_state(config)
        has_pending_interrupt = bool(
            state_snapshot and state_snapshot.tasks and any(t.interrupts for t in state_snapshot.tasks)
        )
        if not has_pending_interrupt:
            await thread_lock_manager.release(chat_id)
            return {"status": "error", "message": "No order pending approval for this chat session."}

        inputs = Command(resume={"approved": request.approved, "reason": request.reason})
    except Exception:
        await thread_lock_manager.release(chat_id)
        raise

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
                    elif (
                        node_name != "summarize_conversation"
                        and isinstance(state_update, dict)
                        and "messages" in state_update
                    ):
                        messages = state_update["messages"]

                        if messages:
                            last_msg = messages[-1]
                            content = extract_text_content(last_msg)
                            payload = {"node": node_name, "content": content}
                            yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"
        except GraphRecursionError as e:
            logger.warning(
                "Recursion limit (%s) exceeded in approval stream for chat_id=%s, business_id=%s: %s",
                RECURSION_LIMIT,
                chat_id,
                business_id,
                e,
            )
            payload = {
                "node": "guardrail",
                "error_type": "recursion_limit",
                "content": "I apologize, but processing this order approval exceeded the maximum allowed steps. Please contact support or retry.",
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in approval stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await thread_lock_manager.release(chat_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{business_id}/chat/{chat_id}")
async def get_chat_history(business_id: str, chat_id: str):
    config = {"configurable": {"thread_id": chat_id}}
    try:
        async with thread_lock_manager.lock(chat_id, timeout=2.0):
            state_snapshot = await graph.aget_state(config)
    except TimeoutError:
        state_snapshot = await graph.aget_state(config)
    
    if not state_snapshot or not state_snapshot.values:
        return {
            "summary": None,
            "messages": [],
            "is_pending_approval": False,
            "pending_approval_details": [],
        }

    summary = state_snapshot.values.get("summary") or None
    messages = state_snapshot.values.get("messages", [])
    
    # Format messages for the frontend
    formatted_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = extract_text_content(msg)
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
        "summary": summary,
        "messages": formatted_messages,
        "is_pending_approval": len(pending_interrupts) > 0,
        "pending_approval_details": pending_interrupts,
    }

