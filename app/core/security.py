import re
from typing import Optional, Tuple

MAX_MESSAGE_LENGTH = 2000

INJECTION_PATTERNS = [
    (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directions|prompts|rules)",
        "instruction_override",
    ),
    (
        r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|directions|rules)",
        "instruction_override",
    ),
    (
        r"forget\s+(all\s+)?(previous|prior|everything)",
        "instruction_override",
    ),
    (
        r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+.*(developer\s+mode|unrestricted|jailbreak|dan|root|admin)",
        "persona_hijack",
    ),
    (
        r"developer\s+mode\s+(enabled|activated|on)",
        "persona_hijack",
    ),
    (
        r"(repeat|reveal|show|print|output|display)\s+(the\s+)?(system\s+prompt|initial\s+prompt|hidden\s+instructions)",
        "prompt_leak",
    ),
    (
        r"what\s+are\s+your\s+(exact\s+)?(system\s+instructions|system\s+prompts)",
        "prompt_leak",
    ),
    (
        r"(bypass|skip|disable)\s+.*(approval|confirmation|safety|guardrail)",
        "safety_bypass",
    ),
    (
        r"system\s+override",
        "system_override",
    ),
]


def detect_prompt_injection(text: str) -> Tuple[bool, Optional[str]]:
    """
    Scans raw text against high-confidence prompt injection and jailbreak patterns.
    Returns (is_injection: bool, attack_type: Optional[str]).
    """
    if not text:
        return False, None

    for pattern, attack_name in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True, attack_name

    return False, None


def sanitize_delimiters(text: str) -> str:
    """
    Escapes custom XML / boundary tags in user input to prevent delimiter breakout attacks.
    """
    if not text:
        return ""

    sanitized = text
    tag_replacements = {
        "<customer_message>": "&lt;customer_message&gt;",
        "</customer_message>": "&lt;/customer_message&gt;",
        "<system>": "&lt;system&gt;",
        "</system>": "&lt;/system&gt;",
        "<tool_call>": "&lt;tool_call&gt;",
        "</tool_call>": "&lt;/tool_call&gt;",
        "<context>": "&lt;context&gt;",
        "</context>": "&lt;/context&gt;",
    }
    for tag, replacement in tag_replacements.items():
        sanitized = re.sub(re.escape(tag), replacement, sanitized, flags=re.IGNORECASE)

    return sanitized


def format_delimited_message(text: str) -> str:
    """
    Wraps sanitized user text within boundary delimiters:
    <customer_message>
    {sanitized_text}
    </customer_message>
    """
    sanitized = sanitize_delimiters(text.strip())
    return f"<customer_message>\n{sanitized}\n</customer_message>"


def extract_raw_content(msg_content: str) -> str:
    """
    Extracts raw user message content by removing enclosing <customer_message> tags if present.
    """
    if not isinstance(msg_content, str):
        return str(msg_content)

    stripped = msg_content.strip()
    match = re.search(
        r"^<customer_message>\s*(.*?)\s*</customer_message>$",
        stripped,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return stripped
