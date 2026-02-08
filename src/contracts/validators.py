"""Validators for parsing and validating LLM outputs.

Provides utilities for:
- Extracting JSON from LLM responses (with markdown code blocks)
- Validating against Pydantic schemas with retries
- Detecting completion signals in outputs
"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# JSON Extraction
# =============================================================================


def extract_json_from_response(response: str) -> str | None:
    """Extract JSON from an LLM response that may contain markdown code blocks.

    Handles:
    - ```json ... ``` blocks
    - ``` ... ``` blocks (no language specified)
    - Raw JSON (starts with { or [)

    Returns:
        Extracted JSON string, or None if no JSON found.
    """
    # Try to find JSON in code blocks first
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            cleaned = match.strip()
            if cleaned.startswith(("{", "[")):
                return cleaned

    # Try raw JSON (find first { or [ and match to closing bracket)
    response_stripped = response.strip()
    if response_stripped.startswith("{"):
        return _extract_balanced_braces(response_stripped, "{", "}")
    elif response_stripped.startswith("["):
        return _extract_balanced_braces(response_stripped, "[", "]")

    # Last resort: find JSON-like content anywhere in the response
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", response)
    if json_match:
        return json_match.group(1)

    return None


def _extract_balanced_braces(text: str, open_char: str, close_char: str) -> str | None:
    """Extract content with balanced braces from the start of text."""
    if not text.startswith(open_char):
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[: i + 1]

    return None


# =============================================================================
# Schema Validation
# =============================================================================


def parse_and_validate(
    response: str,
    schema: type[T],
    max_retries: int = 0,
) -> tuple[T | None, list[str]]:
    """Parse LLM response and validate against a Pydantic schema.

    Args:
        response: Raw LLM response text
        schema: Pydantic model class to validate against
        max_retries: Not used in sync version (placeholder for async retry logic)

    Returns:
        Tuple of (parsed model or None, list of error messages)
    """
    errors: list[str] = []

    # Extract JSON
    json_str = extract_json_from_response(response)
    if json_str is None:
        errors.append("No JSON found in response")
        return None, errors

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        return None, errors

    # Validate against schema
    try:
        model = schema.model_validate(data)
        return model, []
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, errors


def parse_list_of(
    response: str,
    item_schema: type[T],
) -> tuple[list[T], list[str]]:
    """Parse LLM response as a list of items.

    Args:
        response: Raw LLM response text
        item_schema: Pydantic model class for list items

    Returns:
        Tuple of (list of parsed models, list of error messages)
    """
    errors: list[str] = []
    items: list[T] = []

    # Extract JSON
    json_str = extract_json_from_response(response)
    if json_str is None:
        errors.append("No JSON found in response")
        return items, errors

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        return items, errors

    # Ensure it's a list
    if not isinstance(data, list):
        # Maybe it's a single item?
        data = [data]

    # Validate each item
    for i, item_data in enumerate(data):
        try:
            item = item_schema.model_validate(item_data)
            items.append(item)
        except ValidationError as e:
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"Item {i}.{loc}: {err['msg']}")

    return items, errors


# =============================================================================
# Completion Signals
# =============================================================================


COMPLETION_SIGNALS = [
    "<promise>HYPOTHESIS_COMPLETE</promise>",
    "HYPOTHESIS_COMPLETE",
    "[COMPLETE]",
    "<<DONE>>",
]


def detect_completion_signal(response: str) -> bool:
    """Check if the response contains a completion signal.

    Completion signals indicate the LLM believes it has finished
    generating all required content.
    """
    response_upper = response.upper()
    for signal in COMPLETION_SIGNALS:
        if signal.upper() in response_upper:
            return True
    return False


def remove_completion_signals(response: str) -> str:
    """Remove completion signals from a response."""
    result = response
    for signal in COMPLETION_SIGNALS:
        result = result.replace(signal, "")
        result = result.replace(signal.upper(), "")
        result = result.replace(signal.lower(), "")
    return result.strip()


# =============================================================================
# Output Formatting
# =============================================================================


def format_as_json_prompt(schema: type[BaseModel]) -> str:
    """Generate a prompt-friendly JSON schema description.

    This is used to tell the LLM what output format we expect.
    """
    schema_dict = schema.model_json_schema()
    return json.dumps(schema_dict, indent=2)


def create_output_instruction(schema: type[BaseModel], example: BaseModel | None = None) -> str:
    """Create an output instruction block for prompts.

    Args:
        schema: The expected output schema
        example: Optional example instance

    Returns:
        Formatted instruction string
    """
    lines = [
        "<output_format>",
        "Respond with valid JSON matching this schema:",
        "```json",
        format_as_json_prompt(schema),
        "```",
    ]

    if example:
        lines.extend([
            "",
            "Example:",
            "```json",
            example.model_dump_json(indent=2),
            "```",
        ])

    lines.append("</output_format>")
    return "\n".join(lines)
