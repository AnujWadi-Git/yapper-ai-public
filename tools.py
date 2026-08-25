"""Tool-use layer for Yapper AI.

Register a tool by adding an entry to TOOL_SPECS (the JSON schema the LLM sees)
and a matching callable in TOOL_IMPLS (keyed by the same name). Implementations
are plain sync functions that take keyword args and return a string.

Tools run synchronously inside the /chat request, so keep them fast. Anything
that can take minutes does not belong here.
"""

import ast
import json
import operator
from datetime import datetime, timezone

import httpx

# --- Tool specifications (sent to the LLM via the OpenRouter `tools` field) ---

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current UTC date and time. Use when the user asks what time or date it is.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location by name (city, town, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Plain place name only, e.g. 'Paris' or 'St. Louis' (no state or country suffix).",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Look up factual information about a topic, person, place, or thing on "
                "Wikipedia. Use for general knowledge questions the user asks about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to look up, e.g. 'Alan Turing' or 'photosynthesis'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a mathematical expression and return the result. Use for any "
                "arithmetic, so you never guess at math."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '(1234 * 5.5) / 3' or '2 ** 10'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount from one currency to another using live exchange rates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert."},
                    "from_currency": {"type": "string", "description": "3-letter code, e.g. 'USD'."},
                    "to_currency": {"type": "string", "description": "3-letter code, e.g. 'EUR'."},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search recent tech, startup, and science news headlines (from Hacker "
                "News) about a topic. Use for current tech events or trends that static "
                "knowledge would not cover."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search the news for, e.g. 'OpenAI' or 'quantum computing'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# --- Tool implementations (name -> sync callable returning a string) ---

def _get_current_time() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


def _get_weather(location: str) -> str:
    # Uses the free open-meteo geocoding + forecast APIs (no API key required).
    with httpx.Client(timeout=15) as client:
        geo = client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1},
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"No location found matching '{location}'."

        place = results[0]
        forecast = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current_weather": True,
            },
        ).json()

    current = forecast.get("current_weather") or {}
    if not current:
        return f"Weather data unavailable for '{location}'."

    name = place.get("name", location)
    country = place.get("country", "")
    where = f"{name}, {country}".strip(", ")
    return (
        f"Weather in {where}: {current.get('temperature')}°C, "
        f"wind {current.get('windspeed')} km/h."
    )


def _search_wikipedia(query: str) -> str:
    # Uses the free Wikipedia REST summary API (no API key required).
    with httpx.Client(timeout=15, follow_redirects=True) as client:
        # Resolve the query to a page title first, then fetch its summary.
        search = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
            headers={"User-Agent": "Yapper-AI/1.0"},
        ).json()
        hits = search.get("query", {}).get("search") or []
        if not hits:
            return f"No Wikipedia article found for '{query}'."

        title = hits[0]["title"]
        summary = client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": "Yapper-AI/1.0"},
        ).json()

    extract = summary.get("extract")
    if not extract:
        return f"No summary available for '{title}'."
    return f"{title}: {extract}"


# Safe arithmetic: only these AST node/operator types are allowed. No names,
# calls, or attribute access, so there is no code-execution surface.
_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_math_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_eval_math_node(node.left), _eval_math_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _MATH_OPS:
        return _MATH_OPS[type(node.op)](_eval_math_node(node.operand))
    raise ValueError("unsupported expression")


def _calculate(expression: str) -> str:
    try:
        result = _eval_math_node(ast.parse(expression, mode="eval").body)
    except Exception:
        return f"Could not evaluate '{expression}'. Use only numbers and + - * / // % ** ( )."
    return f"{expression} = {result}"


def _convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    src = from_currency.upper()
    dst = to_currency.upper()
    # Free, no-key exchange rate API.
    with httpx.Client(timeout=15) as client:
        data = client.get(f"https://open.er-api.com/v6/latest/{src}").json()

    if data.get("result") != "success":
        return f"Could not get exchange rates for '{src}'."
    rate = (data.get("rates") or {}).get(dst)
    if rate is None:
        return f"Unknown currency '{dst}'."

    converted = amount * rate
    return f"{amount} {src} = {converted:.2f} {dst} (rate {rate})."


def _search_news(query: str) -> str:
    # Uses the free Hacker News (Algolia) search API (no API key required).
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={"query": query, "tags": "story", "hitsPerPage": 5},
            headers={"User-Agent": "Yapper-AI/1.0"},
        )

    try:
        hits = resp.json().get("hits") or []
    except ValueError:
        return f"News service returned an unexpected response for '{query}'."

    lines = []
    for hit in hits:
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        date = (hit.get("created_at") or "")[:10]  # YYYY-MM-DD
        points = hit.get("points", 0)
        lines.append(f"- {title} ({points} points, {date})")

    if not lines:
        return f"No recent tech news found for '{query}'."
    return f"Recent tech news for '{query}':\n" + "\n".join(lines)


TOOL_IMPLS = {
    "get_current_time": _get_current_time,
    "get_weather": _get_weather,
    "search_wikipedia": _search_wikipedia,
    "calculate": _calculate,
    "convert_currency": _convert_currency,
    "search_news": _search_news,
}


def execute_tool_call(tool_call: dict) -> str:
    """Run one tool call from the LLM and return a string result.

    `tool_call` is an OpenRouter/OpenAI tool_call object:
    {"id": ..., "function": {"name": ..., "arguments": "<json string>"}}
    Never raises; failures come back as a string the LLM can read and recover from.
    """
    name = tool_call.get("function", {}).get("name", "")
    raw_args = tool_call.get("function", {}).get("arguments") or "{}"

    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"Error: unknown tool '{name}'."

    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    except json.JSONDecodeError:
        return f"Error: could not parse arguments for '{name}': {raw_args}"

    try:
        return str(impl(**args))
    except Exception as error:  # noqa: BLE001 - surface any failure back to the LLM
        return f"Error running '{name}': {error}"
