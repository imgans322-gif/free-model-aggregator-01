#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

The server reads its brief (MAKEFUN_PROMPT_FILE): the provider directory,
the sealed quality table and the workload manifest. It probes every
provider's catalog route in that provider's own wire, builds one catalog
document, registers it with the env (MAKEFUN_ENV_URL), then serves the
client's twenty requests:

  POST /v1/chat/completions   OpenAI Chat Completions, JSON or SSE
  POST /v1/messages           Anthropic Messages, JSON or SSE
  GET  /v1/models             OpenAI list for Bearer auth, Anthropic for x-api-key

Each request is routed to the best free model that meets its requirements
(tools / vision / JSON mode / minimum context, from the manifest merged with
what the body itself asks for), translated into that provider's wire
(OpenAI, Anthropic or Gemini) with the system prompt, user text, images,
tool names, stop sequences, max_tokens and JSON mode preserved exactly --
the providers echo what they parsed and the client scores that echo.
A refusal (402 / 429 / 5xx) fails over to the next candidate; a provider
that keeps refusing is demoted and finally set aside. `python3 main.py
--list` prints the catalog document, byte for byte the one registered.

Standard library only. Deterministic: no wall clock and no randomness in
anything sent or answered, so a replay reproduces the same state export.
"""

import copy
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
UPSTREAM_TIMEOUT_S = 15.0
REGISTER_TIMEOUT_S = 5.0
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# The most calls any provider can still answer in one trial (the sealed
# budget ranges' upper ends). Once our own counted attempts reach this,
# the provider's free budget is certainly spent and it is not tried again
# while any other candidate remains.
BUDGET_MAX = {
    "groq": 12,
    "cerebras": 6,
    "gemini": 8,
    "zai": 5,
    "openrouter": 12,
    "sambanova": 5,
    "pollinations": 14,
    "cloudflare": 8,
    "ovh": 6,
    "kilol": 16,
}
# Consecutive refusals after which a provider is treated as out (budget
# spent or outage) while alternatives remain.
MAX_CONSECUTIVE_FAILURES = 3

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --------------------------------------------------------------------------
# The brief
# --------------------------------------------------------------------------

def documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    decoder = json.JSONDecoder()
    out = []
    index = text.find("{")
    while index != -1:
        try:
            value, _end = decoder.raw_decode(text, index)
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        except Exception:
            pass
        index = text.find("{", index + 1)
    return out


def string_of(value):
    return value if isinstance(value, str) else ""


def provider_of(value):
    if not is_record(value):
        return None
    pid = string_of(value.get("id"))
    wire = value.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = value.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(value.get(key)) != "":
            base = string_of(value.get(key))
            break
    return {
        "id": pid,
        "name": string_of(value.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": string_of(value.get("key_env")),
        "key": string_of(value.get("key")),
    }


def rating_of(value):
    if not is_record(value):
        return None
    model = string_of(value.get("model"))
    provider = string_of(value.get("provider"))
    if model == "" or provider == "":
        return None
    rating = value.get("rating_milli")
    context = value.get("context_length")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if is_number(rating) else 0,
        "context_length": context if is_number(context) else None,
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is True,
    }


def workload_request_of(value):
    if not is_record(value):
        return None
    rid = string_of(value.get("id"))
    if rid == "":
        return None
    requires = value.get("requires") if is_record(value.get("requires")) else {}
    parsed = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    if is_number(requires.get("min_context")):
        parsed["min_context"] = requires["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": parsed,
    }


def read_brief(path):
    brief = {"providers": [], "ratings": {}, "manifest": [], "error": None}
    if path == "":
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        brief["error"] = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory_doc = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory_doc = doc
        elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory_doc is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for raw in directory_doc["providers"]:
        provider = provider_of(raw)
        if provider is not None:
            brief["providers"].append(provider)
    if ratings_doc is not None:
        for raw in ratings_doc["ratings"]:
            row = rating_of(raw)
            if row is not None:
                brief["ratings"][(row["provider"], row["model"])] = row
    if workload_doc is not None:
        for raw in workload_doc["manifest"]:
            entry = workload_request_of(raw)
            if entry is not None:
                brief["manifest"].append(entry)
    return brief


# --------------------------------------------------------------------------
# Provider catalogs
# --------------------------------------------------------------------------

def join_base(env_url, base):
    text = (base or "").strip()
    if re.match(r"^https?://", text, re.IGNORECASE):
        return text.rstrip("/")
    prefix = env_url.rstrip("/")
    if text == "":
        return prefix
    return prefix + ("" if text.startswith("/") else "/") + text.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers=None, body_obj=None, timeout=UPSTREAM_TIMEOUT_S):
    """One HTTP call. Returns (status, text); HTTP errors return their status."""
    data = None
    sent_headers = dict(headers or {})
    if body_obj is not None:
        data = json.dumps(body_obj).encode("utf-8")
        sent_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=sent_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read().decode("utf-8", "replace")
        except Exception:
            return err.code, ""


def number_or_none(value):
    return value if is_number(value) else None


def zero_priced(model):
    pricing = model.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for item in pricing.values():
        if is_number(item):
            values.append(float(item))
        elif isinstance(item, str) and item.strip() != "":
            try:
                values.append(float(item))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def model_in(wire, value):
    if not is_record(value):
        return None
    if wire == "gemini":
        model_id = string_of(value.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = string_of(value.get("id"))
    if model_id == "":
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        if is_number(value.get(key)):
            context = value[key]
            break
    free = value.get("free")
    if not isinstance(free, bool):
        free = zero_priced(value)
    display = string_of(value.get("display_name")) or string_of(value.get("displayName")) or model_id
    return {
        "id": model_id,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": value.get("tools") if isinstance(value.get("tools"), bool) else None,
        "vision": value.get("vision") if isinstance(value.get("vision"), bool) else None,
        "json": value.get("json") if isinstance(value.get("json"), bool) else None,
    }


def probe_provider(env_url, provider):
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    probe = {"provider": provider, "ok": False, "models": [], "error": None}
    if env_url == "":
        probe["error"] = "MAKEFUN_ENV_URL is not set"
        return probe
    try:
        status, text = http_request("GET", url, headers=headers, timeout=PROBE_TIMEOUT_S)
    except Exception as err:  # connection refused, timeout, DNS: provider is out
        probe["error"] = "GET %s failed: %s" % (url, err)
        return probe
    if status < 200 or status >= 300:
        probe["error"] = "GET %s answered HTTP %d" % (url, status)
        return probe
    try:
        payload = json.loads(text)
    except ValueError:
        probe["error"] = "GET %s answered a body that is not JSON" % url
        return probe
    if not is_record(payload):
        probe["error"] = "GET %s answered a body that is not one JSON object" % url
        return probe
    items = payload.get("models") if provider["wire"] == "gemini" else payload.get("data")
    if not isinstance(items, list):
        items = []
    models = []
    for raw in items:
        model = model_in(provider["wire"], raw)
        if model is not None:
            models.append(model)
    probe["ok"] = True
    probe["models"] = models
    return probe


def probe_all(env_url, providers):
    results = [None] * len(providers)

    def work(index, provider):
        results[index] = probe_provider(env_url, provider)

    threads = []
    for index, provider in enumerate(providers):
        thread = threading.Thread(target=work, args=(index, provider), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(PROBE_TIMEOUT_S + 2)
    for index, probe in enumerate(results):
        if probe is None:
            results[index] = {
                "provider": providers[index],
                "ok": False,
                "models": [],
                "error": "the catalog probe did not finish",
            }
    return results


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def build_catalog(probes, ratings):
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        provider = probe["provider"]
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = ratings.get((provider["id"], model["id"]))
            context = row["context_length"] if row is not None else model["context_length"]
            models.append(
                {"id": model["id"], "free": model_free(model, row), "context_length": context}
            )
        providers.append(
            {
                "id": provider["id"],
                "base_url": provider["base"],
                "key_env": provider["key_env"],
                "wire": provider["wire"],
                "models": models,
            }
        )
    return {"providers": providers}


# --------------------------------------------------------------------------
# Neutral request form: what every client wire is translated through.
# A neutral message keeps the joined text of its text blocks (the exact
# string each provider's parser reconstructs), its images, its tool calls
# and its tool results, so every upstream wire echoes the same fields.
# --------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = blocks_text(content)
        return text if text != "" else json.dumps(content)
    if content is None:
        return ""
    return json.dumps(content)


def parse_arguments(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if is_record(parsed):
                return parsed
        except ValueError:
            pass
    return {}


DATA_URL_RE = re.compile(r"^data:([^,]*);base64,(.*)$", re.DOTALL)


def parse_image_url(url):
    match = DATA_URL_RE.match(url)
    if match:
        media = match.group(1).split(";")[0] or "image/png"
        return {"kind": "data", "media": media, "data": match.group(2)}
    return {"kind": "url", "url": url}


def neutral_message(role, text="", images=None, tool_calls=None, tool_results=None):
    return {
        "role": role,
        "text": text,
        "images": images or [],
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
    }


def neutral_from_openai(body):
    system_parts = []
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = blocks_text(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append(
                neutral_message(
                    "user",
                    tool_results=[
                        {
                            "id": string_of(raw.get("tool_call_id")),
                            "text": tool_result_text(content),
                        }
                    ],
                )
            )
            continue
        text = ""
        images = []
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    url = block["image_url"].get("url")
                    if isinstance(url, str):
                        images.append(parse_image_url(url))
            text = "\n".join(texts)
        calls = []
        for call in raw.get("tool_calls") or []:
            if is_record(call) and is_record(call.get("function")):
                fn = call["function"]
                calls.append(
                    {
                        "id": string_of(call.get("id")),
                        "name": string_of(fn.get("name")),
                        "args": parse_arguments(fn.get("arguments")),
                    }
                )
        messages.append(
            neutral_message(
                "assistant" if role == "assistant" else "user",
                text=text,
                images=images,
                tool_calls=calls,
            )
        )
    return "\n".join(system_parts), messages


def neutral_from_anthropic(body):
    system_text = blocks_text(body.get("system"))
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = "assistant" if raw.get("role") == "assistant" else "user"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append(neutral_message(role, text=content))
            continue
        texts = []
        images = []
        calls = []
        results = []
        if isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                kind = block.get("type")
                if kind == "text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
                elif kind == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        images.append(
                            {
                                "kind": "data",
                                "media": string_of(source.get("media_type")) or "image/png",
                                "data": source["data"],
                            }
                        )
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        images.append({"kind": "url", "url": source["url"]})
                elif kind == "tool_use":
                    calls.append(
                        {
                            "id": string_of(block.get("id")),
                            "name": string_of(block.get("name")),
                            "args": block.get("input") if is_record(block.get("input")) else {},
                        }
                    )
                elif kind == "tool_result":
                    results.append(
                        {
                            "id": string_of(block.get("tool_use_id")),
                            "text": tool_result_text(block.get("content")),
                        }
                    )
        messages.append(
            neutral_message(
                role,
                text="\n".join(texts),
                images=images,
                tool_calls=calls,
                tool_results=results,
            )
        )
    return system_text, messages


def tools_from_openai(body):
    tools = []
    for raw in body.get("tools") or []:
        if is_record(raw) and is_record(raw.get("function")) and isinstance(raw["function"].get("name"), str):
            fn = raw["function"]
            tools.append(
                {
                    "name": fn["name"],
                    "description": string_of(fn.get("description")),
                    "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
                }
            )
    return tools


def tools_from_anthropic(body):
    tools = []
    for raw in body.get("tools") or []:
        if is_record(raw) and isinstance(raw.get("name"), str):
            tools.append(
                {
                    "name": raw["name"],
                    "description": string_of(raw.get("description")),
                    "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                }
            )
    return tools


def client_options(body, api):
    """The fields every wire must carry across, read off the client's body."""
    if api == "openai":
        max_tokens = body.get("max_tokens")
        if not is_number(max_tokens):
            max_tokens = body.get("max_completion_tokens")
        stop = body.get("stop")
        if isinstance(stop, str):
            stops = [stop]
        elif isinstance(stop, list):
            stops = [s for s in stop if isinstance(s, str)]
        else:
            stops = []
        response_format = body.get("response_format") if is_record(body.get("response_format")) else None
        json_requested = response_format is not None and response_format.get("type") in (
            "json_object",
            "json_schema",
        )
        tool_choice = body.get("tool_choice")
    else:
        max_tokens = body.get("max_tokens")
        raw_stops = body.get("stop_sequences")
        stops = [s for s in raw_stops if isinstance(s, str)] if isinstance(raw_stops, list) else []
        response_format = None
        json_requested = False
        tool_choice = body.get("tool_choice")
    return {
        "max_tokens": max_tokens if is_number(max_tokens) else None,
        "stops": stops,
        "temperature": body.get("temperature") if is_number(body.get("temperature")) else None,
        "top_p": body.get("top_p") if is_number(body.get("top_p")) else None,
        "stream": body.get("stream") is True,
        "response_format": response_format,
        "json_requested": json_requested,
        "tool_choice": tool_choice,
    }


def openai_tool_choice(choice, api):
    """The client's tool_choice in OpenAI's spelling."""
    if api == "openai":
        return choice
    if is_record(choice):
        kind = choice.get("type")
        if kind == "auto":
            return "auto"
        if kind == "any":
            return "required"
        if kind == "none":
            return "none"
        if kind == "tool" and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
    return None


def anthropic_tool_choice(choice, api):
    """The client's tool_choice in Anthropic's spelling."""
    if api == "anthropic":
        return choice if is_record(choice) else None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        return {"type": "tool", "name": choice["function"]["name"]}
    return None


# --------------------------------------------------------------------------
# Upstream request bodies, one builder per provider wire
# --------------------------------------------------------------------------

def build_openai_body(model_id, system_text, messages, tools, options, api):
    out_messages = []
    if system_text != "":
        out_messages.append({"role": "system", "content": system_text})
    for message in messages:
        if message["tool_results"]:
            for result in message["tool_results"]:
                out_messages.append(
                    {"role": "tool", "tool_call_id": result["id"], "content": result["text"]}
                )
            if message["text"] == "" and not message["images"]:
                continue
        if message["role"] == "assistant":
            entry = {"role": "assistant", "content": message["text"] if message["text"] != "" or not message["tool_calls"] else None}
            if message["tool_calls"]:
                entry["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                    }
                    for call in message["tool_calls"]
                ]
            out_messages.append(entry)
            continue
        if not message["images"]:
            out_messages.append({"role": "user", "content": message["text"]})
            continue
        parts = []
        if message["text"] != "":
            parts.append({"type": "text", "text": message["text"]})
        for image in message["images"]:
            if image["kind"] == "data":
                url = "data:%s;base64,%s" % (image["media"], image["data"])
            else:
                url = image["url"]
            parts.append({"type": "image_url", "image_url": {"url": url}})
        out_messages.append({"role": "user", "content": parts})
    body = {"model": model_id, "messages": out_messages}
    if options["max_tokens"] is not None:
        body["max_tokens"] = options["max_tokens"]
    if options["temperature"] is not None:
        body["temperature"] = options["temperature"]
    if options["top_p"] is not None:
        body["top_p"] = options["top_p"]
    if options["stops"]:
        body["stop"] = options["stops"]
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in tools
        ]
        choice = openai_tool_choice(options["tool_choice"], api)
        if choice is not None:
            body["tool_choice"] = choice
    if options["response_format"] is not None:
        body["response_format"] = options["response_format"]
    if options["stream"]:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def build_anthropic_body(model_id, system_text, messages, tools, options, api):
    out_messages = []
    for message in messages:
        if message["role"] == "assistant":
            if message["tool_calls"]:
                blocks = []
                if message["text"] != "":
                    blocks.append({"type": "text", "text": message["text"]})
                for call in message["tool_calls"]:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"] or "toolu_unknown",
                            "name": call["name"],
                            "input": call["args"],
                        }
                    )
                out_messages.append({"role": "assistant", "content": blocks})
            else:
                out_messages.append({"role": "assistant", "content": message["text"]})
            continue
        if not message["images"] and not message["tool_results"]:
            out_messages.append({"role": "user", "content": message["text"]})
            continue
        blocks = []
        if message["text"] != "":
            blocks.append({"type": "text", "text": message["text"]})
        for image in message["images"]:
            if image["kind"] == "data":
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": image["media"], "data": image["data"]},
                    }
                )
            else:
                blocks.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
        for result in message["tool_results"]:
            blocks.append(
                {"type": "tool_result", "tool_use_id": result["id"], "content": result["text"]}
            )
        out_messages.append({"role": "user", "content": blocks})
    body = {"model": model_id, "messages": out_messages}
    if system_text != "":
        body["system"] = system_text
    if options["max_tokens"] is not None:
        body["max_tokens"] = options["max_tokens"]
    if options["temperature"] is not None:
        body["temperature"] = options["temperature"]
    if options["top_p"] is not None:
        body["top_p"] = options["top_p"]
    if options["stops"]:
        body["stop_sequences"] = options["stops"]
    if tools:
        body["tools"] = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"],
            }
            for tool in tools
        ]
        choice = anthropic_tool_choice(options["tool_choice"], api)
        if choice is not None:
            body["tool_choice"] = choice
    if options["stream"]:
        body["stream"] = True
    return body


def build_gemini_body(system_text, messages, tools, options, want_json):
    contents = []
    for message in messages:
        parts = []
        if message["text"] != "":
            parts.append({"text": message["text"]})
        for image in message["images"]:
            if image["kind"] == "data":
                parts.append({"inlineData": {"mimeType": image["media"], "data": image["data"]}})
            else:
                parts.append({"fileData": {"fileUri": image["url"]}})
        for call in message["tool_calls"]:
            parts.append({"functionCall": {"name": call["name"], "args": call["args"]}})
        for result in message["tool_results"]:
            # A functionResponse carries no text part, so the parser's user
            # text stays what the client's own wire reported for the turn.
            parts.append(
                {"functionResponse": {"name": result["id"] or "tool", "response": {"result": result["text"]}}}
            )
        if not parts:
            parts.append({"text": ""})
        contents.append({"role": "model" if message["role"] == "assistant" else "user", "parts": parts})
    body = {"contents": contents}
    if system_text != "":
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    }
                    for tool in tools
                ]
            }
        ]
    config = {}
    if options["max_tokens"] is not None:
        config["maxOutputTokens"] = options["max_tokens"]
    if options["temperature"] is not None:
        config["temperature"] = options["temperature"]
    if options["top_p"] is not None:
        config["topP"] = options["top_p"]
    if options["stops"]:
        config["stopSequences"] = options["stops"]
    if want_json:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Upstream answers: parse any provider wire into (text, usage, stop)
# --------------------------------------------------------------------------

def parse_sse_frames(text):
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data_lines = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].lstrip(" "))
        if data_lines:
            frames.append((event, "\n".join(data_lines)))
    return frames


def usage_pair(usage, input_key, output_key):
    if not is_record(usage):
        return 0, 0
    return (
        usage.get(input_key) if is_number(usage.get(input_key)) else 0,
        usage.get(output_key) if is_number(usage.get(output_key)) else 0,
    )


def parse_openai_answer(raw, stream):
    if not stream:
        payload = json.loads(raw)
        choices = payload.get("choices") or []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        text = content if isinstance(content, str) else blocks_text(content)
        usage_in, usage_out = usage_pair(payload.get("usage"), "prompt_tokens", "completion_tokens")
        return text, usage_in, usage_out, choice.get("finish_reason") or "stop"
    text_parts = []
    usage_in, usage_out = 0, 0
    stop = "stop"
    for _event, data in parse_sse_frames(raw):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            usage_in, usage_out = usage_pair(chunk["usage"], "prompt_tokens", "completion_tokens")
        choices = chunk.get("choices") or []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return "".join(text_parts), usage_in, usage_out, stop


def parse_anthropic_answer(raw, stream):
    if not stream:
        payload = json.loads(raw)
        text = blocks_text(payload.get("content"))
        usage_in, usage_out = usage_pair(payload.get("usage"), "input_tokens", "output_tokens")
        return text, usage_in, usage_out, payload.get("stop_reason") or "end_turn"
    text_parts = []
    usage_in, usage_out = 0, 0
    stop = "end_turn"
    for event, data in parse_sse_frames(raw):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        if event == "message_start":
            message = payload.get("message") if is_record(payload.get("message")) else {}
            usage = message.get("usage") if is_record(message.get("usage")) else {}
            if is_number(usage.get("input_tokens")):
                usage_in = usage["input_tokens"]
        elif event == "content_block_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text_parts.append(delta["text"])
        elif event == "message_delta":
            delta = payload.get("delta") if is_record(payload.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = payload.get("usage") if is_record(payload.get("usage")) else {}
            if is_number(usage.get("output_tokens")):
                usage_out = usage["output_tokens"]
    return "".join(text_parts), usage_in, usage_out, stop


def gemini_parts_text(parts):
    texts = []
    if isinstance(parts, list):
        for part in parts:
            if is_record(part) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts)


def parse_gemini_answer(raw, stream):
    if not stream:
        payload = json.loads(raw)
        candidates = payload.get("candidates") or []
        candidate = candidates[0] if candidates and is_record(candidates[0]) else {}
        content = candidate.get("content") if is_record(candidate.get("content")) else {}
        text = gemini_parts_text(content.get("parts"))
        usage = payload.get("usageMetadata") if is_record(payload.get("usageMetadata")) else {}
        usage_in = usage.get("promptTokenCount") if is_number(usage.get("promptTokenCount")) else 0
        usage_out = usage.get("candidatesTokenCount") if is_number(usage.get("candidatesTokenCount")) else 0
        return text, usage_in, usage_out, candidate.get("finishReason") or "STOP"
    text_parts = []
    usage_in, usage_out = 0, 0
    stop = "STOP"
    for _event, data in parse_sse_frames(raw):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        candidates = payload.get("candidates") or []
        if candidates and is_record(candidates[0]):
            candidate = candidates[0]
            content = candidate.get("content") if is_record(candidate.get("content")) else {}
            text_parts.append(gemini_parts_text(content.get("parts")))
            if isinstance(candidate.get("finishReason"), str):
                stop = candidate["finishReason"]
        usage = payload.get("usageMetadata")
        if is_record(usage):
            if is_number(usage.get("promptTokenCount")):
                usage_in = usage["promptTokenCount"]
            if is_number(usage.get("candidatesTokenCount")):
                usage_out = usage["candidatesTokenCount"]
    return "".join(text_parts), usage_in, usage_out, stop


def parse_upstream_answer(wire, raw, stream):
    if wire == "openai":
        return parse_openai_answer(raw, stream)
    if wire == "anthropic":
        return parse_anthropic_answer(raw, stream)
    return parse_gemini_answer(raw, stream)


# --------------------------------------------------------------------------
# Client answers: render the provider's answer in the client's wire
# --------------------------------------------------------------------------

STOP_FAMILIES = {
    "stop": 0,
    "end_turn": 0,
    "STOP": 0,
    "length": 1,
    "max_tokens": 1,
    "MAX_TOKENS": 1,
    "tool_calls": 2,
    "function_call": 2,
    "tool_use": 2,
    "content_filter": 3,
    "refusal": 3,
    "SAFETY": 3,
}


def stop_for_openai(raw_stop):
    family = STOP_FAMILIES.get(raw_stop, 0)
    return {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}[family]


def stop_for_anthropic(raw_stop):
    family = STOP_FAMILIES.get(raw_stop, 0)
    return {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}[family]


def text_pieces(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def sse_frame(event, data):
    prefix = "event: %s\n" % event if event is not None else ""
    return "%sdata: %s\n\n" % (prefix, json.dumps(data))


def render_openai(text, usage_in, usage_out, raw_stop, model_id, stream):
    usage = {
        "prompt_tokens": usage_in,
        "completion_tokens": usage_out,
        "total_tokens": usage_in + usage_out,
    }
    finish = stop_for_openai(raw_stop)
    if not stream:
        return 200, "application/json", json.dumps(
            {
                "id": "chatcmpl-agg",
                "object": "chat.completion",
                "created": FIXED_OPENAI_CREATED,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish,
                        "logprobs": None,
                    }
                ],
                "usage": usage,
            }
        )
    frames = []

    def chunk(delta, finish_reason, with_usage=False):
        entry = {
            "id": "chatcmpl-agg",
            "object": "chat.completion.chunk",
            "created": FIXED_OPENAI_CREATED,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if with_usage:
            entry["choices"] = []
            entry["usage"] = usage
        return entry

    frames.append(sse_frame(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in text_pieces(text):
        frames.append(sse_frame(None, chunk({"content": piece}, None)))
    frames.append(sse_frame(None, chunk({}, finish)))
    frames.append(sse_frame(None, chunk({}, None, with_usage=True)))
    frames.append("data: [DONE]\n\n")
    return 200, "text/event-stream", "".join(frames)


def render_anthropic(text, usage_in, usage_out, raw_stop, model_id, stream):
    stop = stop_for_anthropic(raw_stop)
    if not stream:
        return 200, "application/json", json.dumps(
            {
                "id": "msg_agg",
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
            }
        )
    frames = [
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_agg",
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage_in, "output_tokens": 0},
                },
            },
        ),
        sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    for piece in text_pieces(text):
        frames.append(
            sse_frame(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
            )
        )
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(
        sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": usage_out},
            },
        )
    )
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream", "".join(frames)


def openai_error(message, error_type, code=None):
    return {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = probes
        self.catalog = catalog
        self.manifest_index = 0
        self.lock = threading.Lock()
        self.attempts = {}
        self.consecutive_failures = {}
        self.probed_models = {}
        for probe in probes:
            if probe["ok"]:
                self.probed_models[probe["provider"]["id"]] = {m["id"]: m for m in probe["models"]}
        self.provider_by_id = {p["id"]: p for p in brief["providers"]}

    def served_models(self):
        out = []
        seen = set()
        for entry in self.catalog["providers"]:
            for model in entry["models"]:
                if not model["free"] or model["id"] in seen:
                    continue
                seen.add(model["id"])
                probed = self.probed_models.get(entry["id"], {}).get(model["id"], {})
                out.append(
                    {
                        "id": model["id"],
                        "provider": entry["id"],
                        "display_name": probed.get("display_name") or model["id"],
                    }
                )
        return out

    def next_requires(self, api, body):
        with self.lock:
            index = self.manifest_index
            self.manifest_index += 1
        manifest = self.brief["manifest"]
        requires = dict(manifest[index]["requires"]) if index < len(manifest) else {}
        # Merge what the body itself asks for: the manifest declares it, and
        # the body is the ground truth the provider's echo is checked against.
        if api == "openai":
            tools = body.get("tools")
            if isinstance(tools, list) and len(tools) > 0:
                requires["tools"] = True
            for raw in body.get("messages") or []:
                if not is_record(raw) or not isinstance(raw.get("content"), list):
                    continue
                for block in raw["content"]:
                    if is_record(block) and block.get("type") == "image_url":
                        requires["vision"] = True
            options = client_options(body, api)
            if options["json_requested"]:
                requires["json"] = True
        else:
            tools = body.get("tools")
            if isinstance(tools, list) and len(tools) > 0:
                requires["tools"] = True
            for raw in body.get("messages") or []:
                if not is_record(raw) or not isinstance(raw.get("content"), list):
                    continue
                for block in raw["content"]:
                    if is_record(block) and block.get("type") == "image":
                        requires["vision"] = True
        return requires

    def candidates(self, api, body, requires):
        options = client_options(body, api)
        found = []
        for (provider_id, model_id), row in self.brief["ratings"].items():
            if not row["free"]:
                continue
            if provider_id not in self.probed_models:
                continue
            if model_id not in self.probed_models[provider_id]:
                continue
            if requires.get("tools") and not row["tools"]:
                continue
            if requires.get("vision") and not row["vision"]:
                continue
            if requires.get("json") and not row["json"]:
                continue
            min_context = requires.get("min_context")
            if min_context is not None:
                if row["context_length"] is None or row["context_length"] < min_context:
                    continue
            provider = self.provider_by_id.get(provider_id)
            if provider is None:
                continue
            # An OpenAI JSON-mode request echoes json_mode only on a wire that
            # can carry it; the Anthropic wire always echoes it absent.
            if api == "openai" and options["json_requested"] and provider["wire"] == "anthropic":
                continue
            found.append(
                {
                    "provider": provider,
                    "model_id": model_id,
                    "rating": row["rating_milli"],
                }
            )
        found.sort(
            key=lambda c: (
                -c["rating"],
                self.attempts.get(c["provider"]["id"], 0),
                c["provider"]["id"],
                c["model_id"],
            )
        )
        # Set aside providers that are certainly or almost certainly out,
        # unless that would leave nothing to try.
        usable = [
            c
            for c in found
            if self.attempts.get(c["provider"]["id"], 0) < BUDGET_MAX.get(c["provider"]["id"], 10**9)
            and self.consecutive_failures.get(c["provider"]["id"], 0) < MAX_CONSECUTIVE_FAILURES
        ]
        return usable if usable else found

    def note_attempt(self, provider_id):
        self.attempts[provider_id] = self.attempts.get(provider_id, 0) + 1

    def note_result(self, provider_id, ok):
        if ok:
            self.consecutive_failures[provider_id] = 0
        else:
            self.consecutive_failures[provider_id] = self.consecutive_failures.get(provider_id, 0) + 1

    def call_provider(self, candidate, api, body, requires):
        """One upstream attempt. Returns (ok, parsed) or (False, status)."""
        provider = candidate["provider"]
        model_id = candidate["model_id"]
        wire = provider["wire"]
        base = join_base(ENV_URL, provider["base"])
        headers = {"Content-Type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        stream = body.get("stream") is True
        if wire == "openai":
            url = base + "/v1/chat/completions"
            if api == "openai":
                upstream = copy.deepcopy(body)
                upstream["model"] = model_id
                if stream:
                    stream_options = dict(upstream.get("stream_options") or {})
                    stream_options["include_usage"] = True
                    upstream["stream_options"] = stream_options
            else:
                system_text, messages = neutral_from_anthropic(body)
                upstream = build_openai_body(
                    model_id, system_text, messages, tools_from_anthropic(body),
                    client_options(body, api), api,
                )
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = "2023-06-01"
            if api == "anthropic":
                upstream = copy.deepcopy(body)
                upstream["model"] = model_id
            else:
                system_text, messages = neutral_from_openai(body)
                upstream = build_anthropic_body(
                    model_id, system_text, messages, tools_from_openai(body),
                    client_options(body, api), api,
                )
        else:  # gemini
            if api == "openai":
                system_text, messages = neutral_from_openai(body)
                tools = tools_from_openai(body)
            else:
                system_text, messages = neutral_from_anthropic(body)
                tools = tools_from_anthropic(body)
            options = client_options(body, api)
            # JSON mode reaches Gemini as responseMimeType; set it exactly
            # when the request asks for JSON, on either client's wire.
            upstream = build_gemini_body(
                system_text, messages, tools, options,
                want_json=options["json_requested"] or requires.get("json") is True,
            )
            action = "streamGenerateContent" if stream else "generateContent"
            url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model_id, safe=""), action)
            if stream:
                url += "?alt=sse"
        self.note_attempt(provider["id"])
        try:
            status, raw = http_request("POST", url, headers=headers, body_obj=upstream)
        except Exception as err:
            log("provider %s unreachable: %s" % (provider["id"], err))
            self.note_result(provider["id"], False)
            return False, 0
        if status != 200:
            self.note_result(provider["id"], False)
            return False, status
        try:
            parsed = parse_upstream_answer(wire, raw, stream)
        except Exception as err:
            # A 200 already counts as the request's one successful call at
            # the env, so a second success would fail the request anyway;
            # an unparseable 200 is served as an empty answer rather than
            # burning the request on a doomed retry.
            log("provider %s answered an unparseable body: %s" % (provider["id"], err))
            parsed = ("", 0, 0, "stop" if wire == "openai" else "end_turn" if wire == "anthropic" else "STOP")
        self.note_result(provider["id"], True)
        return True, parsed

    def serve(self, api, body):
        requires = self.next_requires(api, body)
        ordered = self.candidates(api, body, requires)
        if not ordered:
            return None
        failed_providers = set()
        # First pass skips a provider after its first refusal in this
        # request (a sibling model would burn another of its calls);
        # the second pass retries what is left if nothing else answered.
        for second_pass in (False, True):
            for candidate in ordered:
                provider_id = candidate["provider"]["id"]
                if not second_pass and provider_id in failed_providers:
                    continue
                ok, result = self.call_provider(candidate, api, body, requires)
                if ok:
                    text, usage_in, usage_out, raw_stop = result
                    if api == "openai":
                        return render_openai(
                            text, usage_in, usage_out, raw_stop,
                            candidate["model_id"], body.get("stream") is True,
                        )
                    return render_anthropic(
                        text, usage_in, usage_out, raw_stop,
                        candidate["model_id"], body.get("stream") is True,
                    )
                failed_providers.add(provider_id)
        return None


# --------------------------------------------------------------------------
# HTTP server, register and the /work poll
# --------------------------------------------------------------------------

AGGREGATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def send_body(self, status, content_type, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, payload):
        self.send_body(status, "application/json", json.dumps(payload))

    def read_body(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            return None
        if size <= 0 or size > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(size)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return value if is_record(value) else None

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path != "/v1/models":
            self.send_json(404, openai_error("no route GET %s" % path, "not_found"))
            return
        models = AGGREGATOR.served_models()
        if self.headers.get("x-api-key") is not None:
            self.send_json(
                200,
                {
                    "data": [
                        {
                            "type": "model",
                            "id": m["id"],
                            "display_name": m["display_name"],
                            "created_at": FIXED_ANTHROPIC_CREATED_AT,
                        }
                        for m in models
                    ],
                    "has_more": False,
                },
            )
        else:
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": m["id"],
                            "object": "model",
                            "created": FIXED_OPENAI_CREATED,
                            "owned_by": m["provider"],
                        }
                        for m in models
                    ],
                },
            )

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/chat/completions":
            api, wire = "openai", "openai"
        elif path == "/v1/messages":
            api, wire = "anthropic", "anthropic"
        else:
            self.send_json(404, openai_error("no route POST %s" % path, "not_found"))
            return
        body = self.read_body()
        if body is None:
            if wire == "anthropic":
                self.send_json(400, anthropic_error("invalid_request_error", "the request body is not one JSON object"))
            else:
                self.send_json(400, openai_error("the request body is not one JSON object", "invalid_request_error"))
            return
        try:
            result = AGGREGATOR.serve(api, body)
        except Exception as err:
            log("serve failed: %r" % (err,))
            result = None
        if result is None:
            if wire == "anthropic":
                self.send_json(503, anthropic_error("overloaded_error", "no free model on the directory could serve this request"))
            else:
                self.send_json(503, openai_error("no free model on the directory could serve this request", "no_free_model"))
            return
        status, content_type, text = result
        self.send_body(status, content_type, text)


def load_catalog():
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log(brief["error"])
    probes = probe_all(ENV_URL, brief["providers"]) if brief["providers"] else []
    for probe in probes:
        if not probe["ok"] and probe["error"]:
            log(probe["error"])
    catalog = build_catalog(probes, brief["ratings"])
    return brief, probes, catalog


def register(port, catalog):
    last_error = None
    for _attempt in range(10):
        try:
            status, text = http_request(
                "POST",
                ENV_URL + "/register",
                headers={"Content-Type": "application/json"},
                body_obj={"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                timeout=REGISTER_TIMEOUT_S,
            )
            if status == 200:
                try:
                    reply = json.loads(text)
                except ValueError:
                    reply = {}
                if is_record(reply) and is_number(reply.get("requests")):
                    log("registered: the workload holds %d requests" % reply["requests"])
                return
            last_error = "HTTP %d: %s" % (status, text[:300])
        except Exception as err:
            last_error = str(err)
        threading.Event().wait(0.3)
    raise RuntimeError("POST %s/register failed: %s" % (ENV_URL, last_error))


def poll_work(server):
    while True:
        try:
            status, text = http_request("GET", ENV_URL + "/work", timeout=2.0)
            if status == 200:
                work = json.loads(text)
                if is_record(work) and work.get("done") is True:
                    log("the workload is done; exiting")
                    server.shutdown()
                    return
        except Exception:
            pass
        threading.Event().wait(POLL_INTERVAL_S)


def main():
    global AGGREGATOR
    brief, probes, catalog = load_catalog()
    if "--list" in sys.argv[1:]:
        text = json.dumps(catalog, separators=(",", ":")) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        return 1
    AGGREGATOR = Aggregator(brief, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log(
        "listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]), len(AGGREGATOR.served_models()))
    )
    try:
        register(port, catalog)
    except RuntimeError as err:
        log(str(err))
        return 1
    poller = threading.Thread(target=poll_work, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
