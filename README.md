[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/imgans322-gif/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/imgans322-gif/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

# Free-model aggregator — free-model-aggregator-01

One server (`main.py`, Python 3.13, standard library only) that serves the
OpenAI Chat Completions API and the Anthropic Messages API — each in JSON
and in server-sent events — over ten simulated free-tier providers, routing
every request to the best free model that can serve it.

Entry for the make.fun bounty
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01).

## How it works

At startup the server reads its brief (`MAKEFUN_PROMPT_FILE`): the provider
directory, the sealed quality table and the workload manifest. It probes
every provider's catalog route in that provider's own wire (OpenAI,
Anthropic or Gemini), builds one catalog document, listens on an ephemeral
port and registers `{protocol_version, port, catalog}` with the env at
`MAKEFUN_ENV_URL`. It then serves the client's requests one at a time and
exits 0 when `GET /work` reports the workload done.

Routing, per request:

- Requirements come from the manifest row for the request, merged with what
  the body itself asks for (tools present, an image block, a JSON
  `response_format`). Only free models whose quality-table row meets them
  are candidates — a paid model is never a legal pick.
- Candidates are ordered by the table's `rating_milli`, best first, with
  ties going to the provider this server has used least so far.
- The chosen provider is called in its own wire. Requests are translated
  through a neutral form (system prompt, joined user text, images, tool
  names and schemas, stop sequences, `max_tokens`, JSON mode) so the
  provider's echo of what it parsed matches what the client sent, on all
  six client-wire × provider-wire combinations. Same-wire requests pass
  through untouched apart from the model id (and `include_usage` on
  streamed OpenAI calls, so the usage chunk exists to forward).
- A refusal (402, 429 or 5xx) fails over to the next candidate. A provider
  that refuses is skipped for the rest of that request, accumulates a
  consecutive-failure count across requests, and is set aside after three
  in a row — or once our own counted attempts reach the most calls its
  free budget could possibly hold — while any alternative remains.
- The provider's answer is rendered back in the client's wire: text whole,
  usage counts as that wire reports them, and the stop reason mapped
  through its family (`end_turn` ↔ `stop`, `max_tokens` ↔ `length`,
  `tool_use` ↔ `tool_calls`, `refusal` ↔ `content_filter`), streamed with
  the wire's own event order when the client asked for a stream.

`GET /v1/models` answers in the wire its auth header names — the OpenAI
list for `Authorization: Bearer`, the Anthropic list for `x-api-key` — and
lists exactly the free model ids the catalog registered.

## The catalog

    python3 main.py --list

prints the catalog document (one entry per provider: `id`, `base_url`,
`key_env`, `wire`, and its models with `id`, `free` and `context_length`),
byte for byte the document the register body carries. Outside a trial,
with no brief to read, it prints `{"providers":[]}`.

## Determinism

Nothing sent or answered depends on the wall clock or on randomness:
model lists carry fixed timestamps, routing is a function of the brief,
the requests and the provider answers seen so far, and provider probes
are assembled in directory order. A replayed trial therefore leaves the
same state export as the live one.
