# proviras-sdk (Python)

Trace LangGraph and LangChain runs to [Proviras](https://proviras.com).

`ProvirasTracer` is a `langchain_core.tracers.base.BaseTracer` you pass as a callback. It creates one Proviras session per graph invocation and streams each LLM, tool, chain, and retriever run to your dashboard as a `TraceCall`.

## Install

```sh
pip install proviras-sdk
```

`langchain-core>=0.3.0` is pulled in automatically. Your project already has it if you use LangGraph or LangChain.

## Setup

```sh
export PROVIRAS_PARENT_ID=<your-proviras-user-id>
export PROVIRAS_PLATFORM=langgraph
```

| variable | required | description |
|----------|----------|-------------|
| `PROVIRAS_PARENT_ID` | yes | your Proviras user ID |
| `PROVIRAS_PLATFORM` | yes | runtime platform (e.g. `langgraph`, `langchain`) |
| `PROVIRAS_USER_ID` | no | parent agent ID when this agent is spawned by another |

## Usage with LangGraph

```python
from proviras_sdk import ProvirasSdk
from langgraph.graph import StateGraph

sdk = ProvirasSdk()
graph = ...  # build your StateGraph

tracer = sdk.trace("Answer user question")

result = graph.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"callbacks": [tracer]},
)
# session is finalized automatically when the root run ends
```

`sdk.trace(task_description)` returns a `ProvirasTracer`. It:

1. Registers your agent on first run (saves `agentId` to `~/.proviras/config.json`).
2. Creates a session: `POST /api/agent/session`.
3. On each completed LangChain run, posts a trace: `POST /api/agent/session/{id}/trace`.
4. When the root run ends, finalizes the session: `PATCH /api/agent/session`.

Telemetry failures are swallowed — they never break the graph.

## What gets captured

For each LangChain `Run` that completes (`run_type` of `llm`, `tool`, `chain`, or `retriever`):

| field | source |
|-------|--------|
| `runType` | `run.run_type` |
| `stepId` | `run.id` |
| `parentTraceId` | server-issued traceId of the parent run |
| `timestamp` | `run.start_time` |
| `latencyMs` | `run.end_time - run.start_time` |
| `model` | `run.extra.metadata.ls_model_name` |
| `input` / `output` | JSON-encoded `run.inputs` / `run.outputs`, truncated at 8 KB |
| `tokens` | `run.outputs.llm_output.token_usage` (LLM runs only) |
| `error` | `run.error` |

Posting happens once at root-run completion, in tree preorder, so a child's `parentTraceId` is always set correctly before the child is posted.

## Per-invocation surface

```python
sdk.trace("nightly summary", surface="code")
```

`surface` is one of `"cowork" | "chat" | "code" | "api"` and defaults to `"api"`.

## Async LangGraph

LangChain dispatches sync callback handlers on a worker thread when called from async contexts, so `ProvirasTracer` works with `graph.ainvoke()` and `graph.astream()` without changes. Each HTTP call blocks one worker thread for the duration of the request.
