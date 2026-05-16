# proviras-sdk

Trace LangGraph and LangChain runs to [Proviras](https://proviras.com).

`ProvirasTracer` is a LangChain `BaseTracer` you pass as a callback. It creates one Proviras session per graph invocation and streams each LLM, tool, chain, and retriever run to your dashboard as a `TraceCall`.

## Install

```sh
npm install proviras-sdk @langchain/core
```

`@langchain/core` is a peer dependency. Your project already has it if you use LangGraph or LangChain.

## Setup

```sh
export PROVIRAS_PARENT_ID=<your-proviras-user-id>
export PROVIRAS_PLATFORM=langgraph
```

| variable | required | description |
|----------|----------|-------------|
| `PROVIRAS_PARENT_ID` | yes | your Proviras user ID |
| `PROVIRAS_PLATFORM` | yes | runtime platform (e.g. `langgraph`, `langchain-js`) |
| `PROVIRAS_USER_ID` | no | parent agent ID when this agent is spawned by another |

## Usage with LangGraph

```ts
import { ProvirasSdk } from "proviras-sdk";
import { StateGraph } from "@langchain/langgraph";

const sdk = new ProvirasSdk();

const graph = /* build your StateGraph */;

const tracer = await sdk.trace("Answer user question");

const result = await graph.invoke(
  { messages: [{ role: "user", content: "..." }] },
  { callbacks: [tracer] },
);
// session is finalized automatically when the root run ends
```

`sdk.trace(taskDescription)` returns an awaited `ProvirasTracer`. It:

1. Registers your agent on first run (saves `agentId` to `~/.proviras/config.json`)
2. Creates a session: `POST /api/agent/session`
3. On each completed LangChain run, posts a trace: `POST /api/agent/session/{id}/trace`
4. When the root run ends, finalizes the session: `PATCH /api/agent/session/{id}`

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
| `model` | `run.extra.metadata.ls_model_name` (LangChain sets this on LLM runs) |
| `input` / `output` | stringified `run.inputs` / `run.outputs`, truncated at 8 KB |
| `tokens` | `run.outputs.llmOutput.tokenUsage` (LLM runs only) |
| `error` | `run.error` |

Trace posting happens once at root-run completion, in tree preorder, so a child's `parentTraceId` is always set correctly before it's posted.

## Per-invocation surface

```ts
await sdk.trace("nightly summary", { surface: "code" });
```

`surface` is one of `cowork | chat | code | api` and defaults to `api`.

## Server contract

The SDK assumes these endpoints exist on the Proviras server:

| method | path | purpose |
|--------|------|---------|
| `POST` | `/api/agent/register` | one-time agent registration → `{ agentId }` |
| `POST` | `/api/agent/session` | create session with `{ sessionId, agentId, taskDescription, startedAt, status, surface }` |
| `POST` | `/api/agent/session/[sessionId]/trace` | append a `CreateTraceCall` → `{ traceId }` |
| `PATCH` | `/api/agent/session/[sessionId]` | finalize with `{ status, completedAt, totalTokens, totalLatencyMs }` |
