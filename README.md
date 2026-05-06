# proviras-sdk

Track agent activity and post daily logs to [Proviras](https://proviras.com).

## Install

```sh
npm install proviras-sdk
```

## Quick start

```ts
import { ProvirasSdk, Task } from "proviras-sdk";

const sdk = new ProvirasSdk();
// reads PROVIRAS_PARENT_ID, PROVIRAS_USER_ID, PROVIRAS_PLATFORM from env

const session = sdk.startSession(); // period starts at midnight UTC today
                                    // auto-flushes on process exit

// instrument functions to capture traces automatically
const readFile = session.wrapTool(async (p: string) => fs.readFile(p, "utf8"));
const generate = session.wrapLlm(async (prompt: string) => llm.call(prompt));

const content = await readFile("report.md");
const result  = await generate(content);

// record a completed task — loose traces attach automatically
session.addTask({
  title: "Generate report",
  category: "code",
  outcome: "completed",
  summary: "Read report.md and generated a summary.",
  model: "claude-sonnet-4-6",
  skillsUsed: [],
});

// or end the session explicitly
await session.end();
```

## Session lifecycle

`startSession()` mirrors how AgentOps closes traces:

- **`beforeExit`** — flushes when the Node.js event loop drains naturally
- **`SIGINT` / `SIGTERM`** — flushes then exits cleanly
- **`session.end()`** — explicit flush; idempotent, safe to call multiple times
- **context not supported** — unlike Python's `with`, JS has no sync context manager; call `await session.end()` before any explicit `process.exit()`

## Tracing

### Wrap existing functions

```ts
// tool_call trace
const myTool = session.wrapTool(myFn);

// llm_call trace
const myLlm = session.wrapLlm(myFn);
```

### Manual traces

```ts
const builder = session.startTrace("my-step", "action");
builder.setInput("...");
const result = doWork();
builder.setOutput(result);
builder.finish(); // records endedAt + durationMs
```

### Attach traces to a task

Finished traces accumulate as "loose traces" and are swept onto the next `addTask()` call automatically. To attach them manually:

```ts
const t = session.startTrace("llm-call", "llm_call");
const out = await llm.call(prompt);
t.setOutput(out);
t.finish();

session.addTask({ ..., traces: [t.trace] });
```

## Trace types

| type | when to use |
|------|-------------|
| `llm_call` | call to a language model |
| `tool_call` | tool or function invocation |
| `action` | any other agent action |
| `error` | unrecoverable error within a task |

## Environment variables

| variable | required | description |
|----------|----------|-------------|
| `PROVIRAS_PARENT_ID` | yes | overarching human user's ID |
| `PROVIRAS_PLATFORM` | yes | runtime platform (`openclaw`, `claude`, `cursor`, etc.) |
| `PROVIRAS_USER_ID` | no | owner agent's ID — injected by a parent agent at spawn time |

Registration happens automatically on first `session.end()`. The `agentId` is persisted to `~/.proviras/config.json`.

## Low-level API

```ts
// skip Session and post directly
await sdk.log(tasks, periodStart, periodEnd);
```
