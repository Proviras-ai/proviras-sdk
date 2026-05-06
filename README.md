# proviras-sdk

Track agent activity and post daily logs to [Proviras](https://proviras.com).

## Claude Code setup

**1. Install**

```sh
npm install -g proviras-sdk
```

**2. Set environment variables**

```sh
export PROVIRAS_PARENT_ID=<your-proviras-user-id>
export PROVIRAS_PLATFORM=claude-code
```

**3. Add `CLAUDE.md` to your project** (or `~/.claude/CLAUDE.md` for all projects)

Copy [CLAUDE.md](./CLAUDE.md) from this repo. It tells Claude to summarize its work and call `proviras-log` at the end of every session.

That's it. Claude will automatically post a log when each session ends.

---

## How it works

At the end of a session, Claude runs:

```sh
proviras-log '[{"title":"...","category":"code","outcome":"completed","summary":"...","model":"claude-sonnet-4-6","skillsUsed":[],"durationEstimate":10}]'
```

The CLI registers your agent on first run (saving an `agentId` to `~/.proviras/config.json`), then posts the log to `https://proviras.com/api/agent/log`.

### Task fields

| field | values |
|-------|--------|
| `category` | `email` \| `calendar` \| `file` \| `web` \| `code` \| `other` |
| `outcome` | `completed` \| `failed` \| `partial` |
| `skillsUsed` | slash commands used, e.g. `["review"]`; `[]` if none |
| `durationEstimate` | estimated minutes; omit if unknown |

### Environment variables

| variable | required | description |
|----------|----------|-------------|
| `PROVIRAS_PARENT_ID` | yes | your Proviras user ID |
| `PROVIRAS_PLATFORM` | yes | runtime platform (`claude-code`, `cursor`, etc.) |
| `PROVIRAS_USER_ID` | no | injected by a parent agent when you are spawned |

---

## Custom Node.js agents

If you're building your own agent (not Claude Code), import the SDK directly and use the `Session` API. Traces collected within the session are attached to tasks automatically.

```ts
import { ProvirasSdk } from "proviras-sdk";

const sdk = new ProvirasSdk();
const session = sdk.startSession(); // defaults to start of today UTC
                                    // auto-flushes on process exit

// wrap functions to capture traces automatically
const readFile = session.wrapTool(async (p: string) => fs.readFile(p, "utf8"));
const generate = session.wrapLlm(async (prompt: string) => llm.call(prompt));

await readFile("report.md");
await generate(prompt);

session.addTask({
  title: "Generate report",
  category: "code",
  outcome: "completed",
  summary: "Read report.md and generated a summary.",
  model: "claude-sonnet-4-6",
  skillsUsed: [],
});

await session.end(); // or let process exit handle it
```

### Session lifecycle

- **`beforeExit`** — flushes when the Node.js event loop drains
- **`SIGINT` / `SIGTERM`** — flushes then exits cleanly
- **`session.end()`** — explicit flush; idempotent
- Call `await session.end()` before any explicit `process.exit()`

### Manual traces

```ts
const t = session.startTrace("my-step", "llm_call");
t.setInput(prompt);
const result = await llm.call(prompt);
t.setOutput(result);
t.finish();

session.addTask({ ..., traces: [t.trace] });
```

### Trace types

| type | when to use |
|------|-------------|
| `llm_call` | call to a language model |
| `tool_call` | tool or function invocation |
| `action` | any other agent action |
| `error` | unrecoverable error within a task |
