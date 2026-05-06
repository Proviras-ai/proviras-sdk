import * as fs from "fs";
import * as os from "os";
import * as path from "path";

export type TraceType = "llm_call" | "tool_call" | "action" | "error";

export interface Trace {
  type: TraceType;
  name: string;
  startedAt: Date;
  endedAt?: Date;
  input?: string;
  output?: string;
  error?: string;
  metadata?: Record<string, unknown>;
}

export interface Task {
  title: string;
  category: "email" | "calendar" | "file" | "web" | "code" | "other";
  outcome: "completed" | "failed" | "partial";
  summary: string;
  model: string;
  skillsUsed?: string[];
  durationEstimate?: number;
  costEstimate?: string;
  traces?: Trace[];
}

function serializeTrace(t: Trace): Record<string, unknown> {
  const out: Record<string, unknown> = {
    type: t.type,
    name: t.name,
    startedAt: t.startedAt.toISOString(),
  };
  if (t.endedAt) {
    out.endedAt = t.endedAt.toISOString();
    out.durationMs = t.endedAt.getTime() - t.startedAt.getTime();
  }
  if (t.input !== undefined) out.input = t.input;
  if (t.output !== undefined) out.output = t.output;
  if (t.error !== undefined) out.error = t.error;
  if (t.metadata && Object.keys(t.metadata).length > 0) out.metadata = t.metadata;
  return out;
}

export class TraceBuilder {
  private readonly _trace: Trace;

  constructor(name: string, type: TraceType) {
    this._trace = { type, name, startedAt: new Date() };
  }

  setInput(value: string): this {
    this._trace.input = value;
    return this;
  }

  setOutput(value: string): this {
    this._trace.output = value;
    return this;
  }

  setMetadata(metadata: Record<string, unknown>): this {
    this._trace.metadata = { ...this._trace.metadata, ...metadata };
    return this;
  }

  finish(error?: string): Trace {
    this._trace.endedAt = new Date();
    if (error !== undefined) this._trace.error = error;
    return this._trace;
  }

  get trace(): Trace {
    return this._trace;
  }
}

/**
 * Collects tasks and traces for a time period and posts the log when the
 * session ends — explicitly via end(), or automatically on process exit
 * (beforeExit / SIGINT / SIGTERM), mirroring how AgentOps closes traces.
 *
 * Usage (context-style):
 *   const session = sdk.startSession();
 *   session.addTask({ ... });
 *   await session.end();
 *
 * Usage (auto-flush on exit):
 *   const session = sdk.startSession();
 *   // process.beforeExit will call end() automatically
 */
export class Session {
  readonly periodStart: Date;
  private readonly _sdk: ProvirasSdk;
  private readonly _tasks: Task[] = [];
  private readonly _looseTraces: Trace[] = [];
  private _ended = false;

  constructor(sdk: ProvirasSdk, periodStart: Date) {
    this._sdk = sdk;
    this.periodStart = periodStart;

    // Mirror AgentOps' atexit pattern — flush when the event loop drains
    // naturally. For explicit process.exit() callers, end() should be awaited
    // before exiting.
    const flush = () => { void this.end(); };
    process.once("beforeExit", flush);
    process.once("SIGINT", () => { void this.end().then(() => process.exit(0)); });
    process.once("SIGTERM", () => { void this.end().then(() => process.exit(0)); });
  }

  // ── task management ────────────────────────────────────────────────────────

  addTask(task: Task): void {
    const finished = this._looseTraces.filter((t) => t.endedAt !== undefined);
    this._looseTraces.splice(
      0,
      this._looseTraces.length,
      ...this._looseTraces.filter((t) => t.endedAt === undefined)
    );
    this._tasks.push({ ...task, traces: [...(task.traces ?? []), ...finished] });
  }

  // ── trace helpers ──────────────────────────────────────────────────────────

  startTrace(name: string, type: TraceType = "action"): TraceBuilder {
    const builder = new TraceBuilder(name, type);
    this._looseTraces.push(builder.trace);
    return builder;
  }

  /**
   * Wrap a sync or async function as a tool_call trace.
   *
   *   const readFile = session.wrapTool(async (p: string) => fs.readFile(p));
   */
  wrapTool<Args extends unknown[], R>(
    fn: (...args: Args) => R,
    name?: string
  ): (...args: Args) => R {
    const traceName = name ?? fn.name ?? "tool";
    return (...args: Args): R => {
      const builder = this.startTrace(traceName, "tool_call");
      builder.setInput(JSON.stringify(args).slice(0, 500));
      let result: R;
      try {
        result = fn(...args);
      } catch (err) {
        builder.finish(String(err));
        throw err;
      }
      if (result instanceof Promise) {
        return result.then(
          (r) => { builder.setOutput(JSON.stringify(r).slice(0, 500)); builder.finish(); return r; },
          (err) => { builder.finish(String(err)); throw err; }
        ) as unknown as R;
      }
      builder.setOutput(JSON.stringify(result).slice(0, 500));
      builder.finish();
      return result;
    };
  }

  /**
   * Wrap a sync or async function as an llm_call trace.
   *
   *   const generate = session.wrapLlm(async (prompt: string) => llm.call(prompt));
   */
  wrapLlm<Args extends unknown[], R>(
    fn: (...args: Args) => R,
    name?: string
  ): (...args: Args) => R {
    const traceName = name ?? fn.name ?? "llm";
    return (...args: Args): R => {
      const builder = this.startTrace(traceName, "llm_call");
      let result: R;
      try {
        result = fn(...args);
      } catch (err) {
        builder.finish(String(err));
        throw err;
      }
      if (result instanceof Promise) {
        return result.then(
          (r) => { builder.setOutput(JSON.stringify(r).slice(0, 500)); builder.finish(); return r; },
          (err) => { builder.finish(String(err)); throw err; }
        ) as unknown as R;
      }
      builder.setOutput(JSON.stringify(result).slice(0, 500));
      builder.finish();
      return result;
    };
  }

  // ── lifecycle ──────────────────────────────────────────────────────────────

  /** Post the session log. Idempotent — safe to call multiple times. */
  async end(periodEnd?: Date): Promise<boolean> {
    if (this._ended) return true;
    this._ended = true;
    return this._sdk.log(this._tasks, this.periodStart, periodEnd);
  }
}

export class ProvirasSdk {
  private static readonly BASE_URL = "https://proviras.com/api";

  private readonly parentId: string | undefined;
  private readonly userId: string | undefined;
  private readonly platform: string | undefined;
  private readonly configPath: string;
  private _agentId: string | undefined;

  constructor(configPath?: string) {
    this.parentId = process.env.PROVIRAS_PARENT_ID;
    this.userId = process.env.PROVIRAS_USER_ID;
    this.platform = process.env.PROVIRAS_PLATFORM;
    this.configPath =
      configPath ?? path.join(os.homedir(), ".proviras", "config.json");
  }

  get agentId(): string | undefined {
    if (this._agentId) return this._agentId;
    try {
      const data = JSON.parse(fs.readFileSync(this.configPath, "utf8"));
      this._agentId = data.agentId;
    } catch {
      // config not yet written
    }
    return this._agentId;
  }

  async register(): Promise<string> {
    if (this.agentId) return this.agentId;

    if (!this.parentId) throw new Error("PROVIRAS_PARENT_ID is not set");
    if (!this.platform) throw new Error("PROVIRAS_PLATFORM is not set");

    const payload: Record<string, string> = {
      userId: this.parentId,
      name: this.readAgentName(),
      platform: this.platform,
    };
    if (this.userId) payload.parentAgentId = this.userId;

    const response = await this.post("/agent/register", payload);
    const agentId = response.agentId as string;
    if (!agentId) throw new Error(`Registration failed: ${JSON.stringify(response)}`);

    this._agentId = agentId;
    this.saveConfig({ agentId });
    return agentId;
  }

  /**
   * Start a new session. Defaults to the start of today (UTC).
   * The session auto-flushes via beforeExit / SIGINT / SIGTERM.
   *
   *   const session = sdk.startSession();
   *   session.addTask({ ... });
   *   await session.end(); // or let process exit handle it
   */
  startSession(periodStart?: Date): Session {
    const start = periodStart ?? (() => {
      const now = new Date();
      return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
    })();
    return new Session(this, start);
  }

  /** Low-level: post a log directly. Prefer startSession() for new code. */
  async log(tasks: Task[], periodStart: Date, periodEnd?: Date): Promise<boolean> {
    const agentId = await this.register();
    const now = periodEnd ?? new Date();

    const serialized = tasks.map((t) => {
      const entry: Record<string, unknown> = {
        title: t.title,
        category: t.category,
        outcome: t.outcome,
        summary: t.summary,
        model: t.model,
        skillsUsed: t.skillsUsed ?? [],
      };
      if (t.durationEstimate !== undefined) entry.durationEstimate = t.durationEstimate;
      if (t.costEstimate !== undefined) entry.costEstimate = t.costEstimate;
      if (t.traces && t.traces.length > 0) entry.traces = t.traces.map(serializeTrace);
      return entry;
    });

    const payload = {
      agentId,
      loggedAt: now.toISOString(),
      periodStart: periodStart.toISOString(),
      periodEnd: now.toISOString(),
      tasks: serialized,
      heartbeatStatus: tasks.length > 0 ? "active" : "idle",
    };

    try {
      await this.post("/agent/log", payload, { "X-Agent-ID": agentId });
      return true;
    } catch {
      return false;
    }
  }

  private async post(
    endpoint: string,
    payload: unknown,
    headers?: Record<string, string>
  ): Promise<Record<string, unknown>> {
    const response = await fetch(`${ProvirasSdk.BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json() as Promise<Record<string, unknown>>;
  }

  private readAgentName(): string {
    const soulPath = path.join(os.homedir(), ".openclaw", "workspace", "SOUL.md");
    try {
      for (const line of fs.readFileSync(soulPath, "utf8").split("\n")) {
        if (line.startsWith("name:")) return line.split(":", 2)[1].trim();
      }
    } catch {
      // SOUL.md not present
    }
    return "unnamed-agent";
  }

  private saveConfig(data: Record<string, unknown>): void {
    fs.mkdirSync(path.dirname(this.configPath), { recursive: true });
    fs.writeFileSync(this.configPath, JSON.stringify(data, null, 2));
  }
}
