import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { randomUUID } from "crypto";
import { BaseTracer, type Run } from "@langchain/core/tracers/base";

export type Surface = "cowork" | "chat" | "code" | "api";

export interface TokenUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

export interface ProvirasTracerOptions {
  sdk: ProvirasSdk;
  taskDescription: string;
  surface?: Surface;
}

// LangChain Run.run_type can be any string; the server only accepts these four.
const SERVER_RUN_TYPES = new Set(["llm", "tool", "chain", "retriever"]);

const MAX_FIELD_LEN = 8000;

/**
 * LangChain/LangGraph callback handler that streams traces to Proviras.
 *
 * Usage:
 *   const sdk = new ProvirasSdk();
 *   const tracer = await sdk.trace("Answer user question");
 *   await graph.invoke({ input }, { callbacks: [tracer] });
 *
 * The session is created upfront via POST /api/agent/session. Each LangChain
 * run that completes becomes a TraceCall via POST /api/agent/session/{id}/trace.
 * When the root run ends, the session is finalized via PATCH /api/agent/session/{id}.
 *
 * Traces are posted in tree order at the end of the root run (not streamed),
 * so parent traceIds (server-issued) are always available before child traces
 * reference them.
 */
export class ProvirasTracer extends BaseTracer {
  name = "proviras_tracer";

  readonly sessionId: string;
  private readonly sdk: ProvirasSdk;
  private readonly taskDescription: string;
  private readonly surface: Surface;
  private readonly startedAt: Date;
  private readonly traceIdMap: Map<string, string> = new Map();
  private readonly totals: Required<TokenUsage> = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };
  private sessionReady: Promise<void> | null = null;
  private finalized = false;

  static async create(options: ProvirasTracerOptions): Promise<ProvirasTracer> {
    const tracer = new ProvirasTracer(options);
    await tracer.ensureSession();
    return tracer;
  }

  constructor(options: ProvirasTracerOptions) {
    super();
    this.sdk = options.sdk;
    this.taskDescription = options.taskDescription;
    this.surface = options.surface ?? "api";
    this.sessionId = randomUUID();
    this.startedAt = new Date();
  }

  private ensureSession(): Promise<void> {
    if (!this.sessionReady) this.sessionReady = this.createSession();
    return this.sessionReady;
  }

  private async createSession(): Promise<void> {
    const agentId = await this.sdk.register();
    await this.sdk.request("POST", "/agent/session", {
      sessionId: this.sessionId,
      agentId,
      taskDescription: this.taskDescription,
      startedAt: this.startedAt.toISOString(),
      status: "running",
      surface: this.surface,
    }, { "X-Agent-ID": agentId });
  }

  protected async persistRun(run: Run): Promise<void> {
    // Called by BaseTracer only when the root run ends. Walk the tree and
    // post each run in preorder so parents are posted before children.
    await this.ensureSession();
    await this.postRunTree(run);
    await this.finalizeSession(run);
  }

  private async postRunTree(run: Run): Promise<void> {
    await this.postRun(run);
    for (const child of run.child_runs ?? []) {
      await this.postRunTree(child);
    }
  }

  private async postRun(run: Run): Promise<void> {
    if (!SERVER_RUN_TYPES.has(run.run_type)) return;

    const endTime = run.end_time ?? Date.now();
    const latencyMs = Math.max(0, endTime - run.start_time);
    const parentTraceId = run.parent_run_id
      ? this.traceIdMap.get(run.parent_run_id)
      : undefined;

    const tokens = this.extractTokens(run);
    if (tokens) this.accumulateTokens(tokens);

    const payload: Record<string, unknown> = {
      runType: run.run_type,
      stepId: run.id,
      timestamp: new Date(run.start_time).toISOString(),
      latencyMs,
    };
    if (parentTraceId) payload.parentTraceId = parentTraceId;
    const model = this.extractModel(run);
    if (model) payload.model = model;
    const input = this.stringify(run.inputs);
    if (input) payload.input = input;
    const output = this.stringify(run.outputs);
    if (output) payload.output = output;
    if (tokens) payload.tokens = tokens;
    if (run.error) payload.error = String(run.error);

    try {
      const agentId = this.sdk.agentId;
      const response = await this.sdk.request(
        "POST",
        `/agent/session/${this.sessionId}/trace`,
        payload,
        agentId ? { "X-Agent-ID": agentId } : undefined,
      );
      const traceId = response?.traceId;
      if (typeof traceId === "string") this.traceIdMap.set(run.id, traceId);
    } catch {
      // never break the graph because telemetry failed
    }
  }

  private async finalizeSession(rootRun: Run): Promise<void> {
    if (this.finalized) return;
    this.finalized = true;

    const completedAt = new Date();
    const totalLatencyMs = Math.max(
      0,
      (rootRun.end_time ?? completedAt.getTime()) - rootRun.start_time,
    );

    try {
      const agentId = this.sdk.agentId;
      await this.sdk.request(
        "PATCH",
        `/agent/session/${this.sessionId}`,
        {
          sessionId: this.sessionId,
          status: rootRun.error ? "failed" : "completed",
          completedAt: completedAt.toISOString(),
          totalTokens: this.totals,
          totalLatencyMs,
        },
        agentId ? { "X-Agent-ID": agentId } : undefined,
      );
    } catch {
      // swallow
    }
  }

  private accumulateTokens(tokens: TokenUsage): void {
    if (tokens.promptTokens) this.totals.promptTokens += tokens.promptTokens;
    if (tokens.completionTokens) this.totals.completionTokens += tokens.completionTokens;
    if (tokens.totalTokens) this.totals.totalTokens += tokens.totalTokens;
  }

  private extractModel(run: Run): string | undefined {
    const metadata = (run.extra?.metadata ?? {}) as Record<string, unknown>;
    if (typeof metadata.ls_model_name === "string") return metadata.ls_model_name;
    const invocation = (run.extra?.invocation_params ?? {}) as Record<string, unknown>;
    if (typeof invocation.model === "string") return invocation.model;
    if (typeof invocation.model_name === "string") return invocation.model_name;
    const serialized = ((run.serialized ?? {}) as Record<string, unknown>);
    const kwargs = (serialized.kwargs ?? {}) as Record<string, unknown>;
    if (typeof kwargs.model === "string") return kwargs.model;
    if (typeof kwargs.model_name === "string") return kwargs.model_name;
    return undefined;
  }

  private extractTokens(run: Run): TokenUsage | undefined {
    if (run.run_type !== "llm") return undefined;
    const outputs = (run.outputs ?? {}) as Record<string, unknown>;
    const candidates: Array<Record<string, unknown> | undefined> = [
      outputs.llmOutput as Record<string, unknown> | undefined,
      (outputs.llmOutput as Record<string, unknown> | undefined)?.tokenUsage as
        | Record<string, unknown>
        | undefined,
      (outputs.llmOutput as Record<string, unknown> | undefined)?.usage_metadata as
        | Record<string, unknown>
        | undefined,
      outputs.usage_metadata as Record<string, unknown> | undefined,
    ];
    for (const c of candidates) {
      if (!c) continue;
      const usage = this.readUsage(c);
      if (usage) return usage;
    }
    return undefined;
  }

  private readUsage(src: Record<string, unknown>): TokenUsage | undefined {
    const prompt = src.promptTokens ?? src.prompt_tokens ?? src.input_tokens;
    const completion =
      src.completionTokens ?? src.completion_tokens ?? src.output_tokens;
    const total = src.totalTokens ?? src.total_tokens;
    const usage: TokenUsage = {};
    if (typeof prompt === "number") usage.promptTokens = prompt;
    if (typeof completion === "number") usage.completionTokens = completion;
    if (typeof total === "number") usage.totalTokens = total;
    return Object.keys(usage).length > 0 ? usage : undefined;
  }

  private stringify(value: unknown): string | undefined {
    if (value === undefined || value === null) return undefined;
    try {
      const str = typeof value === "string" ? value : JSON.stringify(value);
      if (!str) return undefined;
      return str.length > MAX_FIELD_LEN
        ? str.slice(0, MAX_FIELD_LEN) + "...[truncated]"
        : str;
    } catch {
      return undefined;
    }
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

    const response = await this.request("POST", "/agent/register", payload);
    const agentId = response?.agentId;
    if (typeof agentId !== "string") {
      throw new Error(`Registration failed: ${JSON.stringify(response)}`);
    }

    this._agentId = agentId;
    this.saveConfig({ agentId });
    return agentId;
  }

  /**
   * Create a session-scoped tracer that can be passed to LangGraph as a callback.
   *
   *   const tracer = await sdk.trace("Answer user question");
   *   await graph.invoke(input, { callbacks: [tracer] });
   */
  trace(
    taskDescription: string,
    options?: { surface?: Surface },
  ): Promise<ProvirasTracer> {
    return ProvirasTracer.create({
      sdk: this,
      taskDescription,
      surface: options?.surface,
    });
  }

  async request(
    method: "POST" | "PATCH",
    endpoint: string,
    payload: unknown,
    headers?: Record<string, string>,
  ): Promise<Record<string, unknown>> {
    const response = await fetch(`${ProvirasSdk.BASE_URL}${endpoint}`, {
      method,
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return (await response.json()) as Record<string, unknown>;
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
