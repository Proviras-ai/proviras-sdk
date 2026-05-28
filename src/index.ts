import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { randomUUID } from "crypto";
import { BaseTracer, type Run } from "@langchain/core/tracers/base";

export type Surface = "cowork" | "chat" | "code" | "api";

export interface ProvirasTracerOptions {
  sdk: ProvirasSdk;
  taskDescription: string;
  surface?: Surface;
}

// LangChain Run.run_type can be any string; the server only records these four.
const SERVER_RUN_TYPES = new Set(["llm", "tool", "chain", "retriever"]);

// LangChain wraps user code in synthetic runnable runs. Filter them out of
// node_path so users see their own node names, not LangChain internals.
const SYNTHETIC_CHAIN_NAMES = new Set([
  "RunnableSequence",
  "RunnableParallel",
  "RunnableLambda",
  "RunnableMap",
  "RunnableAssign",
  "RunnablePassthrough",
  "RunnableBinding",
  "RunnableWithFallbacks",
  "ChannelRead",
  "ChannelWrite",
  "__start__",
  "__end__",
]);

/**
 * LangChain/LangGraph callback handler that streams traces to Proviras.
 *
 *   const sdk = new ProvirasSdk();
 *   const tracer = await sdk.trace("Answer user question");
 *   await graph.invoke(input, { callbacks: [tracer] });
 *
 * Traces are posted in tree preorder at root-run completion so each child's
 * `parentTraceId` (server-issued) is set before the child posts.
 */
export class ProvirasTracer extends BaseTracer {
  name = "proviras_tracer";

  readonly sessionId: string;
  private readonly sdk: ProvirasSdk;
  private readonly taskDescription: string;
  private readonly surface: Surface;
  private readonly startedAt: Date;
  private readonly traceIdMap: Map<string, string> = new Map();
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
    await this.sdk.request(
      "POST",
      "/agent/session",
      {
        sessionId: this.sessionId,
        agentId,
        taskDescription: this.taskDescription,
        startedAt: this.startedAt.toISOString(),
        surface: this.surface,
      },
      { "X-Agent-ID": agentId },
    );
  }

  protected async persistRun(run: Run): Promise<void> {
    try {
      await this.ensureSession();
    } catch {
      return; // can't post anything without a session
    }
    await this.postRunTree(run, []);
    await this.finalizeSession(run);
  }

  private async postRunTree(run: Run, parentChainPath: string[]): Promise<void> {
    const isChain = run.run_type === "chain";
    const includeInPath =
      isChain && !!run.parent_run_id && !SYNTHETIC_CHAIN_NAMES.has(run.name);
    const ownPath = includeInPath ? [...parentChainPath, run.name] : parentChainPath;

    await this.postRun(run, ownPath);
    for (const child of run.child_runs ?? []) {
      await this.postRunTree(child, ownPath);
    }
  }

  private async postRun(run: Run, chainPath: string[]): Promise<void> {
    if (!SERVER_RUN_TYPES.has(run.run_type)) return;

    const startedAt = new Date(run.start_time);
    const endTime = run.end_time ?? Date.now();
    const completedAt = new Date(endTime);
    const latencyMs = Math.max(0, endTime - run.start_time);
    const parentTraceId = run.parent_run_id
      ? this.traceIdMap.get(run.parent_run_id)
      : undefined;

    const payload: Record<string, unknown> = {
      stepId: run.id,
      runType: run.run_type,
      name: run.name,
      startedAt: startedAt.toISOString(),
      completedAt: completedAt.toISOString(),
      latencyMs,
      status: run.error ? "error" : "success",
    };
    if (parentTraceId) payload.parentTraceId = parentTraceId;
    if (chainPath.length > 0) payload.nodePath = chainPath.join(".");
    if (run.error) payload.error = String(run.error);

    if (run.run_type === "llm") {
      const llmCall = this.buildLlmCall(run);
      if (llmCall) payload.llmCall = llmCall;
    }

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
      // telemetry failures never break the graph
    }
  }

  private async finalizeSession(rootRun: Run): Promise<void> {
    if (this.finalized) return;
    this.finalized = true;

    try {
      const agentId = this.sdk.agentId;
      const payload: Record<string, unknown> = {
        sessionId: this.sessionId,
        status: rootRun.error ? "error" : "success",
      };
      if (rootRun.error) payload.error = String(rootRun.error);
      await this.sdk.request(
        "PATCH",
        "/agent/session",
        payload,
        agentId ? { "X-Agent-ID": agentId } : undefined,
      );
    } catch {
      // swallow
    }
  }

  private buildLlmCall(run: Run): Record<string, unknown> | undefined {
    const inputs = (run.inputs ?? {}) as Record<string, unknown>;
    const outputs = (run.outputs ?? {}) as Record<string, unknown>;
    const extra = (run.extra ?? {}) as Record<string, unknown>;
    const invocation = (extra.invocation_params ?? {}) as Record<string, unknown>;

    const model = this.extractModel(run);
    const messages = this.extractMessages(inputs);
    const systemPrompt = this.extractSystemPrompt(invocation, messages);
    const tools = this.extractTools(invocation, run);
    const parameters = this.extractParameters(invocation);
    const responseContent = this.extractResponseContent(outputs);
    const stopReason = this.extractStopReason(outputs);
    const usage = this.extractUsage(run);

    if (!model && !messages && !responseContent) return undefined;

    const out: Record<string, unknown> = {};
    if (model) out.model = model;
    if (systemPrompt) out.systemPrompt = systemPrompt;
    if (messages) out.messages = messages;
    if (tools) out.tools = tools;
    if (parameters && Object.keys(parameters).length > 0) out.parameters = parameters;
    if (responseContent !== undefined) out.responseContent = responseContent;
    if (stopReason) out.stopReason = stopReason;
    if (usage?.inputTokens !== undefined) out.inputTokens = usage.inputTokens;
    if (usage?.outputTokens !== undefined) out.outputTokens = usage.outputTokens;
    if (usage?.cacheReadTokens !== undefined) out.cacheReadTokens = usage.cacheReadTokens;
    if (usage?.cacheWriteTokens !== undefined) out.cacheWriteTokens = usage.cacheWriteTokens;
    return out;
  }

  private extractModel(run: Run): string | undefined {
    const metadata = (run.extra?.metadata ?? {}) as Record<string, unknown>;
    if (typeof metadata.ls_model_name === "string") return metadata.ls_model_name;
    const invocation = (run.extra?.invocation_params ?? {}) as Record<string, unknown>;
    if (typeof invocation.model === "string") return invocation.model;
    if (typeof invocation.model_name === "string") return invocation.model_name;
    const serialized = (run.serialized ?? {}) as Record<string, unknown>;
    const kwargs = (serialized.kwargs ?? {}) as Record<string, unknown>;
    if (typeof kwargs.model === "string") return kwargs.model;
    if (typeof kwargs.model_name === "string") return kwargs.model_name;
    return undefined;
  }

  private extractUsage(run: Run):
    | {
        inputTokens?: number;
        outputTokens?: number;
        cacheReadTokens?: number;
        cacheWriteTokens?: number;
      }
    | undefined {
    const outputs = (run.outputs ?? {}) as Record<string, unknown>;
    const llmOutput = outputs.llmOutput as Record<string, unknown> | undefined;
    const candidates: Array<Record<string, unknown> | undefined> = [
      llmOutput,
      llmOutput?.tokenUsage as Record<string, unknown> | undefined,
      llmOutput?.usage_metadata as Record<string, unknown> | undefined,
      llmOutput?.usage as Record<string, unknown> | undefined,
      outputs.usage_metadata as Record<string, unknown> | undefined,
    ];
    for (const c of candidates) {
      if (!c) continue;
      const usage = this.readUsage(c);
      if (usage) return usage;
    }
    return undefined;
  }

  private readUsage(src: Record<string, unknown>):
    | {
        inputTokens?: number;
        outputTokens?: number;
        cacheReadTokens?: number;
        cacheWriteTokens?: number;
      }
    | undefined {
    const input = src.promptTokens ?? src.prompt_tokens ?? src.input_tokens;
    const output = src.completionTokens ?? src.completion_tokens ?? src.output_tokens;
    const cacheRead =
      src.cacheReadInputTokens ??
      src.cache_read_input_tokens ??
      src.cacheReadTokens ??
      src.cache_read_tokens;
    const cacheWrite =
      src.cacheCreationInputTokens ??
      src.cache_creation_input_tokens ??
      src.cacheWriteTokens ??
      src.cache_write_tokens;

    const out: {
      inputTokens?: number;
      outputTokens?: number;
      cacheReadTokens?: number;
      cacheWriteTokens?: number;
    } = {};
    if (typeof input === "number") out.inputTokens = input;
    if (typeof output === "number") out.outputTokens = output;
    if (typeof cacheRead === "number") out.cacheReadTokens = cacheRead;
    if (typeof cacheWrite === "number") out.cacheWriteTokens = cacheWrite;
    return out.inputTokens !== undefined || out.outputTokens !== undefined ? out : undefined;
  }

  private extractMessages(inputs: Record<string, unknown>): unknown {
    if (Array.isArray(inputs.messages)) {
      // Some LangChain LLM runs nest as messages[0] = BaseMessage[]
      const first = (inputs.messages as unknown[])[0];
      if (Array.isArray(first)) return (first as unknown[]).map(serializeMessage);
      return (inputs.messages as unknown[]).map(serializeMessage);
    }
    if (Array.isArray(inputs.prompts)) return inputs.prompts;
    return undefined;
  }

  private extractSystemPrompt(
    invocation: Record<string, unknown>,
    messages: unknown,
  ): string | undefined {
    if (typeof invocation.system === "string") return invocation.system;
    if (Array.isArray(messages)) {
      const sys = (messages as Array<{ role?: string; content?: unknown }>).find(
        (m) => m.role === "system",
      );
      if (sys && typeof sys.content === "string") return sys.content;
    }
    return undefined;
  }

  private extractTools(invocation: Record<string, unknown>, run: Run): unknown {
    if (Array.isArray(invocation.tools)) return invocation.tools;
    const serialized = (run.serialized ?? {}) as Record<string, unknown>;
    const kwargs = (serialized.kwargs ?? {}) as Record<string, unknown>;
    if (Array.isArray(kwargs.tools)) return kwargs.tools;
    return undefined;
  }

  private extractParameters(invocation: Record<string, unknown>): Record<string, unknown> {
    const skip = new Set(["model", "model_name", "messages", "tools", "system", "_type"]);
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(invocation)) {
      if (skip.has(k)) continue;
      out[k] = v;
    }
    return out;
  }

  private extractResponseContent(outputs: Record<string, unknown>): unknown {
    const gens = outputs.generations as unknown[] | undefined;
    if (!Array.isArray(gens) || gens.length === 0) return undefined;
    const first = (Array.isArray(gens[0]) ? (gens[0] as unknown[])[0] : gens[0]) as
      | Record<string, unknown>
      | undefined;
    if (!first) return undefined;
    const message = first.message as Record<string, unknown> | undefined;
    if (message) {
      const content = message.content;
      const toolCalls = message.tool_calls;
      if (toolCalls) return { content, toolCalls };
      return content;
    }
    if (typeof first.text === "string") return first.text;
    return first;
  }

  private extractStopReason(outputs: Record<string, unknown>): string | undefined {
    const gens = outputs.generations as unknown[] | undefined;
    if (Array.isArray(gens) && gens.length > 0) {
      const first = (Array.isArray(gens[0]) ? (gens[0] as unknown[])[0] : gens[0]) as
        | Record<string, unknown>
        | undefined;
      const info = first?.generationInfo as Record<string, unknown> | undefined;
      if (typeof info?.finish_reason === "string") return info.finish_reason as string;
      if (typeof info?.stop_reason === "string") return info.stop_reason as string;
    }
    const llmOutput = outputs.llmOutput as Record<string, unknown> | undefined;
    if (typeof llmOutput?.stop_reason === "string") return llmOutput.stop_reason as string;
    return undefined;
  }
}

function serializeMessage(msg: unknown): Record<string, unknown> {
  if (msg === null || typeof msg !== "object") {
    return { role: "unknown", content: String(msg) };
  }
  const m = msg as Record<string, unknown> & { _getType?: () => string };
  const type =
    typeof m._getType === "function" ? m._getType() : (m.type as string | undefined);
  const role =
    type === "human"
      ? "user"
      : type === "ai"
        ? "assistant"
        : type === "system"
          ? "system"
          : type === "tool"
            ? "tool"
            : type ?? "unknown";
  const out: Record<string, unknown> = { role, content: m.content };
  if (m.tool_calls) out.tool_calls = m.tool_calls;
  if (m.tool_call_id) out.tool_call_id = m.tool_call_id;
  if (m.name) out.name = m.name;
  return out;
}

export class ProvirasSdk {
  private static readonly BASE_URL = "https://www.proviras.com/api";

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
