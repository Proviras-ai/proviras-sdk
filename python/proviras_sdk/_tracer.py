from __future__ import annotations

import urllib.error
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from ._sdk import ProvirasSdk, Surface

# LangChain Run.run_type can be any string; the server only records these four.
SERVER_RUN_TYPES = frozenset({"llm", "tool", "chain", "retriever"})

# LangChain wraps user code in synthetic runnable runs. Filter them out of
# node_path so users see their own node names, not LangChain internals.
SYNTHETIC_CHAIN_NAMES = frozenset(
    {
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
    }
)

PARAMETER_SKIP_KEYS = frozenset(
    {"model", "model_name", "messages", "tools", "system", "_type"}
)


class ProvirasTracer(BaseTracer):
    """LangChain/LangGraph callback that streams traces to Proviras.

    Use :meth:`ProvirasTracer.create` (or :meth:`ProvirasSdk.trace`) to build
    a tracer — that creates the Proviras session up front, and ``session_id``
    is populated from the server's response::

        tracer = ProvirasTracer.create(sdk, "Answer user question")
        graph.invoke(input, config={"callbacks": [tracer]})

    Traces post at root-run completion in tree preorder, so every child's
    ``parentTraceId`` (server-issued) is set before it is posted.
    """

    name: str = "proviras_tracer"

    def __init__(
        self,
        sdk: ProvirasSdk,
        task_description: str,
        *,
        surface: Surface = "api",
    ) -> None:
        super().__init__()
        self._sdk = sdk
        self._task_description = task_description
        self._surface = surface
        self._started_at = datetime.now(timezone.utc)
        self._trace_id_map: dict[str, str] = {}
        self.session_id: str = str(uuid.uuid4())
        self._session_attempted = False
        self._finalized = False

    @classmethod
    def create(
        cls,
        sdk: ProvirasSdk,
        task_description: str,
        *,
        surface: Surface = "api",
    ) -> "ProvirasTracer":
        tracer = cls(sdk, task_description, surface=surface)
        tracer._ensure_session()
        return tracer

    def _ensure_session(self) -> None:
        if self._session_attempted:
            return
        self._session_attempted = True
        try:
            agent_id = self._sdk.register()
            try:
                self._post_session(agent_id)
            except urllib.error.HTTPError as e:
                # request() already cleared the cache on 404/410. Re-register
                # to mint a fresh agent and retry the session create once.
                if e.code in (404, 410) and self._sdk.agent_id is None:
                    agent_id = self._sdk.register()
                    self._post_session(agent_id)
                else:
                    raise
        except Exception:
            # Never break the graph because telemetry failed.
            pass

    def _post_session(self, agent_id: str) -> None:
        self._sdk.request(
            "POST",
            "/agent/session",
            {
                "sessionId": self.session_id,
                "agentId": agent_id,
                "taskDescription": self._task_description,
                "startedAt": _iso(self._started_at),
                "surface": self._surface,
            },
            {"X-Agent-ID": agent_id},
        )

    def _persist_run(self, run: Run) -> None:
        self._ensure_session()
        self._post_run_tree(run, [])
        self._finalize_session(run)

    def _post_run_tree(self, run: Run, parent_chain_path: list[str]) -> None:
        is_chain = run.run_type == "chain"
        include_in_path = (
            is_chain
            and run.parent_run_id is not None
            and run.name not in SYNTHETIC_CHAIN_NAMES
        )
        own_path = (
            parent_chain_path + [run.name] if include_in_path else parent_chain_path
        )
        self._post_run(run, own_path)
        for child in run.child_runs or []:
            self._post_run_tree(child, own_path)

    def _post_run(self, run: Run, chain_path: list[str]) -> None:
        if run.run_type not in SERVER_RUN_TYPES:
            return

        start = run.start_time
        end = run.end_time or datetime.now(timezone.utc)
        latency_ms = max(0, int((end - start).total_seconds() * 1000))
        parent_trace_id = (
            self._trace_id_map.get(str(run.parent_run_id))
            if run.parent_run_id is not None
            else None
        )

        payload: dict[str, Any] = {
            "stepId": str(run.id),
            "runType": run.run_type,
            "name": run.name,
            "startedAt": _iso(start),
            "completedAt": _iso(end),
            "latencyMs": latency_ms,
            "status": "error" if run.error else "success",
        }
        if parent_trace_id:
            payload["parentTraceId"] = parent_trace_id
        if chain_path:
            payload["nodePath"] = ".".join(chain_path)
        if run.error:
            payload["error"] = str(run.error)

        if run.run_type == "llm":
            llm_call = self._build_llm_call(run)
            if llm_call:
                payload["llmCall"] = llm_call

        try:
            agent_id = self._sdk.agent_id
            response = self._sdk.request(
                "POST",
                f"/agent/session/{self.session_id}/trace",
                payload,
                {"X-Agent-ID": agent_id} if agent_id else None,
            )
            trace_id = response.get("traceId")
            if isinstance(trace_id, str):
                self._trace_id_map[str(run.id)] = trace_id
        except Exception:
            # never break the graph on telemetry failure
            pass

    def _finalize_session(self, root_run: Run) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            agent_id = self._sdk.agent_id
            payload: dict[str, Any] = {
                "sessionId": self.session_id,
                "status": "error" if root_run.error else "success",
            }
            if root_run.error:
                payload["error"] = str(root_run.error)
            self._sdk.request(
                "PATCH",
                "/agent/session",
                payload,
                {"X-Agent-ID": agent_id} if agent_id else None,
            )
        except Exception:
            pass

    # ── LLM-call payload extraction ──────────────────────────────────────────

    def _build_llm_call(self, run: Run) -> Optional[dict[str, Any]]:
        inputs = run.inputs if isinstance(run.inputs, dict) else {}
        outputs = run.outputs if isinstance(run.outputs, dict) else {}
        extra = run.extra if isinstance(run.extra, dict) else {}
        invocation = extra.get("invocation_params") or {}
        if not isinstance(invocation, dict):
            invocation = {}

        model = self._extract_model(run)
        messages = self._extract_messages(inputs)
        system_prompt = self._extract_system_prompt(invocation, messages)
        tools = self._extract_tools(invocation, run)
        parameters = self._extract_parameters(invocation)
        response_content = self._extract_response_content(outputs)
        stop_reason = self._extract_stop_reason(outputs)
        usage = self._extract_usage(run)

        if not model and not messages and response_content is None:
            return None

        out: dict[str, Any] = {}
        if model:
            out["model"] = model
        if system_prompt:
            out["systemPrompt"] = system_prompt
        if messages is not None:
            out["messages"] = messages
        if tools is not None:
            out["tools"] = tools
        if parameters:
            out["parameters"] = parameters
        if response_content is not None:
            out["responseContent"] = response_content
        if stop_reason:
            out["stopReason"] = stop_reason
        if usage:
            for k in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"):
                if k in usage:
                    out[k] = usage[k]
        return out

    def _extract_model(self, run: Run) -> Optional[str]:
        extra = run.extra if isinstance(run.extra, dict) else {}
        metadata = extra.get("metadata") if isinstance(extra, dict) else None
        if isinstance(metadata, dict):
            v = metadata.get("ls_model_name")
            if isinstance(v, str):
                return v
        invocation = extra.get("invocation_params") if isinstance(extra, dict) else None
        if isinstance(invocation, dict):
            for k in ("model", "model_name"):
                v = invocation.get(k)
                if isinstance(v, str):
                    return v
        serialized = run.serialized if isinstance(run.serialized, dict) else {}
        kwargs = serialized.get("kwargs") if isinstance(serialized, dict) else None
        if isinstance(kwargs, dict):
            for k in ("model", "model_name"):
                v = kwargs.get(k)
                if isinstance(v, str):
                    return v
        return None

    def _extract_usage(self, run: Run) -> Optional[dict[str, int]]:
        outputs = run.outputs if isinstance(run.outputs, dict) else {}
        llm_output = outputs.get("llmOutput") or outputs.get("llm_output") or {}

        candidates: list[Any] = [llm_output]
        if isinstance(llm_output, dict):
            candidates.extend(
                [
                    llm_output.get("token_usage"),
                    llm_output.get("tokenUsage"),
                    llm_output.get("usage_metadata"),
                    llm_output.get("usage"),
                ]
            )
        candidates.append(outputs.get("usage_metadata"))

        for c in candidates:
            if not isinstance(c, dict):
                continue
            usage = self._read_usage(c)
            if usage:
                return usage
        return None

    @staticmethod
    def _read_usage(src: dict[str, Any]) -> Optional[dict[str, int]]:
        input_tokens = (
            src.get("promptTokens")
            or src.get("prompt_tokens")
            or src.get("input_tokens")
        )
        output_tokens = (
            src.get("completionTokens")
            or src.get("completion_tokens")
            or src.get("output_tokens")
        )
        cache_read = (
            src.get("cacheReadInputTokens")
            or src.get("cache_read_input_tokens")
            or src.get("cacheReadTokens")
            or src.get("cache_read_tokens")
        )
        cache_write = (
            src.get("cacheCreationInputTokens")
            or src.get("cache_creation_input_tokens")
            or src.get("cacheWriteTokens")
            or src.get("cache_write_tokens")
        )
        out: dict[str, int] = {}
        if isinstance(input_tokens, int):
            out["inputTokens"] = input_tokens
        if isinstance(output_tokens, int):
            out["outputTokens"] = output_tokens
        if isinstance(cache_read, int):
            out["cacheReadTokens"] = cache_read
        if isinstance(cache_write, int):
            out["cacheWriteTokens"] = cache_write
        return out if ("inputTokens" in out or "outputTokens" in out) else None

    def _extract_messages(self, inputs: dict[str, Any]) -> Any:
        msgs = inputs.get("messages")
        if isinstance(msgs, list):
            # LangChain LLM runs sometimes nest as messages[0] = list of messages
            if msgs and isinstance(msgs[0], list):
                return [_serialize_message(m) for m in msgs[0]]
            return [_serialize_message(m) for m in msgs]
        prompts = inputs.get("prompts")
        if isinstance(prompts, list):
            return prompts
        return None

    @staticmethod
    def _extract_system_prompt(
        invocation: dict[str, Any], messages: Any
    ) -> Optional[str]:
        sys_param = invocation.get("system")
        if isinstance(sys_param, str):
            return sys_param
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    content = m.get("content")
                    if isinstance(content, str):
                        return content
        return None

    @staticmethod
    def _extract_tools(invocation: dict[str, Any], run: Run) -> Any:
        tools = invocation.get("tools")
        if isinstance(tools, list):
            return tools
        serialized = run.serialized if isinstance(run.serialized, dict) else {}
        kwargs = serialized.get("kwargs") if isinstance(serialized, dict) else None
        if isinstance(kwargs, dict):
            t = kwargs.get("tools")
            if isinstance(t, list):
                return t
        return None

    @staticmethod
    def _extract_parameters(invocation: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in invocation.items() if k not in PARAMETER_SKIP_KEYS}

    @staticmethod
    def _extract_response_content(outputs: dict[str, Any]) -> Any:
        gens = outputs.get("generations")
        if not isinstance(gens, list) or not gens:
            return None
        first = gens[0]
        if isinstance(first, list) and first:
            first = first[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if tool_calls:
                return {"content": content, "toolCalls": tool_calls}
            return content
        text = first.get("text")
        if isinstance(text, str):
            return text
        return first

    @staticmethod
    def _extract_stop_reason(outputs: dict[str, Any]) -> Optional[str]:
        gens = outputs.get("generations")
        if isinstance(gens, list) and gens:
            first = gens[0]
            if isinstance(first, list) and first:
                first = first[0]
            if isinstance(first, dict):
                info = first.get("generationInfo") or first.get("generation_info")
                if isinstance(info, dict):
                    for k in ("finish_reason", "stop_reason"):
                        v = info.get(k)
                        if isinstance(v, str):
                            return v
        llm_output = outputs.get("llmOutput") or outputs.get("llm_output")
        if isinstance(llm_output, dict):
            for k in ("finish_reason", "stop_reason"):
                v = llm_output.get(k)
                if isinstance(v, str):
                    return v
        return None


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Convert a LangChain BaseMessage (or dict-shaped message) to a plain
    JSON-able dict with ``role``/``content`` keys."""
    if not isinstance(msg, dict) and not hasattr(msg, "content"):
        return {"role": "unknown", "content": str(msg)}

    if isinstance(msg, dict):
        role = msg.get("role") or msg.get("type") or "unknown"
        out: dict[str, Any] = {"role": _normalize_role(role), "content": msg.get("content")}
        for k in ("tool_calls", "tool_call_id", "name"):
            if k in msg:
                out[k] = msg[k]
        return out

    # BaseMessage subclass
    get_type = getattr(msg, "type", None)
    if callable(getattr(msg, "_get_type", None)):
        get_type = msg._get_type()
    role = _normalize_role(get_type or "unknown")
    out = {"role": role, "content": getattr(msg, "content", None)}
    for k in ("tool_calls", "tool_call_id", "name"):
        v = getattr(msg, k, None)
        if v is not None:
            out[k] = v
    return out


def _normalize_role(t: str) -> str:
    return {"human": "user", "ai": "assistant"}.get(t, t)


def _iso(dt: datetime) -> str:
    """ISO-8601 with 'Z' suffix to match the JS SDK and the server's
    ``z.string().datetime()`` schema."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
