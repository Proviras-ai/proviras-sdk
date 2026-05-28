from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from ._sdk import ProvirasSdk, Surface

SERVER_RUN_TYPES = frozenset({"llm", "tool", "chain", "retriever"})
MAX_FIELD_LEN = 8000


class ProvirasTracer(BaseTracer):
    """LangChain/LangGraph callback that streams traces to Proviras.

    Use :meth:`ProvirasTracer.create` (or :meth:`ProvirasSdk.trace`) to build
    a tracer — the constructor creates the Proviras session up front so
    ``session_id`` is known before any callbacks fire::

        tracer = ProvirasTracer.create(sdk, "Answer user question")
        graph.invoke(input, config={"callbacks": [tracer]})

    Traces are posted at root-run completion in tree preorder, so every
    child's ``parentTraceId`` (server-issued) is set before it is posted.
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
        self.session_id = str(uuid.uuid4())
        self._started_at = datetime.now(timezone.utc)
        self._trace_id_map: dict[str, str] = {}
        self._totals = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}
        self._session_created = False
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
        if self._session_created:
            return
        try:
            agent_id = self._sdk.register()
            self._sdk.request(
                "POST",
                "/agent/session",
                {
                    "sessionId": self.session_id,
                    "agentId": agent_id,
                    "taskDescription": self._task_description,
                    "startedAt": _iso(self._started_at),
                    "status": "running",
                    "surface": self._surface,
                },
                {"X-Agent-ID": agent_id},
            )
        except Exception:
            # Never break the graph because telemetry failed.
            pass
        finally:
            self._session_created = True

    def _persist_run(self, run: Run) -> None:
        # BaseTracer calls this only for root runs. Walk the tree preorder
        # so parents post (and get a traceId) before children reference them.
        self._ensure_session()
        self._post_run_tree(run)
        self._finalize_session(run)

    def _post_run_tree(self, run: Run) -> None:
        self._post_run(run)
        for child in run.child_runs or []:
            self._post_run_tree(child)

    def _post_run(self, run: Run) -> None:
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

        tokens = self._extract_tokens(run)
        if tokens:
            self._accumulate(tokens)

        payload: dict[str, Any] = {
            "runType": run.run_type,
            "stepId": str(run.id),
            "timestamp": _iso(start),
            "latencyMs": latency_ms,
        }
        if parent_trace_id:
            payload["parentTraceId"] = parent_trace_id
        model = self._extract_model(run)
        if model:
            payload["model"] = model
        input_str = self._stringify(run.inputs)
        if input_str:
            payload["input"] = input_str
        output_str = self._stringify(run.outputs)
        if output_str:
            payload["output"] = output_str
        if tokens:
            payload["tokens"] = tokens
        if run.error:
            payload["error"] = str(run.error)

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
            pass

    def _finalize_session(self, root_run: Run) -> None:
        if self._finalized:
            return
        self._finalized = True

        completed_at = datetime.now(timezone.utc)
        start = root_run.start_time
        end = root_run.end_time or completed_at
        total_latency_ms = max(0, int((end - start).total_seconds() * 1000))

        try:
            agent_id = self._sdk.agent_id
            self._sdk.request(
                "PATCH",
                "/agent/session",
                {
                    "sessionId": self.session_id,
                    "status": "failed" if root_run.error else "completed",
                    "completedAt": _iso(completed_at),
                    "totalTokens": self._totals,
                    "totalLatencyMs": total_latency_ms,
                },
                {"X-Agent-ID": agent_id} if agent_id else None,
            )
        except Exception:
            pass

    def _accumulate(self, tokens: dict[str, int]) -> None:
        for k in ("promptTokens", "completionTokens", "totalTokens"):
            if k in tokens:
                self._totals[k] += tokens[k]

    def _extract_model(self, run: Run) -> Optional[str]:
        extra = run.extra if isinstance(run.extra, dict) else {}
        metadata = extra.get("metadata")
        if isinstance(metadata, dict):
            ls_model = metadata.get("ls_model_name")
            if isinstance(ls_model, str):
                return ls_model
        invocation = extra.get("invocation_params")
        if isinstance(invocation, dict):
            for k in ("model", "model_name"):
                v = invocation.get(k)
                if isinstance(v, str):
                    return v
        serialized = run.serialized if isinstance(run.serialized, dict) else {}
        kwargs = serialized.get("kwargs")
        if isinstance(kwargs, dict):
            for k in ("model", "model_name"):
                v = kwargs.get(k)
                if isinstance(v, str):
                    return v
        return None

    def _extract_tokens(self, run: Run) -> Optional[dict[str, int]]:
        if run.run_type != "llm":
            return None
        outputs = run.outputs if isinstance(run.outputs, dict) else {}
        llm_output = outputs.get("llmOutput") or outputs.get("llm_output")

        candidates: list[Any] = [llm_output, outputs.get("usage_metadata"), outputs]
        if isinstance(llm_output, dict):
            candidates.extend(
                [
                    llm_output.get("token_usage"),
                    llm_output.get("tokenUsage"),
                    llm_output.get("usage_metadata"),
                ]
            )

        for c in candidates:
            if not isinstance(c, dict):
                continue
            usage = self._read_usage(c)
            if usage:
                return usage
        return None

    @staticmethod
    def _read_usage(src: dict[str, Any]) -> Optional[dict[str, int]]:
        prompt = (
            src.get("promptTokens")
            or src.get("prompt_tokens")
            or src.get("input_tokens")
        )
        completion = (
            src.get("completionTokens")
            or src.get("completion_tokens")
            or src.get("output_tokens")
        )
        total = src.get("totalTokens") or src.get("total_tokens")
        usage: dict[str, int] = {}
        if isinstance(prompt, int):
            usage["promptTokens"] = prompt
        if isinstance(completion, int):
            usage["completionTokens"] = completion
        if isinstance(total, int):
            usage["totalTokens"] = total
        return usage or None

    @staticmethod
    def _stringify(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            s = value if isinstance(value, str) else json.dumps(value, default=str)
        except (TypeError, ValueError):
            return None
        if not s:
            return None
        return s if len(s) <= MAX_FIELD_LEN else s[:MAX_FIELD_LEN] + "...[truncated]"


def _iso(dt: datetime) -> str:
    """ISO-8601 with 'Z' suffix to match the JS SDK and the server's
    ``z.string().datetime()`` schema."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
