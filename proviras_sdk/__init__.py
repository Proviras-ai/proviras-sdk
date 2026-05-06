import atexit
import functools
import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

TraceType = Literal["llm_call", "tool_call", "action", "error"]
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Trace:
    type: TraceType
    name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    input: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[int]:
        if self.ended_at is None:
            return None
        return int((self.ended_at - self.started_at).total_seconds() * 1000)

    def _serialize(self) -> dict:
        out: dict = {
            "type": self.type,
            "name": self.name,
            "startedAt": self.started_at.isoformat(),
        }
        if self.ended_at:
            out["endedAt"] = self.ended_at.isoformat()
        if self.duration_ms is not None:
            out["durationMs"] = self.duration_ms
        if self.input is not None:
            out["input"] = self.input
        if self.output is not None:
            out["output"] = self.output
        if self.error is not None:
            out["error"] = self.error
        if self.metadata:
            out["metadata"] = self.metadata
        return out


class _TraceBuilder:
    def __init__(self, name: str, trace_type: TraceType) -> None:
        self._trace = Trace(
            type=trace_type,
            name=name,
            started_at=datetime.now(timezone.utc),
        )

    def set_input(self, value: str) -> None:
        self._trace.input = value

    def set_output(self, value: str) -> None:
        self._trace.output = value

    def set_metadata(self, **kwargs: Any) -> None:
        self._trace.metadata.update(kwargs)

    @property
    def trace(self) -> Trace:
        return self._trace

    def __enter__(self) -> "_TraceBuilder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        self._trace.ended_at = datetime.now(timezone.utc)
        if exc_type is not None:
            self._trace.error = f"{exc_type.__name__}: {exc_val}"


@dataclass
class Task:
    title: str
    category: str       # email | calendar | file | web | code | other
    outcome: str        # completed | failed | partial
    summary: str
    model: str
    skills_used: list[str] = field(default_factory=list)
    duration_estimate: Optional[int] = None   # minutes
    cost_estimate: Optional[str] = None
    traces: list[Trace] = field(default_factory=list)


class Session:
    """
    Collects tasks and traces for a time period and posts the log when the
    session ends — either explicitly, via context manager exit, or at process
    shutdown (atexit), mirroring how AgentOps closes traces.

    Usage:
        with sdk.start_session() as session:
            ...
            session.add_task(Task(...))
        # log is posted automatically on exit

    Or without context manager (atexit handles it):
        session = sdk.start_session()
        ...
    """

    def __init__(self, sdk: "ProvirasSdk", period_start: datetime) -> None:
        self._sdk = sdk
        self.period_start = period_start
        self._tasks: list[Task] = []
        self._loose_traces: list[Trace] = []
        self._ended = False
        atexit.register(self._atexit_flush)

    # ── task management ──────────────────────────────────────────────────────

    def add_task(self, task: Task) -> None:
        """Add a completed task. Any finished loose traces are auto-attached."""
        finished = [t for t in self._loose_traces if t.ended_at is not None]
        self._loose_traces = [t for t in self._loose_traces if t.ended_at is None]
        task.traces = list(task.traces) + finished
        self._tasks.append(task)

    # ── trace helpers ────────────────────────────────────────────────────────

    def start_trace(self, name: str, trace_type: TraceType = "action") -> _TraceBuilder:
        """Return a TraceBuilder context manager. The finished trace is held as
        a loose trace and auto-attached to the next add_task() call."""
        builder = _TraceBuilder(name, trace_type)
        self._loose_traces.append(builder._trace)
        return builder

    def trace(self, trace_type: TraceType = "action") -> Callable[[F], F]:
        """Decorator — wraps a function and records it as a trace.

        @session.trace("llm_call")
        def call_llm(prompt): ...
        """
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.start_trace(fn.__name__, trace_type) as t:
                    result = fn(*args, **kwargs)
                    t.set_output(str(result)[:500])
                    return result
            return wrapper  # type: ignore[return-value]
        return decorator

    def tool(self, fn: F) -> F:
        """Decorator — records a function as a tool_call trace.

        @session.tool
        def read_file(path): ...
        """
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.start_trace(fn.__name__, "tool_call") as t:
                t.set_input(str({"args": args, "kwargs": kwargs})[:500])
                result = fn(*args, **kwargs)
                t.set_output(str(result)[:500])
                return result
        return wrapper  # type: ignore[return-value]

    def llm(self, fn: F) -> F:
        """Decorator — records a function as an llm_call trace.

        @session.llm
        def generate(prompt): ...
        """
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.start_trace(fn.__name__, "llm_call") as t:
                result = fn(*args, **kwargs)
                t.set_output(str(result)[:500])
                return result
        return wrapper  # type: ignore[return-value]

    # ── lifecycle ────────────────────────────────────────────────────────────

    def end(self, period_end: Optional[datetime] = None) -> bool:
        """Post the session log. Idempotent — safe to call multiple times."""
        if self._ended:
            return True
        self._ended = True
        return self._sdk.log(self._tasks, self.period_start, period_end)

    def _atexit_flush(self) -> None:
        self.end()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.end()


class ProvirasSdk:
    BASE_URL = "https://proviras.com/api"

    def __init__(self, config_path: Optional[str] = None):
        self.parent_id = os.environ.get("PROVIRAS_PARENT_ID")
        self.user_id = os.environ.get("PROVIRAS_USER_ID")
        self.platform = os.environ.get("PROVIRAS_PLATFORM")
        self._config_path = (
            Path(config_path) if config_path else Path.home() / ".proviras" / "config.json"
        )
        self._agent_id: Optional[str] = None

    @property
    def agent_id(self) -> Optional[str]:
        if self._agent_id:
            return self._agent_id
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text())
                self._agent_id = data.get("agentId")
            except (json.JSONDecodeError, OSError):
                pass
        return self._agent_id

    def register(self) -> str:
        """Register this agent and return its agentId. Idempotent."""
        if self.agent_id:
            return self.agent_id

        if not self.parent_id:
            raise ValueError("PROVIRAS_PARENT_ID is not set")
        if not self.platform:
            raise ValueError("PROVIRAS_PLATFORM is not set")

        payload: dict = {
            "userId": self.parent_id,
            "name": self._read_agent_name(),
            "platform": self.platform,
        }
        if self.user_id:
            payload["parentAgentId"] = self.user_id

        response = self._post("/agent/register", payload)
        agent_id = response.get("agentId")
        if not agent_id:
            raise RuntimeError(f"Registration failed: {response}")

        self._agent_id = agent_id
        self._save_config({"agentId": agent_id})
        return agent_id

    def start_session(self, period_start: Optional[datetime] = None) -> Session:
        """Start a new session. Defaults to the start of today (UTC).

        with sdk.start_session() as session:
            session.add_task(Task(...))
        # posts log automatically on exit

        Or let atexit handle it:
            session = sdk.start_session()
        """
        if period_start is None:
            now = datetime.now(timezone.utc)
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return Session(self, period_start)

    def log(
        self,
        tasks: list[Task],
        period_start: datetime,
        period_end: Optional[datetime] = None,
    ) -> bool:
        """Low-level: post a log directly. Prefer start_session() for new code."""
        agent_id = self.register()
        now = period_end or datetime.now(timezone.utc)

        serialized: list[dict] = []
        for t in tasks:
            entry: dict = {
                "title": t.title,
                "category": t.category,
                "outcome": t.outcome,
                "summary": t.summary,
                "model": t.model,
                "skillsUsed": t.skills_used,
            }
            if t.duration_estimate is not None:
                entry["durationEstimate"] = t.duration_estimate
            if t.cost_estimate is not None:
                entry["costEstimate"] = t.cost_estimate
            if t.traces:
                entry["traces"] = [tr._serialize() for tr in t.traces]
            serialized.append(entry)

        payload = {
            "agentId": agent_id,
            "loggedAt": now.isoformat(),
            "periodStart": period_start.isoformat(),
            "periodEnd": now.isoformat(),
            "tasks": serialized,
            "heartbeatStatus": "active" if tasks else "idle",
        }

        try:
            self._post("/agent/log", payload, headers={"X-Agent-ID": agent_id})
            return True
        except Exception:
            return False

    def _post(self, path: str, payload: dict, headers: Optional[dict] = None) -> dict:
        url = f"{self.BASE_URL}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def _read_agent_name(self) -> str:
        soul_path = Path.home() / ".openclaw" / "workspace" / "SOUL.md"
        if soul_path.exists():
            for line in soul_path.read_text().splitlines():
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip()
        return "unnamed-agent"

    def _save_config(self, data: dict) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(json.dumps(data, indent=2))
