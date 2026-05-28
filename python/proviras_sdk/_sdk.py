from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from ._tracer import ProvirasTracer

Surface = Literal["cowork", "chat", "code", "api"]


class ProvirasSdk:
    """Proviras client.

    Reads ``PROVIRAS_PARENT_ID``, ``PROVIRAS_PLATFORM``, and optional
    ``PROVIRAS_USER_ID`` from the environment. Caches an ``agentId`` in
    ``~/.proviras/config.json`` after first registration so subsequent runs
    don't re-register.
    """

    BASE_URL = "https://proviras.com/api"

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._parent_id = os.environ.get("PROVIRAS_PARENT_ID")
        self._user_id = os.environ.get("PROVIRAS_USER_ID")
        self._platform = os.environ.get("PROVIRAS_PLATFORM")
        self._config_path = config_path or Path.home() / ".proviras" / "config.json"
        self._agent_id: Optional[str] = None

    @property
    def agent_id(self) -> Optional[str]:
        if self._agent_id:
            return self._agent_id
        try:
            data = json.loads(self._config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        cached = data.get("agentId")
        if isinstance(cached, str):
            self._agent_id = cached
        return self._agent_id

    def register(self) -> str:
        cached = self.agent_id
        if cached:
            return cached
        if not self._parent_id:
            raise RuntimeError("PROVIRAS_PARENT_ID is not set")
        if not self._platform:
            raise RuntimeError("PROVIRAS_PLATFORM is not set")

        payload: dict[str, str] = {
            "userId": self._parent_id,
            "name": self._read_agent_name(),
            "platform": self._platform,
        }
        if self._user_id:
            payload["parentAgentId"] = self._user_id

        response = self.request("POST", "/agent/register", payload)
        agent_id = response.get("agentId")
        if not isinstance(agent_id, str):
            raise RuntimeError(f"Registration failed: {response!r}")

        self._agent_id = agent_id
        self._save_config({"agentId": agent_id})
        return agent_id

    def trace(
        self,
        task_description: str,
        *,
        surface: Surface = "api",
    ) -> "ProvirasTracer":
        """Create a session-scoped tracer for a single graph invocation.

            tracer = sdk.trace("Answer user question")
            graph.invoke(input, config={"callbacks": [tracer]})
        """
        from ._tracer import ProvirasTracer

        return ProvirasTracer.create(self, task_description, surface=surface)

    def request(
        self,
        method: str,
        endpoint: str,
        payload: Any,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.BASE_URL}{endpoint}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}") from e
        return json.loads(raw) if raw else {}

    def _read_agent_name(self) -> str:
        soul_path = Path.home() / ".openclaw" / "workspace" / "SOUL.md"
        try:
            for line in soul_path.read_text().splitlines():
                if line.startswith("name:"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return "unnamed-agent"

    def _save_config(self, data: dict[str, Any]) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(json.dumps(data, indent=2))
