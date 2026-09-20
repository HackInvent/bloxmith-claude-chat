#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Role: Verifies Claude Chat block behavior.
# File Name: F5.20_claude_chat_block.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2026-05-19
# -----------------------------------------------------------------------------

"""F5.20 - Claude Chat block.

The test runs against a local Anthropic-compatible Messages API endpoint and
verifies instruction input, auth headers, response/raw JSON outputs, chat
history, UI rendering, API-key masking, and both runtime modes.
"""

# Test cases:
# - FB1/FB2/FB4/FB8 - Run text -> Claude Chat -> display in centralized and active runtime, verify /v1/messages request and output propagation.
# - FB3 - Execute the block twice with a run_data store and verify the second request includes the previous exchange.
# - FB5/FB6 - Refuse missing keys and oversized prompts before network calls and ensure api_key never leaks in logs/errors.
# - FB7 - Render modal, inspector, node-card, and assets with model/API fields and masked api_key.

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from typing import Any
import json
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from blocs.claude_chat.block import ClaudeChatBlock
from bloxsmith_app.block_runtime import BlockRuntimeContext
from bloxsmith_app.block_ui import render_block_inspector_panel, render_block_modal, render_block_node_card
from bloxsmith_app.run_data_store import RunDataStore
from ui_smoke_common import (
    create_run_api,
    data_edge,
    display_node,
    expect,
    graph_payload,
    isolated_server,
    text_node,
    wait_for_run_terminal,
)
from urllib.parse import quote
from block_test_packages import install_test_package, release_key, surface_payload


SECRET = "sk-ant-test-claude-chat-secret"
ANSWER = "hello from the fake claude chat"


class FakeClaudeChatHttpServer(ThreadingHTTPServer):
    requests_log: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeClaudeChatHandler)
        self.requests_log = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class FakeClaudeChatHandler(BaseHTTPRequestHandler):
    server: FakeClaudeChatHttpServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            payload = {}
        self.server.requests_log.append(
            {
                "path": self.path,
                "api_key": self.headers.get("x-api-key") or "",
                "anthropic_version": self.headers.get("anthropic-version") or "",
                "content_type": self.headers.get("Content-Type") or "",
                "payload": payload,
            }
        )
        response = json.dumps(
            {
                "id": "msg_test_claude_chat",
                "type": "message",
                "role": "assistant",
                "model": payload.get("model"),
                "content": [{"type": "text", "text": ANSWER}],
                "usage": {"input_tokens": 5, "output_tokens": 4},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        return


class FakeClaudeServer:
    def __enter__(self) -> FakeClaudeChatHttpServer:
        self.server = FakeClaudeChatHttpServer()
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def claude_chat_node(api_base_url: str, *, max_prompt_chars: int = 250000) -> dict[str, Any]:
    return {
        "id": "claude-chat-1",
        "kind": "claude_chat",
        "title": "Claude Chat test",
        "position": {"x": 360, "y": 120},
        "inputs": [
            {"id": 1, "name": "instruction", "title": "Instruction", "accepts": ["message/*", "text/plain"], "multiplicity": "many"}
        ],
        "outputs": [
            {
                "id": 1,
                "name": "response",
                "title": "Response",
                "emits": ["message/*", "text/plain"],
                "multiplicity": "many",
            },
            {
                "id": 2,
                "name": "raw_json",
                "title": "Raw JSON",
                "emits": ["application/json", "message/*"],
                "multiplicity": "many",
            },
        ],
        "config": {
            "model": "claude-sonnet-4-20250514",
            "api_key": SECRET,
            "api_base_url": api_base_url,
            "anthropic_version": "2023-06-01",
            "system_instruction": "Tu es concis.",
            "max_tokens": 256,
            "temperature": "",
            "history_turns": 5,
            "timeout_sec": 10,
            "max_prompt_chars": max_prompt_chars,
        },
    }


def run_claude_chat_case(runtime_mode: str, fake_server: FakeClaudeChatHttpServer) -> dict[str, Any]:
    with isolated_server() as server:
        # Surfaces are release assets: a bundled kind serves none of them.
        model = install_test_package(server, "claude_chat")
        key = quote(release_key(model), safe="")
        served = lambda payload, suffix: next(
            asset["path"] for asset in payload["assets"] if asset["path"].endswith(suffix))
        document = graph_payload(
            f"F5 Claude Chat {runtime_mode}",
            [
                text_node("text-1", "Instruction Claude", "hello claude chat", 80, 120),
                claude_chat_node(fake_server.base_url),
                display_node("display-1", "Affichage", 700, 120),
            ],
            [
                data_edge("edge-text-chat", "text-1", 1, "claude-chat-1", 1),
                data_edge("edge-chat-display", "claude-chat-1", 1, "display-1", 1),
            ],
        )
        created = create_run_api(server, document, runtime_mode=runtime_mode)
        run = wait_for_run_terminal(server, str(created.get("run_id") or ""), timeout_sec=25)

    logs = "\n".join(run.get("logs", []))
    node_logs = "\n".join(run.get("node_logs", {}).get("claude-chat-1", []))
    expect(run.get("status") == "success", f"The Claude Chat {runtime_mode} run must succeed.")
    expect(
        run.get("output_values", {}).get("claude-chat-1:1", {}).get("value") == ANSWER,
        "The text response must leave on the response port.",
    )
    raw_json = run.get("output_values", {}).get("claude-chat-1:2", {}).get("value") or ""
    expect("msg_test_claude_chat" in raw_json and ANSWER in raw_json, "The raw JSON response must be published on raw_json.")
    expect(SECRET not in logs and SECRET not in node_logs, "The API key must not appear in the logs.")
    expect("fallback centralized" not in logs, "The run must not fall back to centralized.")
    if runtime_mode == "zeromq_active":
        expect(
            run.get("results", {}).get("claude-chat-1", {}).get("transport") == "zeromq_active",
            "claude_chat must run through zeromq_active.",
        )
    return run


def test_http_requests(fake_server: FakeClaudeChatHttpServer) -> None:
    """TC1 - Verify Anthropic-compatible Messages API requests."""

    expect(len(fake_server.requests_log) >= 2, "The fake Claude endpoint must receive one request per run.")
    for request in fake_server.requests_log:
        payload = request["payload"]
        expect(request["path"] == "/v1/messages", "The block must call /v1/messages.")
        expect(request["api_key"] == SECRET, "The block must send the API key as x-api-key.")
        expect(request["anthropic_version"] == "2023-06-01", "The block must send anthropic-version.")
        expect(request["content_type"] == "application/json", "The block must send JSON.")
        expect(payload.get("model") == "claude-sonnet-4-20250514", "The Claude model must be sent.")
        expect(payload.get("max_tokens") == 256, "max_tokens must be sent.")
        expect("hello claude chat" in str(payload.get("messages") or ""), "The instruction must be in messages.")
        expect(payload.get("system") == "Tu es concis.", "The system instruction must be sent.")


def test_conversation_history(fake_server: FakeClaudeChatHttpServer) -> None:
    """TC2 - Verify run-local conversation memory keeps the previous exchange."""

    before = len(fake_server.requests_log)
    block = ClaudeChatBlock()
    store = RunDataStore()
    common_context = {
        "run_id": "history-run",
        "node_id": "claude-chat-history",
        "kind": "claude_chat",
        "title": "Claude Chat history",
        "input_content_types": {"instruction": "text/plain"},
        "input_ports": (SimpleNamespace(id=1, name="instruction"),),
        "output_ports": (
            SimpleNamespace(id=1, name="response", emits=("message/*", "text/plain")),
            SimpleNamespace(id=2, name="raw_json", emits=("application/json", "message/*")),
        ),
        "root_dir": ROOT,
        "run_dir": ROOT,
        "store": store,
        "config": {
            "api_key": SECRET,
            "api_base_url": fake_server.base_url,
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 128,
            "history_turns": 1,
            "max_prompt_chars": 10000,
        },
    }
    first = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            inputs={"instruction": "premiere question"},
            input_message="premiere question",
        )
    )
    expect(first.status == "success", "The first call with memory must succeed.")
    second = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            inputs={"instruction": "seconde question"},
            input_message="seconde question",
        )
    )
    expect(second.status == "success", "The second call with memory must succeed.")
    requests = fake_server.requests_log[before:]
    expect(len(requests) == 2, "The memory test must produce two API calls.")
    second_messages = requests[1]["payload"].get("messages")
    expect(isinstance(second_messages, list) and len(second_messages) == 3, "The second call must include the previous exchange and the new question.")
    expect(second_messages[0]["role"] == "user" and "premiere question" in second_messages[0]["content"], "The first question must be restored.")
    expect(second_messages[1]["role"] == "assistant" and ANSWER in second_messages[1]["content"], "The first response must be restored.")
    expect(second_messages[2]["role"] == "user" and "seconde question" in second_messages[2]["content"], "The new question must close the payload.")


def test_prompt_guards(fake_server: FakeClaudeChatHttpServer) -> None:
    """TC3 - Verify missing keys and prompt limits fail before network calls."""

    before = len(fake_server.requests_log)
    block = ClaudeChatBlock()
    common_context = {
        "run_id": "unit-run",
        "node_id": "claude-chat-guard",
        "kind": "claude_chat",
        "title": "Claude Chat guard",
        "inputs": {"instruction": "0123456789"},
        "input_content_types": {"instruction": "text/plain"},
        "input_message": "0123456789",
        "input_ports": (SimpleNamespace(id=1, name="instruction"),),
        "output_ports": (
            SimpleNamespace(id=1, name="response", emits=("message/*", "text/plain")),
            SimpleNamespace(id=2, name="raw_json", emits=("application/json", "message/*")),
        ),
        "root_dir": ROOT,
        "run_dir": ROOT,
    }
    missing_key = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            config={"api_key": "", "api_base_url": fake_server.base_url, "max_prompt_chars": 100},
        )
    )
    expect(missing_key.status == "failed", "A missing API key must fail.")
    expect("api_key" in missing_key.error, "The error must mention the missing API key.")

    oversized = block.execute_runtime(
        BlockRuntimeContext(
            **common_context,
            config={"api_key": SECRET, "api_base_url": fake_server.base_url, "max_prompt_chars": 8},
        )
    )
    expect(oversized.status == "failed", "A prompt that is too long must fail.")
    expect("trop long" in oversized.error, "The error must explain the prompt limit.")
    expect(len(fake_server.requests_log) == before, "Local errors must not call the fake endpoint.")


def test_claude_chat_ui_contract(fake_server: FakeClaudeChatHttpServer) -> None:
    """TC4 - Render Claude Chat block-owned modal, inspector, node-card, and assets."""

    node = claude_chat_node(fake_server.base_url)
    rendered = render_block_modal("claude_chat", {"node": node, "runtime": {}})
    html = str(rendered.get("html") or "")
    assets = rendered.get("assets") or []
    css = (ROOT / "blocs/claude_chat/assets/css/block_modal.css").read_text(encoding="utf-8")
    js = (ROOT / "blocs/claude_chat/assets/js/block_modal.js").read_text(encoding="utf-8")

    expect("cw-claude-chat-modal" in html, "The Claude Chat modal must come from the block.")
    expect('data-block-runtime-refresh="autonomous"' in html, "The Claude Chat modal must own its runtime refresh.")
    expect('data-claude-chat-tab-id="setup"' in html, "The modal must expose the Setup tab.")
    expect('data-claude-chat-tab-id="last-response"' in html, "The modal must expose the Last response tab.")
    expect('data-block-config-field="model"' in html, "Le modele must be editable.")
    expect('data-block-config-field="api_key"' in html, "The API key must be editable.")
    expect('data-block-config-field="history_turns"' in html, "Le nombre d'echanges memorises must be editable.")
    expect(SECRET not in html, "The API key must not be rendered in clear text in the modal.")
    expect("claude-sonnet-4-20250514" in html and "claude-3-5-haiku-latest" in html, "The Claude models must be offered.")
    expect(".claude-chat-modal-panel[hidden]" in css, "The CSS must hide the inactive panels.")
    expect("export function mount" in js, "The JS must mount the modal through the block UI registry.")

    inspector = render_block_inspector_panel("claude_chat", {"node": node})
    inspector_html = str(inspector.get("html") or "")
    expect("cw-claude-chat-inspector" in inspector_html, "The Claude Chat inspector must come from the block.")
    expect('data-block-config-field="system_instruction"' in inspector_html, "The inspector must edit the system instruction.")
    expect('data-block-config-field="history_turns"' in inspector_html, "The inspector must edit the conversation memory.")
    expect(SECRET not in inspector_html, "The API key must not be rendered in clear text in the inspector.")

    card = render_block_node_card("claude_chat", {"node": node})
    card_html = str(card.get("html") or "")
    expect("data-claude-chat-node-card" in card_html, "The Claude Chat node card must come from the block.")


def main() -> None:
    with FakeClaudeServer() as fake_server:
        test_claude_chat_ui_contract(fake_server)
        run_claude_chat_case("centralized", fake_server)
        run_claude_chat_case("zeromq_active", fake_server)
        test_http_requests(fake_server)
        test_conversation_history(fake_server)
        test_prompt_guards(fake_server)
    print("[ok] F5.20_claude_chat_block")


if __name__ == "__main__":
    main()
