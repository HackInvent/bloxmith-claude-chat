# -----------------------------------------------------------------------------
# Role: Implements the Claude Chat API block runtime and UI contract.
# File Name: block.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2026-05-19
# -----------------------------------------------------------------------------

from __future__ import annotations

from html import escape
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
import json
import os
import re
import time

from bloxsmith_app.block_api import (
    append_chat_exchange,
    APPLICATION_JSON,
    BlockDefinition,
    BlockRuntimeContext,
    BlockRuntimeOutput,
    BlockRuntimeResult,
    load_chat_history,
    normalize_history_turns,
    render_inspector_template,
    render_node_card_template,
    TEXT_PLAIN,
)


CLAUDE_CHAT_MODELS = (
    "claude-opus-4-1-20250805",
    "claude-opus-4-1",
    "claude-opus-4-20250514",
    "claude-opus-4-0",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-haiku-4-5",
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
)
DEFAULT_CLAUDE_CHAT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_PROMPT_CHARS = 250_000
MAX_TIMEOUT_SEC = 3600
MAX_PROMPT_CHARS = 1_000_000
MAX_TOKENS_LIMIT = 200_000


# Functional behavior:
# FB1 - Read the current instruction from the input port and reject empty instructions before network calls.
# FB2 - Call Anthropic Messages API at /v1/messages with the configured Claude chat model.
# FB3 - Reuse the last configured user/assistant exchanges from run-local chat history and append the new exchange.
# FB4 - Emit normalized assistant text on text outputs and the full response JSON on raw_json outputs.
# FB5 - Refuse missing API keys and oversized message payloads before calling the API.
# FB6 - Capture model, endpoint, usage, request id, and response metadata without leaking api_key.
# FB7 - Render Claude model/API configuration through block-owned inspector, modal, and node card UI.
# FB8 - Run through the generic runtime path used by both centralized and zeromq_active execution modes.
class ClaudeChatBlockError(ValueError):
    """Raised when the Claude Chat block cannot generate a response."""


class ClaudeChatBlock(BlockDefinition):
    """Autonomous block implementation for `ClaudeChatBlock`."""
    kind = "claude_chat"

    def render_node_card(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the Claude Chat canvas card from the block-owned template."""

        config = self._ui_config(node)
        return render_node_card_template(
            block=self,
            node=node,
            node_classes=["claude-chat-node"],
            replacements={
                "title": node.get("title") or self.default_title(),
                "summary": self._truncate(config["system_instruction"] or "Instruction via input", 62),
                "model": config["model"],
                "history_turns": str(config["history_turns"]),
            },
        )

    def render_modal(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the Claude Chat modal with setup and diagnostics tabs."""

        payload = payload or {}
        title = str(node.get("title") or self.default_title())
        config = self._ui_config(node)
        template = (self.directory / "block_modal.html").read_text(encoding="utf-8")
        replacements = {
            "node_id": escape(str(node.get("id") or ""), quote=True),
            "node_title": escape(title),
            "node_kind": escape(self.kind, quote=True),
            "node_kind_title": escape(str(self.model.get("title") or self.default_title())),
            "modal_tabs_html": self._render_modal_tabs(node, title, config, payload),
        }
        html = template
        for key, value in replacements.items():
            html = html.replace(f"{{{{ {key} }}}}", str(value))
        return {"html": html, "context": {"node_id": str(node.get("id") or ""), "node_kind": self.kind}}

    def render_inspector_panel(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the Claude Chat inspector with generic field bindings."""

        config = self._ui_config(node)
        template = (self.directory / "inspector_panel.html").read_text(encoding="utf-8")
        html = render_inspector_template(
            template=(
                template
                .replace("{{ model_options }}", self._select_options(CLAUDE_CHAT_MODELS, config["model"]))
                .replace("{{ api_key_placeholder }}", "Cle configuree" if config["api_key"] else "sk-ant-...")
                .replace("{{ api_base_url }}", escape(config["api_base_url"], quote=True))
                .replace("{{ anthropic_version }}", escape(config["anthropic_version"], quote=True))
                .replace("{{ system_instruction }}", escape(config["system_instruction"]))
                .replace("{{ max_tokens }}", str(config["max_tokens"]))
                .replace("{{ temperature }}", escape(config["temperature"], quote=True))
                .replace("{{ history_turns }}", str(config["history_turns"]))
                .replace("{{ timeout_sec }}", str(config["timeout_sec"]))
                .replace("{{ max_prompt_chars }}", str(config["max_prompt_chars"]))
            ),
            node={**node, "type": self.kind, "kind": self.kind},
            payload=payload,
        )
        return {
            "html": html,
            "context": {
                "node_id": str(node.get("id") or ""),
                "api_key_configured": bool(config["api_key"]),
                "full_panel": True,
            },
        }

    def execute_runtime(self, context: BlockRuntimeContext) -> BlockRuntimeResult:
        """Call Anthropic Messages API and publish text/raw JSON outputs."""

        logs: list[str] = []
        started = time.perf_counter()
        try:
            config = self.normalize_config(context.config)
            instruction = self._current_instruction(context)
            history = load_chat_history(context, turns=int(config["history_turns"]))
            messages = [*history, {"role": "user", "content": instruction}]
            request_chars = self._messages_char_count(messages) + len(config["system_instruction"])
            if not config["api_key"]:
                raise ClaudeChatBlockError("api_key Anthropic manquante.")
            if not instruction.strip():
                raise ClaudeChatBlockError("Instruction Claude Chat vide.")
            if request_chars > int(config["max_prompt_chars"]):
                raise ClaudeChatBlockError(
                    "Prompt Claude Chat trop long: "
                    f"{request_chars} caracteres > limite {config['max_prompt_chars']}."
                )

            endpoint = self._messages_endpoint(config["api_base_url"])
            logs.append(
                f"[claude-chat] {context.node_id}: model={config['model']} "
                f"endpoint={endpoint} prompt_chars={request_chars} "
                f"history_turns={len(history) // 2} timeout={config['timeout_sec']}s."
            )
            response = self._post_message(messages=messages, config=config, endpoint=endpoint)
            answer = self._extract_output_text(response)
            updated_history = append_chat_exchange(
                context,
                user_message=instruction,
                assistant_message=answer,
                turns=int(config["history_turns"]),
            )
            raw_json = json.dumps(response, ensure_ascii=False, indent=2)
            outputs = self._runtime_outputs(context, answer=answer, raw_json=raw_json)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            request_id = str(response.get("id") or "")
            logs.append(
                f"[done] Claude Chat {context.node_id}: {len(answer)} caractere(s) "
                f"usage_input={usage.get('input_tokens', '?')} usage_output={usage.get('output_tokens', '?')}."
            )
            return BlockRuntimeResult(
                status="success",
                outputs=outputs,
                logs=logs,
                last_message=answer,
                content_type=TEXT_PLAIN,
                worker_received=answer or "-",
                metadata={
                    "claude_chat": {
                        "model": config["model"],
                        "endpoint": endpoint,
                        "request_id": request_id,
                        "usage": usage,
                        "duration": round(time.perf_counter() - started, 3),
                        "history_turns": len(updated_history) // 2,
                    },
                    "last_claude_chat_request": self._last_request_preview(config=config, messages=messages),
                },
            )
        except ClaudeChatBlockError as exc:
            message = self._mask_secret(str(exc), context.config.get("api_key") if isinstance(context.config, dict) else "")
            logs.append(f"[claude-chat-error] {context.node_id}: {message}")
            return BlockRuntimeResult(
                status="failed",
                outputs=[],
                logs=logs,
                error=message,
                exit_code=1,
                last_message=message,
                content_type=TEXT_PLAIN,
                worker_received="-",
            )

    def normalize_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        """Return safe Claude Chat runtime configuration from raw node config."""

        raw = config if isinstance(config, dict) else {}
        return {
            "model": self._normalize_model(raw.get("model")),
            "api_key": str(raw.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or "").strip(),
            "api_base_url": self._normalize_base_url(raw.get("api_base_url")),
            "anthropic_version": str(raw.get("anthropic_version") or DEFAULT_ANTHROPIC_VERSION).strip() or DEFAULT_ANTHROPIC_VERSION,
            "system_instruction": str(raw.get("system_instruction") or "").strip(),
            "max_tokens": self._normalize_int(
                raw.get("max_tokens"),
                default=DEFAULT_MAX_TOKENS,
                minimum=1,
                maximum=MAX_TOKENS_LIMIT,
            ),
            "temperature": self._normalize_temperature(raw.get("temperature")),
            "history_turns": normalize_history_turns(raw.get("history_turns"), default=5),
            "timeout_sec": self._normalize_int(
                raw.get("timeout_sec"),
                default=DEFAULT_TIMEOUT_SEC,
                minimum=1,
                maximum=MAX_TIMEOUT_SEC,
            ),
            "max_prompt_chars": self._normalize_int(
                raw.get("max_prompt_chars"),
                default=DEFAULT_MAX_PROMPT_CHARS,
                minimum=1,
                maximum=MAX_PROMPT_CHARS,
            ),
        }

    def _post_message(self, *, messages: list[dict[str, str]], config: dict[str, Any], endpoint: str) -> dict[str, Any]:
        """Send one JSON request to Anthropic Messages API."""

        payload: dict[str, Any] = {
            "model": config["model"],
            "max_tokens": int(config["max_tokens"]),
            "messages": messages,
        }
        if config["system_instruction"]:
            payload["system"] = config["system_instruction"]
        if config["temperature"] != "":
            payload["temperature"] = float(config["temperature"])

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "x-api-key": config["api_key"],
                "anthropic-version": config["anthropic_version"],
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urlrequest.urlopen(request, timeout=int(config["timeout_sec"])) as response:
                response_body = response.read().decode("utf-8", errors="replace")
        except urlerror.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise ClaudeChatBlockError(
                f"Claude Chat HTTP {exc.code}: {self._mask_secret(response_body, config['api_key'])[:800]}"
            ) from exc
        except urlerror.URLError as exc:
            raise ClaudeChatBlockError(f"Claude Chat API inaccessible: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ClaudeChatBlockError(f"Timeout Claude Chat apres {config['timeout_sec']}s.") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ClaudeChatBlockError(f"Reponse Claude Chat non JSON: {response_body[:600]}") from exc
        if not isinstance(parsed, dict):
            raise ClaudeChatBlockError("Reponse Claude Chat invalide.")
        return parsed

    def _current_instruction(self, context: BlockRuntimeContext) -> str:
        """Return the current user instruction from the first input."""

        for key in ("instruction", "in", "1"):
            value = str(context.input_value(key) or "").strip()
            if value:
                return value
        return str(context.input_message or "").strip()

    def _runtime_outputs(self, context: BlockRuntimeContext, *, answer: str, raw_json: str) -> list[BlockRuntimeOutput]:
        """Map the Claude response text and raw JSON to declared output ports."""

        outputs: list[BlockRuntimeOutput] = []
        for output_port in context.output_ports:
            port_id = int(getattr(output_port, "id", 0) or 0)
            port_name = str(getattr(output_port, "name", "") or "")
            emits = getattr(output_port, "emits", getattr(output_port, "accepts", ()))
            emits_values = tuple(str(item) for item in emits) if isinstance(emits, (list, tuple)) else ()
            is_raw_json = port_name == "raw_json" or APPLICATION_JSON in emits_values
            outputs.append(
                BlockRuntimeOutput(
                    port_id=port_id,
                    port_name=port_name,
                    value=raw_json if is_raw_json else answer,
                    content_type=APPLICATION_JSON if is_raw_json else TEXT_PLAIN,
                )
            )
        return outputs

    def _extract_output_text(self, response: dict[str, Any]) -> str:
        """Extract assistant text from the Anthropic Messages API shape."""

        collected: list[str] = []
        content = response.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    collected.append(str(item["text"]))
        elif isinstance(content, str):
            collected.append(content)
        return "\n".join(part for part in collected if part)

    def _ui_config(self, node: dict[str, Any]) -> dict[str, Any]:
        """Return normalized values used by the inspector, modal, and node card."""

        raw_config = node.get("config") if isinstance(node.get("config"), dict) else {}
        return self.normalize_config(raw_config)

    def _render_modal_tabs(
        self,
        node: dict[str, Any],
        title: str,
        config: dict[str, Any],
        payload: dict[str, Any],
    ) -> str:
        """Render the tabbed Claude Chat modal content."""

        node_dom_id = self._safe_dom_id(str(node.get("id") or "claude-chat"))
        tabs = [
            self._render_tab(
                tab_id="setup",
                tab_dom_id=f"claude-chat-{node_dom_id}-tab-setup",
                panel_dom_id=f"claude-chat-{node_dom_id}-panel-setup",
                title="Setup",
                subtitle="API et memoire",
                selected=True,
            ),
            self._render_tab(
                tab_id="last-response",
                tab_dom_id=f"claude-chat-{node_dom_id}-tab-last-response",
                panel_dom_id=f"claude-chat-{node_dom_id}-panel-last-response",
                title="Last response",
                subtitle="Reponse brute",
                selected=False,
            ),
        ]
        panels = [
            self._render_setup_panel(node_dom_id=node_dom_id, selected=True, title=title, config=config, node=node),
            self._render_last_response_panel(
                node_dom_id=node_dom_id,
                selected=False,
                response=self._ui_last_response(payload),
            ),
        ]
        return (
            '<div class="claude-chat-modal-body" data-claude-chat-modal-tabs>'
            '<nav class="claude-chat-modal-tablist" role="tablist" aria-label="Configuration Claude Chat">'
            + "".join(tabs)
            + '</nav>'
            + '<div class="claude-chat-modal-panels">'
            + "".join(panels)
            + '</div>'
            + '</div>'
        )

    def _render_setup_panel(
        self,
        *,
        node_dom_id: str,
        selected: bool,
        title: str,
        config: dict[str, Any],
        node: dict[str, Any],
    ) -> str:
        """Render identity, model config, memory, and ports in the modal."""

        panel_id = f"claude-chat-{node_dom_id}-panel-setup"
        tab_id = f"claude-chat-{node_dom_id}-tab-setup"
        return (
            '<section class="claude-chat-modal-panel" data-claude-chat-modal-panel '
            f'data-claude-chat-tab-id="setup" id="{escape(panel_id, quote=True)}" role="tabpanel" '
            f'aria-labelledby="{escape(tab_id, quote=True)}"{"" if selected else " hidden"}>'
            '<div class="claude-chat-panel-card">'
            '<div class="claude-chat-grid">'
            f'{self._render_title_field(title)}'
            '<div class="field-group">'
            '<label>Modele</label>'
            f'<select data-block-config-field="model">{self._select_options(CLAUDE_CHAT_MODELS, config["model"])}</select>'
            '</div>'
            '<div class="field-group">'
            '<label>API base URL</label>'
            f'<input data-block-config-field="api_base_url" type="text" autocomplete="off" spellcheck="false" value="{escape(config["api_base_url"], quote=True)}" />'
            '</div>'
            '<div class="field-group">'
            '<label>API key</label>'
            f'<input data-block-config-field="api_key" data-block-skip-empty="true" type="password" autocomplete="off" spellcheck="false" placeholder="{escape("Cle configuree" if config["api_key"] else "sk-ant-...", quote=True)}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Anthropic version</label>'
            f'<input data-block-config-field="anthropic_version" type="text" autocomplete="off" spellcheck="false" value="{escape(config["anthropic_version"], quote=True)}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Max tokens</label>'
            f'<input data-block-config-field="max_tokens" data-block-value-type="integer" type="number" min="1" max="{MAX_TOKENS_LIMIT}" step="1" value="{config["max_tokens"]}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Temperature</label>'
            f'<input data-block-config-field="temperature" type="text" autocomplete="off" spellcheck="false" placeholder="vide = defaut" value="{escape(config["temperature"], quote=True)}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Conversation history</label>'
            f'<input data-block-config-field="history_turns" data-block-value-type="integer" type="number" min="0" max="50" step="1" value="{config["history_turns"]}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Timeout secondes</label>'
            f'<input data-block-config-field="timeout_sec" data-block-value-type="integer" type="number" min="1" max="{MAX_TIMEOUT_SEC}" step="1" value="{config["timeout_sec"]}" />'
            '</div>'
            '<div class="field-group">'
            '<label>Limite prompt</label>'
            f'<input data-block-config-field="max_prompt_chars" data-block-value-type="integer" type="number" min="1" max="{MAX_PROMPT_CHARS}" step="1000" value="{config["max_prompt_chars"]}" />'
            '</div>'
            '</div>'
            '<div class="field-group claude-chat-system">'
            '<label>System instruction</label>'
            '<textarea data-block-config-field="system_instruction" rows="7" spellcheck="false" '
            'placeholder="Role, ton, contraintes globales du modele.">'
            f'{escape(config["system_instruction"])}'
            '</textarea>'
            '</div>'
            '<div class="field-hint">Wire a source to the <code>instruction</code> port. That value becomes the user message courant.</div>'
            f'{self._render_ports_summary(node)}'
            '</div>'
            '</section>'
        )

    def _render_last_response_panel(self, *, node_dom_id: str, selected: bool, response: str) -> str:
        """Render the latest raw Claude response panel."""

        panel_id = f"claude-chat-{node_dom_id}-panel-last-response"
        tab_id = f"claude-chat-{node_dom_id}-tab-last-response"
        source_id = f"{panel_id}-source"
        empty_class = " hidden" if response else ""
        response_class = "" if response else " hidden"
        return (
            '<section class="claude-chat-modal-panel" data-claude-chat-modal-panel '
            f'data-claude-chat-tab-id="last-response" id="{escape(panel_id, quote=True)}" role="tabpanel" '
            f'aria-labelledby="{escape(tab_id, quote=True)}"{"" if selected else " hidden"}>'
            '<div class="claude-chat-panel-card">'
            '<div class="claude-chat-grid">'
            '<div>'
            '<span class="group-label">Derniere reponse</span>'
            '<h3>Last response</h3>'
            '<p class="field-hint">Raw JSON response kept in the last runtime state.</p>'
            '</div>'
            f'<button class="ghost-btn{response_class}" data-block-modal-copy="#{escape(source_id, quote=True)}" type="button">Copier</button>'
            '</div>'
            f'<p class="claude-chat-empty{empty_class}">No Claude Chat response recorded for this block.</p>'
            f'<pre class="claude-chat-last-response-output{response_class}" id="{escape(source_id, quote=True)}" data-block-modal-copy-source>{escape(response)}</pre>'
            '</div>'
            '</section>'
        )

    def _render_title_field(self, title: str) -> str:
        """Render the editable title field without duplicating the modal Apply button."""

        return (
            '<div class="field-group">'
            '<label>Block name</label>'
            f'<input data-block-title-field type="text" autocomplete="off" value="{escape(title, quote=True)}" />'
            '</div>'
        )

    def _render_ports_summary(self, node: dict[str, Any]) -> str:
        """Render a compact modal summary of declared input and output ports."""

        inputs = [port for port in (node.get("inputs") or []) if isinstance(port, dict)]
        outputs = [port for port in (node.get("outputs") or []) if isinstance(port, dict)]
        rows = []
        for label, ports in (("Inputs", inputs), ("Outputs", outputs)):
            cells = "".join(
                f'<code>#{escape(str(port.get("id") or ""))} {escape(str(port.get("name") or ""))}</code>'
                for port in ports
            )
            rows.append(f'<div class="field-hint"><strong>{label}</strong> {cells or "aucun port"}</div>')
        return "".join(rows)

    def _render_tab(
        self,
        *,
        tab_id: str,
        tab_dom_id: str,
        panel_dom_id: str,
        title: str,
        subtitle: str,
        selected: bool,
    ) -> str:
        """Render one modal tab button."""

        return (
            '<button class="claude-chat-modal-tab" data-claude-chat-modal-tab '
            f'data-claude-chat-tab-id="{escape(tab_id, quote=True)}" id="{escape(tab_dom_id, quote=True)}" '
            f'type="button" role="tab" aria-selected="{str(selected).lower()}" '
            f'aria-controls="{escape(panel_dom_id, quote=True)}" tabindex="{0 if selected else -1}">'
            f'<span>{escape(title)}</span>'
            f'<small>{escape(subtitle)}</small>'
            '</button>'
        )

    def _ui_last_response(self, payload: dict[str, Any]) -> str:
        """Return the latest raw JSON response from runtime payload outputs."""

        runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
        result = runtime.get("result") if isinstance(runtime.get("result"), dict) else {}
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        for output in outputs.values():
            if not isinstance(output, dict):
                continue
            if str(output.get("port_name") or "").strip() == "raw_json":
                value = output.get("value") or output.get("last_message")
                return str(value or "")
        return ""

    def _last_request_preview(self, *, config: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a metadata-safe preview of the API request."""

        return {
            "endpoint": self._messages_endpoint(config["api_base_url"]),
            "model": config["model"],
            "prompt_chars": self._messages_char_count(messages) + len(config["system_instruction"]),
            "messages": len(messages),
            "history_turns": config["history_turns"],
            "max_tokens": config["max_tokens"],
        }

    def _select_options(self, values: tuple[str, ...], selected: str) -> str:
        """Render select options and keep custom configured model values selectable."""

        options = list(values)
        if selected and selected not in options:
            options.insert(0, selected)
        return "\n".join(
            f'<option value="{escape(value)}"{" selected" if value == selected else ""}>{escape(value)}</option>'
            for value in options
        )

    def _normalize_model(self, value: Any) -> str:
        """Return the configured Claude model id without blocking new provider model names."""

        model = str(value or DEFAULT_CLAUDE_CHAT_MODEL).strip()
        return model or DEFAULT_CLAUDE_CHAT_MODEL

    def _normalize_base_url(self, value: Any) -> str:
        """Return a normalized Anthropic-compatible API base URL."""

        base_url = str(value or DEFAULT_ANTHROPIC_BASE_URL).strip().rstrip("/")
        return base_url or DEFAULT_ANTHROPIC_BASE_URL

    def _messages_endpoint(self, base_url: str) -> str:
        """Return the Messages API endpoint for the configured base URL."""

        return f"{self._normalize_base_url(base_url)}/v1/messages"

    def _normalize_int(self, raw_value: Any, *, default: int, minimum: int, maximum: int) -> int:
        """Return a bounded integer config value."""

        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    def _normalize_temperature(self, value: Any) -> str:
        """Return an optional bounded temperature value serialized for the UI."""

        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = float(text)
        except (TypeError, ValueError):
            return ""
        return str(max(0.0, min(1.0, parsed)))

    def _safe_dom_id(self, value: str) -> str:
        """Normalize a value so it can be embedded in modal DOM ids."""

        normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip())
        return normalized.strip("-") or "claude-chat"

    def _truncate(self, value: str, max_length: int) -> str:
        """Return a compact one-line preview for the node card."""

        text = str(value or "").replace("\n", " ").strip()
        return text if len(text) <= max_length else f"{text[: max_length - 1]}..."

    def _mask_secret(self, text: str, secret: str) -> str:
        """Mask API key occurrences in runtime errors and logs."""

        if not secret:
            return text
        return str(text).replace(str(secret), "***")

    def _messages_char_count(self, messages: list[dict[str, str]]) -> int:
        """Return the total text size sent to the chat API."""

        return sum(len(str(message.get("content") or "")) for message in messages)
