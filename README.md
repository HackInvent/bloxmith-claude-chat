# Claude Chat Block

<!-- block-metadata:start -->
[![Block version: unversioned](https://img.shields.io/badge/block-unversioned-lightgrey)](model.json)
[![BloxSmith compatibility: 1.0.9](https://img.shields.io/badge/BloxSmith-1.0.9-brightgreen)](compatibility.json)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Verified BloxSmith versions: **1.0.9** (bundled-block tests; see [test evidence](compatibility.json)).
<!-- block-metadata:end -->


## Role

`claude_chat` calls the Anthropic Messages API and emits a Claude assistant response.

## Files

- `block.py`: instruction reading, Anthropic API request, conversation history, response parsing, API-key masking, runtime outputs, and UI rendering.
- `model.json`: instruction input, response/raw JSON outputs, API config, and runtime capabilities.
- `inspector_panel.html`: block-owned inspector UI for model/API settings.
- `block_modal.html`: setup-first modal UI with last-response diagnostics.
- `assets/css/block_modal.css`: modal layout and diagnostics styles.
- `assets/js/block_modal.js`: modal tab keyboard/click behavior.
- `node_card.html`: block-owned canvas card body.
- `tests/F5.20_claude_chat_block.py`: behavior tests for centralized and zeromq_active execution.

## Ports

- Inputs:
  - `instruction` (`id: 1`): required user instruction; accepts generic messages, text, and JSON.
- Outputs:
  - `response` (`id: 1`): emits assistant text as `text/plain`.
  - `raw_json` (`id: 2`): emits the full Anthropic JSON response as `application/json`.

## Configuration

- `model`: Claude chat model id. Default: `claude-sonnet-4-20250514`.
- `api_key`: Anthropic API key. If empty at runtime, `ANTHROPIC_API_KEY` is used.
- `api_base_url`: Anthropic-compatible API root. Default: `https://api.anthropic.com`.
- `anthropic_version`: Anthropic API version header. Default: `2023-06-01`.
- `system_instruction`: optional system prompt sent as the Messages API `system` field.
- `max_tokens`: required maximum response token count sent to the API.
- `temperature`: optional value between `0` and `1`; empty omits the parameter.
- `history_turns`: number of previous user/assistant exchanges stored in the current run and resent on the next call. `0` disables memory.
- `timeout_sec`: HTTP timeout.
- `max_prompt_chars`: local payload guard before the HTTP request.

## Runtime Behavior

`execute_runtime()` reads the current instruction from the input port, loads the configured number of previous exchanges, then posts JSON to:

```text
POST <api_base_url>/v1/messages
```

The request body contains `model`, `max_tokens`, and `messages`. Optional `system` and `temperature` are included only when configured.

The block extracts assistant text from the Anthropic `content[].text` response shape. The raw JSON response is also emitted for diagnostics or downstream parsing.

Conversation history is stored in the current run data under an internal `chat_history.claude_chat.<node_id>` key. A new full run starts empty; run resume or active runtime waves can reuse the restored history.

The same implementation runs in One Shot Simulation (`centralized`) and Active Runtime (`zeromq_active`) through the generic block executor.

## UI Behavior

The inspector and modal expose system instruction, Claude model selection, API base URL, API key, Anthropic version, max tokens, temperature, history turn count, timeout, and prompt guard.

API keys are not rendered back in clear text. Leaving the API key field empty keeps the existing value when the generic frontend binding skips empty sensitive fields. Editable prompt and API settings stay local until the user clicks **Apply**.

The modal declares `data-block-runtime-refresh="autonomous"`; block-owned JS keeps tabs, diagnostics, focus, and draft fields stable during runtime polling.

## Maintenance Notes

Claude Chat behavior belongs in this block. Do not add Claude Chat-specific branches to the orchestrator or active runtime worker; use the generic block executor contract instead.

## Compatibility policy

[compatibility.json](compatibility.json) records HackInvent's verified BloxSmith versions and test evidence. Only the versions listed above have been verified, using the block-owned suites in a **bundled-block test installation**. This is not a certification of managed-package installation, every browser/OS, or live provider availability. Other framework versions are unverified, not necessarily incompatible.

The block-version badge follows `model.json`, not a published Git tag. `unversioned` means that no block release version is declared; no number is inferred from the framework version. The framework still uses `model.json` for its runtime/install contract; the tester-owned JSON does not replace it. Official integration tests run in the private `bloxmith-blocs` workspace. Test helpers and the proprietary framework are not bundled in this public block repository.
