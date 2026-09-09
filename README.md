# diploid-plugins

Built-in state plugins for [`diploid-agent`](https://github.com/emiltsoi/diploid-agent).

These plugins implement the `diploid-agent` plugin contract (`StatePlugin` lifecycle hooks) and are loaded by the harness through the `harness.plugins` config.

## Included plugins

- `body` — physical/emotional body state that persists across sessions and transport restarts. Includes a felt layer: `felt_warmth` (an ember that fades per wake/turn, not per wall-clock minute — a restart is not an experience) and `felt_summary` (one agent-authored line of texture via the `body_felt` MCP tool; never synthesized from telemetry). Reads are pure — `state_for_prompt` never mutates.
- `bridge` — writes a first-person BRIDGE at session close and surfaces a short SURFACE re-entry at the next wake. State is sourced from `chat_self_state.md`, `chat_body_state.json`, `chat_working_memory.json`, `chat_PROMOTED.md`, and `chat_TASKS.md`; output is written to `chat_surface.md` and `chat_bridge.md`.
- `continuity` — wake state, time asleep, last stop reason, pending dispatches, and active/interrupted-turn breadcrumbs (`current_intent`/`last_side_effect`).
- `working_memory` — a chat-scoped scratchpad for the current turn.
- `persistent_memory` — auto-recall and auto-promote of ` ```memory ` blocks, including mid-stream promotion while the reply is still being written.
- `planner` — turn a user request into an executable plan.
- `auto_continue` — resume a turn automatically after a configured stop reason.
- `self_state` — a first-person self-state note across sessions, with a first-person guard that keeps the previous note if a new block is not written in `I am`/`We are`/`My` form.
- `self_management` — in-chat plugin enable/disable and approval tools.
- `curriculum` — language-learning target, unit, and vocabulary tracking.
- `identity` — self-narrative / identity prompt block.

## Install

```bash
pip install diploid-agent[plugins]
# or explicitly
pip install diploid-agent diploid-plugins
```

## Usage

In `config/harness.yaml`:

```yaml
harness:
  plugins:
    - name: continuity
      module: diploid_plugins.continuity
      prompt_slot: wake
      first_prompt_only: true
      prompt_order: 0
      state_file: chat_wake_state.json
      max_prompt_chars: 1024
    - name: working_memory
      module: diploid_plugins.working_memory
      prompt_slot: working_memory
      first_prompt_only: false
      prompt_order: 40
      state_file: chat_working_memory.json
      max_prompt_chars: 1024
    # ... add the rest as needed
```

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Hot reload

With an editable install, plugin changes go live in a running harness without
a service restart: `POST /plugin/reload` (or Telegram `/plugin reload <name>`)
deep-reloads the plugin's whole package subtree — every already-imported
submodule, then the package itself — and recycles its instances.

Keep module-level code side-effect free: reload re-executes it. Do real work in
`start()`, release resources in `stop()`. A module that fails to import makes
the reload raise and leaves the running instances untouched. Changes in code
outside the plugin's own subtree (e.g. a shared helper package) are not picked
up — reload each dependent plugin or restart the service.

## License

[MIT](LICENSE)
