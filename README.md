# diploid-plugins

Built-in state plugins for [`diploid-agent`](https://github.com/emiltsoi/diploid-agent).

These plugins implement the `diploid-agent` plugin contract (`StatePlugin` lifecycle hooks) and are loaded by the harness through the `harness.plugins` config.

## Included plugins

- `body` — physical/emotional body state that persists across sessions and transport restarts.
- `continuity` — wake state, time asleep, last stop reason, and pending dispatches.
- `working_memory` — a chat-scoped scratchpad for the current turn.
- `persistent_memory` — auto-recall and auto-promote of `memory` blocks.
- `planner` — turn a user request into an executable plan.
- `auto_continue` — resume a turn automatically after a configured stop reason.
- `self_state` — a first-person self-state note across sessions.
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

## License

[MIT](LICENSE)
