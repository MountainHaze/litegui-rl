# Contributing to ClawGUI

Thank you for your interest in contributing! ClawGUI is a research framework for GUI agents — any contribution that makes it more capable, reproducible, or easier to use is welcome.


## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Module-Specific Guidelines](#module-specific-guidelines)
  - [ClawGUI-Eval — Adding a New Model](#clawgui-eval--adding-a-new-model)
  - [ClawGUI-RL — Adding a New Environment](#clawgui-rl--adding-a-new-environment)
  - [ClawGUI-Agent — Adding a New Device or Model Adapter](#clawgui-agent--adding-a-new-device-or-model-adapter)
- [Code Style](#code-style)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)


## Ways to Contribute

| Area | Examples |
|------|---------|
| **Bug fixes** | Fix a crash, incorrect coordinate parsing, broken script |
| **New model support** | Add a new VLM to ClawGUI-Eval or a new GUI model adapter to ClawGUI-Agent |
| **New environments** | Add a new RL training environment (cloud phones, desktop, web) to ClawGUI-RL |
| **Reproduction results** | Run an existing model on a new benchmark and share results |
| **Documentation** | Fix typos, improve explanations, add examples |
| **Tooling** | Improve scripts, add diagnostics, CI improvements |


## Getting Started

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/ClawGUI.git
   cd ClawGUI
   ```

2. **Set up the module** you want to work on. Each module has its own environment:

   ```bash
   # ClawGUI-Eval
   cd clawgui-eval
   conda create -n opengui python=3.12 -y && conda activate opengui
   pip install -r requirements.txt

   # ClawGUI-Agent
   cd clawgui-agent
   uv venv .venv --python 3.12 && source .venv/bin/activate
   uv pip install -e . && uv pip install -e nanobot/

   # ClawGUI-RL
   cd clawgui-rl
   conda create -n opengui-rl python=3.12 -y && conda activate opengui-rl
   pip install -e .
   ```

3. **Create a branch** for your work:

   ```bash
   git checkout -b feat/your-feature-name
   # or
   git checkout -b fix/issue-number-short-description
   ```


## Development Workflow

1. Make your changes in the appropriate module directory.
2. Test locally (see module-specific guidelines below).
3. Update documentation — README files, docstrings, parameter tables — if your change affects user-facing behavior.
4. Commit with a descriptive message:

   ```
   feat(eval): add UI-TARS 2.0 inferencer

   Extends Qwen2.5-VL with updated prompt format and smart_resize
   coordinate handling. Reproduction results on ScreenSpot-Pro: 52.1
   (official: 51.8).
   ```

5. Push and open a Pull Request against the `master` branch.


## Module-Specific Guidelines

### ClawGUI-Eval — Adding a New Model

Adding a new model to ClawGUI-Eval involves four files. See the existing inferencers for reference.

**Step 1 — Create the inferencer.**

Create `inference/<name>_inferencer.py`. Inherit from `BaseInferencer` (or an existing inferencer if the model shares an architecture, e.g. UI-TARS extends `Qwen25VLInferencer`).

Implement four methods:

| Method | Responsibility |
|--------|---------------|
| `_init_model()` | Load model and processor/tokenizer |
| `_build_prompt()` | Construct the input message list for one sample |
| `_generate()` | Call `model.generate()` and decode the output |
| `_post_process()` | Parse the raw text output into a `(x, y)` coordinate |

Key things to get right:
- **Coordinate system**: know whether the model outputs `[0, 1000]`, absolute pixels, `[0, 999]`, or `[0, 1]`. Use the matching normalization in `_post_process()`.
- **Input order** (`tv_or_vt`): check the official implementation for whether text or image comes first in the message list.
- **System prompt**: check whether the model requires a tool-call system prompt or injects it into the user turn.
- **Image resolution**: match `MIN_PIXELS` / `MAX_PIXELS` to the official values.

**Step 2 — Register the inferencer.**

Add it to `inference/__init__.py`:

```python
from .your_model_inferencer import YourModelInferencer

INFERENCER_REGISTRY = {
    ...
    "your_model": YourModelInferencer,
}
```

**Step 3 — Add prompt injection.**

Add the model's prompt format to `data/convert_any_models.py` and generate the model-specific data files for any benchmarks you are supporting.

**Step 4 — Add judge parsing.**

Add an output parser for your model to `judge/grounding_judge.py` (and `osworld_g_judge.py` if supporting OSWorld-G).

**Step 5 — Add launch scripts.**

Create `scripts/infer/transformers/<name>_run_transformers.sh` and (optionally) `scripts/infer/api/<name>_run_api.sh`.

**Step 6 — Report reproduction results.**

Run the full Infer → Judge → Metric pipeline on at least ScreenSpot-Pro and include your results in the PR description. If official numbers are available, compute the reproduction rate.


### ClawGUI-RL — Adding a New Environment

New environments follow the interface in `agent_system/environments/env_package/`. Use `mobileworld/` or `realdevice/` as a reference.

Each environment package must implement:

```python
class YourEnv:
    def reset(self, task: str) -> dict:
        """Reset environment, return initial observation."""

    def step(self, action: dict) -> tuple[dict, float, bool]:
        """Execute action, return (observation, reward, done)."""

    def close(self) -> None:
        """Clean up resources."""
```

Where `observation` is a dict containing at minimum `{"screenshot": <PIL.Image>}`.

Register your environment in the environment factory and add a sample training script under `examples/`.


### ClawGUI-Agent — Adding a New Device or Model Adapter

**New device backend:** Create a new directory under `phone_agent/` (e.g. `phone_agent/mydevice/`) implementing the same interface as `phone_agent/adb/`. Register it in `phone_agent/device_factory.py`.

**New model adapter:** Add a new adapter class in `phone_agent/model/adapters.py`. The adapter is responsible for:
1. Formatting the prompt (system prompt + history + screenshot + task instruction)
2. Parsing the model's raw text output into a structured action dict: `{"action": "tap", "x": 500, "y": 300}` (or equivalent)
3. Normalizing output coordinates to absolute device pixels

Add the new `promptTemplateStyle` value to the configuration documentation in `README.md` and `README_CN.md`.


## Code Style

- **Python**: Follow the existing style of the file you're editing. We don't enforce a strict linter, but aim for readable code with meaningful variable names.
- **Shell scripts**: Use `#!/bin/bash` and keep parameter declarations at the top.
- **No unnecessary dependencies**: Avoid adding new top-level dependencies unless necessary. If you do, update `requirements.txt` or `pyproject.toml` in the relevant module.
- **No breaking changes to existing interfaces** without discussion in an issue first.


## Submitting a Pull Request

1. Fill out the [PR template](.github/PULL_REQUEST_TEMPLATE.md) completely.
2. Keep PRs focused — one feature or fix per PR. Large refactors should be discussed in an issue first.
3. If your PR adds a new model or environment, include concrete test results in the description.
4. A maintainer will review your PR. Please be responsive to feedback — PRs with no activity for 30 days may be closed.


## Reporting Issues

Use the GitHub Issue templates:

- [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) — for crashes, incorrect behavior, or broken scripts
- [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) — for new capabilities or improvements
- [New Model Support](.github/ISSUE_TEMPLATE/new_model.md) — to request or track adding a new model to ClawGUI-Eval

Before opening a new issue, please search existing issues to avoid duplicates.


## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
