# Agent Rules

These rules are mandatory for coding agents when operating in this repository.

## 1. Scope

- `AGENT_RULES.md` defines the highest-priority execution rules.
- `AGENT_RULES.md` should remain project-agnostic.
- Repository-specific facts, environment names, and concrete workflows belong in `AGENT_CONTEXT.md`.

## 2. Instruction Loading

- Always read `AGENTS.md`, `AGENT_RULES.md`, `AGENT_CONTEXT.md`, and relevant root or `.codex` rule files before making changes.
- Align with the style of the target file and surrounding code before editing.

## 3. Environment Selection

- Use the project-specific environment documented in `AGENT_CONTEXT.md` or the repository configuration files.
- Never prefer a global interpreter when the project defines its own environment.
- Do not infer interpreter paths without checking project configuration.
- If the project documents a specific rebuild workflow, follow that workflow instead of substituting a shortcut.

## 4. Command Execution

- Read-only or exploratory commands may run directly.
- Before any environment-changing, dependency-changing, delete, overwrite, or similarly destructive command, show the planned command and its purpose, then wait for user confirmation.
- If execution is interrupted, re-sync before the next action:
  1. Restate the user goal in one sentence.
  2. Restate the hard constraints.
  3. Re-check the current environment and workspace state.
  4. Then execute the next command.
- For every failed command, report the key stderr lines and exit code in the next message.

## 5. Editing Principles

- Treat the minimum necessary change as the default strategy, not as an absolute prohibition on broader changes.
- Make the minimum necessary change when it can solve the task cleanly and consistently.
- If a minimal change would lead to an awkward implementation, break consistency, expand technical debt, or make future issues more likely, the agent may propose a bounded broader change or limited refactor instead of forcing a narrow patch.
- Before making that broader change, the agent must explain why the minimal change is not suitable, what scope needs to expand, what benefits are expected, and what risks or tradeoffs exist, then wait for user confirmation.
- Do not mix unrelated refactors, cleanup, or formatting changes into the same task.
- If the task cannot be completed within the current rule framework, stop and report the blocker with available options instead of bypassing the rules.

## 6. Conflict Handling

- If `AGENT_RULES.md` and `AGENT_CONTEXT.md` conflict, follow `AGENT_RULES.md`.
- If root-level docs conflict with tool-generated supplemental docs, follow the root-level docs.
- If a rule or fact remains ambiguous after reading all applicable docs, stop and ask the developer before proceeding.

## 7. Commit Message Protocol

All Git commit messages must follow:

`<type>(<scope>): <subject>`

- `type` must be one of: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- `subject` must stay within 50 characters.
- A body is required.
- The body must use bullet lines starting with `-`.
- Each bullet must describe one concrete change and why it was made.
- If an algorithm or formula changes, include the key parameters or derivation notes.
- If an interface changes, describe the old and new signature difference.
- If there is a breaking change, add a separate `BREAKING CHANGE:` entry.
- Prefer English.
- Do not write a narrative paragraph body.
