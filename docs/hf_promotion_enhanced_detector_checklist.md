# STD Enhanced-Detector -> HF Promotion Checklist

This playbook solidifies the reusable promotion path for `STD` enhanced-detector plugins. It is based on real upgrade work from:

- `plugins/switch_inspection`
- `plugins/transformer_inspection`
- `plugins/capacitor_inspection`
- `plugins/busbar_inspection` as the mature structural reference

It is intentionally limited to governance promotion. Do not use it to justify cross-plugin algorithm edits.

## 1. Fixed Scope

Only apply this checklist when all scan-layer conditions below are true:

```yaml
rule_id: std_enhanced_detector_hf_promotion_v1
applies_if:
  governance_level: STD
  object_type: plugin-enhanced-detector
  has_agent_skills: true
required_local_fact_sources:
  - plugin.py
  - detector_enhanced.py
  - manifest.json
  - tests/
  - configs/
preferred_scan_fields:
  - object_type
  - governance_level
  - has_agent_skills
  - has_claude_commands
  - scripts_present
```

If `object_type != plugin-enhanced-detector`, stop and hand off to another promotion path.

## 2. Why This Plugin Type Goes First

Enhanced-detector plugins are the best `STD -> HF` upgrade targets when their fact sources are complete.

- They usually already have the most stable core contract pair: `plugin.py + detector_enhanced.py`.
- They usually already expose the main governance anchors that automation needs: `manifest.json`, `tests/`, `configs/`.
- Their upgrade work is mostly structural: align skills, commands, scripts, and writeback.
- They usually do not require platform-layer or cross-plugin edits to become usable by low-prompt-cost agents.

This is exactly what held for `switch_inspection` and `transformer_inspection`, and it is also why `capacitor_inspection` could be advanced to `准HF` without rewriting algorithm semantics.

## 3. Priority Promotion Gate

A `STD` plugin is a priority HF promotion target only when the following are all true:

1. `governance_level == STD`
2. `object_type == plugin-enhanced-detector`
3. `has_agent_skills == true`
4. Local fact sources exist and are real:
   - `plugin.py`
   - `detector_enhanced.py`
   - `manifest.json`
   - `tests/`
   - `configs/`
5. The plugin can be validated inside its own directory without cross-plugin edits:
   - import or smoke path exists
   - `tests/` is not empty boilerplate
   - promotion does not require changing `platform_core/`, `ui/`, root governance files, or another plugin

`has_claude_commands` and `scripts_present` affect effort, but do not block candidacy by themselves. They tell the promotion agent where to start.

### 3.1 How To Interpret the Preferred Scan Fields

Use the scan fields below to decide both priority and the first action.

- `has_claude_commands == false`
  - still eligible for promotion
  - first action is Step 2, not algorithm work
- `has_claude_commands == true`
  - the plugin is structurally closer to HF
  - first check is whether commands still contain stale template residue
- `scripts_present < 2`
  - promotion is still possible, but Step 3 is the main work item
  - do not claim HF before the four-script loop is real
- `scripts_present in [2, 3]`
  - medium-effort candidate
  - unify script roles before touching README or PROJECT_CARD language
- `scripts_present >= 4`
  - good signal, but only if the scripts are executable and truth-based
  - placeholder or fake-green scripts still trigger a hard stop

## 4. Hard Stop Conditions

If any condition below is true, stop the HF promotion and report the blocker instead of faking closure.

### 4.1 Tests too weak

Stop when any of these are true:

- `tests/` exists but is empty or nearly empty
- there is no runnable contract test for `plugin.py` or `detector_enhanced.py`
- `run_targeted_tests.sh` cannot map modules to real test files
- the only validation path is a README claim with no executable evidence

### 4.2 Core fact source missing

Stop when any of these are missing:

- `plugin.py`
- `detector_enhanced.py`
- `manifest.json`
- `tests/`
- `configs/`

Do not replace missing fact sources with guesses copied from another plugin.

### 4.3 Scripts not executable

Stop when any of these are true:

- required scripts are missing and cannot be added truthfully from local facts
- a script exists but only echoes success
- a script points to nonexistent tests or files
- a script requires tools or paths that do not exist locally and has no explicit skip behavior

### 4.4 Cross-plugin or platform changes required

Stop when HF closure would require edits outside the target plugin directory.

Typical stop examples:

- plugin contract only works after changing shared SDK or platform code
- task routing only makes sense after modifying root router logic
- commands depend on another plugin's scripts or fixtures

## 5. Fixed Upgrade Order

Always follow this order. Do not skip ahead.

### Step 1. Calibrate `.agent_skills/00~08`

Goal: make skills true, compressed, and directly usable.

Required checks:

- `00_project_context.md` matches the real directory, tests, configs, and detector entry points
- `02_algorithm_contract.md` matches `plugin.py`, `detector_enhanced.py`, and `manifest.json`
- `03_test_strategy.md` matches real tests and scripts, not planned tests
- `04_quality_audit.md` lists real current risks, not generic audit boilerplate
- `05_security_boundary.md` reflects actual network/file/process boundaries
- `08_task_routing.md` is execution-oriented, not summary prose

Compression rule:

- remove duplicated explanations already covered by README or PROJECT_CARD
- remove claims about replay, platform integration, or field performance that cannot be executed locally

### Step 2. Connect `.claude/commands`

Required commands:

- `.claude/commands/implement.md`
- `.claude/commands/repair.md`
- `.claude/commands/audit.md`

Command rules:

- every command must route through `.agent_skills/08_task_routing.md`
- commands must not mention missing scripts
- commands must not mention `CLAUDE.md` or generic template text that does not exist in the plugin
- commands must reflect the plugin's real module split, not another detector's business rules

### Step 3. Unify the Four Governance Scripts

Required scripts:

- `scripts/run_targeted_tests.sh`
- `scripts/run_regression_tests.sh`
- `scripts/run_quality_gate.sh`
- `scripts/collect_root_cause.sh`

Script responsibilities are fixed.

`run_targeted_tests.sh`

- fast module-scoped test entry
- must support only real module names
- must fail fast when a selected module has no backing test file

`run_regression_tests.sh`

- full local regression gate
- should chain: targeted gate -> full pytest -> demo smoke if present -> optional regression marker tests -> optional static/security tools
- optional tools such as `mypy`, `flake8`, `bandit`, benchmark scripts must skip explicitly when unavailable
- must not pretend replay coverage exists when it does not

`run_quality_gate.sh`

- governance asset presence check
- anti-pattern scan for production modules
- verify that `.agent_skills/08_task_routing.md` mentions all four scripts
- verify commands are routed through `08_task_routing.md`
- run regression gate as its execution core

`collect_root_cause.sh`

- collect minimum reproducibility evidence under `data/root_cause/<timestamp>/`
- capture env, config, routing docs, audit docs, and current plugin-local git diff
- run targeted/regression commands and store logs without pretending diagnosis is automatic

### Step 4. Calibrate `08_task_routing.md`

This file is the execution hub, not the summary page.

It must include:

- shared pre-read
- per-task read-next for `implement`, `repair`, `audit`
- exact script entrypoints
- escalation path
- writeback destination

Minimum structure:

1. shared pre-read
2. `implement`
3. `repair`
4. `audit`
5. quick map table

Execution rules:

- `implement` must point to `run_targeted_tests.sh <module>`
- `repair` must point to reproduction first, then targeted tests, then `collect_root_cause.sh` if needed
- `audit` must start with `run_quality_gate.sh`
- evidence path must be explicit when `collect_root_cause.sh` is used

### Step 5. Write Back Into `04` and `07`

Promotion is not complete until the plugin writes back what it learned.

Required writeback:

- `04_quality_audit.md`: current blockers, residual risks, and unsupported claims
- `07_learning_log.md`: new failure patterns, repair notes, promotion decisions, and remaining manual gaps

Do not end the upgrade with only scripts and commands changed.

## 6. Minimum Truth Standard for `README.md` and `PROJECT_CARD.md`

These files must exist in at least minimal real form before steady HF can be claimed.

They must state:

- what the plugin actually does today
- what the main runtime entry is
- what tests and scripts are real
- what is still missing
- whether replay data exists or not

Do not copy another plugin's domain contract.

## 7. Promotion Outcome Decision

Use this decision tree after the upgrade work.

### 7.1 Can enter HF

Mark the plugin as `HF` only when all of these are true:

- `.agent_skills/00~08` are real and compressed
- commands exist and route through `08_task_routing.md`
- all four scripts exist and are executable
- `run_quality_gate.sh` passes
- plugin tests are real enough to exercise the current main path
- README and PROJECT_CARD are aligned with facts
- no cross-plugin change is required to keep the plugin truthful

### 7.2 Can only enter `准HF`

Mark as `准HF` when the governance loop is real, but one key fact source for steady HF is still missing.

This is the right label when:

- the four scripts exist and run
- commands and routing are real
- skills/docs are aligned
- but a domain-critical validation source is still missing

Example from real upgrade work:

- `capacitor_inspection` reached `准HF` because the governance loop is real, but there is still no replay corpus proving the intrusion-positive path beyond contract tests

### 7.3 Must remain STD

Keep the plugin at `STD` when any hard stop condition remains unresolved.

Typical examples:

- tests are still too weak
- scripts are still placeholders
- README and PROJECT_CARD still overclaim runtime capability
- promotion would require platform-layer edits

## 8. What We Learned From the Sample Plugins

`switch_inspection`

- proves the value of executable `08_task_routing.md`
- shows that the quality gate should verify routing and command freshness, not only run tests

`transformer_inspection`

- proves HF is still possible without a replay corpus when the current main path is covered by real contracts, demo smoke, and honest documentation
- shows that optional tools must be opt-in or explicitly skipped

`capacitor_inspection`

- proves a plugin can complete the governance loop yet still stop at `准HF`
- shows that replay absence is blocking only when it leaves the plugin's main differentiating path under-evidenced

`busbar_inspection`

- remains the useful structural reference for the four-script split and command/task-routing shape
- should not be copied for transformer, switch, or capacitor business semantics

## 9. HF Promotion Checklist

Use this checklist before an agent starts, during execution, and before declaring the result.

### 9.1 Preflight

- [ ] `governance_level == STD`
- [ ] `object_type == plugin-enhanced-detector`
- [ ] `has_agent_skills == true`
- [ ] `plugin.py` exists
- [ ] `detector_enhanced.py` exists
- [ ] `manifest.json` exists
- [ ] `tests/` exists and contains real test files
- [ ] `configs/` exists
- [ ] upgrade can be completed inside the plugin directory only

### 9.2 Step 1: skills

- [ ] `00_project_context.md` matches real directory facts
- [ ] `02_algorithm_contract.md` matches runtime contract
- [ ] `03_test_strategy.md` matches real tests and scripts
- [ ] `04_quality_audit.md` contains real current risks
- [ ] `08_task_routing.md` is execution-oriented
- [ ] stale or duplicated content is removed

### 9.3 Step 2: commands

- [ ] `implement.md` exists
- [ ] `repair.md` exists
- [ ] `audit.md` exists
- [ ] all commands reference `.agent_skills/08_task_routing.md`
- [ ] no command mentions nonexistent files or generic template residue

### 9.4 Step 3: scripts

- [ ] `run_targeted_tests.sh` exists and fails fast on missing module tests
- [ ] `run_regression_tests.sh` exists and runs a truthful local regression chain
- [ ] `run_quality_gate.sh` exists and checks governance assets plus routing/command freshness
- [ ] `collect_root_cause.sh` exists and writes evidence under `data/root_cause/<timestamp>/`
- [ ] all four scripts are executable

### 9.5 Step 4: task routing

- [ ] `08_task_routing.md` includes shared pre-read
- [ ] `implement` points to `run_targeted_tests.sh <module>`
- [ ] `repair` includes reproduce -> targeted -> regression/root-cause flow
- [ ] `audit` starts with `run_quality_gate.sh`
- [ ] writeback targets are named explicitly

### 9.6 Step 5: writeback

- [ ] `04_quality_audit.md` updated with residual risks
- [ ] `07_learning_log.md` updated with promotion decisions and new failure patterns

### 9.7 Exit decision

- [ ] no hard stop condition remains
- [ ] local scripts pass or skip truthfully
- [ ] README and PROJECT_CARD do not overclaim
- [ ] outcome labeled correctly as `HF`, `准HF`, or `stay STD`

## 10. Recommended Agent Behavior

Future automation or agents should follow this fail-closed policy:

1. evaluate the scan-layer gate
2. verify local fact sources
3. stop immediately on any hard stop condition
4. execute the five upgrade steps in order
5. classify the result as `HF`, `准HF`, or `stay STD`
6. record the specific blocker if the plugin does not reach HF

This rule set is designed to be consumed by future promotion scripts, `sync_agent_commands.py`, and routing agents without relying on plugin name heuristics.
