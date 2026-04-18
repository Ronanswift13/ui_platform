# 05 Security Boundary

## File Boundary

- Writable:
  - `plugins/capacitor_inspection/**`
- Read-only reference:
  - `plugins/switch_inspection/**`
  - `plugins/transformer_inspection/**`
  - `plugins/busbar_inspection/**`
- Blocked:
  - any other plugin
  - `ui/`
  - `platform_core/`
  - `darkbreaker_sdk/`
  - root-level governance files

## Runtime Boundary

- No external network calls.
- No raw inspection image persistence into the repo.
- No control-command generation.
- No hidden dependency on another plugin being present at runtime.

## Security Checks

```bash
rg -n "requests\.|http://|https://|urllib|aiohttp" \
  plugin.py detector_enhanced.py standalone demo \
  scripts/run_targeted_tests.sh scripts/run_regression_tests.sh scripts/collect_root_cause.sh
rg -n "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" \
  scripts/run_targeted_tests.sh scripts/run_regression_tests.sh scripts/collect_root_cause.sh
./scripts/run_quality_gate.sh
```

## Special Note

`data/results.db` is runtime state and should not be edited by governance work.
