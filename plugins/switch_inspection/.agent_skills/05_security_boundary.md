# 05 Security Boundary

## File Boundary

- Writable:
  - `plugins/switch_inspection/**`
- Read-only reference:
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
rg -n "requests\.|http://|https://|urllib|aiohttp" plugin.py detector_enhanced.py switch_consistency.py standalone scripts
rg -n "rm -rf|sudo |os\.remove\(|shutil\.rmtree\(" scripts
./scripts/run_quality_gate.sh
```

## Special Note

`switch_consistency.py` may summarize evidence conflicts, but any result from it remains advisory only.
