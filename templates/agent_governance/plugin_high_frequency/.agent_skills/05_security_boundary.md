# 05 安全边界 — {{PLUGIN_DISPLAY_NAME}}

## 文件边界

- 可读写: `plugins/{{PLUGIN_NAME}}/` 内所有文件
- 只读: `platform_core/`、`darkbreaker_sdk/`
- 禁止: 其他插件目录、系统配置

## 行为边界

- 不访问外部网络
- 不持久化原始数据到未授权目录
- 日志中不输出敏感标识符原文
- 不执行任意系统命令

## 安全检查

```bash
# 检查潜在安全问题
bandit -r . -ll -q --exclude __pycache__,tests 2>/dev/null || echo "bandit not installed"
```

## 阻断条件

命中以下任一条件 → 停止并报告:
1. 发现硬编码凭证或密钥
2. 发现未授权的文件系统访问
3. 发现网络请求到非白名单地址
