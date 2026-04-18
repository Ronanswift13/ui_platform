# 05_security_boundary

## 1. 当前安全边界

1. 插件职责应限定为“订阅/接收事件 + 归一化 + 存储 + 分析触发”。
2. 当前不应接入任何自动控制、远方下发、联动操作指令。
3. 配置中的协议凭据不得写入真实账号密码。
4. 拓扑 YAML 属于资产映射数据，不得混入真实敏感秘钥。

## 2. 协议类插件的额外边界

1. 外部连接只允许来自配置声明的协议适配器。
2. 日志中不得回显密码字段。
3. 协议未连接成功时，不应伪装成“数据采集正常”。
4. 默认配置中的 `host/port/username/password` 只能作为示例或空值。

## 3. 当前可执行安全检查

```bash
# 1) 检查配置中是否有真实敏感值
rg -n "password:|username:|token:|secret:" configs plugin.py

# 2) 检查是否出现控制指令/写操作意图
rg -n "write|control|operate|trip_command|close_command" plugin.py

# 3) 检查是否把密码打印到日志
rg -n "password|username|secret|token" plugin.py
```

## 4. 当前阻断条件

1. 默认配置出现真实凭据。
2. 插件新增控制命令能力。
3. 日志中输出敏感配置或协议凭据。
