# Terminal Execute Prompt（busbar_inspection）

1. 确认工作目录为 `plugins/busbar_inspection`。
2. 优先运行模块化脚本：`run_targeted_tests.sh`。
3. 避免跨插件递归命令，除非任务明确要求。
4. 出错先定位首个失败边界，再修改代码。
5. 输出中必须保留执行命令与退出码。
