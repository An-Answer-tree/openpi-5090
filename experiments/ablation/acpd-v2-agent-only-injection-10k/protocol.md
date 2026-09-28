# ACPD-v2 Agentview-only 注入 10K 推理消融

预注册日期：2026-09-29。性质：低成本、冻结 checkpoint 的闭环验证；不重新训练。

## 目的

离线 gate probe 在 H9 的 10K 和 30K checkpoint 上都观察到 agentview-only 分支的
flow MSE 低于关闭注入，而 wrist-only 分支方向相反。单点区间均跨 0，因此本实验
只检验该方向是否能在 LIBERO 闭环任务中复现。

## 设置

| 项目 | 设置 |
|---|---|
| checkpoint | H9-Fixed backview 10K，`9999` |
| 处理 | 推理时保留 agentview contribution，屏蔽 wrist contribution |
| 对照 | 同 checkpoint 完整注入 H9；另有全关闭注入任务 `136866` |
| 评测 | 四套 LIBERO，各500回合，共2000回合；固定 seed 和 episode 顺序 |
| 资源 | 四个单卡 array task，每卡24G；不训练、不写 checkpoint |

## 判据

以 pooled paired success 为主指标。若 agentview-only 相对完整注入提高至少
`1.5` 个百分点且配对区间下界高于 0，则继续验证 H9 30K；否则不扩展训练。
该实验只能判断推理时分支选择，不能证明 contribution loss 或 teacher 蒸馏本身
的因果作用。

## 证据边界

离线 probe 的 agentview-only 优势很小，且置信区间跨 0；本实验的闭环结果优先于
离线 MSE。mask 只改变推理配置，默认 H9 配置 `(True, True)` 和已有 checkpoint
参数树不变。
