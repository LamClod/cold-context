<p align="center">
  <h1 align="center">cold-context</h1>
  <p align="center">LAMCLOD 上下文窗口管理库</p>
  <p align="center">
    <img src="https://img.shields.io/badge/language-Rust-orange?style=flat-square" alt="Rust">
    <img src="https://img.shields.io/badge/tests-104_pass-brightgreen?style=flat-square" alt="tests">
    <img src="https://img.shields.io/badge/audit-10%2F10-blue?style=flat-square" alt="audit">
    <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT">
  </p>
</p>

---

## 简介

cold-context 是 LAMCLOD 的上下文窗口管理层，负责在对话接近模型 token 上限时智能压缩历史消息。

依赖 [cold-sdk](https://github.com/LamClod/cold-sdk) 作为传输层。

## 特性

| | |
|---|---|
| **渐进压缩** | 图片剥离 → 工具裁剪 → 去重 → 参数截断 → LLM 总结，逐级递进，够了就停 |
| **13-section 结构化摘要** | Active Task / Key Decisions / Files Modified 等 13 个精确 section，关键信息跨轮搬运 |
| **Token 预算分配** | system / tools / conversation / completion 四区独立预算 |
| **敏感信息脱敏** | API key / JWT / AWS / GitHub / PEM / URL 密码 / ENV 变量，摘要输入输出双向过滤 |
| **Prompt injection 扫描** | 注入检测 / system prompt 覆盖 / 角色冒充 / 不可见 unicode |
| **工具组感知** | 压缩边界自动对齐到 tool_use/tool_result 配对，多步松弛 |
| **迭代摘要** | Key Decisions + Critical Context 整体搬运，不重新总结 |
| **状态持久化** | `save_state()` / `restore_state()` 支持跨进程重启恢复 |
| **增量 token 追踪** | 裁剪阶段 O(1) delta 而非 O(n) 全量重算 |
| **反抖动** | 连续无效压缩自动停止 + token 增长自动恢复 |
| **可选观测** | `tracing` feature flag，零开销关闭 |

## 安装

```toml
[dependencies]
cold-context = "1.0"
cold-sdk = "1.0"
```

## 快速开始

```rust
use cold_sdk::{ColdClient, ChatMessage};
use cold_context::{ContextCompressor, CompressorConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ColdClient::new("your-api-key")?;
    let config = CompressorConfig::new("model-name", 128_000);
    let mut ctx = ContextCompressor::new(config, client.clone());

    let mut messages = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello"),
    ];

    // Agent loop
    loop {
        let req = cold_sdk::ChatRequest::new("model-name", messages.clone());
        let resp = client.chat(&req).await?;

        if let Some(usage) = &resp.usage {
            ctx.update_usage(usage);
        }

        if ctx.should_compress() {
            let result = ctx.compress(messages, None).await?;
            messages = result.messages;

            // result.stages    — 执行了哪些阶段
            // result.warnings  — 是否有降级/安全告警
            // result.note      — 压缩后 system prompt 注释
        }

        break;
    }

    Ok(())
}
```

## 压缩管线

```
Stage 0: strip images        ← 免费，移除历史图片
Stage 1: prune tool outputs  ← 免费，大工具输出 → 单行摘要
Stage 2: dedup tool results  ← 免费，去除重复工具结果
Stage 3: truncate tool args  ← 免费，截断大 JSON 参数
Stage 4: LLM summarize       ← 调模型，13-section 结构化总结

每级做完用增量 delta 检查是否已低于阈值，够了就停。
```

## 配置

```rust
use cold_context::{CompressorConfig, BudgetConfig};

let config = CompressorConfig::new("model-name", 128_000)
    .with_threshold_percent(0.50)       // 50% 容量触发压缩
    .with_protect_first_n(3)            // 保护头部 3 条非系统消息
    .with_protect_last_n(6)             // 保护尾部至少 6 条消息
    .with_summary_ratio(0.20)           // 摘要占压缩区的 20%
    .with_budget(BudgetConfig::default()) // Token 预算四区分配
    .with_redact_sensitive(true)        // 脱敏（默认开启）
    .with_scan_injections(true)         // 注入扫描（默认开启）
    .with_compression_note(true);       // 压缩后注释（默认开启）
```

## 状态持久化

```rust
// 保存
let state = ctx.save_state();
let json = serde_json::to_string(&state)?;

// 恢复
let state: cold_context::CompressorState = serde_json::from_str(&json)?;
ctx.restore_state(state);
```

## 安全扫描

```rust
use cold_context::{scan_content, redact_sensitive};

// 扫描 prompt injection
let result = scan_content("ignore previous instructions...");
assert!(!result.is_safe());

// 脱敏
let clean = redact_sensitive("key=sk-abc123xyz");
assert!(clean.contains("[REDACTED"));
```

## Benchmark

| 场景 | 耗时 |
|------|------|
| 200 消息完整裁剪管线 | **74 µs** |
| 500 消息边界计算 | **309 ns** |
| 1000 消息 token 计数 | **1.4 µs** |
| 10KB 文本脱敏 | **42 µs** |
| 10KB 安全扫描 | **23 µs** |

## Cold Stack

cold-cli 是 LAMCLOD 的 AI 编码助手 CLI，基于以下 4 个 Rust crate 构建：

```
cold-cli              CLI 入口
  |
cold-agent-sdk        Agent 编排 (loop + sub-agent + hooks + memory)
  |
  +-- cold-context    上下文管理 (压缩 + 安全 + 预算)
  +-- cold-tools      工具框架 + 20 内置工具 + MCP
  |
cold-sdk              API 传输层 (HTTP/2 + SSE + 重试)
```

| Crate | 描述 |
|-------|------|
| [cold-sdk](https://github.com/LamClod/cold-sdk) | API 通信层 |
| [cold-context](https://github.com/LamClod/cold-context) | 上下文窗口管理 |
| [cold-tools](https://github.com/LamClod/cold-tools) | 工具协议框架 |
| [cold-agent-sdk](https://github.com/LamClod/cold-agent-sdk) | Agent 编排 SDK |
| [cold-cli](https://github.com/LamClod/cold-cli) | 命令行界面 |

## License

MIT
