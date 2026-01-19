# any-llm

A unified Rust interface for LLM providers. Inspired by [mozilla-ai/any-llm](https://github.com/mozilla-ai/any-llm) and ported from [any-llm-ts](https://github.com/r/any-llm-ts).

## Features

- 🔄 **Unified API** - Same interface for all providers
- 🌐 **Remote Providers** - OpenAI, Anthropic
- 🏠 **Local Providers** - Ollama, Llamafile
- 📡 **Streaming** - Full streaming support
- 🔧 **Tool Calling** - Function/tool calling support
- 🦀 **Type Safe** - Full Rust type safety
- 🔌 **Extensible** - Easy to add new providers
- 📦 **Async** - Built on tokio for async operations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
any-llm = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

### Simple API

```rust
use any_llm::{completion, Message, CompletionRequest, MessageContent};

#[tokio::main]
async fn main() -> Result<(), any_llm::AnyLLMError> {
    // Use provider:model format
    let response = completion(CompletionRequest {
        model: "openai:gpt-4o".to_string(),
        messages: vec![Message::user("Hello!")],
        ..Default::default()
    }).await?;

    if let MessageContent::Text(text) = &response.choices[0].message.content {
        println!("{}", text);
    }
    Ok(())
}
```

### Streaming

```rust
use any_llm::{completion_stream, Message, CompletionRequest};
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<(), any_llm::AnyLLMError> {
    let mut stream = completion_stream(CompletionRequest {
        model: "anthropic:claude-3-5-sonnet-20241022".to_string(),
        messages: vec![Message::user("Tell me a story")],
        ..Default::default()
    }).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(content) = &chunk.choices[0].delta.content {
            print!("{}", content);
        }
    }
    Ok(())
}
```

### Class-based API

```rust
use any_llm::{AnyLLM, Message, CompletionRequest, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), any_llm::AnyLLMError> {
    // Create a reusable instance
    let llm = AnyLLM::create("openai", ProviderConfig {
        api_key: Some(Some(std::env::var("OPENAI_API_KEY").ok())),
        ..Default::default()
    })?;

    // Check availability
    let available = llm.is_available().await;

    // List models
    let models = llm.list_models().await?;

    // Make completions (model name without provider prefix)
    let response = llm.completion(CompletionRequest {
        model: "gpt-4o-mini".to_string(),
        messages: vec![Message::user("Hello!")],
        ..Default::default()
    }).await?;

    Ok(())
}
```

### Local LLMs with Ollama

```rust
use any_llm::{completion, Message, CompletionRequest, list_models};

#[tokio::main]
async fn main() -> Result<(), any_llm::AnyLLMError> {
    // List available Ollama models
    let models = list_models("ollama", None).await?;
    println!("Available models: {:?}", models.iter().map(|m| &m.id).collect::<Vec<_>>());

    // Use a local model
    let response = completion(CompletionRequest {
        model: "ollama:llama3.2".to_string(),
        messages: vec![Message::user("Hello!")],
        ..Default::default()
    }).await?;

    Ok(())
}
```

### Tool Calling

```rust
use any_llm::{completion, Message, CompletionRequest, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), any_llm::AnyLLMError> {
    let weather_tool = Tool::function(
        "get_weather",
        Some("Get the current weather in a given location".to_string()),
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state"
                }
            },
            "required": ["location"]
        }),
    );

    let response = completion(CompletionRequest {
        model: "openai:gpt-4o".to_string(),
        messages: vec![Message::user("What's the weather in San Francisco?")],
        tools: Some(vec![weather_tool]),
        ..Default::default()
    }).await?;

    if let Some(tool_calls) = &response.choices[0].message.tool_calls {
        for tool_call in tool_calls {
            println!("Function: {}", tool_call.function.name);
            println!("Arguments: {}", tool_call.function.arguments);
        }
    }

    Ok(())
}
```

## Supported Providers

| Provider | Streaming | Tools | Vision | List Models |
|----------|-----------|-------|--------|-------------|
| OpenAI | ✅ | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ | ❌ (static) |
| Ollama | ✅ | ✅ | ✅ | ✅ |
| Llamafile | ✅ | ✅ | ❌ | ✅ |

## API Reference

### Functions

- `completion(request)` - Create a chat completion
- `completion_stream(request)` - Create a streaming chat completion
- `list_models(provider, config)` - List available models
- `check_provider(provider, config)` - Check if a provider is available
- `get_supported_providers()` - Get list of supported provider names

### AnyLLM Class

```rust
let llm = AnyLLM::create("openai", config)?;

// Methods
llm.is_available().await;
llm.list_models().await?;
llm.completion(request).await?;
llm.completion_stream(request).await?;
```

### Error Handling

```rust
use any_llm::{AnyLLMError, is_rate_limit_error};

match completion(request).await {
    Ok(response) => { /* handle response */ }
    Err(AnyLLMError::MissingApiKey { provider, env_key }) => {
        eprintln!("Set {} environment variable", env_key);
    }
    Err(AnyLLMError::RateLimit { provider, retry_after }) => {
        eprintln!("Rate limited, retry after {:?}s", retry_after);
    }
    Err(AnyLLMError::ProviderUnavailable { provider, reason }) => {
        eprintln!("Provider {} unavailable: {:?}", provider, reason);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Environment Variables

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Ollama | (none required) |
| Llamafile | (none required) |

## Development

```bash
# Run tests
cargo test

# Run tests with Ollama (requires Ollama running)
cargo test --features ollama

# Build
cargo build --release
```

## License

MIT
