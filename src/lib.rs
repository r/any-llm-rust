//! # any-llm
//!
//! A unified Rust interface for LLM providers.
//!
//! Supports both remote providers (OpenAI, Anthropic) and local providers (Ollama, llamafile).
//!
//! Inspired by [mozilla-ai/any-llm](https://github.com/mozilla-ai/any-llm) and
//! [any-llm-ts](https://github.com/r/any-llm-ts).
//!
//! ## Features
//!
//! - 🔄 **Unified API** - Same interface for all providers
//! - 🌐 **Remote Providers** - OpenAI, Anthropic, Mistral, Groq
//! - 🏠 **Local Providers** - Ollama, llamafile
//! - 📡 **Streaming** - Full streaming support
//! - 🔧 **Tool Calling** - Function/tool calling support
//! - 🦀 **Type Safe** - Full Rust type safety
//! - 🔌 **Extensible** - Easy to add new providers
//! - 📦 **Official SDKs** - Built on official provider SDKs for reliability
//!
//! ## Quick Start
//!
//! ### Simple API
//!
//! ```rust,no_run
//! use any_llm::{completion, Message, CompletionRequest, MessageContent};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), any_llm::AnyLLMError> {
//!     // Use provider:model format
//!     let response = completion(CompletionRequest {
//!         model: "openai:gpt-4o".to_string(),
//!         messages: vec![Message::user("Hello!")],
//!         ..Default::default()
//!     }).await?;
//!
//!     if let MessageContent::Text(text) = &response.choices[0].message.content {
//!         println!("{}", text);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Streaming
//!
//! ```rust,no_run
//! use any_llm::{completion_stream, Message, CompletionRequest};
//! use futures::StreamExt;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), any_llm::AnyLLMError> {
//!     let mut stream = completion_stream(CompletionRequest {
//!         model: "anthropic:claude-3-5-sonnet-20241022".to_string(),
//!         messages: vec![Message::user("Tell me a story")],
//!         ..Default::default()
//!     }).await?;
//!
//!     while let Some(chunk) = stream.next().await {
//!         let chunk = chunk?;
//!         if let Some(content) = &chunk.choices[0].delta.content {
//!             print!("{}", content);
//!         }
//!     }
//!     Ok(())
//! }
//! ```
//!
//! ### Class-based API
//!
//! ```rust,no_run
//! use any_llm::{AnyLLM, Message, CompletionRequest, ProviderConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), any_llm::AnyLLMError> {
//!     // Create a reusable instance
//!     let llm = AnyLLM::create("openai", ProviderConfig {
//!         api_key: Some(std::env::var("OPENAI_API_KEY").ok()),
//!         ..Default::default()
//!     })?;
//!
//!     // Check availability
//!     let available = llm.is_available().await;
//!
//!     // List models
//!     let models = llm.list_models().await?;
//!
//!     // Make completions (model name without provider prefix)
//!     let response = llm.completion(CompletionRequest {
//!         model: "gpt-4o-mini".to_string(),
//!         messages: vec![Message::user("Hello!")],
//!         ..Default::default()
//!     }).await?;
//!
//!     Ok(())
//! }
//! ```

// =============================================================================
// Package Info
// =============================================================================

/// Package version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Package name
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");

// =============================================================================
// Modules
// =============================================================================

pub mod types;
pub mod errors;
pub mod registry;
pub mod api;
pub mod providers;

// =============================================================================
// Re-exports
// =============================================================================

// Main API
pub use api::{
    completion,
    completion_stream,
    list_models,
    check_provider,
    get_supported_providers,
    AnyLLM,
};

// Types
pub use types::{
    // Provider types
    LLMProviderType,
    ProviderConfig,
    ProviderMetadata,
    
    // Message types
    MessageRole,
    Message,
    MessageContent,
    TextContentPart,
    ImageContentPart,
    ContentPart,
    ToolCall,
    Tool,
    ToolFunction,
    
    // Request/Response types
    CompletionRequest,
    CompletionChoice,
    CompletionUsage,
    ChatCompletion,
    
    // Streaming types
    ChunkDelta,
    ChunkChoice,
    ChatCompletionChunk,
    
    // Model types
    ModelInfo,
    
    // Utility types
    ParsedModel,
    ProviderStatus,
};

// Errors
pub use errors::{
    AnyLLMError,
    AnyLLMErrorCode,
    is_any_llm_error,
    is_rate_limit_error,
    wrap_error,
};

// Registry (for advanced usage)
pub use registry::{
    register_provider,
    get_provider_constructor,
    has_provider,
    get_registered_providers,
    create_provider,
    parse_model_string,
    resolve_provider_and_model,
    get_provider_for_model,
    clear_provider_cache,
};

// Providers (for advanced usage)
pub use providers::{
    Provider,
    ProviderConstructor,
};

#[cfg(feature = "openai")]
pub use providers::openai::{OpenAIProvider, OpenAICompatibleProvider};

#[cfg(feature = "anthropic")]
pub use providers::anthropic::AnthropicProvider;

#[cfg(feature = "ollama")]
pub use providers::ollama::OllamaProvider;

#[cfg(feature = "llamafile")]
pub use providers::llamafile::LlamafileProvider;
