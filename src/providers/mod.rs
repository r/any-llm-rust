//! Provider implementations for any-llm.
//!
//! This module contains the base Provider trait and implementations for
//! various LLM providers.

mod base;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "llamafile")]
pub mod llamafile;

// Re-exports
pub use base::{Provider, ProviderConstructor};

#[cfg(feature = "openai")]
pub use openai::{OpenAIProvider, OpenAICompatibleProvider};

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;

#[cfg(feature = "llamafile")]
pub use llamafile::LlamafileProvider;
