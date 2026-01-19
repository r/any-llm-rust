//! Main API for any-llm.
//!
//! Provides both function-based and class-based interfaces for LLM completions.
//! Follows the patterns from mozilla-ai/any-llm and any-llm-ts.

use futures::Stream;
use std::pin::Pin;

use crate::errors::AnyLLMError;
use crate::providers::Provider;
use crate::registry::{
    create_provider, get_registered_providers, has_provider, parse_model_string,
    register_provider, resolve_provider_and_model,
};
use crate::types::{
    ChatCompletion, ChatCompletionChunk, CompletionRequest, ModelInfo, ProviderConfig,
    ProviderStatus,
};

// =============================================================================
// Provider Registration (called at initialization)
// =============================================================================

/// Initialize built-in providers.
/// This is called automatically when the crate is used.
fn init_providers() {
    use std::sync::Once;
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        #[cfg(feature = "ollama")]
        register_provider("ollama", crate::providers::ollama::OllamaProvider::constructor);

        #[cfg(feature = "openai")]
        register_provider("openai", crate::providers::openai::OpenAIProvider::constructor);

        #[cfg(feature = "anthropic")]
        register_provider("anthropic", crate::providers::anthropic::AnthropicProvider::constructor);

        #[cfg(feature = "llamafile")]
        register_provider("llamafile", crate::providers::llamafile::LlamafileProvider::constructor);
    });
}

// =============================================================================
// Direct API Functions
// =============================================================================

/// Create a chat completion.
///
/// This is the simplest way to make an LLM request. A new provider instance
/// is created for each call (stateless).
///
/// # Example
///
/// ```rust,no_run
/// use any_llm::{completion, Message, CompletionRequest};
///
/// # async fn example() -> Result<(), any_llm::AnyLLMError> {
/// // With provider:model format
/// let response = completion(CompletionRequest {
///     model: "openai:gpt-4o".to_string(),
///     messages: vec![Message::user("Hello!")],
///     ..Default::default()
/// }).await?;
///
/// // With separate provider parameter
/// let response = completion(CompletionRequest {
///     model: "gpt-4o".to_string(),
///     provider: Some("openai".to_string()),
///     messages: vec![Message::user("Hello!")],
///     ..Default::default()
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn completion(request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
    init_providers();

    let resolved = resolve_provider_and_model(&request.model, request.provider.as_deref())?;

    let config = ProviderConfig {
        api_key: request.api_key.clone().map(Some),
        base_url: request.api_base.clone(),
        ..Default::default()
    };

    let provider = create_provider(&resolved.provider, config)?;

    // Update request with resolved model
    let resolved_request = CompletionRequest {
        model: resolved.model,
        stream: Some(false),
        ..request
    };

    provider.completion(resolved_request).await
}

/// Create a streaming chat completion.
///
/// Returns an async stream of completion chunks.
///
/// # Example
///
/// ```rust,no_run
/// use any_llm::{completion_stream, Message, CompletionRequest};
/// use futures::StreamExt;
///
/// # async fn example() -> Result<(), any_llm::AnyLLMError> {
/// let mut stream = completion_stream(CompletionRequest {
///     model: "openai:gpt-4o".to_string(),
///     messages: vec![Message::user("Hello!")],
///     ..Default::default()
/// }).await?;
///
/// while let Some(chunk) = stream.next().await {
///     let chunk = chunk?;
///     if let Some(content) = &chunk.choices[0].delta.content {
///         print!("{}", content);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub async fn completion_stream(
    request: CompletionRequest,
) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError>
{
    init_providers();

    let resolved = resolve_provider_and_model(&request.model, request.provider.as_deref())?;

    let config = ProviderConfig {
        api_key: request.api_key.clone().map(Some),
        base_url: request.api_base.clone(),
        ..Default::default()
    };

    let provider = create_provider(&resolved.provider, config)?;

    // Update request with resolved model
    let resolved_request = CompletionRequest {
        model: resolved.model,
        stream: Some(true),
        ..request
    };

    provider.completion_stream(resolved_request).await
}

/// List available models from a provider.
///
/// # Example
///
/// ```rust,no_run
/// use any_llm::{list_models, ProviderConfig};
///
/// # async fn example() -> Result<(), any_llm::AnyLLMError> {
/// let models = list_models("openai", None).await?;
/// for model in models {
///     println!("{}", model.id);
/// }
/// # Ok(())
/// # }
/// ```
pub async fn list_models(
    provider: &str,
    config: Option<ProviderConfig>,
) -> Result<Vec<ModelInfo>, AnyLLMError> {
    init_providers();

    let instance = create_provider(provider, config.unwrap_or_default())?;
    instance.list_models().await
}

/// Check if a provider is available.
///
/// # Example
///
/// ```rust,no_run
/// use any_llm::{check_provider, ProviderConfig};
///
/// # async fn example() -> Result<(), any_llm::AnyLLMError> {
/// let status = check_provider("ollama", None).await;
/// if status.available {
///     println!("Ollama is running!");
///     if let Some(models) = status.models {
///         println!("Available models: {:?}", models.iter().map(|m| &m.id).collect::<Vec<_>>());
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub async fn check_provider(
    provider: &str,
    config: Option<ProviderConfig>,
) -> ProviderStatus {
    init_providers();

    match create_provider(provider, config.unwrap_or_default()) {
        Ok(instance) => {
            let available = instance.is_available().await;

            let models = if available && instance.supports_list_models() {
                instance.list_models().await.ok()
            } else {
                None
            };

            ProviderStatus {
                provider: instance.provider_name().to_string(),
                available,
                error: None,
                version: None,
                models,
            }
        }
        Err(e) => ProviderStatus {
            provider: provider.to_lowercase(),
            available: false,
            error: Some(e.to_string()),
            version: None,
            models: None,
        },
    }
}

/// Get list of supported provider names.
///
/// # Example
///
/// ```rust
/// use any_llm::get_supported_providers;
///
/// let providers = get_supported_providers();
/// println!("Supported providers: {:?}", providers);
/// ```
pub fn get_supported_providers() -> Vec<String> {
    init_providers();
    get_registered_providers()
}

// =============================================================================
// Class-based API
// =============================================================================

/// AnyLLM struct for working with LLM providers.
///
/// Use this when you need to reuse a provider instance or want more control.
///
/// # Example
///
/// ```rust,no_run
/// use any_llm::{AnyLLM, Message, CompletionRequest, ProviderConfig};
///
/// # async fn example() -> Result<(), any_llm::AnyLLMError> {
/// let llm = AnyLLM::create("openai", ProviderConfig {
///     api_key: Some(Some("sk-...".to_string())),
///     ..Default::default()
/// })?;
///
/// let response = llm.completion(CompletionRequest {
///     model: "gpt-4o".to_string(),
///     messages: vec![Message::user("Hello!")],
///     ..Default::default()
/// }).await?;
/// # Ok(())
/// # }
/// ```
pub struct AnyLLM {
    provider: Box<dyn Provider>,
    provider_name: String,
}

impl AnyLLM {
    /// Create an AnyLLM instance for a provider.
    pub fn create(provider: &str, config: ProviderConfig) -> Result<Self, AnyLLMError> {
        init_providers();

        let instance = create_provider(provider, config)?;
        let provider_name = instance.provider_name().to_string();

        Ok(Self {
            provider: instance,
            provider_name,
        })
    }

    /// Get list of supported providers.
    pub fn get_supported_providers() -> Vec<String> {
        init_providers();
        get_registered_providers()
    }

    /// Check if a provider is supported.
    pub fn has_provider(name: &str) -> bool {
        init_providers();
        has_provider(name)
    }

    /// Parse a model string into provider and model parts.
    pub fn parse_model_string(model: &str) -> Option<crate::types::ParsedModel> {
        init_providers();
        parse_model_string(model)
    }

    /// Get the provider name.
    pub fn name(&self) -> &str {
        &self.provider_name
    }

    /// Check if this provider is available.
    pub async fn is_available(&self) -> bool {
        self.provider.is_available().await
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> {
        self.provider.list_models().await
    }

    /// Create a chat completion.
    ///
    /// Note: The model parameter should NOT include a provider prefix.
    pub async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
        let request = CompletionRequest {
            stream: Some(false),
            ..request
        };
        self.provider.completion(request).await
    }

    /// Create a streaming chat completion.
    pub async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError>
    {
        let request = CompletionRequest {
            stream: Some(true),
            ..request
        };
        self.provider.completion_stream(request).await
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Message, MessageContent, MessageRole};

    // =========================================================================
    // API Functions Tests (matching api.test.ts)
    // =========================================================================

    mod get_supported_providers_tests {
        use super::*;

        #[test]
        fn should_return_list_of_registered_providers() {
            let providers = get_supported_providers();

            assert!(providers.iter().any(|p| p == "openai"));
            assert!(providers.iter().any(|p| p == "anthropic"));
            assert!(providers.iter().any(|p| p == "ollama"));
            assert!(providers.iter().any(|p| p == "llamafile"));
        }
    }

    // =========================================================================
    // AnyLLM Class Tests
    // =========================================================================

    mod any_llm_class_tests {
        use super::*;

        #[test]
        fn should_list_supported_providers() {
            let providers = AnyLLM::get_supported_providers();

            assert!(providers.iter().any(|p| p == "openai"));
            assert!(providers.iter().any(|p| p == "anthropic"));
        }

        #[test]
        fn should_check_if_provider_exists() {
            init_providers();
            
            assert!(AnyLLM::has_provider("openai"));
            assert!(AnyLLM::has_provider("anthropic"));
            assert!(AnyLLM::has_provider("ollama"));
            assert!(!AnyLLM::has_provider("nonexistent"));
        }

        #[test]
        fn should_parse_model_strings() {
            init_providers();
            
            let result = AnyLLM::parse_model_string("openai:gpt-4o");

            assert!(result.is_some());
            let parsed = result.unwrap();
            assert_eq!(parsed.provider, "openai");
            assert_eq!(parsed.model, "gpt-4o");
        }

        #[test]
        fn should_return_none_for_invalid_model_strings() {
            init_providers();
            
            let result = AnyLLM::parse_model_string("just-a-model");

            assert!(result.is_none());
        }
    }

    // =========================================================================
    // Model String Parsing Tests
    // =========================================================================

    mod model_string_parsing_tests {
        use super::*;

        #[test]
        fn should_parse_provider_model_format() {
            init_providers();
            
            let resolved = resolve_provider_and_model("openai:my-model", None);

            assert!(resolved.is_ok());
            let parsed = resolved.unwrap();
            assert_eq!(parsed.provider, "openai");
            assert_eq!(parsed.model, "my-model");
        }

        #[test]
        fn should_handle_model_with_colons() {
            init_providers();
            
            // Some model names have colons (e.g., ollama models like llama3.2:latest)
            let resolved = resolve_provider_and_model("ollama:model:with:colons", None);

            assert!(resolved.is_ok());
            let parsed = resolved.unwrap();
            assert_eq!(parsed.provider, "ollama");
            assert_eq!(parsed.model, "model:with:colons");
        }

        #[test]
        fn should_use_provider_param_when_model_has_no_prefix() {
            init_providers();
            
            let resolved = resolve_provider_and_model("simple-model", Some("openai"));

            assert!(resolved.is_ok());
            let parsed = resolved.unwrap();
            assert_eq!(parsed.provider, "openai");
            assert_eq!(parsed.model, "simple-model");
        }

        #[test]
        fn should_error_when_no_provider_specified() {
            init_providers();
            
            let resolved = resolve_provider_and_model("just-a-model", None);

            assert!(resolved.is_err());
        }
    }

    // =========================================================================
    // Integration Tests (require actual providers)
    // =========================================================================

    #[tokio::test]
    async fn check_provider_returns_unavailable_for_unknown() {
        let status = check_provider("unknown-provider", None).await;

        assert!(!status.available);
        assert!(status.error.is_some());
    }

    #[tokio::test]
    async fn check_provider_ollama_without_running() {
        // This test checks that we handle the case where Ollama isn't running
        // It should return unavailable, not error
        let status = check_provider("ollama", Some(ProviderConfig {
            base_url: Some("http://localhost:99999".to_string()), // Invalid port
            ..Default::default()
        })).await;

        assert!(!status.available);
    }
}
