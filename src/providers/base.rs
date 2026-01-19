//! Base provider trait for any-llm.
//!
//! All LLM providers must implement this trait.
//! Follows the patterns from mozilla-ai/any-llm and any-llm-ts.

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::errors::AnyLLMError;
use crate::types::{
    ChatCompletion, ChatCompletionChunk, CompletionRequest, ModelInfo, ProviderConfig,
    ProviderMetadata,
};

// =============================================================================
// Provider Constructor Type
// =============================================================================

/// Type for a provider constructor function.
pub type ProviderConstructor = fn(ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError>;

// =============================================================================
// Provider Trait
// =============================================================================

/// Abstract base trait for LLM providers.
///
/// Implementations should:
/// - Implement all metadata methods
/// - Implement the completion methods
/// - Handle their own API key management
///
/// # Example
///
/// ```ignore
/// use any_llm::providers::Provider;
/// use async_trait::async_trait;
///
/// struct MyProvider {
///     api_key: Option<String>,
///     base_url: String,
/// }
///
/// #[async_trait]
/// impl Provider for MyProvider {
///     fn provider_name(&self) -> &str { "my_provider" }
///     fn env_api_key_name(&self) -> &str { "MY_PROVIDER_API_KEY" }
///     // ... implement other methods
/// }
/// ```
#[async_trait]
pub trait Provider: Send + Sync {
    // =========================================================================
    // Provider Metadata (required)
    // =========================================================================

    /// Provider identifier (e.g., 'openai', 'anthropic').
    fn provider_name(&self) -> &str;

    /// Environment variable name for the API key.
    fn env_api_key_name(&self) -> &str;

    /// URL to provider documentation.
    fn provider_documentation_url(&self) -> &str;

    /// Default API base URL.
    fn api_base(&self) -> &str;

    // =========================================================================
    // Feature Flags (with defaults)
    // =========================================================================

    /// Whether provider supports streaming completions.
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Whether provider supports tool/function calling.
    fn supports_tools(&self) -> bool {
        true
    }

    /// Whether provider supports vision/image inputs.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Whether provider supports listing models.
    fn supports_list_models(&self) -> bool {
        false
    }

    /// Whether provider exposes reasoning/thinking content.
    fn supports_reasoning(&self) -> bool {
        false
    }

    // =========================================================================
    // Metadata
    // =========================================================================

    /// Get provider metadata.
    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            name: self.provider_name().to_string(),
            env_key: self.env_api_key_name().to_string(),
            doc_url: self.provider_documentation_url().to_string(),
            streaming: self.supports_streaming(),
            reasoning: self.supports_reasoning(),
            completion: true,
            embedding: false,
            image: self.supports_vision(),
            list_models: self.supports_list_models(),
        }
    }

    // =========================================================================
    // Core Methods (required)
    // =========================================================================

    /// Create a chat completion.
    ///
    /// # Arguments
    ///
    /// * `request` - The completion request
    ///
    /// # Returns
    ///
    /// The completion response.
    async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError>;

    /// Create a streaming chat completion.
    ///
    /// # Arguments
    ///
    /// * `request` - The completion request
    ///
    /// # Returns
    ///
    /// An async stream of completion chunks.
    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError>;

    /// Check if the provider is available.
    ///
    /// For local providers, this checks if the service is running.
    /// For remote providers, this validates the API key.
    async fn is_available(&self) -> bool;

    // =========================================================================
    // Optional Methods (with defaults)
    // =========================================================================

    /// List available models from this provider.
    ///
    /// Override in providers that support model listing.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> {
        if !self.supports_list_models() {
            return Ok(vec![]);
        }
        Err(AnyLLMError::invalid_request(format!(
            "list_models not implemented for {}",
            self.provider_name()
        )))
    }
}

// =============================================================================
// Helper Functions for Providers
// =============================================================================

/// Resolve API key from config or environment.
pub fn resolve_api_key(config_key: Option<&str>, env_var_name: &str) -> Option<String> {
    // First check config
    if let Some(key) = config_key {
        if !key.is_empty() {
            return Some(key.to_string());
        }
    }

    // Then check environment
    std::env::var(env_var_name).ok()
}

/// Check if API key is required and present.
pub fn validate_api_key(
    api_key: &Option<String>,
    provider_name: &str,
    env_var_name: &str,
    requires_api_key: bool,
) -> Result<(), AnyLLMError> {
    if requires_api_key && api_key.is_none() {
        return Err(AnyLLMError::missing_api_key(provider_name, env_var_name));
    }
    Ok(())
}

/// Get base URL from config or default.
pub fn resolve_base_url(config_url: Option<&str>, default_url: &str) -> String {
    config_url
        .filter(|s| !s.is_empty())
        .unwrap_or(default_url)
        .to_string()
}

/// Get timeout from config or default (in milliseconds).
pub fn resolve_timeout(config_timeout: Option<u64>, default_ms: u64) -> u64 {
    config_timeout.unwrap_or(default_ms)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_api_key_from_config() {
        let key = resolve_api_key(Some("test-key"), "TEST_API_KEY");
        assert_eq!(key, Some("test-key".to_string()));
    }

    #[test]
    fn test_resolve_api_key_empty_config() {
        let key = resolve_api_key(Some(""), "NONEXISTENT_VAR_12345");
        assert_eq!(key, None);
    }

    #[test]
    fn test_resolve_api_key_none_config() {
        // This will check env var, which likely doesn't exist
        let key = resolve_api_key(None, "NONEXISTENT_VAR_12345");
        assert_eq!(key, None);
    }

    #[test]
    fn test_validate_api_key_present() {
        let result = validate_api_key(
            &Some("key".to_string()),
            "test",
            "TEST_KEY",
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_api_key_missing_required() {
        let result = validate_api_key(&None, "test", "TEST_KEY", true);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AnyLLMError::MissingApiKey { provider, env_key } => {
                assert_eq!(provider, "test");
                assert_eq!(env_key, "TEST_KEY");
            }
            _ => panic!("Expected MissingApiKey error"),
        }
    }

    #[test]
    fn test_validate_api_key_not_required() {
        let result = validate_api_key(&None, "test", "TEST_KEY", false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_base_url_from_config() {
        let url = resolve_base_url(Some("https://custom.api.com"), "https://default.api.com");
        assert_eq!(url, "https://custom.api.com");
    }

    #[test]
    fn test_resolve_base_url_default() {
        let url = resolve_base_url(None, "https://default.api.com");
        assert_eq!(url, "https://default.api.com");
    }

    #[test]
    fn test_resolve_base_url_empty_config() {
        let url = resolve_base_url(Some(""), "https://default.api.com");
        assert_eq!(url, "https://default.api.com");
    }

    #[test]
    fn test_resolve_timeout_from_config() {
        let timeout = resolve_timeout(Some(30000), 60000);
        assert_eq!(timeout, 30000);
    }

    #[test]
    fn test_resolve_timeout_default() {
        let timeout = resolve_timeout(None, 60000);
        assert_eq!(timeout, 60000);
    }
}
