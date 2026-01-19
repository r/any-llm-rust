//! Provider registry for any-llm.
//!
//! Handles dynamic provider loading and instantiation.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;

use crate::errors::AnyLLMError;
use crate::providers::Provider;
use crate::types::{ParsedModel, ProviderConfig};

// =============================================================================
// Provider Constructor Type
// =============================================================================

/// Type for a provider constructor function.
pub type ProviderConstructor = fn(ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError>;

// =============================================================================
// Registry State
// =============================================================================

/// Registry of provider constructors.
static PROVIDER_REGISTRY: Lazy<RwLock<HashMap<String, ProviderConstructor>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Cache of provider instances (for reuse).
static PROVIDER_CACHE: Lazy<RwLock<HashMap<String, Arc<dyn Provider>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

// =============================================================================
// Registration Functions
// =============================================================================

/// Register a provider.
///
/// # Arguments
///
/// * `name` - Provider identifier (case-insensitive)
/// * `constructor` - Provider constructor function
///
/// # Example
///
/// ```ignore
/// use any_llm::{register_provider, ProviderConfig};
///
/// fn my_provider_constructor(config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
///     Ok(Box::new(MyProvider::new(config)?))
/// }
///
/// register_provider("my_provider", my_provider_constructor);
/// ```
pub fn register_provider(name: &str, constructor: ProviderConstructor) {
    let mut registry = PROVIDER_REGISTRY.write().expect("Failed to lock registry");
    registry.insert(name.to_lowercase(), constructor);
}

/// Get a registered provider constructor.
///
/// # Arguments
///
/// * `name` - Provider identifier (case-insensitive)
///
/// # Returns
///
/// The provider constructor, or None if not found.
pub fn get_provider_constructor(name: &str) -> Option<ProviderConstructor> {
    let registry = PROVIDER_REGISTRY.read().expect("Failed to lock registry");
    registry.get(&name.to_lowercase()).copied()
}

/// Check if a provider is registered.
///
/// # Arguments
///
/// * `name` - Provider identifier (case-insensitive)
///
/// # Returns
///
/// True if the provider is registered.
pub fn has_provider(name: &str) -> bool {
    let registry = PROVIDER_REGISTRY.read().expect("Failed to lock registry");
    registry.contains_key(&name.to_lowercase())
}

/// Get all registered provider names.
///
/// # Returns
///
/// Array of provider names.
pub fn get_registered_providers() -> Vec<String> {
    let registry = PROVIDER_REGISTRY.read().expect("Failed to lock registry");
    registry.keys().cloned().collect()
}

// =============================================================================
// Provider Creation
// =============================================================================

/// Create a provider instance.
///
/// # Arguments
///
/// * `name` - Provider identifier (case-insensitive)
/// * `config` - Provider configuration
///
/// # Returns
///
/// Provider instance.
///
/// # Errors
///
/// Returns `UnsupportedProviderError` if provider is not registered.
pub fn create_provider(name: &str, config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
    let normalized_name = name.to_lowercase();
    
    let constructor = get_provider_constructor(&normalized_name)
        .ok_or_else(|| {
            AnyLLMError::unsupported_provider(name, get_registered_providers())
        })?;
    
    constructor(config)
}

/// Create a provider instance with optional caching.
///
/// # Arguments
///
/// * `name` - Provider identifier (case-insensitive)
/// * `config` - Provider configuration
/// * `use_cache` - Whether to use cached instances
///
/// # Returns
///
/// Provider instance (wrapped in Arc if cached).
pub fn create_provider_cached(
    name: &str,
    config: ProviderConfig,
    use_cache: bool,
) -> Result<Arc<dyn Provider>, AnyLLMError> {
    let normalized_name = name.to_lowercase();
    
    if use_cache {
        // Generate cache key from name and config
        let cache_key = format!(
            "{}:{}",
            normalized_name,
            serde_json::to_string(&config).unwrap_or_default()
        );
        
        // Check cache first
        {
            let cache = PROVIDER_CACHE.read().expect("Failed to lock cache");
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(Arc::clone(cached));
            }
        }
        
        // Create new instance
        let provider = create_provider(&normalized_name, config)?;
        let arc_provider: Arc<dyn Provider> = Arc::from(provider);
        
        // Store in cache
        {
            let mut cache = PROVIDER_CACHE.write().expect("Failed to lock cache");
            cache.insert(cache_key, Arc::clone(&arc_provider));
        }
        
        Ok(arc_provider)
    } else {
        let provider = create_provider(&normalized_name, config)?;
        Ok(Arc::from(provider))
    }
}

/// Clear the provider cache.
pub fn clear_provider_cache() {
    let mut cache = PROVIDER_CACHE.write().expect("Failed to lock cache");
    cache.clear();
}

// =============================================================================
// Model String Parsing
// =============================================================================

/// Parse a model string into provider and model parts.
///
/// Supports formats:
/// - "provider:model" (new format, recommended)
/// - "provider/model" (legacy format, only if provider is registered)
/// - "model" (requires separate provider parameter)
///
/// # Arguments
///
/// * `model` - The model string
///
/// # Returns
///
/// Parsed model parts, or None if no provider prefix.
///
/// # Example
///
/// ```
/// use any_llm::parse_model_string;
///
/// let parsed = parse_model_string("openai:gpt-4o");
/// assert!(parsed.is_some());
/// let parsed = parsed.unwrap();
/// assert_eq!(parsed.provider, "openai");
/// assert_eq!(parsed.model, "gpt-4o");
/// ```
pub fn parse_model_string(model: &str) -> Option<ParsedModel> {
    // Check for colon format first (preferred)
    if let Some(colon_index) = model.find(':') {
        let provider = &model[..colon_index];
        let model_id = &model[colon_index + 1..];
        
        if !provider.is_empty() && !model_id.is_empty() {
            return Some(ParsedModel {
                provider: provider.to_lowercase(),
                model: model_id.to_string(),
            });
        }
    }
    
    // Check for slash format (legacy)
    if let Some(slash_index) = model.find('/') {
        // Only treat as provider/model if the first part looks like a provider name
        // (not a model family like "meta-llama/...")
        let potential_provider = &model[..slash_index].to_lowercase();
        
        if has_provider(potential_provider) {
            let model_id = &model[slash_index + 1..];
            if !model_id.is_empty() {
                return Some(ParsedModel {
                    provider: potential_provider.clone(),
                    model: model_id.to_string(),
                });
            }
        }
    }
    
    // No provider prefix found
    None
}

/// Resolve provider and model from request parameters.
///
/// # Arguments
///
/// * `model` - Model string (may include provider prefix)
/// * `provider` - Explicit provider (optional)
///
/// # Returns
///
/// Resolved provider and model.
///
/// # Errors
///
/// Returns error if provider cannot be determined.
pub fn resolve_provider_and_model(
    model: &str,
    provider: Option<&str>,
) -> Result<ParsedModel, AnyLLMError> {
    // If explicit provider is given, use it
    if let Some(provider) = provider {
        return Ok(ParsedModel {
            provider: provider.to_lowercase(),
            model: model.to_string(),
        });
    }
    
    // Try to parse from model string
    if let Some(parsed) = parse_model_string(model) {
        return Ok(parsed);
    }
    
    // Cannot determine provider
    Err(AnyLLMError::invalid_request(format!(
        "Cannot determine provider for model '{}'. Use 'provider:model' format or pass explicit provider parameter.",
        model
    )))
}

/// Get a provider instance, resolving from model string if needed.
///
/// # Arguments
///
/// * `model` - Model string (may include provider prefix)
/// * `provider` - Explicit provider (optional)
/// * `config` - Provider configuration
///
/// # Returns
///
/// Object with provider instance and resolved model name.
pub fn get_provider_for_model(
    model: &str,
    provider: Option<&str>,
    config: ProviderConfig,
) -> Result<(Box<dyn Provider>, String), AnyLLMError> {
    let resolved = resolve_provider_and_model(model, provider)?;
    let provider_instance = create_provider(&resolved.provider, config)?;
    
    Ok((provider_instance, resolved.model))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Provider;
    use crate::types::{
        ChatCompletion, ChatCompletionChunk, CompletionRequest, ModelInfo, ProviderMetadata,
    };
    use async_trait::async_trait;
    use std::pin::Pin;
    use futures::Stream;

    // Mock provider for testing
    struct MockProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn provider_name(&self) -> &str {
            &self.name
        }

        fn env_api_key_name(&self) -> &str {
            "MOCK_API_KEY"
        }

        fn provider_documentation_url(&self) -> &str {
            "https://mock.test"
        }

        fn api_base(&self) -> &str {
            "https://api.mock.test"
        }

        fn supports_streaming(&self) -> bool {
            true
        }

        fn supports_tools(&self) -> bool {
            true
        }

        fn supports_vision(&self) -> bool {
            false
        }

        fn supports_list_models(&self) -> bool {
            true
        }

        fn supports_reasoning(&self) -> bool {
            false
        }

        fn metadata(&self) -> ProviderMetadata {
            ProviderMetadata {
                name: self.name.clone(),
                env_key: "MOCK_API_KEY".to_string(),
                doc_url: "https://mock.test".to_string(),
                streaming: true,
                reasoning: false,
                completion: true,
                embedding: false,
                image: false,
                list_models: true,
            }
        }

        async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
            Ok(ChatCompletion {
                id: "mock-123".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: request.model,
                provider: Some("mock".to_string()),
                choices: vec![],
                usage: None,
            })
        }

        async fn completion_stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError> {
            Ok(Box::pin(futures::stream::empty()))
        }

        async fn is_available(&self) -> bool {
            true
        }

        async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> {
            Ok(vec![])
        }
    }

    fn mock_constructor(config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
        Ok(Box::new(MockProvider {
            name: "mock".to_string(),
        }))
    }

    fn setup_mock_provider() {
        register_provider("mock", mock_constructor);
    }

    #[test]
    fn test_register_and_has_provider() {
        setup_mock_provider();
        
        assert!(has_provider("mock"));
        assert!(has_provider("MOCK")); // Case-insensitive
        assert!(!has_provider("nonexistent"));
    }

    #[test]
    fn test_get_registered_providers() {
        setup_mock_provider();
        
        let providers = get_registered_providers();
        assert!(providers.contains(&"mock".to_string()));
    }

    #[test]
    fn test_create_provider() {
        setup_mock_provider();
        
        let provider = create_provider("mock", ProviderConfig::default());
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().provider_name(), "mock");
    }

    #[test]
    fn test_create_provider_unsupported() {
        let result = create_provider("nonexistent", ProviderConfig::default());
        assert!(result.is_err());
        
        let err = result.err().unwrap();
        match err {
            AnyLLMError::UnsupportedProvider { provider, .. } => {
                assert_eq!(provider, "nonexistent");
            }
            _ => panic!("Expected UnsupportedProvider error"),
        }
    }

    #[test]
    fn test_parse_model_string_colon_format() {
        let parsed = parse_model_string("openai:gpt-4o");
        assert!(parsed.is_some());
        
        let parsed = parsed.unwrap();
        assert_eq!(parsed.provider, "openai");
        assert_eq!(parsed.model, "gpt-4o");
    }

    #[test]
    fn test_parse_model_string_with_colons_in_model() {
        // Some model names have colons (e.g., ollama models like llama3.2:latest)
        let parsed = parse_model_string("ollama:model:with:colons");
        assert!(parsed.is_some());
        
        let parsed = parsed.unwrap();
        assert_eq!(parsed.provider, "ollama");
        assert_eq!(parsed.model, "model:with:colons");
    }

    #[test]
    fn test_parse_model_string_no_prefix() {
        let parsed = parse_model_string("gpt-4o");
        assert!(parsed.is_none());
    }

    #[test]
    fn test_parse_model_string_slash_format_with_registered_provider() {
        setup_mock_provider();
        
        let parsed = parse_model_string("mock/some-model");
        assert!(parsed.is_some());
        
        let parsed = parsed.unwrap();
        assert_eq!(parsed.provider, "mock");
        assert_eq!(parsed.model, "some-model");
    }

    #[test]
    fn test_parse_model_string_slash_format_unregistered() {
        // Should not parse as provider/model if provider isn't registered
        let parsed = parse_model_string("meta-llama/some-model");
        assert!(parsed.is_none());
    }

    #[test]
    fn test_resolve_provider_and_model_with_explicit_provider() {
        let resolved = resolve_provider_and_model("gpt-4o", Some("openai"));
        assert!(resolved.is_ok());
        
        let resolved = resolved.unwrap();
        assert_eq!(resolved.provider, "openai");
        assert_eq!(resolved.model, "gpt-4o");
    }

    #[test]
    fn test_resolve_provider_and_model_from_model_string() {
        let resolved = resolve_provider_and_model("openai:gpt-4o", None);
        assert!(resolved.is_ok());
        
        let resolved = resolved.unwrap();
        assert_eq!(resolved.provider, "openai");
        assert_eq!(resolved.model, "gpt-4o");
    }

    #[test]
    fn test_resolve_provider_and_model_no_provider() {
        let resolved = resolve_provider_and_model("gpt-4o", None);
        assert!(resolved.is_err());
    }

    #[test]
    fn test_clear_provider_cache() {
        // Just verify it doesn't panic
        clear_provider_cache();
    }
}
