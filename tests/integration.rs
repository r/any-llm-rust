//! Integration tests for any-llm.
//!
//! These tests require actual LLM providers to be available.
//! Tests are skipped if providers are not available.

use any_llm::{
    check_provider, completion, completion_stream, get_supported_providers, list_models,
    AnyLLM, CompletionRequest, Message, ProviderConfig,
};
use futures::StreamExt;

// =============================================================================
// Test Utilities
// =============================================================================

/// Check if a provider is available for testing.
async fn provider_available(name: &str) -> bool {
    let status = check_provider(name, None).await;
    status.available
}

/// Skip test if provider not available.
macro_rules! skip_if_unavailable {
    ($provider:expr) => {
        if !provider_available($provider).await {
            eprintln!(
                "Skipping test: {} provider not available",
                $provider
            );
            return;
        }
    };
}

// =============================================================================
// Provider Discovery Tests
// =============================================================================

#[test]
fn test_get_supported_providers() {
    let providers = get_supported_providers();
    
    assert!(providers.contains(&"openai".to_string()));
    assert!(providers.contains(&"anthropic".to_string()));
    assert!(providers.contains(&"ollama".to_string()));
    assert!(providers.contains(&"llamafile".to_string()));
}

// =============================================================================
// Basic Completion Tests (require actual providers)
// =============================================================================

#[tokio::test]
async fn test_ollama_completion_if_available() {
    skip_if_unavailable!("ollama");

    let models = list_models("ollama", None).await;
    if models.is_err() || models.as_ref().unwrap().is_empty() {
        eprintln!("Skipping: no Ollama models installed");
        return;
    }

    let model = &models.unwrap()[0].id;
    
    let response = completion(CompletionRequest {
        model: format!("ollama:{}", model),
        messages: vec![Message::user("Say 'hello' and nothing else.")],
        max_tokens: Some(10),
        ..Default::default()
    })
    .await;

    assert!(response.is_ok(), "Completion failed: {:?}", response.err());
    let response = response.unwrap();
    assert!(!response.choices.is_empty());
}

#[tokio::test]
async fn test_ollama_streaming_if_available() {
    skip_if_unavailable!("ollama");

    let models = list_models("ollama", None).await;
    if models.is_err() || models.as_ref().unwrap().is_empty() {
        eprintln!("Skipping: no Ollama models installed");
        return;
    }

    let model = &models.unwrap()[0].id;
    
    let stream = completion_stream(CompletionRequest {
        model: format!("ollama:{}", model),
        messages: vec![Message::user("Say 'hi'.")],
        max_tokens: Some(10),
        ..Default::default()
    })
    .await;

    assert!(stream.is_ok(), "Stream creation failed: {:?}", stream.err());
    
    let mut stream = stream.unwrap();
    let mut received_content = false;
    
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok(), "Stream chunk failed: {:?}", chunk.err());
        let chunk = chunk.unwrap();
        if chunk.choices.first().and_then(|c| c.delta.content.as_ref()).is_some() {
            received_content = true;
        }
    }
    
    assert!(received_content, "No content received in stream");
}

// =============================================================================
// AnyLLM Class Tests
// =============================================================================

#[tokio::test]
async fn test_any_llm_class_ollama_if_available() {
    skip_if_unavailable!("ollama");

    let llm = AnyLLM::create("ollama", ProviderConfig::default());
    assert!(llm.is_ok(), "Failed to create AnyLLM: {:?}", llm.err());
    
    let llm = llm.unwrap();
    assert_eq!(llm.name(), "ollama");
    
    let available = llm.is_available().await;
    assert!(available);
    
    let models = llm.list_models().await;
    assert!(models.is_ok(), "Failed to list models: {:?}", models.err());
}

// =============================================================================
// Model String Parsing Tests
// =============================================================================

#[test]
fn test_parse_model_string() {
    let parsed = AnyLLM::parse_model_string("openai:gpt-4o");
    assert!(parsed.is_some());
    let parsed = parsed.unwrap();
    assert_eq!(parsed.provider, "openai");
    assert_eq!(parsed.model, "gpt-4o");

    let parsed = AnyLLM::parse_model_string("anthropic:claude-3-5-sonnet-20241022");
    assert!(parsed.is_some());
    let parsed = parsed.unwrap();
    assert_eq!(parsed.provider, "anthropic");
    assert_eq!(parsed.model, "claude-3-5-sonnet-20241022");

    // Model with colons (like ollama tags)
    let parsed = AnyLLM::parse_model_string("ollama:llama3.2:latest");
    assert!(parsed.is_some());
    let parsed = parsed.unwrap();
    assert_eq!(parsed.provider, "ollama");
    assert_eq!(parsed.model, "llama3.2:latest");

    // No provider prefix
    let parsed = AnyLLM::parse_model_string("gpt-4o");
    assert!(parsed.is_none());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_invalid_provider_error() {
    let result = completion(CompletionRequest {
        model: "nonexistent:model".to_string(),
        messages: vec![Message::user("Hello")],
        ..Default::default()
    })
    .await;

    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(err.to_string().contains("not supported"));
}

#[tokio::test]
async fn test_missing_provider_error() {
    let result = completion(CompletionRequest {
        model: "just-a-model".to_string(), // No provider prefix
        messages: vec![Message::user("Hello")],
        ..Default::default()
    })
    .await;

    assert!(result.is_err());
}

// =============================================================================
// Provider Availability Tests
// =============================================================================

#[tokio::test]
async fn test_check_provider_unavailable() {
    // Test with invalid base URL to ensure provider is unavailable
    let status = check_provider(
        "ollama",
        Some(ProviderConfig {
            base_url: Some("http://localhost:99999".to_string()),
            ..Default::default()
        }),
    )
    .await;

    assert!(!status.available);
}
