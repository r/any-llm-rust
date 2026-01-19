//! Error types for any-llm.
//!
//! Provides structured errors following the patterns from mozilla-ai/any-llm.
//! Enhanced to handle errors from official provider SDKs.

use thiserror::Error;

// =============================================================================
// Error Codes
// =============================================================================

/// Error codes for any-llm errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyLLMErrorCode {
    /// API key is missing.
    MissingApiKey,
    /// Provider is not supported.
    UnsupportedProvider,
    /// Request failed.
    RequestFailed,
    /// Rate limited.
    RateLimited,
    /// Invalid request.
    InvalidRequest,
    /// Provider unavailable.
    ProviderUnavailable,
    /// Model not found.
    ModelNotFound,
    /// Timeout.
    Timeout,
}

impl AnyLLMErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MissingApiKey => "MISSING_API_KEY",
            Self::UnsupportedProvider => "UNSUPPORTED_PROVIDER",
            Self::RequestFailed => "REQUEST_FAILED",
            Self::RateLimited => "RATE_LIMITED",
            Self::InvalidRequest => "INVALID_REQUEST",
            Self::ProviderUnavailable => "PROVIDER_UNAVAILABLE",
            Self::ModelNotFound => "MODEL_NOT_FOUND",
            Self::Timeout => "TIMEOUT",
        }
    }
}

impl std::fmt::Display for AnyLLMErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// =============================================================================
// Main Error Type
// =============================================================================

/// Custom error type for any-llm errors.
#[derive(Debug, Error)]
pub enum AnyLLMError {
    /// API key is missing for a provider.
    #[error("API key not found for provider '{provider}'. Please set the {env_key} environment variable or pass api_key in the request.")]
    MissingApiKey {
        provider: String,
        env_key: String,
    },

    /// Provider is not supported.
    #[error("Provider '{provider}' is not supported. Supported providers: {}", supported.join(", "))]
    UnsupportedProvider {
        provider: String,
        supported: Vec<String>,
    },

    /// Request to a provider failed.
    #[error("Request to {provider} failed: {message}")]
    ProviderRequest {
        provider: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
        status_code: Option<u16>,
    },

    /// Rate limited by provider.
    #[error("{}", rate_limit_message(.provider, .retry_after))]
    RateLimit {
        provider: String,
        retry_after: Option<u64>,
    },

    /// Provider is unavailable.
    #[error("Provider '{provider}' is unavailable{}", reason.as_ref().map(|r| format!(": {}", r)).unwrap_or_default())]
    ProviderUnavailable {
        provider: String,
        reason: Option<String>,
    },

    /// Request timed out.
    #[error("Request to {provider} timed out after {timeout_ms}ms")]
    Timeout {
        provider: String,
        timeout_ms: u64,
    },

    /// Model not found.
    #[error("Model '{model}' not found for provider '{provider}'")]
    ModelNotFound {
        provider: String,
        model: String,
    },

    /// Invalid request.
    #[error("Invalid request: {message}")]
    InvalidRequest {
        message: String,
    },

    /// Generic error wrapper.
    #[error("{0}")]
    Other(String),
}

fn rate_limit_message(provider: &str, retry_after: &Option<u64>) -> String {
    match retry_after {
        Some(seconds) => format!("Rate limited by {}. Retry after {} seconds.", provider, seconds),
        None => format!("Rate limited by {}.", provider),
    }
}

impl AnyLLMError {
    /// Get the error code for this error.
    pub fn code(&self) -> AnyLLMErrorCode {
        match self {
            Self::MissingApiKey { .. } => AnyLLMErrorCode::MissingApiKey,
            Self::UnsupportedProvider { .. } => AnyLLMErrorCode::UnsupportedProvider,
            Self::ProviderRequest { .. } => AnyLLMErrorCode::RequestFailed,
            Self::RateLimit { .. } => AnyLLMErrorCode::RateLimited,
            Self::ProviderUnavailable { .. } => AnyLLMErrorCode::ProviderUnavailable,
            Self::Timeout { .. } => AnyLLMErrorCode::Timeout,
            Self::ModelNotFound { .. } => AnyLLMErrorCode::ModelNotFound,
            Self::InvalidRequest { .. } => AnyLLMErrorCode::InvalidRequest,
            Self::Other(_) => AnyLLMErrorCode::RequestFailed,
        }
    }

    /// Get the provider name if available.
    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::MissingApiKey { provider, .. } => Some(provider),
            Self::UnsupportedProvider { provider, .. } => Some(provider),
            Self::ProviderRequest { provider, .. } => Some(provider),
            Self::RateLimit { provider, .. } => Some(provider),
            Self::ProviderUnavailable { provider, .. } => Some(provider),
            Self::Timeout { provider, .. } => Some(provider),
            Self::ModelNotFound { provider, .. } => Some(provider),
            Self::InvalidRequest { .. } => None,
            Self::Other(_) => None,
        }
    }

    /// Get the HTTP status code if available.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::ProviderRequest { status_code, .. } => *status_code,
            Self::RateLimit { .. } => Some(429),
            _ => None,
        }
    }

    // =========================================================================
    // Constructors (matching TypeScript error classes)
    // =========================================================================

    /// Create a MissingApiKey error.
    pub fn missing_api_key(provider: impl Into<String>, env_key: impl Into<String>) -> Self {
        Self::MissingApiKey {
            provider: provider.into(),
            env_key: env_key.into(),
        }
    }

    /// Create an UnsupportedProvider error.
    pub fn unsupported_provider(provider: impl Into<String>, supported: Vec<String>) -> Self {
        Self::UnsupportedProvider {
            provider: provider.into(),
            supported,
        }
    }

    /// Create a ProviderRequest error.
    pub fn provider_request(
        provider: impl Into<String>,
        message: impl Into<String>,
        status_code: Option<u16>,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::ProviderRequest {
            provider: provider.into(),
            message: message.into(),
            source,
            status_code,
        }
    }

    /// Create a RateLimit error.
    pub fn rate_limit(provider: impl Into<String>, retry_after: Option<u64>) -> Self {
        Self::RateLimit {
            provider: provider.into(),
            retry_after,
        }
    }

    /// Create a ProviderUnavailable error.
    pub fn provider_unavailable(provider: impl Into<String>, reason: Option<String>) -> Self {
        Self::ProviderUnavailable {
            provider: provider.into(),
            reason,
        }
    }

    /// Create a Timeout error.
    pub fn timeout(provider: impl Into<String>, timeout_ms: u64) -> Self {
        Self::Timeout {
            provider: provider.into(),
            timeout_ms,
        }
    }

    /// Create a ModelNotFound error.
    pub fn model_not_found(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self::ModelNotFound {
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Create an InvalidRequest error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }
}

// =============================================================================
// Type Guards
// =============================================================================

/// Check if an error is an any-llm error.
/// Works with AnyLLMError directly.
pub fn is_any_llm_error(_error: &AnyLLMError) -> bool {
    true
}

/// Check if an error indicates rate limiting.
pub fn is_rate_limit_error(error: &AnyLLMError) -> bool {
    matches!(error, AnyLLMError::RateLimit { .. })
}

/// Check if an error is a timeout error.
pub fn is_timeout_error(error: &AnyLLMError) -> bool {
    matches!(error, AnyLLMError::Timeout { .. })
}

/// Check if an error is due to missing API key.
pub fn is_missing_api_key_error(error: &AnyLLMError) -> bool {
    matches!(error, AnyLLMError::MissingApiKey { .. })
}

/// Check if an error is a connection error (provider not running).
pub fn is_connection_error(error: &dyn std::error::Error) -> bool {
    let message = error.to_string().to_lowercase();
    message.contains("econnrefused")
        || message.contains("fetch failed")
        || message.contains("network")
        || message.contains("connection")
        || message.contains("connection refused")
}

// =============================================================================
// Error Wrapping
// =============================================================================

/// Information extracted from an SDK error.
pub struct SdkErrorInfo {
    pub status: Option<u16>,
    pub message: String,
}

/// Check if an error looks like an OpenAI SDK error (has status field).
pub fn extract_sdk_error_info(error: &dyn std::error::Error) -> Option<SdkErrorInfo> {
    // Try to extract status code from error message or debug representation
    let message = error.to_string();
    let debug = format!("{:?}", error);
    
    // Common patterns for SDK errors
    let status = if debug.contains("status: 429") || message.contains("429") {
        Some(429)
    } else if debug.contains("status: 401") || message.contains("401") || message.to_lowercase().contains("unauthorized") {
        Some(401)
    } else if debug.contains("status: 403") || message.contains("403") || message.to_lowercase().contains("forbidden") {
        Some(403)
    } else if debug.contains("status: 404") || message.contains("404") {
        Some(404)
    } else if debug.contains("status: 500") || message.contains("500") {
        Some(500)
    } else {
        None
    };
    
    if status.is_some() {
        Some(SdkErrorInfo { status, message })
    } else {
        None
    }
}

/// Wrap an unknown error into an AnyLLMError.
/// Enhanced to handle errors from official provider SDKs.
pub fn wrap_error(error: Box<dyn std::error::Error + Send + Sync>, provider: &str) -> AnyLLMError {
    // If it's already an AnyLLMError, return it
    if let Some(any_llm_error) = error.downcast_ref::<AnyLLMError>() {
        return match any_llm_error {
            AnyLLMError::MissingApiKey { provider, env_key } => {
                AnyLLMError::missing_api_key(provider.clone(), env_key.clone())
            }
            AnyLLMError::UnsupportedProvider { provider, supported } => {
                AnyLLMError::unsupported_provider(provider.clone(), supported.clone())
            }
            AnyLLMError::RateLimit { provider, retry_after } => {
                AnyLLMError::rate_limit(provider.clone(), *retry_after)
            }
            AnyLLMError::ProviderUnavailable { provider, reason } => {
                AnyLLMError::provider_unavailable(provider.clone(), reason.clone())
            }
            AnyLLMError::Timeout { provider, timeout_ms } => {
                AnyLLMError::timeout(provider.clone(), *timeout_ms)
            }
            _ => AnyLLMError::provider_request(provider, error.to_string(), None, None),
        };
    }

    let message = error.to_string();
    let message_lower = message.to_lowercase();
    
    // Check for SDK errors with status codes
    if let Some(info) = extract_sdk_error_info(error.as_ref()) {
        match info.status {
            Some(429) => return AnyLLMError::rate_limit(provider, None),
            Some(401) | Some(403) => {
                return AnyLLMError::missing_api_key(
                    provider,
                    format!("{}_API_KEY", provider.to_uppercase()),
                )
            }
            Some(status) => {
                return AnyLLMError::provider_request(provider, info.message, Some(status), None)
            }
            None => {}
        }
    }
    
    // Handle connection errors (local providers not running)
    if is_connection_error(error.as_ref()) {
        return AnyLLMError::provider_unavailable(
            provider,
            Some(format!("Cannot connect to {}. Is it running?", provider)),
        );
    }
    
    // Check for common error patterns in message
    if message_lower.contains("rate limit") || message_lower.contains("too many requests") {
        return AnyLLMError::rate_limit(provider, None);
    }
    
    if message_lower.contains("unauthorized")
        || message_lower.contains("invalid api key")
        || message_lower.contains("authentication")
    {
        return AnyLLMError::missing_api_key(
            provider,
            format!("{}_API_KEY", provider.to_uppercase()),
        );
    }
    
    if message_lower.contains("timeout") || message_lower.contains("timed out") {
        return AnyLLMError::timeout(provider, 0);
    }
    
    // Default to provider request error
    AnyLLMError::provider_request(provider, message, None, Some(error))
}

/// Wrap a standard error into an AnyLLMError (convenience function).
pub fn wrap_std_error<E: std::error::Error + Send + Sync + 'static>(
    error: E,
    provider: &str,
) -> AnyLLMError {
    wrap_error(Box::new(error), provider)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Error Classes Tests (matching errors.test.ts)
    // =========================================================================

    mod missing_api_key_error {
        use super::*;

        #[test]
        fn should_create_error_with_correct_message() {
            let error = AnyLLMError::missing_api_key("openai", "OPENAI_API_KEY");
            
            assert_eq!(error.code(), AnyLLMErrorCode::MissingApiKey);
            assert_eq!(error.provider(), Some("openai"));
            
            let message = error.to_string();
            assert!(message.contains("openai"));
            assert!(message.contains("OPENAI_API_KEY"));
        }
    }

    mod unsupported_provider_error {
        use super::*;

        #[test]
        fn should_create_error_with_supported_providers_list() {
            let error = AnyLLMError::unsupported_provider(
                "invalid",
                vec!["openai".to_string(), "anthropic".to_string()],
            );
            
            assert_eq!(error.code(), AnyLLMErrorCode::UnsupportedProvider);
            
            let message = error.to_string();
            assert!(message.contains("invalid"));
            assert!(message.contains("openai"));
            assert!(message.contains("anthropic"));
        }
    }

    mod provider_request_error {
        use super::*;

        #[test]
        fn should_create_error_with_status_code() {
            let error = AnyLLMError::provider_request("openai", "Bad request", Some(400), None);
            
            assert_eq!(error.code(), AnyLLMErrorCode::RequestFailed);
            assert_eq!(error.provider(), Some("openai"));
            assert_eq!(error.status_code(), Some(400));
        }

        #[test]
        fn should_create_error_with_cause() {
            let cause = std::io::Error::new(std::io::ErrorKind::Other, "Original error");
            let error = AnyLLMError::provider_request(
                "anthropic",
                "Request failed",
                Some(500),
                Some(Box::new(cause)),
            );
            
            assert_eq!(error.provider(), Some("anthropic"));
        }
    }

    mod rate_limit_error {
        use super::*;

        #[test]
        fn should_create_error_without_retry_after() {
            let error = AnyLLMError::rate_limit("openai", None);
            
            assert_eq!(error.code(), AnyLLMErrorCode::RateLimited);
            assert_eq!(error.status_code(), Some(429));
        }

        #[test]
        fn should_create_error_with_retry_after() {
            let error = AnyLLMError::rate_limit("anthropic", Some(60));
            
            let message = error.to_string();
            assert!(message.contains("60 seconds"));
        }
    }

    mod provider_unavailable_error {
        use super::*;

        #[test]
        fn should_create_error_with_reason() {
            let error = AnyLLMError::provider_unavailable("ollama", Some("Not running".to_string()));
            
            assert_eq!(error.code(), AnyLLMErrorCode::ProviderUnavailable);
            
            let message = error.to_string();
            assert!(message.contains("Not running"));
        }

        #[test]
        fn should_create_error_without_reason() {
            let error = AnyLLMError::provider_unavailable("llamafile", None);
            
            let message = error.to_string();
            assert!(message.contains("llamafile"));
            assert!(message.contains("unavailable"));
        }
    }

    mod timeout_error {
        use super::*;

        #[test]
        fn should_create_error_with_timeout_duration() {
            let error = AnyLLMError::timeout("openai", 30000);
            
            assert_eq!(error.code(), AnyLLMErrorCode::Timeout);
            
            let message = error.to_string();
            assert!(message.contains("30000ms"));
        }
    }

    // =========================================================================
    // Type Guards Tests
    // =========================================================================

    mod type_guards {
        use super::*;

        #[test]
        fn is_rate_limit_error_returns_true_for_rate_limit() {
            let error = AnyLLMError::rate_limit("test", None);
            assert!(is_rate_limit_error(&error));
        }

        #[test]
        fn is_rate_limit_error_returns_false_for_other_errors() {
            let error = AnyLLMError::timeout("test", 1000);
            assert!(!is_rate_limit_error(&error));
        }

        #[test]
        fn is_timeout_error_returns_true_for_timeout() {
            let error = AnyLLMError::timeout("test", 1000);
            assert!(is_timeout_error(&error));
        }

        #[test]
        fn is_missing_api_key_error_returns_true_for_missing_key() {
            let error = AnyLLMError::missing_api_key("test", "TEST_KEY");
            assert!(is_missing_api_key_error(&error));
        }

        #[test]
        fn is_connection_error_detects_connection_refused() {
            let error = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "ECONNREFUSED");
            assert!(is_connection_error(&error));
        }

        #[test]
        fn is_connection_error_detects_network_errors() {
            let error = std::io::Error::new(std::io::ErrorKind::Other, "network error");
            assert!(is_connection_error(&error));
        }

        #[test]
        fn is_connection_error_returns_false_for_other_errors() {
            let error = std::io::Error::new(std::io::ErrorKind::Other, "Invalid JSON");
            assert!(!is_connection_error(&error));
        }
    }

    // =========================================================================
    // wrap_error Tests
    // =========================================================================

    mod wrap_error_tests {
        use super::*;

        #[test]
        fn should_convert_connection_errors_to_provider_unavailable() {
            let connection_error = std::io::Error::new(
                std::io::ErrorKind::ConnectionRefused,
                "ECONNREFUSED",
            );
            let wrapped = wrap_std_error(connection_error, "ollama");
            
            assert!(matches!(wrapped, AnyLLMError::ProviderUnavailable { .. }));
        }

        #[test]
        fn should_detect_rate_limit_from_error_message() {
            let error = std::io::Error::new(
                std::io::ErrorKind::Other,
                "Rate limit exceeded, please try again later",
            );
            let wrapped = wrap_std_error(error, "test");
            
            assert!(matches!(wrapped, AnyLLMError::RateLimit { .. }));
        }

        #[test]
        fn should_detect_auth_errors_from_error_message() {
            let error = std::io::Error::new(std::io::ErrorKind::Other, "Unauthorized access");
            let wrapped = wrap_std_error(error, "test");
            
            assert!(matches!(wrapped, AnyLLMError::MissingApiKey { .. }));
        }

        #[test]
        fn should_detect_timeout_from_message() {
            let error = std::io::Error::new(std::io::ErrorKind::TimedOut, "Request timed out");
            let wrapped = wrap_std_error(error, "test");
            
            assert!(matches!(wrapped, AnyLLMError::Timeout { .. }));
        }

        #[test]
        fn should_wrap_unknown_errors_as_provider_request() {
            let error = std::io::Error::new(std::io::ErrorKind::Other, "Something went wrong");
            let wrapped = wrap_std_error(error, "test");
            
            assert!(matches!(wrapped, AnyLLMError::ProviderRequest { .. }));
            assert!(wrapped.to_string().contains("Something went wrong"));
        }
    }

    // =========================================================================
    // Error Inheritance Tests
    // =========================================================================

    mod error_inheritance {
        use super::*;

        #[test]
        fn all_errors_implement_error_trait() {
            let errors: Vec<Box<dyn std::error::Error>> = vec![
                Box::new(AnyLLMError::missing_api_key("test", "KEY")),
                Box::new(AnyLLMError::unsupported_provider("test", vec![])),
                Box::new(AnyLLMError::provider_request("test", "msg", None, None)),
                Box::new(AnyLLMError::rate_limit("test", None)),
                Box::new(AnyLLMError::provider_unavailable("test", None)),
                Box::new(AnyLLMError::timeout("test", 1000)),
            ];

            for error in errors {
                // All errors should have a non-empty message
                assert!(!error.to_string().is_empty());
            }
        }

        #[test]
        fn errors_have_correct_error_codes() {
            assert_eq!(
                AnyLLMError::missing_api_key("test", "KEY").code(),
                AnyLLMErrorCode::MissingApiKey
            );
            assert_eq!(
                AnyLLMError::unsupported_provider("test", vec![]).code(),
                AnyLLMErrorCode::UnsupportedProvider
            );
            assert_eq!(
                AnyLLMError::provider_request("test", "msg", None, None).code(),
                AnyLLMErrorCode::RequestFailed
            );
            assert_eq!(
                AnyLLMError::rate_limit("test", None).code(),
                AnyLLMErrorCode::RateLimited
            );
            assert_eq!(
                AnyLLMError::provider_unavailable("test", None).code(),
                AnyLLMErrorCode::ProviderUnavailable
            );
            assert_eq!(
                AnyLLMError::timeout("test", 1000).code(),
                AnyLLMErrorCode::Timeout
            );
        }
    }
}
