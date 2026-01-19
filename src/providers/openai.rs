//! OpenAI LLM Provider for any-llm.
//!
//! Uses the official async-openai SDK for robust API interactions.
//!
//! @see https://platform.openai.com/docs/api-reference
//! @see https://github.com/64bit/async-openai

use async_openai::{
    config::OpenAIConfig,
    Client,
    types::chat::{
        ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestUserMessage,
        ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestToolMessage,
        ChatCompletionRequestUserMessageContent,
        ChatCompletionRequestUserMessageContentPart,
        ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestMessageContentPartImage,
        CreateChatCompletionRequest,
        ChatCompletionTool,
        ChatCompletionTools,
        ChatCompletionMessageToolCall,
        ChatCompletionMessageToolCalls,
        ChatCompletionNamedToolChoice,
        ChatCompletionToolChoiceOption,
        ToolChoiceOptions,
        FunctionObject,
        FunctionName,
        ImageUrl as OAIImageUrl,
        ImageDetail as OAIImageDetail,
    },
};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::errors::AnyLLMError;
use crate::providers::base::{resolve_api_key, resolve_base_url, resolve_timeout, Provider};
use crate::types::{
    ChatCompletion, ChatCompletionChunk, ChunkChoice, ChunkDelta, CompletionChoice,
    CompletionRequest, CompletionUsage, FinishReason, Message, MessageContent, MessageRole,
    ModelInfo, ProviderConfig, ProviderMetadata, Tool, ToolCall, ToolCallFunction,
    PartialToolCall, PartialToolCallFunction,
};

// =============================================================================
// OpenAI Provider Implementation
// =============================================================================

/// OpenAI provider using the official async-openai SDK.
pub struct OpenAIProvider {
    client: Client<OpenAIConfig>,
    api_key: Option<String>,
    base_url: String,
}

impl OpenAIProvider {
    pub const PROVIDER_NAME: &'static str = "openai";
    pub const ENV_API_KEY_NAME: &'static str = "OPENAI_API_KEY";
    pub const PROVIDER_DOCUMENTATION_URL: &'static str = "https://platform.openai.com/docs";
    pub const API_BASE: &'static str = "https://api.openai.com/v1";

    pub fn new(config: ProviderConfig) -> Result<Self, AnyLLMError> {
        let api_key = resolve_api_key(
            config.api_key.as_ref().and_then(|k| k.as_deref()),
            Self::ENV_API_KEY_NAME,
        );
        let base_url = resolve_base_url(config.base_url.as_deref(), Self::API_BASE);
        let _timeout_ms = resolve_timeout(config.timeout, 60000);

        // Build OpenAI config
        let mut openai_config = OpenAIConfig::new();
        
        if let Some(ref key) = api_key {
            openai_config = openai_config.with_api_key(key);
        }
        
        if base_url != Self::API_BASE {
            openai_config = openai_config.with_api_base(&base_url);
        }

        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            api_key,
            base_url,
        })
    }

    pub fn constructor(config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
        Ok(Box::new(Self::new(config)?))
    }

    fn model_supports_tools(model_id: &str) -> bool {
        !model_id.starts_with("o1-")
    }

    fn model_supports_vision(model_id: &str) -> bool {
        let vision_models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision", "o1"];
        vision_models.iter().any(|m| model_id.contains(m))
    }

    /// Convert our messages to OpenAI SDK format.
    fn convert_messages(messages: &[Message]) -> Result<Vec<ChatCompletionRequestMessage>, AnyLLMError> {
        let mut result = Vec::new();

        for msg in messages {
            let openai_msg = match msg.role {
                MessageRole::System => {
                    let content = match &msg.content {
                        MessageContent::Text(s) => s.clone(),
                        _ => String::new(),
                    };
                    ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage {
                        content: content.into(),
                        name: msg.name.clone(),
                    })
                }
                MessageRole::User => {
                    match &msg.content {
                        MessageContent::Text(s) => {
                            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(s.clone()),
                                name: msg.name.clone(),
                            })
                        }
                        MessageContent::Parts(parts) => {
                            let content_parts: Vec<ChatCompletionRequestUserMessageContentPart> = parts
                                .iter()
                                .filter_map(|part| {
                                    match part {
                                        crate::types::ContentPart::Text(text_part) => {
                                            Some(ChatCompletionRequestUserMessageContentPart::Text(
                                                ChatCompletionRequestMessageContentPartText {
                                                    text: text_part.text.clone(),
                                                }
                                            ))
                                        }
                                        crate::types::ContentPart::Image(img_part) => {
                                            let detail = img_part.image_url.detail.as_ref()
                                                .and_then(|d| match d.as_str() {
                                                    "low" => Some(OAIImageDetail::Low),
                                                    "high" => Some(OAIImageDetail::High),
                                                    _ => Some(OAIImageDetail::Auto),
                                                });
                                            Some(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                                                ChatCompletionRequestMessageContentPartImage {
                                                    image_url: OAIImageUrl {
                                                        url: img_part.image_url.url.clone(),
                                                        detail,
                                                    }
                                                }
                                            ))
                                        }
                                    }
                                })
                                .collect();

                            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Array(content_parts),
                                name: msg.name.clone(),
                            })
                        }
                        MessageContent::Null => {
                            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                                content: ChatCompletionRequestUserMessageContent::Text(String::new()),
                                name: msg.name.clone(),
                            })
                        }
                    }
                }
                MessageRole::Assistant => {
                    let content = match &msg.content {
                        MessageContent::Text(s) => Some(s.clone()),
                        MessageContent::Null => None,
                        MessageContent::Parts(_) => None,
                    };

                    let tool_calls = msg.tool_calls.as_ref().map(|tcs| {
                        tcs.iter()
                            .map(|tc| ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                                id: tc.id.clone(),
                                function: async_openai::types::chat::FunctionCall {
                                    name: tc.function.name.clone(),
                                    arguments: tc.function.arguments.clone(),
                                },
                            }))
                            .collect()
                    });

                    ChatCompletionRequestMessage::Assistant(ChatCompletionRequestAssistantMessage {
                        content: content.map(|c| c.into()),
                        name: msg.name.clone(),
                        tool_calls,
                        ..Default::default()
                    })
                }
                MessageRole::Tool => {
                    let content = match &msg.content {
                        MessageContent::Text(s) => s.clone(),
                        _ => serde_json::to_string(&msg.content).unwrap_or_default(),
                    };
                    ChatCompletionRequestMessage::Tool(ChatCompletionRequestToolMessage {
                        content: content.into(),
                        tool_call_id: msg.tool_call_id.clone().unwrap_or_default(),
                    })
                }
            };
            result.push(openai_msg);
        }

        Ok(result)
    }

    /// Convert our tools to OpenAI SDK format.
    fn convert_tools(tools: &[Tool]) -> Vec<ChatCompletionTools> {
        tools
            .iter()
            .map(|tool| {
                ChatCompletionTools::Function(ChatCompletionTool {
                    function: FunctionObject {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone(),
                        parameters: Some(tool.function.parameters.clone()),
                        strict: None,
                    },
                })
            })
            .collect()
    }

    /// Convert tool choice to OpenAI SDK format.
    fn convert_tool_choice(choice: &crate::types::ToolChoice) -> Option<ChatCompletionToolChoiceOption> {
        match choice {
            crate::types::ToolChoice::None => Some(ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::None)),
            crate::types::ToolChoice::Auto => Some(ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto)),
            crate::types::ToolChoice::Required => Some(ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Required)),
            crate::types::ToolChoice::Function { function, .. } => {
                Some(ChatCompletionToolChoiceOption::Function(ChatCompletionNamedToolChoice {
                    function: FunctionName {
                        name: function.name.clone(),
                    },
                }))
            }
        }
    }

    /// Handle SDK errors and convert to our error types.
    fn handle_error(error: async_openai::error::OpenAIError) -> AnyLLMError {
        let message = error.to_string();
        let message_lower = message.to_lowercase();

        if message_lower.contains("rate limit") || message.contains("429") {
            return AnyLLMError::rate_limit(Self::PROVIDER_NAME, None);
        }

        if message_lower.contains("unauthorized")
            || message_lower.contains("invalid api key")
            || message.contains("401")
        {
            return AnyLLMError::missing_api_key(Self::PROVIDER_NAME, Self::ENV_API_KEY_NAME);
        }

        if message_lower.contains("timeout") || message_lower.contains("timed out") {
            return AnyLLMError::timeout(Self::PROVIDER_NAME, 0);
        }

        AnyLLMError::provider_request(Self::PROVIDER_NAME, message, None, None)
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn provider_name(&self) -> &str { Self::PROVIDER_NAME }
    fn env_api_key_name(&self) -> &str { Self::ENV_API_KEY_NAME }
    fn provider_documentation_url(&self) -> &str { Self::PROVIDER_DOCUMENTATION_URL }
    fn api_base(&self) -> &str { &self.base_url }

    fn supports_streaming(&self) -> bool { true }
    fn supports_tools(&self) -> bool { true }
    fn supports_vision(&self) -> bool { true }
    fn supports_list_models(&self) -> bool { true }
    fn supports_reasoning(&self) -> bool { false }

    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            name: Self::PROVIDER_NAME.to_string(),
            env_key: Self::ENV_API_KEY_NAME.to_string(),
            doc_url: Self::PROVIDER_DOCUMENTATION_URL.to_string(),
            streaming: true,
            reasoning: false,
            completion: true,
            embedding: false,
            image: true,
            list_models: true,
        }
    }

    async fn is_available(&self) -> bool {
        if self.api_key.is_none() && std::env::var(Self::ENV_API_KEY_NAME).is_err() {
            return false;
        }

        // Try to list models as a health check
        match self.client.models().list().await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> {
        let response = self
            .client
            .models()
            .list()
            .await
            .map_err(Self::handle_error)?;

        Ok(response
            .data
            .into_iter()
            .filter(|m| m.id.contains("gpt") || m.id.contains("o1") || m.id.contains("o3"))
            .map(|model| ModelInfo {
                id: model.id.clone(),
                object: "model".to_string(),
                created: Some(model.created as u64),
                owned_by: Some(model.owned_by),
                provider: Some("openai".to_string()),
                context_length: None,
                supports_tools: Some(Self::model_supports_tools(&model.id)),
                supports_vision: Some(Self::model_supports_vision(&model.id)),
            })
            .collect())
    }

    async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
        let messages = Self::convert_messages(&request.messages)?;

        let mut openai_request = CreateChatCompletionRequest {
            model: request.model.clone(),
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            n: request.n.map(|n| n as u8),
            stream: Some(false),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            user: request.user.clone(),
            seed: request.seed,
            ..Default::default()
        };

        // Add max_tokens
        if let Some(max_tokens) = request.max_tokens {
            if request.model.starts_with("o1") || request.model.starts_with("o3") {
                openai_request.max_completion_tokens = Some(max_tokens);
            } else {
                openai_request.max_tokens = Some(max_tokens);
            }
        }

        // Add tools
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                openai_request.tools = Some(Self::convert_tools(tools));
            }
        }

        // Add tool_choice
        if let Some(ref choice) = request.tool_choice {
            openai_request.tool_choice = Self::convert_tool_choice(choice);
        }

        let response = self
            .client
            .chat()
            .create(openai_request)
            .await
            .map_err(Self::handle_error)?;

        // Convert response
        Ok(ChatCompletion {
            id: response.id,
            object: "chat.completion".to_string(),
            created: response.created as u64,
            model: response.model,
            provider: Some("openai".to_string()),
            choices: response
                .choices
                .into_iter()
                .map(|choice| CompletionChoice {
                    index: choice.index,
                    message: Message {
                        role: MessageRole::Assistant,
                        content: choice
                            .message
                            .content
                            .map(MessageContent::Text)
                            .unwrap_or(MessageContent::Null),
                        tool_calls: choice.message.tool_calls.map(|tcs| {
                            tcs.into_iter()
                                .filter_map(|tc| {
                                    match tc {
                                        ChatCompletionMessageToolCalls::Function(f) => Some(ToolCall {
                                            id: f.id,
                                            call_type: "function".to_string(),
                                            function: ToolCallFunction {
                                                name: f.function.name,
                                                arguments: f.function.arguments,
                                            },
                                        }),
                                        ChatCompletionMessageToolCalls::Custom(_) => None,
                                    }
                                })
                                .collect()
                        }),
                        tool_call_id: None,
                        name: None,
                    },
                    finish_reason: choice.finish_reason.map(|fr| match fr {
                        async_openai::types::chat::FinishReason::Stop => FinishReason::Stop,
                        async_openai::types::chat::FinishReason::Length => FinishReason::Length,
                        async_openai::types::chat::FinishReason::ToolCalls => FinishReason::ToolCalls,
                        async_openai::types::chat::FinishReason::ContentFilter => FinishReason::ContentFilter,
                        _ => FinishReason::Stop,
                    }),
                    logprobs: None,
                })
                .collect(),
            usage: response.usage.map(|u| CompletionUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            }),
        })
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError>
    {
        let messages = Self::convert_messages(&request.messages)?;

        let mut openai_request = CreateChatCompletionRequest {
            model: request.model.clone(),
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(true),
            presence_penalty: request.presence_penalty,
            frequency_penalty: request.frequency_penalty,
            ..Default::default()
        };

        // Add max_tokens
        if let Some(max_tokens) = request.max_tokens {
            if request.model.starts_with("o1") || request.model.starts_with("o3") {
                openai_request.max_completion_tokens = Some(max_tokens);
            } else {
                openai_request.max_tokens = Some(max_tokens);
            }
        }

        // Add tools
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                openai_request.tools = Some(Self::convert_tools(tools));
            }
        }

        let mut stream = self
            .client
            .chat()
            .create_stream(openai_request)
            .await
            .map_err(Self::handle_error)?;

        let stream = try_stream! {
            use futures::StreamExt;
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(response) => {
                        for choice in response.choices {
                            let chunk = ChatCompletionChunk {
                                id: response.id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: response.created as u64,
                                model: response.model.clone(),
                                choices: vec![ChunkChoice {
                                    index: choice.index,
                                    delta: ChunkDelta {
                                        role: choice.delta.role.map(|_| MessageRole::Assistant),
                                        content: choice.delta.content,
                                        tool_calls: choice.delta.tool_calls.map(|tcs| {
                                            tcs.into_iter()
                                                .map(|tc| PartialToolCall {
                                                    id: tc.id,
                                                    call_type: Some("function".to_string()),
                                                    function: tc.function.map(|f| {
                                                        PartialToolCallFunction {
                                                            name: f.name,
                                                            arguments: f.arguments,
                                                        }
                                                    }),
                                                })
                                                .collect()
                                        }),
                                        reasoning: None,
                                    },
                                    finish_reason: choice.finish_reason.map(|fr| match fr {
                                        async_openai::types::chat::FinishReason::Stop => FinishReason::Stop,
                                        async_openai::types::chat::FinishReason::Length => FinishReason::Length,
                                        async_openai::types::chat::FinishReason::ToolCalls => FinishReason::ToolCalls,
                                        async_openai::types::chat::FinishReason::ContentFilter => FinishReason::ContentFilter,
                                        _ => FinishReason::Stop,
                                    }),
                                }],
                            };
                            yield chunk;
                        }
                    }
                    Err(e) => {
                        Err(Self::handle_error(e))?;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// =============================================================================
// OpenAI-Compatible Provider
// =============================================================================

/// Base class for OpenAI-compatible providers.
/// Providers like Groq, Together, OpenRouter can use this.
pub struct OpenAICompatibleProvider {
    inner: OpenAIProvider,
    provider_name: String,
    env_key_name: String,
    api_base: String,
}

impl OpenAICompatibleProvider {
    pub fn new(
        provider_name: impl Into<String>,
        api_base: impl Into<String>,
        env_key_name: impl Into<String>,
        config: ProviderConfig,
    ) -> Result<Self, AnyLLMError> {
        let provider_name = provider_name.into();
        let api_base = api_base.into();
        let env_key_name = env_key_name.into();

        let mut config = config;
        if config.base_url.is_none() {
            config.base_url = Some(api_base.clone());
        }

        let inner = OpenAIProvider::new(config)?;

        Ok(Self {
            inner,
            provider_name,
            env_key_name,
            api_base,
        })
    }
}

#[async_trait]
impl Provider for OpenAICompatibleProvider {
    fn provider_name(&self) -> &str { &self.provider_name }
    fn env_api_key_name(&self) -> &str { &self.env_key_name }
    fn provider_documentation_url(&self) -> &str { OpenAIProvider::PROVIDER_DOCUMENTATION_URL }
    fn api_base(&self) -> &str { &self.api_base }
    fn supports_streaming(&self) -> bool { self.inner.supports_streaming() }
    fn supports_tools(&self) -> bool { self.inner.supports_tools() }
    fn supports_vision(&self) -> bool { self.inner.supports_vision() }
    fn supports_list_models(&self) -> bool { self.inner.supports_list_models() }
    fn supports_reasoning(&self) -> bool { self.inner.supports_reasoning() }

    async fn is_available(&self) -> bool { self.inner.is_available().await }
    async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> { self.inner.list_models().await }

    async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
        let mut result = self.inner.completion(request).await?;
        result.provider = Some(self.provider_name.clone());
        Ok(result)
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError> {
        self.inner.completion_stream(request).await
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_metadata() {
        let provider = OpenAIProvider::new(ProviderConfig {
            api_key: Some(Some("test-key".to_string())),
            ..Default::default()
        }).unwrap();

        assert_eq!(provider.provider_name(), "openai");
        assert_eq!(provider.env_api_key_name(), "OPENAI_API_KEY");
        assert_eq!(provider.api_base(), "https://api.openai.com/v1");
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = OpenAIProvider::new(ProviderConfig {
            api_key: Some(Some("test-key".to_string())),
            ..Default::default()
        }).unwrap();

        assert!(provider.supports_streaming());
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
        assert!(provider.supports_list_models());
    }

    #[test]
    fn test_model_supports_tools() {
        assert!(OpenAIProvider::model_supports_tools("gpt-4o"));
        assert!(!OpenAIProvider::model_supports_tools("o1-preview"));
    }

    #[test]
    fn test_model_supports_vision() {
        assert!(OpenAIProvider::model_supports_vision("gpt-4o"));
        assert!(!OpenAIProvider::model_supports_vision("gpt-3.5-turbo"));
    }

    #[tokio::test]
    async fn test_is_available_without_key() {
        std::env::remove_var("OPENAI_API_KEY");
        let provider = OpenAIProvider::new(ProviderConfig::default()).unwrap();
        assert!(!provider.is_available().await);
    }
}
