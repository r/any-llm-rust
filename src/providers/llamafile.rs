//! Llamafile LLM Provider for any-llm.
//!
//! Llamafile provides an OpenAI-compatible API on localhost:8080.
//! This provider uses the async-openai SDK with a custom baseURL.
//!
//! @see https://github.com/Mozilla-Ocho/llamafile

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
        CreateChatCompletionRequest,
        ChatCompletionTool,
        ChatCompletionTools,
        ChatCompletionMessageToolCall,
        ChatCompletionMessageToolCalls,
        FunctionObject,
    },
};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use crate::errors::AnyLLMError;
use crate::providers::base::{resolve_base_url, resolve_timeout, Provider};
use crate::types::{
    ChatCompletion, ChatCompletionChunk, ChunkChoice, ChunkDelta, CompletionChoice,
    CompletionRequest, CompletionUsage, FinishReason, Message, MessageContent, MessageRole,
    ModelInfo, ProviderConfig, ProviderMetadata, Tool, ToolCall, ToolCallFunction,
    PartialToolCall, PartialToolCallFunction,
};

// =============================================================================
// Llamafile Provider Implementation
// =============================================================================

/// Llamafile provider using the async-openai SDK (OpenAI-compatible API).
pub struct LlamafileProvider {
    client: Client<OpenAIConfig>,
    base_url: String,
}

impl LlamafileProvider {
    pub const PROVIDER_NAME: &'static str = "llamafile";
    pub const ENV_API_KEY_NAME: &'static str = "LLAMAFILE_API_KEY";
    pub const PROVIDER_DOCUMENTATION_URL: &'static str = "https://github.com/Mozilla-Ocho/llamafile";
    pub const API_BASE: &'static str = "http://localhost:8080/v1";

    pub fn new(config: ProviderConfig) -> Result<Self, AnyLLMError> {
        let base_url = resolve_base_url(config.base_url.as_deref(), Self::API_BASE);
        let _timeout_ms = resolve_timeout(config.timeout, 60000);

        // Build OpenAI config pointed at llamafile
        // Llamafile doesn't require an API key, but the SDK needs one
        let openai_config = OpenAIConfig::new()
            .with_api_key("not-needed")
            .with_api_base(&base_url);

        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            base_url,
        })
    }

    pub fn constructor(config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
        Ok(Box::new(Self::new(config)?))
    }

    /// Convert our messages to OpenAI SDK format.
    fn convert_messages(messages: &[Message]) -> Vec<ChatCompletionRequestMessage> {
        messages
            .iter()
            .map(|msg| match msg.role {
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
                    // Flatten multimodal to text only for llamafile
                    let content = match &msg.content {
                        MessageContent::Text(s) => s.clone(),
                        MessageContent::Parts(parts) => {
                            parts
                                .iter()
                                .filter_map(|p| match p {
                                    crate::types::ContentPart::Text(t) => Some(t.text.clone()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join(" ")
                        }
                        MessageContent::Null => String::new(),
                    };
                    ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                        content: ChatCompletionRequestUserMessageContent::Text(content),
                        name: msg.name.clone(),
                    })
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
            })
            .collect()
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

    /// Handle SDK errors and convert to our error types.
    fn handle_error(error: async_openai::error::OpenAIError) -> AnyLLMError {
        let message = error.to_string();
        let message_lower = message.to_lowercase();

        if message_lower.contains("connection refused") || message_lower.contains("econnrefused") {
            return AnyLLMError::provider_unavailable(
                Self::PROVIDER_NAME,
                Some("Llamafile is not running. Start your llamafile first.".to_string()),
            );
        }

        if message_lower.contains("timeout") || message_lower.contains("timed out") {
            return AnyLLMError::timeout(Self::PROVIDER_NAME, 0);
        }

        AnyLLMError::provider_request(Self::PROVIDER_NAME, message, None, None)
    }
}

#[async_trait]
impl Provider for LlamafileProvider {
    fn provider_name(&self) -> &str { Self::PROVIDER_NAME }
    fn env_api_key_name(&self) -> &str { Self::ENV_API_KEY_NAME }
    fn provider_documentation_url(&self) -> &str { Self::PROVIDER_DOCUMENTATION_URL }
    fn api_base(&self) -> &str { &self.base_url }

    fn supports_streaming(&self) -> bool { true }
    fn supports_tools(&self) -> bool { true }
    fn supports_vision(&self) -> bool { false } // Depends on model
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
            image: false,
            list_models: true,
        }
    }

    async fn is_available(&self) -> bool {
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
            .map(|model| ModelInfo {
                id: model.id.clone(),
                object: "model".to_string(),
                created: Some(model.created as u64),
                owned_by: Some(model.owned_by),
                provider: Some("llamafile".to_string()),
                context_length: None,
                supports_tools: Some(true),
                supports_vision: Some(false),
            })
            .collect())
    }

    async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
        let messages = Self::convert_messages(&request.messages);

        let mut openai_request = CreateChatCompletionRequest {
            model: if request.model.is_empty() { "default".to_string() } else { request.model.clone() },
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(false),
            ..Default::default()
        };

        // Add max_tokens
        if let Some(max_tokens) = request.max_tokens {
            openai_request.max_tokens = Some(max_tokens);
        }

        // Add tools
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                openai_request.tools = Some(Self::convert_tools(tools));
            }
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
            provider: Some("llamafile".to_string()),
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
        let messages = Self::convert_messages(&request.messages);

        let mut openai_request = CreateChatCompletionRequest {
            model: if request.model.is_empty() { "default".to_string() } else { request.model.clone() },
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(true),
            ..Default::default()
        };

        // Add max_tokens
        if let Some(max_tokens) = request.max_tokens {
            openai_request.max_tokens = Some(max_tokens);
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

        let model = request.model.clone();

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
                                model: if response.model.is_empty() { 
                                    if model.is_empty() { "default".to_string() } else { model.clone() }
                                } else { 
                                    response.model.clone() 
                                },
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
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_metadata() {
        let provider = LlamafileProvider::new(ProviderConfig::default()).unwrap();
        assert_eq!(provider.provider_name(), "llamafile");
        assert_eq!(provider.api_base(), "http://localhost:8080/v1");
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = LlamafileProvider::new(ProviderConfig::default()).unwrap();
        assert!(provider.supports_streaming());
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision()); // Depends on model
        assert!(provider.supports_list_models());
    }
}
