//! Ollama LLM Provider for any-llm.
//!
//! Uses the official ollama-rs SDK for robust API interactions.
//! Ollama runs locally and provides access to open-source models.
//!
//! @see https://github.com/ollama/ollama
//! @see https://github.com/pepperoni21/ollama-rs

use async_stream::try_stream;
use async_trait::async_trait;
use futures::Stream;
use ollama_rs::{
    generation::chat::{
        request::ChatMessageRequest, ChatMessage, MessageRole as OllamaRole,
    },
    generation::tools::{ToolInfo, ToolFunctionInfo, ToolType},
    models::ModelOptions,
    Ollama,
};
use schemars::Schema;
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
// Ollama Provider Implementation
// =============================================================================

/// Ollama provider using the official ollama-rs SDK.
pub struct OllamaProvider {
    client: Ollama,
    base_url: String,
}

impl OllamaProvider {
    pub const PROVIDER_NAME: &'static str = "ollama";
    pub const ENV_API_KEY_NAME: &'static str = "OLLAMA_API_KEY";
    pub const PROVIDER_DOCUMENTATION_URL: &'static str = "https://github.com/ollama/ollama";
    pub const API_BASE: &'static str = "http://localhost:11434";

    pub fn new(config: ProviderConfig) -> Result<Self, AnyLLMError> {
        let base_url = resolve_base_url(config.base_url.as_deref(), Self::API_BASE);
        let _timeout_ms = resolve_timeout(config.timeout, 120000);

        // Parse host and port from base_url
        let url = url::Url::parse(&base_url)
            .map_err(|e| AnyLLMError::invalid_request(format!("Invalid base URL: {}", e)))?;

        let host = format!("{}://{}", url.scheme(), url.host_str().unwrap_or("localhost"));
        let port = url.port().unwrap_or(11434);

        let client = Ollama::new(host, port);

        Ok(Self {
            client,
            base_url,
        })
    }

    pub fn constructor(config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
        Ok(Box::new(Self::new(config)?))
    }

    fn model_supports_tools(model_name: &str) -> bool {
        let name = model_name.to_lowercase();
        let no_tool_support = ["mistral:7b-instruct", "mistral:7b", "llama2", "codellama"];
        if no_tool_support.iter().any(|m| name.contains(m)) {
            return false;
        }
        let tool_models = ["llama3.1", "llama3.2", "llama3.3", "mistral-nemo", "mistral-large", "mixtral", "qwen2.5", "phi4", "granite3", "command-r"];
        tool_models.iter().any(|m| name.contains(m))
    }

    fn model_supports_vision(model_name: &str) -> bool {
        let vision_models = ["llava", "bakllava", "llama3.2-vision", "moondream"];
        vision_models.iter().any(|m| model_name.to_lowercase().contains(m))
    }

    /// Convert messages to Ollama SDK format.
    fn convert_messages(messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    MessageRole::System => OllamaRole::System,
                    MessageRole::User => OllamaRole::User,
                    MessageRole::Assistant => OllamaRole::Assistant,
                    MessageRole::Tool => OllamaRole::Tool,
                };

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

                // Extract images from parts for vision models
                let images: Option<Vec<ollama_rs::generation::images::Image>> =
                    if let MessageContent::Parts(parts) = &msg.content {
                        let imgs: Vec<_> = parts
                            .iter()
                            .filter_map(|p| match p {
                                crate::types::ContentPart::Image(i) => {
                                    let url = &i.image_url.url;
                                    if url.starts_with("data:image/") {
                                        url.split(",").nth(1).map(|b64| {
                                            ollama_rs::generation::images::Image::from_base64(b64)
                                        })
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            })
                            .collect();
                        if imgs.is_empty() { None } else { Some(imgs) }
                    } else {
                        None
                    };

                let mut chat_msg = ChatMessage::new(role, content);

                if let Some(imgs) = images {
                    chat_msg = chat_msg.with_images(imgs);
                }

                chat_msg
            })
            .collect()
    }

    /// Convert tools to Ollama SDK format.
    fn convert_tools(tools: &[Tool]) -> Vec<ToolInfo> {
        tools
            .iter()
            .map(|tool| {
                // Convert our JSON schema to schemars Schema
                let parameters: Schema = serde_json::from_value(tool.function.parameters.clone())
                    .unwrap_or_else(|_| Schema::default());

                ToolInfo {
                    tool_type: ToolType::Function,
                    function: ToolFunctionInfo {
                        name: tool.function.name.clone(),
                        description: tool.function.description.clone().unwrap_or_default(),
                        parameters,
                    },
                }
            })
            .collect()
    }

    /// Handle SDK errors and convert to our error types.
    fn handle_error(error: ollama_rs::error::OllamaError) -> AnyLLMError {
        let message = error.to_string();
        let message_lower = message.to_lowercase();

        if message_lower.contains("connection refused") || message_lower.contains("econnrefused") {
            return AnyLLMError::provider_unavailable(
                Self::PROVIDER_NAME,
                Some("Ollama is not running. Start it with: ollama serve".to_string()),
            );
        }

        if message_lower.contains("timeout") || message_lower.contains("timed out") {
            return AnyLLMError::timeout(Self::PROVIDER_NAME, 0);
        }

        AnyLLMError::provider_request(Self::PROVIDER_NAME, message, None, None)
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn provider_name(&self) -> &str { Self::PROVIDER_NAME }
    fn env_api_key_name(&self) -> &str { Self::ENV_API_KEY_NAME }
    fn provider_documentation_url(&self) -> &str { Self::PROVIDER_DOCUMENTATION_URL }
    fn api_base(&self) -> &str { &self.base_url }

    fn supports_streaming(&self) -> bool { true }
    fn supports_tools(&self) -> bool { true }
    fn supports_vision(&self) -> bool { true }
    fn supports_list_models(&self) -> bool { true }
    fn supports_reasoning(&self) -> bool { true }

    fn metadata(&self) -> ProviderMetadata {
        ProviderMetadata {
            name: Self::PROVIDER_NAME.to_string(),
            env_key: Self::ENV_API_KEY_NAME.to_string(),
            doc_url: Self::PROVIDER_DOCUMENTATION_URL.to_string(),
            streaming: true,
            reasoning: true,
            completion: true,
            embedding: false,
            image: true,
            list_models: true,
        }
    }

    async fn is_available(&self) -> bool {
        match self.client.list_local_models().await {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> {
        let response = self
            .client
            .list_local_models()
            .await
            .map_err(Self::handle_error)?;

        Ok(response
            .into_iter()
            .map(|model| ModelInfo {
                id: model.name.clone(),
                object: "model".to_string(),
                created: None,
                owned_by: Some("ollama".to_string()),
                provider: Some("ollama".to_string()),
                context_length: None,
                supports_tools: Some(Self::model_supports_tools(&model.name)),
                supports_vision: Some(Self::model_supports_vision(&model.name)),
            })
            .collect())
    }

    async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
        let messages = Self::convert_messages(&request.messages);

        let mut chat_request = ChatMessageRequest::new(request.model.clone(), messages);

        // Add tools if provided
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                chat_request = chat_request.tools(Self::convert_tools(tools));
            }
        }

        // Add options
        let mut options = ModelOptions::default();
        
        if let Some(temp) = request.temperature {
            options = options.temperature(temp);
        }
        
        if let Some(top_p) = request.top_p {
            options = options.top_p(top_p);
        }
        
        if let Some(max_tokens) = request.max_tokens {
            options = options.num_predict(max_tokens as i32);
        }
        
        if let Some(seed) = request.seed {
            options = options.seed(seed as i32);
        }
        
        if let Some(ref stop) = request.stop {
            let stop_seqs = match stop {
                crate::types::StopSequence::Single(s) => vec![s.clone()],
                crate::types::StopSequence::Multiple(v) => v.clone(),
            };
            options = options.stop(stop_seqs);
        }

        chat_request = chat_request.options(options);

        let response = self
            .client
            .send_chat_messages(chat_request)
            .await
            .map_err(Self::handle_error)?;

        // Convert tool calls (tool_calls is Vec<ToolCall>, not Option<Vec<ToolCall>>)
        let tool_calls = if !response.message.tool_calls.is_empty() {
            Some(response.message.tool_calls.iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{}", i),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name: tc.function.name.clone(),
                        arguments: serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
                    },
                })
                .collect::<Vec<_>>())
        } else {
            None
        };

        let finish_reason = if tool_calls.is_some() {
            Some(FinishReason::ToolCalls)
        } else {
            Some(FinishReason::Stop)
        };

        Ok(ChatCompletion {
            id: format!("ollama-{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model: response.model.clone(),
            provider: Some("ollama".to_string()),
            choices: vec![CompletionChoice {
                index: 0,
                message: Message {
                    role: MessageRole::Assistant,
                    content: if response.message.content.is_empty() {
                        MessageContent::Null
                    } else {
                        MessageContent::Text(response.message.content)
                    },
                    tool_calls,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason,
                logprobs: None,
            }],
            usage: response.final_data.map(|fd| CompletionUsage {
                prompt_tokens: fd.prompt_eval_count as u32,
                completion_tokens: fd.eval_count as u32,
                total_tokens: (fd.prompt_eval_count + fd.eval_count) as u32,
            }),
        })
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError>
    {
        let messages = Self::convert_messages(&request.messages);

        let mut chat_request = ChatMessageRequest::new(request.model.clone(), messages);

        // Add tools if provided
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                chat_request = chat_request.tools(Self::convert_tools(tools));
            }
        }

        // Add options
        let mut options = ModelOptions::default();
        
        if let Some(temp) = request.temperature {
            options = options.temperature(temp);
        }
        
        if let Some(top_p) = request.top_p {
            options = options.top_p(top_p);
        }
        
        if let Some(max_tokens) = request.max_tokens {
            options = options.num_predict(max_tokens as i32);
        }

        chat_request = chat_request.options(options);

        let mut stream = self
            .client
            .send_chat_messages_stream(chat_request)
            .await
            .map_err(Self::handle_error)?;

        let model = request.model.clone();
        let id = format!("ollama-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0));
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let stream = try_stream! {
            use futures::StreamExt;
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(response) => {
                        let content = if response.message.content.is_empty() {
                            None
                        } else {
                            Some(response.message.content.clone())
                        };

                        let tool_calls = if !response.message.tool_calls.is_empty() {
                            Some(response.message.tool_calls.iter()
                                .enumerate()
                                .map(|(i, tc)| PartialToolCall {
                                    id: Some(format!("call_{}", i)),
                                    call_type: Some("function".to_string()),
                                    function: Some(PartialToolCallFunction {
                                        name: Some(tc.function.name.clone()),
                                        arguments: Some(serde_json::to_string(&tc.function.arguments).unwrap_or_default()),
                                    }),
                                })
                                .collect::<Vec<_>>())
                        } else {
                            None
                        };

                        let finish_reason = if response.done {
                            if tool_calls.is_some() {
                                Some(FinishReason::ToolCalls)
                            } else {
                                Some(FinishReason::Stop)
                            }
                        } else {
                            None
                        };

                        yield ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: response.model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content,
                                    tool_calls,
                                    reasoning: None,
                                },
                                finish_reason,
                            }],
                        };
                    }
                    Err(_e) => {
                        // Stream error, yield nothing but continue
                        // (ollama-rs stream errors are not recoverable)
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
        let provider = OllamaProvider::new(ProviderConfig::default()).unwrap();
        assert_eq!(provider.provider_name(), "ollama");
        assert_eq!(provider.api_base(), "http://localhost:11434");
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = OllamaProvider::new(ProviderConfig::default()).unwrap();
        assert!(provider.supports_streaming());
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
        assert!(provider.supports_list_models());
    }

    #[test]
    fn test_model_supports_tools() {
        assert!(OllamaProvider::model_supports_tools("llama3.1"));
        assert!(OllamaProvider::model_supports_tools("llama3.2:latest"));
        assert!(!OllamaProvider::model_supports_tools("llama2"));
    }

    #[test]
    fn test_model_supports_vision() {
        assert!(OllamaProvider::model_supports_vision("llava:latest"));
        assert!(!OllamaProvider::model_supports_vision("llama3.2"));
    }
}
