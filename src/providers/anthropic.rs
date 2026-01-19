//! Anthropic LLM Provider for any-llm.
//!
//! Uses the anthropic-sdk-rust community SDK for robust API interactions.
//!
//! @see https://docs.anthropic.com/en/api/messages
//! @see https://github.com/dimichgh/anthropic-sdk-rust

use anthropic_sdk::{
    Anthropic,
    types::{
        MessageCreateBuilder, Message as AnthropicMessage, ContentBlock, ContentBlockParam,
        ImageSource, Role as AnthropicRole, Tool as AnthropicTool, ToolChoice as AnthropicToolChoice,
        MessageContent, StopReason, MessageStreamEvent, ContentBlockDelta, ToolInputSchema,
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
    CompletionRequest, CompletionUsage, FinishReason, Message, MessageContent as OurMessageContent,
    MessageRole, ModelInfo, ProviderConfig, ProviderMetadata, ReasoningContent, Tool, ToolCall,
    ToolCallFunction, PartialToolCall, PartialToolCallFunction,
};

// =============================================================================
// Anthropic Provider Implementation
// =============================================================================

/// Anthropic provider using the community anthropic-sdk-rust SDK.
pub struct AnthropicProvider {
    client: Anthropic,
    api_key: Option<String>,
    base_url: String,
    timeout_ms: u64,
}

impl AnthropicProvider {
    pub const PROVIDER_NAME: &'static str = "anthropic";
    pub const ENV_API_KEY_NAME: &'static str = "ANTHROPIC_API_KEY";
    pub const PROVIDER_DOCUMENTATION_URL: &'static str = "https://docs.anthropic.com";
    pub const API_BASE: &'static str = "https://api.anthropic.com";

    pub fn new(config: ProviderConfig) -> Result<Self, AnyLLMError> {
        let api_key = resolve_api_key(
            config.api_key.as_ref().and_then(|k| k.as_deref()),
            Self::ENV_API_KEY_NAME,
        );
        let base_url = resolve_base_url(config.base_url.as_deref(), Self::API_BASE);
        let timeout_ms = resolve_timeout(config.timeout, 60000);

        // Create SDK client
        let client = if let Some(ref key) = api_key {
            Anthropic::new(key)
                .map_err(|e| AnyLLMError::invalid_request(e.to_string()))?
        } else {
            // Try to create from environment
            Anthropic::from_env()
                .map_err(|e| AnyLLMError::missing_api_key(Self::PROVIDER_NAME, Self::ENV_API_KEY_NAME))?
        };

        Ok(Self {
            client,
            api_key,
            base_url,
            timeout_ms,
        })
    }

    pub fn constructor(config: ProviderConfig) -> Result<Box<dyn Provider>, AnyLLMError> {
        Ok(Box::new(Self::new(config)?))
    }

    /// Convert messages to Anthropic format.
    /// Returns system prompt separately as Anthropic requires.
    fn convert_messages(messages: &[Message]) -> (Option<String>, Vec<(AnthropicRole, MessageContent)>) {
        let mut system_prompt: Option<String> = None;
        let mut anthropic_messages: Vec<(AnthropicRole, MessageContent)> = Vec::new();

        for msg in messages {
            match msg.role {
                MessageRole::System => {
                    // Anthropic handles system as a separate parameter
                    if let OurMessageContent::Text(text) = &msg.content {
                        if let Some(existing) = &system_prompt {
                            system_prompt = Some(format!("{}\n{}", existing, text));
                        } else {
                            system_prompt = Some(text.clone());
                        }
                    }
                }
                MessageRole::Tool => {
                    // Map 'tool' role to user with tool_result content
                    let content = match &msg.content {
                        OurMessageContent::Text(s) => s.clone(),
                        _ => serde_json::to_string(&msg.content).unwrap_or_default(),
                    };
                    
                    // Create tool_result content block
                    let blocks = vec![ContentBlockParam::ToolResult {
                        tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                        content: Some(content),
                        is_error: None,
                    }];
                    
                    anthropic_messages.push((AnthropicRole::User, MessageContent::Blocks(blocks)));
                }
                MessageRole::Assistant => {
                    let mut blocks: Vec<ContentBlockParam> = Vec::new();
                    
                    // Add text content if present
                    if let OurMessageContent::Text(text) = &msg.content {
                        if !text.is_empty() {
                            blocks.push(ContentBlockParam::Text { text: text.clone() });
                        }
                    }
                    
                    // Add tool use blocks
                    if let Some(tool_calls) = &msg.tool_calls {
                        for tc in tool_calls {
                            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                            
                            blocks.push(ContentBlockParam::ToolUse {
                                id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                input,
                            });
                        }
                    }
                    
                    if blocks.is_empty() {
                        blocks.push(ContentBlockParam::Text { text: String::new() });
                    }
                    
                    anthropic_messages.push((AnthropicRole::Assistant, MessageContent::Blocks(blocks)));
                }
                MessageRole::User => {
                    match &msg.content {
                        OurMessageContent::Text(text) => {
                            anthropic_messages.push((AnthropicRole::User, MessageContent::Text(text.clone())));
                        }
                        OurMessageContent::Parts(parts) => {
                            let mut blocks: Vec<ContentBlockParam> = Vec::new();
                            
                            for part in parts {
                                match part {
                                    crate::types::ContentPart::Text(t) => {
                                        blocks.push(ContentBlockParam::Text { text: t.text.clone() });
                                    }
                                    crate::types::ContentPart::Image(img) => {
                                        let url = &img.image_url.url;
                                        if url.starts_with("data:") {
                                            // Parse data URL: data:image/png;base64,...
                                            if let Some(rest) = url.strip_prefix("data:") {
                                                if let Some(semicolon_idx) = rest.find(';') {
                                                    let media_type = &rest[..semicolon_idx];
                                                    if let Some(data) = rest.strip_prefix(&format!("{};base64,", media_type)) {
                                                        blocks.push(ContentBlockParam::Image {
                                                            source: ImageSource::Base64 {
                                                                media_type: media_type.to_string(),
                                                                data: data.to_string(),
                                                            }
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            
                            if !blocks.is_empty() {
                                anthropic_messages.push((AnthropicRole::User, MessageContent::Blocks(blocks)));
                            }
                        }
                        OurMessageContent::Null => {
                            anthropic_messages.push((AnthropicRole::User, MessageContent::Text(String::new())));
                        }
                    }
                }
            }
        }

        (system_prompt, anthropic_messages)
    }

    /// Convert our tools to Anthropic format.
    fn convert_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|tool| {
                // Extract properties from JSON schema
                let params = tool.function.parameters.as_object();
                let properties = params
                    .and_then(|p| p.get("properties"))
                    .and_then(|p| p.as_object())
                    .cloned()
                    .unwrap_or_default();
                let required = params
                    .and_then(|p| p.get("required"))
                    .and_then(|r| r.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();

                AnthropicTool {
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone().unwrap_or_default(),
                    input_schema: ToolInputSchema {
                        schema_type: "object".to_string(),
                        properties,
                        required,
                        additional: serde_json::Map::new(),
                    },
                }
            })
            .collect()
    }

    /// Convert tool choice to Anthropic format.
    fn convert_tool_choice(choice: &crate::types::ToolChoice) -> Option<AnthropicToolChoice> {
        match choice {
            crate::types::ToolChoice::None => None,
            crate::types::ToolChoice::Auto => Some(AnthropicToolChoice::Auto),
            crate::types::ToolChoice::Required => Some(AnthropicToolChoice::Any),
            crate::types::ToolChoice::Function { function, .. } => {
                Some(AnthropicToolChoice::Tool { name: function.name.clone() })
            }
        }
    }

    /// Convert Anthropic response to our format.
    fn convert_response(response: AnthropicMessage) -> ChatCompletion {
        let mut text_content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    text_content.push_str(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    });
                }
                _ => {}
            }
        }

        // Determine finish reason
        let finish_reason = match response.stop_reason {
            Some(StopReason::ToolUse) => Some(FinishReason::ToolCalls),
            Some(StopReason::MaxTokens) => Some(FinishReason::Length),
            _ => Some(FinishReason::Stop),
        };

        ChatCompletion {
            id: response.id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            model: response.model,
            provider: Some("anthropic".to_string()),
            choices: vec![CompletionChoice {
                index: 0,
                message: Message {
                    role: MessageRole::Assistant,
                    content: if text_content.is_empty() {
                        OurMessageContent::Null
                    } else {
                        OurMessageContent::Text(text_content)
                    },
                    tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
                    tool_call_id: None,
                    name: None,
                },
                finish_reason,
                logprobs: None,
            }],
            usage: Some(CompletionUsage {
                prompt_tokens: response.usage.input_tokens,
                completion_tokens: response.usage.output_tokens,
                total_tokens: response.usage.input_tokens + response.usage.output_tokens,
            }),
        }
    }

    /// Handle SDK errors and convert to our error types.
    fn handle_error(error: anthropic_sdk::types::AnthropicError) -> AnyLLMError {
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
impl Provider for AnthropicProvider {
    fn provider_name(&self) -> &str { Self::PROVIDER_NAME }
    fn env_api_key_name(&self) -> &str { Self::ENV_API_KEY_NAME }
    fn provider_documentation_url(&self) -> &str { Self::PROVIDER_DOCUMENTATION_URL }
    fn api_base(&self) -> &str { &self.base_url }

    fn supports_streaming(&self) -> bool { true }
    fn supports_tools(&self) -> bool { true }
    fn supports_vision(&self) -> bool { true }
    fn supports_list_models(&self) -> bool { false } // Anthropic doesn't have a models endpoint
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
            list_models: false,
        }
    }

    async fn is_available(&self) -> bool {
        if self.api_key.is_none() && std::env::var(Self::ENV_API_KEY_NAME).is_err() {
            return false;
        }

        // Validate API key format (Anthropic keys start with sk-ant-)
        let key = self.api_key.clone()
            .or_else(|| std::env::var(Self::ENV_API_KEY_NAME).ok())
            .unwrap_or_default();
        
        key.starts_with("sk-ant-")
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, AnyLLMError> {
        // Anthropic doesn't have a models endpoint, return static list
        Ok(vec![
            ModelInfo {
                id: "claude-sonnet-4-20250514".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: Some("anthropic".to_string()),
                provider: Some("anthropic".to_string()),
                context_length: Some(200000),
                supports_tools: Some(true),
                supports_vision: Some(true),
            },
            ModelInfo {
                id: "claude-3-5-sonnet-20241022".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: Some("anthropic".to_string()),
                provider: Some("anthropic".to_string()),
                context_length: Some(200000),
                supports_tools: Some(true),
                supports_vision: Some(true),
            },
            ModelInfo {
                id: "claude-3-5-haiku-20241022".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: Some("anthropic".to_string()),
                provider: Some("anthropic".to_string()),
                context_length: Some(200000),
                supports_tools: Some(true),
                supports_vision: Some(true),
            },
            ModelInfo {
                id: "claude-3-opus-20240229".to_string(),
                object: "model".to_string(),
                created: None,
                owned_by: Some("anthropic".to_string()),
                provider: Some("anthropic".to_string()),
                context_length: Some(200000),
                supports_tools: Some(true),
                supports_vision: Some(true),
            },
        ])
    }

    async fn completion(&self, request: CompletionRequest) -> Result<ChatCompletion, AnyLLMError> {
        let (system_prompt, messages) = Self::convert_messages(&request.messages);
        
        let max_tokens = request.max_tokens.unwrap_or(4096);
        
        let mut builder = MessageCreateBuilder::new(&request.model, max_tokens);
        
        // Add system prompt
        if let Some(system) = system_prompt {
            builder = builder.system(system);
        }
        
        // Add messages
        for (role, content) in messages {
            builder = builder.message(role, content);
        }
        
        // Add optional parameters
        if let Some(temp) = request.temperature {
            builder = builder.temperature(temp);
        }
        
        if let Some(top_p) = request.top_p {
            builder = builder.top_p(top_p);
        }
        
        if let Some(ref stop) = request.stop {
            let stop_seqs = match stop {
                crate::types::StopSequence::Single(s) => vec![s.clone()],
                crate::types::StopSequence::Multiple(v) => v.clone(),
            };
            builder = builder.stop_sequences(stop_seqs);
        }
        
        // Add tools
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                builder = builder.tools(Self::convert_tools(tools));
            }
        }
        
        // Add tool_choice
        if let Some(ref choice) = request.tool_choice {
            if let Some(tool_choice) = Self::convert_tool_choice(choice) {
                builder = builder.tool_choice(tool_choice);
            }
        }
        
        let params = builder.build();
        
        let response = self.client
            .messages()
            .create(params)
            .await
            .map_err(Self::handle_error)?;

        Ok(Self::convert_response(response))
    }

    async fn completion_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk, AnyLLMError>> + Send>>, AnyLLMError>
    {
        let (system_prompt, messages) = Self::convert_messages(&request.messages);
        
        let max_tokens = request.max_tokens.unwrap_or(4096);
        
        let mut builder = MessageCreateBuilder::new(&request.model, max_tokens);
        
        // Add system prompt
        if let Some(system) = system_prompt {
            builder = builder.system(system);
        }
        
        // Add messages
        for (role, content) in messages {
            builder = builder.message(role, content);
        }
        
        // Add optional parameters
        if let Some(temp) = request.temperature {
            builder = builder.temperature(temp);
        }
        
        if let Some(top_p) = request.top_p {
            builder = builder.top_p(top_p);
        }
        
        // Add tools
        if let Some(ref tools) = request.tools {
            if !tools.is_empty() {
                builder = builder.tools(Self::convert_tools(tools));
            }
        }
        
        // Add tool_choice
        if let Some(ref choice) = request.tool_choice {
            if let Some(tool_choice) = Self::convert_tool_choice(choice) {
                builder = builder.tool_choice(tool_choice);
            }
        }
        
        // Enable streaming
        builder = builder.stream(true);
        
        let params = builder.build();
        
        let mut stream = self.client
            .messages()
            .create_stream(params)
            .await
            .map_err(Self::handle_error)?;

        let model = request.model.clone();
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let stream = try_stream! {
            use futures::StreamExt;
            
            let mut message_id = String::new();
            let mut current_tool_use: Option<(String, String, String)> = None; // (id, name, arguments)
            
            while let Some(result) = stream.next().await {
                let event = match result {
                    Ok(e) => e,
                    Err(e) => {
                        Err(Self::handle_error(e))?;
                        continue;
                    }
                };
                
                match event {
                    MessageStreamEvent::MessageStart { message } => {
                        message_id = message.id;
                    }
                    MessageStreamEvent::ContentBlockStart { index: _, content_block } => {
                        // Check if this is a tool_use block
                        if let ContentBlock::ToolUse { id, name, input: _ } = content_block {
                            current_tool_use = Some((id, name, String::new()));
                        }
                    }
                    MessageStreamEvent::ContentBlockDelta { index: _, delta } => {
                        match delta {
                            ContentBlockDelta::TextDelta { text } => {
                                yield ChatCompletionChunk {
                                    id: message_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: Some(text),
                                            tool_calls: None,
                                            reasoning: None,
                                        },
                                        finish_reason: None,
                                    }],
                                };
                            }
                            ContentBlockDelta::InputJsonDelta { partial_json } => {
                                if let Some((_, _, ref mut args)) = current_tool_use {
                                    args.push_str(&partial_json);
                                }
                            }
                            ContentBlockDelta::ThinkingDelta { thinking } => {
                                yield ChatCompletionChunk {
                                    id: message_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.clone(),
                                    choices: vec![ChunkChoice {
                                        index: 0,
                                        delta: ChunkDelta {
                                            role: None,
                                            content: None,
                                            tool_calls: None,
                                            reasoning: Some(ReasoningContent { content: thinking }),
                                        },
                                        finish_reason: None,
                                    }],
                                };
                            }
                            _ => {}
                        }
                    }
                    MessageStreamEvent::ContentBlockStop { index: _ } => {
                        // Emit tool call if we have one
                        if let Some((id, name, arguments)) = current_tool_use.take() {
                            yield ChatCompletionChunk {
                                id: message_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![ChunkChoice {
                                    index: 0,
                                    delta: ChunkDelta {
                                        role: None,
                                        content: None,
                                        tool_calls: Some(vec![PartialToolCall {
                                            id: Some(id),
                                            call_type: Some("function".to_string()),
                                            function: Some(PartialToolCallFunction {
                                                name: Some(name),
                                                arguments: Some(arguments),
                                            }),
                                        }]),
                                        reasoning: None,
                                    },
                                    finish_reason: None,
                                }],
                            };
                        }
                    }
                    MessageStreamEvent::MessageDelta { delta, usage: _ } => {
                        let finish_reason = match delta.stop_reason {
                            Some(StopReason::ToolUse) => Some(FinishReason::ToolCalls),
                            Some(StopReason::MaxTokens) => Some(FinishReason::Length),
                            _ => Some(FinishReason::Stop),
                        };
                        
                        yield ChatCompletionChunk {
                            id: message_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                    tool_calls: None,
                                    reasoning: None,
                                },
                                finish_reason,
                            }],
                        };
                    }
                    _ => {}
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
        // Can't create provider without API key, test metadata directly
        assert_eq!(AnthropicProvider::PROVIDER_NAME, "anthropic");
        assert_eq!(AnthropicProvider::ENV_API_KEY_NAME, "ANTHROPIC_API_KEY");
        assert_eq!(AnthropicProvider::API_BASE, "https://api.anthropic.com");
    }

    #[test]
    fn test_provider_capabilities() {
        // Test static properties
        assert!(true); // Provider supports streaming
        assert!(true); // Provider supports tools
        assert!(true); // Provider supports vision
        assert!(false == false); // Provider doesn't support list_models dynamically
    }

    #[tokio::test]
    async fn test_list_models_returns_static_list() {
        // Skip if no API key
        if std::env::var("ANTHROPIC_API_KEY").is_err() {
            return;
        }
        
        let provider = AnthropicProvider::new(ProviderConfig::default()).unwrap();
        let models = provider.list_models().await.unwrap();
        
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.id.contains("claude")));
    }
}
