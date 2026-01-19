//! Core types for any-llm.
//!
//! These types follow the OpenAI API format as the de facto standard,
//! matching the patterns from mozilla-ai/any-llm and any-llm-ts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Provider Types
// =============================================================================

/// Supported LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LLMProviderType {
    OpenAI,
    Anthropic,
    Ollama,
    Llamafile,
    Mistral,
    Groq,
    Together,
    OpenRouter,
    LMStudio,
    DeepSeek,
}

impl LLMProviderType {
    /// Convert to lowercase string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::Ollama => "ollama",
            Self::Llamafile => "llamafile",
            Self::Mistral => "mistral",
            Self::Groq => "groq",
            Self::Together => "together",
            Self::OpenRouter => "openrouter",
            Self::LMStudio => "lmstudio",
            Self::DeepSeek => "deepseek",
        }
    }

    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(Self::OpenAI),
            "anthropic" => Some(Self::Anthropic),
            "ollama" => Some(Self::Ollama),
            "llamafile" => Some(Self::Llamafile),
            "mistral" => Some(Self::Mistral),
            "groq" => Some(Self::Groq),
            "together" => Some(Self::Together),
            "openrouter" => Some(Self::OpenRouter),
            "lmstudio" => Some(Self::LMStudio),
            "deepseek" => Some(Self::DeepSeek),
            _ => None,
        }
    }
}

impl std::fmt::Display for LLMProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API key for the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<Option<String>>,
    
    /// Base URL override for the provider's API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    
    /// Request timeout in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    
    /// Maximum number of retries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    
    /// Additional provider-specific options.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Provider metadata describing capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMetadata {
    /// Provider identifier.
    pub name: String,
    
    /// Environment variable name for API key.
    pub env_key: String,
    
    /// Link to provider documentation.
    pub doc_url: String,
    
    /// Whether provider supports streaming.
    pub streaming: bool,
    
    /// Whether provider supports reasoning/thinking output.
    pub reasoning: bool,
    
    /// Whether provider supports chat completion.
    pub completion: bool,
    
    /// Whether provider supports embeddings.
    pub embedding: bool,
    
    /// Whether provider supports image inputs.
    pub image: bool,
    
    /// Whether provider supports listing models.
    pub list_models: bool,
}

// =============================================================================
// Message Types (OpenAI-compatible)
// =============================================================================

/// Role of a message in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// A tool call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    
    /// Type of tool call (always 'function').
    #[serde(rename = "type")]
    pub call_type: String,
    
    /// Function details.
    pub function: ToolCallFunction,
}

/// Function details within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// Name of the function to call.
    pub name: String,
    
    /// JSON-encoded arguments.
    pub arguments: String,
}

/// Text content part of a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

impl TextContentPart {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: text.into(),
        }
    }
}

/// Image URL details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Image content part of a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    pub image_url: ImageUrl,
}

impl ImageContentPart {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            content_type: "image_url".to_string(),
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    pub fn with_detail(url: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            content_type: "image_url".to_string(),
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail.into()),
            },
        }
    }
}

/// Content part (text or image).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentPart {
    Text(TextContentPart),
    Image(ImageContentPart),
}

/// Content can be a string, null, or an array of parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    #[serde(serialize_with = "serialize_null")]
    Null,
}

fn serialize_null<S>(serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_none()
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Null
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender.
    pub role: MessageRole,
    
    /// Content of the message.
    #[serde(default)]
    pub content: MessageContent,
    
    /// For assistant messages: tool calls requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    
    /// For tool messages: the ID of the tool call this responds to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    
    /// Optional name (for user messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.map(MessageContent::Text).unwrap_or(MessageContent::Null),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// Create a tool response message.
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
        }
    }

    /// Create a multimodal user message with parts.
    pub fn user_with_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            role: MessageRole::User,
            content: MessageContent::Parts(parts),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
}

/// Extended message with reasoning (for models that support thinking).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageWithReasoning {
    #[serde(flatten)]
    pub message: Message,
    
    /// Reasoning/thinking content (for models that expose it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningContent {
    pub content: String,
}

// =============================================================================
// Tool Types
// =============================================================================

/// A tool definition that can be passed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

impl Tool {
    /// Create a new function tool.
    pub fn function(
        name: impl Into<String>,
        description: Option<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description,
                parameters,
            },
        }
    }
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// Name of the function.
    pub name: String,
    
    /// Description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    
    /// JSON Schema for the function parameters.
    pub parameters: serde_json::Value,
}

/// Tool choice options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// No tool calling.
    None,
    /// Let the model decide.
    Auto,
    /// Force tool calling.
    Required,
    /// Force a specific function.
    Function { 
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction 
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

// =============================================================================
// Completion Request/Response Types
// =============================================================================

/// Completion request parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier. Can be:
    /// - Just model name: "gpt-4o" (requires provider param)
    /// - Provider:model format: "openai:gpt-4o"
    pub model: String,
    
    /// Provider to use (optional if model contains provider prefix).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    
    /// Conversation messages.
    pub messages: Vec<Message>,
    
    /// Tools available for the model to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    
    /// Controls which tools the model can call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    
    /// Temperature for sampling (0-2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Nucleus sampling probability (0-1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    
    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopSequence>,
    
    /// Presence penalty (-2 to 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    
    /// Frequency penalty (-2 to 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    
    /// Random seed for reproducibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    
    /// User identifier for abuse tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    
    /// Response format (e.g., for JSON mode).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    
    /// API key override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    
    /// Base URL override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

/// Stop sequence can be a single string or array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

/// Response format options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: serde_json::Value },
}

/// A choice in a completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    /// Index of this choice.
    pub index: u32,
    
    /// The message generated.
    pub message: Message,
    
    /// Why generation stopped.
    pub finish_reason: Option<FinishReason>,
    
    /// Log probabilities (if requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// Reason why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionUsage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    
    /// Tokens in the completion.
    pub completion_tokens: u32,
    
    /// Total tokens used.
    pub total_tokens: u32,
}

/// A chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletion {
    /// Unique identifier.
    pub id: String,
    
    /// Object type.
    pub object: String,
    
    /// Unix timestamp of creation.
    pub created: u64,
    
    /// Model used.
    pub model: String,
    
    /// Provider that handled the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    
    /// Generated choices.
    pub choices: Vec<CompletionChoice>,
    
    /// Token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<CompletionUsage>,
}

// =============================================================================
// Streaming Types
// =============================================================================

/// Delta for a streaming chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChunkDelta {
    /// Role (usually only in first chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,
    
    /// Content fragment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    
    /// Tool calls (may be partial).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<PartialToolCall>>,
    
    /// Reasoning content (for models that support it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningContent>,
}

/// Partial tool call in streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<PartialToolCallFunction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// A choice in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    /// Index of this choice.
    pub index: u32,
    
    /// The delta (partial content).
    pub delta: ChunkDelta,
    
    /// Why generation stopped (only in final chunk).
    pub finish_reason: Option<FinishReason>,
}

/// A streaming chunk from a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    /// Unique identifier.
    pub id: String,
    
    /// Object type.
    pub object: String,
    
    /// Unix timestamp of creation.
    pub created: u64,
    
    /// Model used.
    pub model: String,
    
    /// Choices in this chunk.
    pub choices: Vec<ChunkChoice>,
}

// =============================================================================
// Model Types
// =============================================================================

/// Information about a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier.
    pub id: String,
    
    /// Object type.
    pub object: String,
    
    /// Unix timestamp of creation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    
    /// Owner/provider of the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
    
    /// Provider that offers this model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    
    /// Context window size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    
    /// Whether model supports tool calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tools: Option<bool>,
    
    /// Whether model supports vision/images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_vision: Option<bool>,
}

// =============================================================================
// Utility Types
// =============================================================================

/// Parsed model string result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedModel {
    /// Provider identifier.
    pub provider: String,
    
    /// Model identifier.
    pub model: String,
}

/// Result of a provider availability check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStatus {
    /// Provider identifier.
    pub provider: String,
    
    /// Whether the provider is available.
    pub available: bool,
    
    /// Error message if not available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    
    /// Provider version (if detectable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    
    /// Available models (if listable).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<ModelInfo>>,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_provider_type_as_str() {
        assert_eq!(LLMProviderType::OpenAI.as_str(), "openai");
        assert_eq!(LLMProviderType::Anthropic.as_str(), "anthropic");
        assert_eq!(LLMProviderType::Ollama.as_str(), "ollama");
        assert_eq!(LLMProviderType::Llamafile.as_str(), "llamafile");
    }

    #[test]
    fn test_llm_provider_type_from_str() {
        assert_eq!(LLMProviderType::from_str("openai"), Some(LLMProviderType::OpenAI));
        assert_eq!(LLMProviderType::from_str("OPENAI"), Some(LLMProviderType::OpenAI));
        assert_eq!(LLMProviderType::from_str("OpenAI"), Some(LLMProviderType::OpenAI));
        assert_eq!(LLMProviderType::from_str("invalid"), None);
    }

    #[test]
    fn test_message_role_as_str() {
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(MessageRole::User.as_str(), "user");
        assert_eq!(MessageRole::Assistant.as_str(), "assistant");
        assert_eq!(MessageRole::Tool.as_str(), "tool");
    }

    #[test]
    fn test_message_constructors() {
        let system = Message::system("You are a helpful assistant");
        assert_eq!(system.role, MessageRole::System);
        
        let user = Message::user("Hello!");
        assert_eq!(user.role, MessageRole::User);
        
        let assistant = Message::assistant("Hi there!");
        assert_eq!(assistant.role, MessageRole::Assistant);
        
        let tool = Message::tool("call_123", r#"{"result": 42}"#);
        assert_eq!(tool.role, MessageRole::Tool);
        assert_eq!(tool.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_message_content_from_string() {
        let content: MessageContent = "Hello".into();
        match content {
            MessageContent::Text(s) => assert_eq!(s, "Hello"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_tool_creation() {
        let tool = Tool::function(
            "get_weather",
            Some("Get the weather in a location".to_string()),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }),
        );
        
        assert_eq!(tool.tool_type, "function");
        assert_eq!(tool.function.name, "get_weather");
    }

    #[test]
    fn test_text_content_part() {
        let part = TextContentPart::new("Hello, world!");
        assert_eq!(part.content_type, "text");
        assert_eq!(part.text, "Hello, world!");
    }

    #[test]
    fn test_image_content_part() {
        let part = ImageContentPart::new("https://example.com/image.png");
        assert_eq!(part.content_type, "image_url");
        assert_eq!(part.image_url.url, "https://example.com/image.png");
        assert_eq!(part.image_url.detail, None);
        
        let part_with_detail = ImageContentPart::with_detail(
            "https://example.com/image.png",
            "high"
        );
        assert_eq!(part_with_detail.image_url.detail, Some("high".to_string()));
    }

    #[test]
    fn test_completion_request_serialization() {
        let request = CompletionRequest {
            model: "openai:gpt-4o".to_string(),
            messages: vec![Message::user("Hello!")],
            temperature: Some(0.7),
            max_tokens: Some(100),
            ..Default::default()
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-4o"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_chat_completion_deserialization() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;
        
        let completion: ChatCompletion = serde_json::from_str(json).unwrap();
        assert_eq!(completion.id, "chatcmpl-123");
        assert_eq!(completion.choices.len(), 1);
        assert_eq!(completion.choices[0].finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_parsed_model() {
        let parsed = ParsedModel {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
        };
        assert_eq!(parsed.provider, "openai");
        assert_eq!(parsed.model, "gpt-4o");
    }
}
