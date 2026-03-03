package responses

import (
	"testing"

	"github.com/tidwall/gjson"
)

// TestConvertClaudeRequestToOpenAIResponses_BasicTextMessage tests basic text message conversion
func TestConvertClaudeRequestToOpenAIResponses_BasicTextMessage(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [
			{"role": "user", "content": "Hello, world!"}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// Check model
	if got := gjson.Get(outputStr, "model").String(); got != "gpt-5.3-codex" {
		t.Errorf("Expected model 'gpt-5.3-codex', got '%s'", got)
	}

	// Check max_output_tokens (Claude max_tokens -> Responses API max_output_tokens)
	if got := gjson.Get(outputStr, "max_output_tokens").Int(); got != 4096 {
		t.Errorf("Expected max_output_tokens 4096, got %d", got)
	}

	// Check that max_tokens is NOT present (Responses API uses max_output_tokens)
	if gjson.Get(outputStr, "max_tokens").Exists() {
		t.Error("max_tokens should not be present in output")
	}

	// Check input array has a message
	input := gjson.Get(outputStr, "input")
	if !input.IsArray() || len(input.Array()) != 1 {
		t.Fatalf("Expected input array with 1 item, got %d", len(input.Array()))
	}

	// Check message role
	if got := gjson.Get(outputStr, "input.0.role").String(); got != "user" {
		t.Errorf("Expected role 'user', got '%s'", got)
	}

	// Check content type is input_text for user messages
	if got := gjson.Get(outputStr, "input.0.content.0.type").String(); got != "input_text" {
		t.Errorf("Expected content type 'input_text', got '%s'", got)
	}

	// Check content text
	if got := gjson.Get(outputStr, "input.0.content.0.text").String(); got != "Hello, world!" {
		t.Errorf("Expected text 'Hello, world!', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_AssistantMessage tests assistant message role mapping
func TestConvertClaudeRequestToOpenAIResponses_AssistantMessage(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [
			{"role": "user", "content": "Hi"},
			{"role": "assistant", "content": "Hello!"},
			{"role": "user", "content": "How are you?"}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// Assistant message should use output_text type
	if got := gjson.Get(outputStr, "input.1.content.0.type").String(); got != "output_text" {
		t.Errorf("Expected assistant content type 'output_text', got '%s'", got)
	}

	// User messages should use input_text type
	if got := gjson.Get(outputStr, "input.0.content.0.type").String(); got != "input_text" {
		t.Errorf("Expected user content type 'input_text', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_SystemMessage tests system message -> developer role conversion
func TestConvertClaudeRequestToOpenAIResponses_SystemMessage(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"system": "You are a helpful assistant.",
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// System should be converted to developer role as first input item
	if got := gjson.Get(outputStr, "input.0.role").String(); got != "developer" {
		t.Errorf("Expected role 'developer', got '%s'", got)
	}

	if got := gjson.Get(outputStr, "input.0.content.0.text").String(); got != "You are a helpful assistant." {
		t.Errorf("Expected system text, got '%s'", got)
	}

	// User message should be second
	if got := gjson.Get(outputStr, "input.1.role").String(); got != "user" {
		t.Errorf("Expected role 'user', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_SystemArray tests array-format system message conversion
func TestConvertClaudeRequestToOpenAIResponses_SystemArray(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"system": [
			{"type": "text", "text": "Part 1."},
			{"type": "text", "text": "Part 2."}
		],
		"messages": [
			{"role": "user", "content": "Hello"}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// Both parts should be in the developer message content
	if got := gjson.Get(outputStr, "input.0.role").String(); got != "developer" {
		t.Errorf("Expected role 'developer', got '%s'", got)
	}
	if got := gjson.Get(outputStr, "input.0.content.0.text").String(); got != "Part 1." {
		t.Errorf("Expected first text 'Part 1.', got '%s'", got)
	}
	if got := gjson.Get(outputStr, "input.0.content.1.text").String(); got != "Part 2." {
		t.Errorf("Expected second text 'Part 2.', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_ToolConversion tests Claude tools -> Responses API tools conversion
func TestConvertClaudeRequestToOpenAIResponses_ToolConversion(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [{"role": "user", "content": "Use a tool"}],
		"tools": [
			{
				"name": "get_weather",
				"description": "Get the weather",
				"input_schema": {
					"type": "object",
					"properties": {
						"location": {"type": "string"}
					},
					"required": ["location"]
				}
			}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// Check tools array
	tools := gjson.Get(outputStr, "tools")
	if !tools.IsArray() || len(tools.Array()) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(tools.Array()))
	}

	// Check tool type is function (Responses API format)
	if got := gjson.Get(outputStr, "tools.0.type").String(); got != "function" {
		t.Errorf("Expected tool type 'function', got '%s'", got)
	}

	// Check name
	if got := gjson.Get(outputStr, "tools.0.name").String(); got != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got '%s'", got)
	}

	// Check that input_schema was converted to parameters
	if !gjson.Get(outputStr, "tools.0.parameters").Exists() {
		t.Error("Expected parameters field in tool")
	}
	if got := gjson.Get(outputStr, "tools.0.parameters.type").String(); got != "object" {
		t.Errorf("Expected parameters type 'object', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_ToolUseAndResult tests tool_use and tool_result conversion
func TestConvertClaudeRequestToOpenAIResponses_ToolUseAndResult(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [
			{"role": "user", "content": "What's the weather?"},
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"location": "NYC"}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 72°F"}
				]
			}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	input := gjson.Get(outputStr, "input")
	if !input.IsArray() {
		t.Fatal("Expected input to be an array")
	}

	// Find function_call item
	found := false
	for _, item := range input.Array() {
		if item.Get("type").String() == "function_call" {
			found = true
			if got := item.Get("call_id").String(); got != "call_123" {
				t.Errorf("Expected call_id 'call_123', got '%s'", got)
			}
			if got := item.Get("name").String(); got != "get_weather" {
				t.Errorf("Expected name 'get_weather', got '%s'", got)
			}
			break
		}
	}
	if !found {
		t.Error("Expected to find a function_call item in input")
	}

	// Find function_call_output item
	found = false
	for _, item := range input.Array() {
		if item.Get("type").String() == "function_call_output" {
			found = true
			if got := item.Get("call_id").String(); got != "call_123" {
				t.Errorf("Expected call_id 'call_123', got '%s'", got)
			}
			if got := item.Get("output").String(); got != "Sunny, 72°F" {
				t.Errorf("Expected output 'Sunny, 72°F', got '%s'", got)
			}
			break
		}
	}
	if !found {
		t.Error("Expected to find a function_call_output item in input")
	}
}

// TestConvertClaudeRequestToOpenAIResponses_StreamOption tests that stream is properly set
func TestConvertClaudeRequestToOpenAIResponses_StreamOption(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	// Test stream=true
	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, true)
	if got := gjson.GetBytes(output, "stream").Bool(); !got {
		t.Error("Expected stream to be true")
	}

	// Test stream=false
	output = ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	if got := gjson.GetBytes(output, "stream").Bool(); got {
		t.Error("Expected stream to be false")
	}
}

// TestConvertClaudeRequestToOpenAIResponses_NoStreamOptions tests that stream_options is not set
func TestConvertClaudeRequestToOpenAIResponses_NoStreamOptions(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"stream_options": {"include_usage": true},
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, true)
	outputStr := string(output)

	if gjson.Get(outputStr, "stream_options").Exists() {
		t.Error("stream_options should not be in output (not supported by Responses API)")
	}
}

// TestConvertClaudeRequestToOpenAIResponses_TemperatureAndTopP tests parameter passthrough
func TestConvertClaudeRequestToOpenAIResponses_TemperatureAndTopP(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"temperature": 0.7,
		"top_p": 0.9,
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	if got := gjson.Get(outputStr, "temperature").Float(); got != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", got)
	}
	if got := gjson.Get(outputStr, "top_p").Float(); got != 0.9 {
		t.Errorf("Expected top_p 0.9, got %f", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_StopSequences tests stop_sequences -> stop conversion
func TestConvertClaudeRequestToOpenAIResponses_StopSequences(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"stop_sequences": ["STOP", "END"],
		"messages": [{"role": "user", "content": "hi"}]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	stop := gjson.Get(outputStr, "stop")
	if !stop.IsArray() || len(stop.Array()) != 2 {
		t.Fatalf("Expected 2 stop sequences, got %v", stop.Raw)
	}
	if got := stop.Array()[0].String(); got != "STOP" {
		t.Errorf("Expected first stop 'STOP', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_ToolChoiceAuto tests tool_choice auto mapping
func TestConvertClaudeRequestToOpenAIResponses_ToolChoiceAuto(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"name": "test", "description": "test", "input_schema": {}}],
		"tool_choice": {"type": "auto"}
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	if got := gjson.Get(outputStr, "tool_choice").String(); got != "auto" {
		t.Errorf("Expected tool_choice 'auto', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_ToolChoiceAny tests tool_choice any -> required mapping
func TestConvertClaudeRequestToOpenAIResponses_ToolChoiceAny(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"name": "test", "description": "test", "input_schema": {}}],
		"tool_choice": {"type": "any"}
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	if got := gjson.Get(outputStr, "tool_choice").String(); got != "required" {
		t.Errorf("Expected tool_choice 'required', got '%s'", got)
	}
}

// TestConvertClaudeRequestToOpenAIResponses_ThinkingSkipped tests that thinking blocks are skipped in content
func TestConvertClaudeRequestToOpenAIResponses_ThinkingSkipped(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [
			{
				"role": "assistant",
				"content": [
					{"type": "thinking", "thinking": "Let me think..."},
					{"type": "text", "text": "Here is my answer."}
				]
			}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// Only the text content should be in the input, thinking should be skipped
	inputItems := gjson.Get(outputStr, "input")
	if !inputItems.IsArray() {
		t.Fatal("Expected input to be an array")
	}

	// Verify that there's a message with just the text content
	for _, item := range inputItems.Array() {
		if item.Get("type").String() == "message" {
			content := item.Get("content")
			if content.IsArray() {
				for _, c := range content.Array() {
					if c.Get("type").String() == "thinking" {
						t.Error("thinking block should be skipped in output")
					}
				}
			}
		}
	}
}

// TestConvertClaudeRequestToOpenAIResponses_EmptyMessages tests handling of empty messages
func TestConvertClaudeRequestToOpenAIResponses_EmptyMessages(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": []
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	input := gjson.Get(outputStr, "input")
	if !input.IsArray() {
		t.Error("Expected input to be an array")
	}
	if len(input.Array()) != 0 {
		t.Errorf("Expected empty input array, got %d items", len(input.Array()))
	}
}

// TestConvertClaudeRequestToOpenAIResponses_ArrayContent tests array-format message content
func TestConvertClaudeRequestToOpenAIResponses_ArrayContent(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5.3-codex",
		"max_tokens": 4096,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "First part"},
					{"type": "text", "text": "Second part"}
				]
			}
		]
	}`)

	output := ConvertClaudeRequestToOpenAIResponses("gpt-5.3-codex", inputJSON, false)
	outputStr := string(output)

	// Check both text parts are present
	if got := gjson.Get(outputStr, "input.0.content.0.text").String(); got != "First part" {
		t.Errorf("Expected first text 'First part', got '%s'", got)
	}
	if got := gjson.Get(outputStr, "input.0.content.1.text").String(); got != "Second part" {
		t.Errorf("Expected second text 'Second part', got '%s'", got)
	}
}
