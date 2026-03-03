package responses

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// TestConvertOpenAIResponsesToClaude_TextDelta tests streaming text delta conversion
func TestConvertOpenAIResponsesToClaude_TextDelta(t *testing.T) {
	ctx := context.Background()

	// First event: response.created to trigger message_start
	var state any
	createdPayload := `data: {"type":"response.created","response":{"id":"resp_123","model":"gpt-5.3-codex"}}`
	results := ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(createdPayload), &state)

	if len(results) == 0 {
		t.Fatal("Expected at least one result for response.created")
	}

	// Check message_start was emitted
	found := false
	for _, r := range results {
		if strings.Contains(r, "message_start") {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected message_start event in results")
	}

	// Text delta event
	textPayload := `data: {"type":"response.output_text.delta","delta":"Hello","item_id":"msg_1","output_index":0,"content_index":0}`
	results = ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(textPayload), &state)

	// Should have content_block_start + content_block_delta
	foundDelta := false
	for _, r := range results {
		if strings.Contains(r, "content_block_delta") && strings.Contains(r, "text_delta") {
			parsed := extractDataJSON(r)
			if got := gjson.Get(parsed, "delta.text").String(); got != "Hello" {
				t.Errorf("Expected delta text 'Hello', got '%s'", got)
			}
			foundDelta = true
		}
	}
	if !foundDelta {
		t.Error("Expected content_block_delta with text_delta")
	}
}

// TestConvertOpenAIResponsesToClaude_ReasoningDelta tests thinking/reasoning delta conversion
func TestConvertOpenAIResponsesToClaude_ReasoningDelta(t *testing.T) {
	ctx := context.Background()

	var state any
	// response.created
	created := `data: {"type":"response.created","response":{"id":"resp_456","model":"gpt-5.3-codex"}}`
	ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(created), &state)

	// reasoning_summary_part.added
	partAdded := `data: {"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,"summary_index":0}`
	results := ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(partAdded), &state)

	foundThinkingStart := false
	for _, r := range results {
		if strings.Contains(r, "content_block_start") && strings.Contains(r, "thinking") {
			foundThinkingStart = true
		}
	}
	if !foundThinkingStart {
		t.Error("Expected content_block_start with thinking type")
	}

	// reasoning_summary_text.delta
	textDelta := `data: {"type":"response.reasoning_summary_text.delta","delta":"Let me think...","item_id":"rs_1","output_index":0,"summary_index":0}`
	results = ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(textDelta), &state)

	foundThinkingDelta := false
	for _, r := range results {
		if strings.Contains(r, "thinking_delta") {
			parsed := extractDataJSON(r)
			if got := gjson.Get(parsed, "delta.thinking").String(); got != "Let me think..." {
				t.Errorf("Expected thinking delta 'Let me think...', got '%s'", got)
			}
			foundThinkingDelta = true
		}
	}
	if !foundThinkingDelta {
		t.Error("Expected content_block_delta with thinking_delta")
	}

	// reasoning_summary_part.done
	partDone := `data: {"type":"response.reasoning_summary_part.done","item_id":"rs_1","output_index":0,"summary_index":0}`
	results = ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(partDone), &state)

	foundThinkingStop := false
	for _, r := range results {
		if strings.Contains(r, "content_block_stop") {
			foundThinkingStop = true
		}
	}
	if !foundThinkingStop {
		t.Error("Expected content_block_stop for thinking")
	}
}

// TestConvertOpenAIResponsesToClaude_FunctionCall tests tool call / function call conversion
func TestConvertOpenAIResponsesToClaude_FunctionCall(t *testing.T) {
	ctx := context.Background()

	var state any
	// response.created
	created := `data: {"type":"response.created","response":{"id":"resp_789","model":"gpt-5.3-codex"}}`
	ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(created), &state)

	// function call output_item.added
	itemAdded := `data: {"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_abc","name":"get_weather","status":"in_progress","arguments":""}}`
	results := ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(itemAdded), &state)

	foundToolStart := false
	for _, r := range results {
		if strings.Contains(r, "content_block_start") && strings.Contains(r, "tool_use") {
			parsed := extractDataJSON(r)
			if got := gjson.Get(parsed, "content_block.name").String(); got != "get_weather" {
				t.Errorf("Expected tool name 'get_weather', got '%s'", got)
			}
			if got := gjson.Get(parsed, "content_block.id").String(); got != "call_abc" {
				t.Errorf("Expected tool id 'call_abc', got '%s'", got)
			}
			foundToolStart = true
		}
	}
	if !foundToolStart {
		t.Error("Expected content_block_start with tool_use")
	}

	// function_call_arguments.delta
	argsDelta := `data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":1,"delta":"{\"location\":\"NYC\"}"}`
	results = ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(argsDelta), &state)

	foundArgsDelta := false
	for _, r := range results {
		if strings.Contains(r, "input_json_delta") {
			foundArgsDelta = true
		}
	}
	if !foundArgsDelta {
		t.Error("Expected content_block_delta with input_json_delta")
	}
}

// TestConvertOpenAIResponsesToClaude_Completed tests response.completed handling
func TestConvertOpenAIResponsesToClaude_Completed(t *testing.T) {
	ctx := context.Background()

	var state any
	// response.created
	created := `data: {"type":"response.created","response":{"id":"resp_done","model":"gpt-5.3-codex"}}`
	ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(created), &state)

	// text delta
	textPayload := `data: {"type":"response.output_text.delta","delta":"Hi","item_id":"msg_1","output_index":0,"content_index":0}`
	ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(textPayload), &state)

	// response.completed (no explicit stop_reason -> defaults to end_turn)
	completed := `data: {"type":"response.completed","response":{"id":"resp_done","model":"gpt-5.3-codex","usage":{"input_tokens":100,"output_tokens":50,"input_tokens_details":{"cached_tokens":0}}}}`
	results := ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(completed), &state)

	foundMessageStop := false
	for _, r := range results {
		if strings.Contains(r, "message_stop") {
			parsed := extractDataJSON(r)
			if got := gjson.Get(parsed, "message.stop_reason").String(); got != "end_turn" {
				t.Errorf("Expected stop_reason 'end_turn', got '%s'", got)
			}
			if got := gjson.Get(parsed, "message.usage.input_tokens").Int(); got != 100 {
				t.Errorf("Expected input_tokens 100, got %d", got)
			}
			if got := gjson.Get(parsed, "message.usage.output_tokens").Int(); got != 50 {
				t.Errorf("Expected output_tokens 50, got %d", got)
			}
			foundMessageStop = true
		}
	}
	if !foundMessageStop {
		t.Error("Expected message_stop event")
	}
}

// TestConvertOpenAIResponsesToClaude_CompletedWithCachedTokens tests cached token adjustment
func TestConvertOpenAIResponsesToClaude_CompletedWithCachedTokens(t *testing.T) {
	ctx := context.Background()

	var state any
	created := `data: {"type":"response.created","response":{"id":"resp_cache","model":"gpt-5.3-codex"}}`
	ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(created), &state)

	completed := `data: {"type":"response.completed","response":{"id":"resp_cache","usage":{"input_tokens":200,"output_tokens":50,"input_tokens_details":{"cached_tokens":150}}}}`
	results := ConvertOpenAIResponsesToClaude(ctx, "gpt-5.3-codex", nil, nil, []byte(completed), &state)

	for _, r := range results {
		if strings.Contains(r, "message_stop") {
			parsed := extractDataJSON(r)
			// input_tokens should be reduced by cached_tokens
			if got := gjson.Get(parsed, "message.usage.input_tokens").Int(); got != 50 {
				t.Errorf("Expected input_tokens 50 (200-150), got %d", got)
			}
			if got := gjson.Get(parsed, "message.usage.cache_read_input_tokens").Int(); got != 150 {
				t.Errorf("Expected cache_read_input_tokens 150, got %d", got)
			}
		}
	}
}

// TestConvertOpenAIResponsesToClaude_InvalidInput tests handling of invalid/non-SSE input
func TestConvertOpenAIResponsesToClaude_InvalidInput(t *testing.T) {
	ctx := context.Background()
	var state any
	state = &responsesToClaudeState{
		TextBlockIndex:    -1,
		OutputIndexToTool: make(map[int]*responsesToClaudeToolState),
		ItemIDToTool:      make(map[string]*responsesToClaudeToolState),
	}

	// Not a data: line
	results := ConvertOpenAIResponsesToClaude(ctx, "model", nil, nil, []byte("event: ping"), &state)
	if results != nil {
		t.Errorf("Expected nil for non-data line, got %v", results)
	}

	// [DONE] marker
	results = ConvertOpenAIResponsesToClaude(ctx, "model", nil, nil, []byte("data: [DONE]"), &state)
	if results != nil {
		t.Errorf("Expected nil for [DONE], got %v", results)
	}

	// Invalid JSON
	results = ConvertOpenAIResponsesToClaude(ctx, "model", nil, nil, []byte("data: not-json"), &state)
	if results != nil {
		t.Errorf("Expected nil for invalid JSON, got %v", results)
	}
}

// TestConvertOpenAIResponsesToClaudeNonStream_BasicResponse tests non-streaming response conversion
func TestConvertOpenAIResponsesToClaudeNonStream_BasicResponse(t *testing.T) {
	ctx := context.Background()
	inputJSON := []byte(`{
		"id": "resp_123",
		"model": "gpt-5.3-codex",
		"output": [
			{
				"type": "message",
				"content": [
					{"type": "output_text", "text": "Hello, world!"}
				]
			}
		],
		"usage": {
			"input_tokens": 100,
			"output_tokens": 50
		}
	}`)

	result := ConvertOpenAIResponsesToClaudeNonStream(ctx, "gpt-5.3-codex", nil, nil, inputJSON, nil)

	if got := gjson.Get(result, "id").String(); got != "resp_123" {
		t.Errorf("Expected id 'resp_123', got '%s'", got)
	}
	if got := gjson.Get(result, "model").String(); got != "gpt-5.3-codex" {
		t.Errorf("Expected model 'gpt-5.3-codex', got '%s'", got)
	}
	if got := gjson.Get(result, "role").String(); got != "assistant" {
		t.Errorf("Expected role 'assistant', got '%s'", got)
	}

	// Check content
	content := gjson.Get(result, "content")
	if !content.IsArray() || len(content.Array()) != 1 {
		t.Fatalf("Expected 1 content item, got %d", len(content.Array()))
	}
	if got := content.Array()[0].Get("type").String(); got != "text" {
		t.Errorf("Expected content type 'text', got '%s'", got)
	}
	if got := content.Array()[0].Get("text").String(); got != "Hello, world!" {
		t.Errorf("Expected text 'Hello, world!', got '%s'", got)
	}

	// Check usage
	if got := gjson.Get(result, "usage.input_tokens").Int(); got != 100 {
		t.Errorf("Expected input_tokens 100, got %d", got)
	}
	if got := gjson.Get(result, "usage.output_tokens").Int(); got != 50 {
		t.Errorf("Expected output_tokens 50, got %d", got)
	}

	// Check stop_reason
	if got := gjson.Get(result, "stop_reason").String(); got != "end_turn" {
		t.Errorf("Expected stop_reason 'end_turn', got '%s'", got)
	}
}

// TestConvertOpenAIResponsesToClaudeNonStream_WithToolUse tests non-streaming tool use conversion
func TestConvertOpenAIResponsesToClaudeNonStream_WithToolUse(t *testing.T) {
	ctx := context.Background()
	inputJSON := []byte(`{
		"id": "resp_tool",
		"model": "gpt-5.3-codex",
		"output": [
			{
				"type": "function_call",
				"call_id": "call_123",
				"name": "get_weather",
				"arguments": "{\"location\":\"NYC\"}"
			}
		],
		"usage": {"input_tokens": 50, "output_tokens": 25}
	}`)

	result := ConvertOpenAIResponsesToClaudeNonStream(ctx, "gpt-5.3-codex", nil, nil, inputJSON, nil)

	content := gjson.Get(result, "content")
	if !content.IsArray() || len(content.Array()) != 1 {
		t.Fatalf("Expected 1 content item, got %d", len(content.Array()))
	}

	toolUse := content.Array()[0]
	if got := toolUse.Get("type").String(); got != "tool_use" {
		t.Errorf("Expected type 'tool_use', got '%s'", got)
	}
	if got := toolUse.Get("id").String(); got != "call_123" {
		t.Errorf("Expected id 'call_123', got '%s'", got)
	}
	if got := toolUse.Get("name").String(); got != "get_weather" {
		t.Errorf("Expected name 'get_weather', got '%s'", got)
	}
	if got := toolUse.Get("input.location").String(); got != "NYC" {
		t.Errorf("Expected input.location 'NYC', got '%s'", got)
	}

	// Check stop_reason is tool_use
	if got := gjson.Get(result, "stop_reason").String(); got != "tool_use" {
		t.Errorf("Expected stop_reason 'tool_use', got '%s'", got)
	}
}

// TestConvertOpenAIResponsesToClaudeNonStream_WithReasoning tests reasoning/thinking conversion
func TestConvertOpenAIResponsesToClaudeNonStream_WithReasoning(t *testing.T) {
	ctx := context.Background()
	inputJSON := []byte(`{
		"id": "resp_reason",
		"model": "gpt-5.3-codex",
		"output": [
			{
				"type": "reasoning",
				"summary": [
					{"type": "summary_text", "text": "Let me think about this..."}
				]
			},
			{
				"type": "message",
				"content": [
					{"type": "output_text", "text": "Here is my answer."}
				]
			}
		],
		"usage": {"input_tokens": 100, "output_tokens": 80}
	}`)

	result := ConvertOpenAIResponsesToClaudeNonStream(ctx, "gpt-5.3-codex", nil, nil, inputJSON, nil)

	content := gjson.Get(result, "content")
	if !content.IsArray() || len(content.Array()) != 2 {
		t.Fatalf("Expected 2 content items (thinking + text), got %d", len(content.Array()))
	}

	// First should be thinking
	if got := content.Array()[0].Get("type").String(); got != "thinking" {
		t.Errorf("Expected first content type 'thinking', got '%s'", got)
	}
	if got := content.Array()[0].Get("thinking").String(); got != "Let me think about this..." {
		t.Errorf("Expected thinking text, got '%s'", got)
	}

	// Second should be text
	if got := content.Array()[1].Get("type").String(); got != "text" {
		t.Errorf("Expected second content type 'text', got '%s'", got)
	}
	if got := content.Array()[1].Get("text").String(); got != "Here is my answer." {
		t.Errorf("Expected text 'Here is my answer.', got '%s'", got)
	}
}

// TestConvertOpenAIResponsesToClaudeNonStream_CachedTokens tests cached token adjustment in non-stream
func TestConvertOpenAIResponsesToClaudeNonStream_CachedTokens(t *testing.T) {
	ctx := context.Background()
	inputJSON := []byte(`{
		"id": "resp_cached",
		"model": "gpt-5.3-codex",
		"output": [
			{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
		],
		"usage": {
			"input_tokens": 200,
			"output_tokens": 50,
			"input_tokens_details": {"cached_tokens": 150}
		}
	}`)

	result := ConvertOpenAIResponsesToClaudeNonStream(ctx, "gpt-5.3-codex", nil, nil, inputJSON, nil)

	// input_tokens should be reduced by cached_tokens
	if got := gjson.Get(result, "usage.input_tokens").Int(); got != 50 {
		t.Errorf("Expected input_tokens 50 (200-150), got %d", got)
	}
	if got := gjson.Get(result, "usage.cache_read_input_tokens").Int(); got != 150 {
		t.Errorf("Expected cache_read_input_tokens 150, got %d", got)
	}
}

// extractDataJSON extracts the JSON payload from an SSE event string formatted by emitEvent()
func extractDataJSON(event string) string {
	// emitEvent format: "event: <type>\ndata: <json>"
	parts := strings.SplitN(event, "\ndata: ", 2)
	if len(parts) == 2 {
		return parts[1]
	}
	return event
}
