package responses

import (
	"bytes"
	"context"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type responsesToClaudeState struct {
	MessageStarted    bool
	MessageStopSent   bool
	TextBlockStarted  bool
	TextBlockIndex    int
	NextContentIndex  int
	HasToolUse        bool
	ReasoningActive   bool
	ReasoningIndex    int
	OutputIndexToTool map[int]*responsesToClaudeToolState
	ItemIDToTool      map[string]*responsesToClaudeToolState
}

type responsesToClaudeToolState struct {
	Index int
	ID    string
	Name  string
}

// ConvertOpenAIResponsesToClaude converts OpenAI Responses API SSE events to Claude SSE format.
func ConvertOpenAIResponsesToClaude(ctx context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &responsesToClaudeState{
			TextBlockIndex:    -1,
			OutputIndexToTool: make(map[int]*responsesToClaudeToolState),
			ItemIDToTool:      make(map[string]*responsesToClaudeToolState),
		}
	}
	state := (*param).(*responsesToClaudeState)

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return nil
	}
	payload := bytes.TrimSpace(rawJSON[5:])
	if bytes.Equal(payload, []byte("[DONE]")) {
		return nil
	}
	if !gjson.ValidBytes(payload) {
		return nil
	}

	event := gjson.GetBytes(payload, "type").String()
	results := make([]string, 0, 4)

	ensureMessageStart := func() {
		if state.MessageStarted {
			return
		}
		messageStart := `{"type":"message_start","message":{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`
		messageStart, _ = sjson.Set(messageStart, "message.id", gjson.GetBytes(payload, "response.id").String())
		messageStart, _ = sjson.Set(messageStart, "message.model", gjson.GetBytes(payload, "response.model").String())
		results = append(results, emitEvent("message_start", messageStart))
		state.MessageStarted = true
	}

	startTextBlockIfNeeded := func() {
		if state.TextBlockStarted {
			return
		}
		if state.TextBlockIndex < 0 {
			state.TextBlockIndex = state.NextContentIndex
			state.NextContentIndex++
		}
		contentBlockStart := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
		contentBlockStart, _ = sjson.Set(contentBlockStart, "index", state.TextBlockIndex)
		results = append(results, emitEvent("content_block_start", contentBlockStart))
		state.TextBlockStarted = true
	}

	stopTextBlockIfNeeded := func() {
		if !state.TextBlockStarted {
			return
		}
		contentBlockStop := `{"type":"content_block_stop","index":0}`
		contentBlockStop, _ = sjson.Set(contentBlockStop, "index", state.TextBlockIndex)
		results = append(results, emitEvent("content_block_stop", contentBlockStop))
		state.TextBlockStarted = false
		state.TextBlockIndex = -1
	}

	resolveTool := func(itemID string, outputIndex int) *responsesToClaudeToolState {
		if itemID != "" {
			if tool, ok := state.ItemIDToTool[itemID]; ok {
				return tool
			}
		}
		if tool, ok := state.OutputIndexToTool[outputIndex]; ok {
			if itemID != "" {
				state.ItemIDToTool[itemID] = tool
			}
			return tool
		}
		return nil
	}

	switch event {
	case "response.created":
		ensureMessageStart()

	case "response.output_text.delta":
		ensureMessageStart()
		startTextBlockIfNeeded()
		delta := gjson.GetBytes(payload, "delta").String()
		if delta != "" {
			contentDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`
			contentDelta, _ = sjson.Set(contentDelta, "index", state.TextBlockIndex)
			contentDelta, _ = sjson.Set(contentDelta, "delta.text", delta)
			results = append(results, emitEvent("content_block_delta", contentDelta))
		}

	case "response.reasoning_summary_part.added":
		ensureMessageStart()
		state.ReasoningActive = true
		state.ReasoningIndex = state.NextContentIndex
		state.NextContentIndex++
		thinkingStart := `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`
		thinkingStart, _ = sjson.Set(thinkingStart, "index", state.ReasoningIndex)
		results = append(results, emitEvent("content_block_start", thinkingStart))

	case "response.reasoning_summary_text.delta":
		if state.ReasoningActive {
			delta := gjson.GetBytes(payload, "delta").String()
			if delta != "" {
				thinkingDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}`
				thinkingDelta, _ = sjson.Set(thinkingDelta, "index", state.ReasoningIndex)
				thinkingDelta, _ = sjson.Set(thinkingDelta, "delta.thinking", delta)
				results = append(results, emitEvent("content_block_delta", thinkingDelta))
			}
		}

	case "response.reasoning_summary_part.done":
		if state.ReasoningActive {
			thinkingStop := `{"type":"content_block_stop","index":0}`
			thinkingStop, _ = sjson.Set(thinkingStop, "index", state.ReasoningIndex)
			results = append(results, emitEvent("content_block_stop", thinkingStop))
			state.ReasoningActive = false
		}

	case "response.output_item.added":
		if gjson.GetBytes(payload, "item.type").String() != "function_call" {
			break
		}
		ensureMessageStart()
		stopTextBlockIfNeeded()
		state.HasToolUse = true
		tool := &responsesToClaudeToolState{
			Index: state.NextContentIndex,
			ID:    gjson.GetBytes(payload, "item.call_id").String(),
			Name:  gjson.GetBytes(payload, "item.name").String(),
		}
		if tool.ID == "" {
			tool.ID = gjson.GetBytes(payload, "item.id").String()
		}
		state.NextContentIndex++
		outputIndex := int(gjson.GetBytes(payload, "output_index").Int())
		state.OutputIndexToTool[outputIndex] = tool
		if itemID := gjson.GetBytes(payload, "item.id").String(); itemID != "" {
			state.ItemIDToTool[itemID] = tool
		}
		contentBlockStart := `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`
		contentBlockStart, _ = sjson.Set(contentBlockStart, "index", tool.Index)
		contentBlockStart, _ = sjson.Set(contentBlockStart, "content_block.id", tool.ID)
		contentBlockStart, _ = sjson.Set(contentBlockStart, "content_block.name", tool.Name)
		results = append(results, emitEvent("content_block_start", contentBlockStart))

	case "response.output_item.delta":
		item := gjson.GetBytes(payload, "item")
		if item.Get("type").String() != "function_call" {
			break
		}
		tool := resolveTool(item.Get("id").String(), int(gjson.GetBytes(payload, "output_index").Int()))
		if tool == nil {
			break
		}
		partial := gjson.GetBytes(payload, "delta").String()
		if partial == "" {
			partial = item.Get("arguments").String()
		}
		if partial == "" {
			break
		}
		inputDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`
		inputDelta, _ = sjson.Set(inputDelta, "index", tool.Index)
		inputDelta, _ = sjson.Set(inputDelta, "delta.partial_json", partial)
		results = append(results, emitEvent("content_block_delta", inputDelta))

	case "response.function_call_arguments.delta":
		// Copilot sends tool call arguments via this event type
		itemID := gjson.GetBytes(payload, "item_id").String()
		outputIndex := int(gjson.GetBytes(payload, "output_index").Int())
		tool := resolveTool(itemID, outputIndex)
		if tool == nil {
			break
		}
		partial := gjson.GetBytes(payload, "delta").String()
		if partial == "" {
			break
		}
		inputDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`
		inputDelta, _ = sjson.Set(inputDelta, "index", tool.Index)
		inputDelta, _ = sjson.Set(inputDelta, "delta.partial_json", partial)
		results = append(results, emitEvent("content_block_delta", inputDelta))

	case "response.output_item.done":
		if gjson.GetBytes(payload, "item.type").String() != "function_call" {
			break
		}
		tool := resolveTool(gjson.GetBytes(payload, "item.id").String(), int(gjson.GetBytes(payload, "output_index").Int()))
		if tool == nil {
			break
		}
		contentBlockStop := `{"type":"content_block_stop","index":0}`
		contentBlockStop, _ = sjson.Set(contentBlockStop, "index", tool.Index)
		results = append(results, emitEvent("content_block_stop", contentBlockStop))

	case "response.completed":
		ensureMessageStart()
		stopTextBlockIfNeeded()
		if !state.MessageStopSent {
			stopReason := "end_turn"
			if state.HasToolUse {
				stopReason = "tool_use"
			} else if sr := gjson.GetBytes(payload, "response.stop_reason").String(); sr == "max_tokens" || sr == "stop" {
				stopReason = sr
			}
			inputTokens := gjson.GetBytes(payload, "response.usage.input_tokens").Int()
			outputTokens := gjson.GetBytes(payload, "response.usage.output_tokens").Int()
			cachedTokens := gjson.GetBytes(payload, "response.usage.input_tokens_details.cached_tokens").Int()
			if cachedTokens > 0 && inputTokens >= cachedTokens {
				inputTokens -= cachedTokens
			}
			messageStop := `{"type":"message_stop","message":{"usage":{"input_tokens":0,"output_tokens":0},"stop_reason":""}}`
			messageStop, _ = sjson.Set(messageStop, "message.usage.input_tokens", inputTokens)
			messageStop, _ = sjson.Set(messageStop, "message.usage.output_tokens", outputTokens)
			if cachedTokens > 0 {
				messageStop, _ = sjson.Set(messageStop, "message.usage.cache_read_input_tokens", cachedTokens)
			}
			messageStop, _ = sjson.Set(messageStop, "message.stop_reason", stopReason)
			results = append(results, emitEvent("message_stop", messageStop))
			state.MessageStopSent = true
		}
	}

	return results
}

// ConvertOpenAIResponsesToClaudeNonStream converts a non-streaming OpenAI Responses API response to Claude format.
func ConvertOpenAIResponsesToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	root := gjson.ParseBytes(rawJSON)
	out := `{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}`
	out, _ = sjson.Set(out, "id", root.Get("id").String())
	out, _ = sjson.Set(out, "model", root.Get("model").String())

	hasToolUse := false
	if output := root.Get("output"); output.Exists() && output.IsArray() {
		for _, item := range output.Array() {
			switch item.Get("type").String() {
			case "reasoning":
				var thinkingText string
				if summary := item.Get("summary"); summary.Exists() && summary.IsArray() {
					var parts []string
					for _, part := range summary.Array() {
						if txt := part.Get("text").String(); txt != "" {
							parts = append(parts, txt)
						}
					}
					thinkingText = strings.Join(parts, "")
				}
				if thinkingText == "" {
					if content := item.Get("content"); content.Exists() && content.IsArray() {
						var parts []string
						for _, part := range content.Array() {
							if txt := part.Get("text").String(); txt != "" {
								parts = append(parts, txt)
							}
						}
						thinkingText = strings.Join(parts, "")
					}
				}
				if thinkingText != "" {
					block := `{"type":"thinking","thinking":""}`
					block, _ = sjson.Set(block, "thinking", thinkingText)
					out, _ = sjson.SetRaw(out, "content.-1", block)
				}

			case "message":
				if content := item.Get("content"); content.Exists() && content.IsArray() {
					for _, part := range content.Array() {
						if part.Get("type").String() != "output_text" {
							continue
						}
						text := part.Get("text").String()
						if text == "" {
							continue
						}
						block := `{"type":"text","text":""}`
						block, _ = sjson.Set(block, "text", text)
						out, _ = sjson.SetRaw(out, "content.-1", block)
					}
				}

			case "function_call":
				hasToolUse = true
				toolUse := `{"type":"tool_use","id":"","name":"","input":{}}`
				toolID := item.Get("call_id").String()
				if toolID == "" {
					toolID = item.Get("id").String()
				}
				toolUse, _ = sjson.Set(toolUse, "id", toolID)
				toolUse, _ = sjson.Set(toolUse, "name", item.Get("name").String())
				if args := item.Get("arguments").String(); args != "" && gjson.Valid(args) {
					argObj := gjson.Parse(args)
					if argObj.IsObject() {
						toolUse, _ = sjson.SetRaw(toolUse, "input", argObj.Raw)
					}
				}
				out, _ = sjson.SetRaw(out, "content.-1", toolUse)
			}
		}
	}

	inputTokens := root.Get("usage.input_tokens").Int()
	outputTokens := root.Get("usage.output_tokens").Int()
	cachedTokens := root.Get("usage.input_tokens_details.cached_tokens").Int()
	if cachedTokens > 0 && inputTokens >= cachedTokens {
		inputTokens -= cachedTokens
	}
	out, _ = sjson.Set(out, "usage.input_tokens", inputTokens)
	out, _ = sjson.Set(out, "usage.output_tokens", outputTokens)
	if cachedTokens > 0 {
		out, _ = sjson.Set(out, "usage.cache_read_input_tokens", cachedTokens)
	}
	if hasToolUse {
		out, _ = sjson.Set(out, "stop_reason", "tool_use")
	} else if sr := root.Get("stop_reason").String(); sr == "max_tokens" || sr == "stop" {
		out, _ = sjson.Set(out, "stop_reason", sr)
	} else {
		out, _ = sjson.Set(out, "stop_reason", "end_turn")
	}
	return out
}
