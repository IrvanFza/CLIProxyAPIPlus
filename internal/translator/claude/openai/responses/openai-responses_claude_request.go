package responses

import (
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertClaudeRequestToOpenAIResponses transforms a Claude Messages API request
// into an OpenAI Responses API request format.
// It supports:
// - messages[] -> input[] (with role mapping)
// - system -> input[] with role="developer"
// - max_tokens -> max_output_tokens
// - tools (Claude format) -> tools (Responses API format)
// - thinking -> reasoning.effort
// - stream_options removal (not supported in Responses API)
func ConvertClaudeRequestToOpenAIResponses(modelName string, inputRawJSON []byte, stream bool) []byte {
	rawJSON := inputRawJSON

	out := `{"model":"","input":[]}`

	root := gjson.ParseBytes(rawJSON)

	// Model
	out, _ = sjson.Set(out, "model", modelName)

	// Max tokens
	if maxTokens := root.Get("max_tokens"); maxTokens.Exists() {
		out, _ = sjson.Set(out, "max_output_tokens", maxTokens.Int())
	}

	if temp := root.Get("temperature"); temp.Exists() {
		out, _ = sjson.Set(out, "temperature", temp.Float())
	}

	if topP := root.Get("top_p"); topP.Exists() {
		out, _ = sjson.Set(out, "top_p", topP.Float())
	}

	if stopSequences := root.Get("stop_sequences"); stopSequences.Exists() && stopSequences.IsArray() {
		var stops []string
		stopSequences.ForEach(func(_, value gjson.Result) bool {
			stops = append(stops, value.String())
			return true
		})
		if len(stops) > 0 {
			out, _ = sjson.Set(out, "stop", stops)
		}
	}

	// Stream
	out, _ = sjson.Set(out, "stream", stream)

	// Convert Claude thinking config to OpenAI reasoning.effort
	if thinkingConfig := root.Get("thinking"); thinkingConfig.Exists() && thinkingConfig.IsObject() {
		if thinkingType := thinkingConfig.Get("type"); thinkingType.Exists() {
			switch thinkingType.String() {
			case "enabled":
				if budgetTokens := thinkingConfig.Get("budget_tokens"); budgetTokens.Exists() {
					budget := int(budgetTokens.Int())
					if effort, ok := thinking.ConvertBudgetToLevel(budget); ok && effort != "" {
						out, _ = sjson.Set(out, "reasoning.effort", effort)
					}
				} else {
					if effort, ok := thinking.ConvertBudgetToLevel(-1); ok && effort != "" {
						out, _ = sjson.Set(out, "reasoning.effort", effort)
					}
				}
			case "adaptive", "auto":
				out, _ = sjson.Set(out, "reasoning.effort", string(thinking.LevelXHigh))
			case "disabled":
				if effort, ok := thinking.ConvertBudgetToLevel(0); ok && effort != "" {
					out, _ = sjson.Set(out, "reasoning.effort", effort)
				}
			}
		}
	}

	inputArr := "[]"

	// system -> developer role
	if system := root.Get("system"); system.Exists() {
		var systemParts []string
		if system.IsArray() {
			system.ForEach(func(_, part gjson.Result) bool {
				if txt := part.Get("text").String(); txt != "" {
					systemParts = append(systemParts, txt)
				}
				return true
			})
		} else if system.Type == gjson.String {
			systemParts = append(systemParts, system.String())
		}
		if len(systemParts) > 0 {
			msg := `{"type":"message","role":"developer","content":[]}`
			for _, txt := range systemParts {
				part := `{"type":"input_text","text":""}`
				part, _ = sjson.Set(part, "text", txt)
				msg, _ = sjson.SetRaw(msg, "content.-1", part)
			}
			inputArr, _ = sjson.SetRaw(inputArr, "-1", msg)
		}
	}

	// messages -> input
	if messages := root.Get("messages"); messages.Exists() && messages.IsArray() {
		messages.ForEach(func(_, message gjson.Result) bool {
			role := message.Get("role").String()
			contentResult := message.Get("content")

			if !contentResult.Exists() {
				return true
			}

			if contentResult.Type == gjson.String {
				textType := "input_text"
				if role == "assistant" {
					textType = "output_text"
				}
				item := `{"type":"message","role":"","content":[]}`
				item, _ = sjson.Set(item, "role", role)
				part := `{"type":"","text":""}`
				part, _ = sjson.Set(part, "type", textType)
				part, _ = sjson.Set(part, "text", contentResult.String())
				item, _ = sjson.SetRaw(item, "content.-1", part)
				inputArr, _ = sjson.SetRaw(inputArr, "-1", item)
				return true
			}

			if !contentResult.IsArray() {
				return true
			}

			var msgParts []string
			for _, c := range contentResult.Array() {
				cType := c.Get("type").String()
				switch cType {
				case "text":
					textType := "input_text"
					if role == "assistant" {
						textType = "output_text"
					}
					part := `{"type":"","text":""}`
					part, _ = sjson.Set(part, "type", textType)
					part, _ = sjson.Set(part, "text", c.Get("text").String())
					msgParts = append(msgParts, part)

				case "image":
					source := c.Get("source")
					if source.Exists() {
						data := source.Get("data").String()
						mediaType := source.Get("media_type").String()
						if mediaType == "" {
							mediaType = "application/octet-stream"
						}
						if data != "" {
							part := `{"type":"input_image","image_url":""}`
							part, _ = sjson.Set(part, "image_url", "data:"+mediaType+";base64,"+data)
							msgParts = append(msgParts, part)
						}
					}

				case "document":
					source := c.Get("source")
					if source.Exists() {
						data := source.Get("data").String()
						mediaType := source.Get("media_type").String()
						if mediaType == "" {
							mediaType = "application/octet-stream"
						}
						if data != "" {
							part := `{"type":"input_file","file_data":""}`
							part, _ = sjson.Set(part, "file_data", "data:"+mediaType+";base64,"+data)
							msgParts = append(msgParts, part)
						}
					}

				case "tool_use":
					if len(msgParts) > 0 {
						item := `{"type":"message","role":"","content":[]}`
						item, _ = sjson.Set(item, "role", role)
						for _, p := range msgParts {
							item, _ = sjson.SetRaw(item, "content.-1", p)
						}
						inputArr, _ = sjson.SetRaw(inputArr, "-1", item)
						msgParts = nil
					}
					fc := `{"type":"function_call","call_id":"","name":"","arguments":""}`
					fc, _ = sjson.Set(fc, "call_id", c.Get("id").String())
					fc, _ = sjson.Set(fc, "name", c.Get("name").String())
					if inputRaw := c.Get("input"); inputRaw.Exists() {
						fc, _ = sjson.Set(fc, "arguments", inputRaw.Raw)
					}
					inputArr, _ = sjson.SetRaw(inputArr, "-1", fc)

				case "tool_result":
					if len(msgParts) > 0 {
						item := `{"type":"message","role":"","content":[]}`
						item, _ = sjson.Set(item, "role", role)
						for _, p := range msgParts {
							item, _ = sjson.SetRaw(item, "content.-1", p)
						}
						inputArr, _ = sjson.SetRaw(inputArr, "-1", item)
						msgParts = nil
					}
					fco := `{"type":"function_call_output","call_id":"","output":""}`
					fco, _ = sjson.Set(fco, "call_id", c.Get("tool_use_id").String())

					resultContent := c.Get("content")
					if resultContent.Type == gjson.String {
						fco, _ = sjson.Set(fco, "output", resultContent.String())
					} else if resultContent.IsArray() {
						var resultParts []string
						for _, rc := range resultContent.Array() {
							if txt := rc.Get("text").String(); txt != "" {
								resultParts = append(resultParts, txt)
							}
						}
						fco, _ = sjson.Set(fco, "output", strings.Join(resultParts, "\n"))
					} else if resultContent.Exists() {
						fco, _ = sjson.Set(fco, "output", resultContent.String())
					}
					inputArr, _ = sjson.SetRaw(inputArr, "-1", fco)

				case "thinking", "redacted_thinking":
				}
			}

			if len(msgParts) > 0 {
				item := `{"type":"message","role":"","content":[]}`
				item, _ = sjson.Set(item, "role", role)
				for _, p := range msgParts {
					item, _ = sjson.SetRaw(item, "content.-1", p)
				}
				inputArr, _ = sjson.SetRaw(inputArr, "-1", item)
			}

			return true
		})
	}

	out, _ = sjson.SetRaw(out, "input", inputArr)

	// Tools mapping: Claude tools -> Responses API tools
	if tools := root.Get("tools"); tools.Exists() && tools.IsArray() {
		var toolsJSON = "[]"
		tools.ForEach(func(_, tool gjson.Result) bool {
			name := tool.Get("name").String()
			description := tool.Get("description").String()

			if name == "" {
				return true // Skip tools without names
			}

			respToolJSON := `{"type":"function","name":"","description":"","parameters":{}}`
			respToolJSON, _ = sjson.Set(respToolJSON, "name", name)
			respToolJSON, _ = sjson.Set(respToolJSON, "description", description)

			if inputSchema := tool.Get("input_schema"); inputSchema.Exists() && inputSchema.IsObject() {
				respToolJSON, _ = sjson.SetRaw(respToolJSON, "parameters", inputSchema.Raw)
			}

			toolsJSON, _ = sjson.SetRaw(toolsJSON, "-1", respToolJSON)
			return true
		})

		if gjson.Parse(toolsJSON).IsArray() && len(gjson.Parse(toolsJSON).Array()) > 0 {
			out, _ = sjson.SetRaw(out, "tools", toolsJSON)
		}
	}

	if toolChoice := root.Get("tool_choice"); toolChoice.Exists() {
		switch toolChoice.Get("type").String() {
		case "auto":
			out, _ = sjson.Set(out, "tool_choice", "auto")
		case "any":
			out, _ = sjson.Set(out, "tool_choice", "required")
		case "tool":
			toolName := toolChoice.Get("name").String()
			if toolName != "" {
				tcJSON := `{"type":"function","function":{"name":""}}`
				tcJSON, _ = sjson.Set(tcJSON, "function.name", toolName)
				out, _ = sjson.SetRaw(out, "tool_choice", tcJSON)
			}
		default:
			out, _ = sjson.Set(out, "tool_choice", "auto")
		}
	}

	return []byte(out)
}
