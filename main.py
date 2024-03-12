import xmltodict
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from prompt_constructors import *

app = FastAPI()


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    print("REQUEST BODY: ", body)
    return await call_next(request)


@app.get("/models")
async def list_models() -> JSONResponse:
    return JSONResponse(content={"data": [
        {"id": "claude-3-sonnet-20240229", "name": "Anthropic Claude 3 Sonnet"},
        {"id": "anthropic.claude-3-sonnet-20240229-v1:0", "name": "AWS Bedrock Anthropic Claude 3 Sonnet"},
        {"id": "claude-3-opus-20240229", "name": "Anthropic Claude 3 Opus"},
    ]
    })


@app.post("/chat/completions")
async def completions(request: Request) -> StreamingResponse:
    system: str | None = ""
    data = await request.body()
    data = json.loads(data)
    max_tokens = 4096
    try:
        tools = data["tools"]
        system += construct_tool_use_system_prompt(tools)
    except KeyError as e:
        print("an error happened with tools - key 'tools' not present ")
        tools = []

    messages = data["messages"]
    tool_inputs_xml = ""
    tool_outputs_xml = ""
    for message in messages:
        try:
            if 'role' in message.keys() and message["role"] == "system":
                system += message["content"] + "\n"
        except Exception as e:
            print("an error happened with system message: ", e)

        try:
            if 'role' in message.keys() and message["role"] == "tool":
                tool_outputs_xml = construct_tool_outputs_message([message], None)
                # message["role"] = "delete"
        except Exception as e:
            print("an error happened mapping tool response: ", e)

        try:
            if 'role' in message.keys() and message["role"] == "assistant":
                tool_inputs = []
                for tool_call in message["tool_calls"]:
                    tool_inputs.append({
                        "tool_name": tool_call["function"]["name"],
                        "tool_arguments": tool_call["function"]["arguments"],
                    })
                tool_inputs_xml = construct_tool_inputs_message(message["content"], tool_inputs)
                print("PRINT TOOL INPUTS: ", tool_inputs)
        #                 message["role"] = "delete"

        except Exception as e:
            print("an error happened mapping tool_calls: ", e)

    messages = [d for d in messages if d.get('role') != 'system']
    messages = [d for d in messages if d.get('role') != 'assistant']
    messages = [d for d in messages if d.get('role') != 'tool']

    if tool_inputs_xml != "":
        content = tool_inputs_xml
        if tool_outputs_xml != "":
            content += "\n" + tool_outputs_xml
        print("LIST COMPLETE TOOL MESSAGE: ", content)
        messages.append({
            "role": "assistant",
            "content": content,
        })

    # print("SYSTEM: ", system)
    print("MESSAGES: ", messages)

    if data["model"].startswith("anthropic."):
        client = AsyncAnthropicBedrock()
    else:
        client = AsyncAnthropic()

    async with client.messages.stream(
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            model=data["model"],
            temperature=data["temperature"],
            stop_sequences=["</function_calls>"],
    ) as stream:
        accumulated = await stream.get_final_message()
        print("accumulated response: ", accumulated.model_dump_json())

    response = "data: " + translate_response(accumulated.json()) + "\n\n"

    return StreamingResponse(response, media_type="application/x-ndjson")


def translate_response(response) -> str:
    data = json.loads(response)
    finish_reason = None
    parsed_tool_calls = []
    for message in data["content"]:
        if message["text"].startswith("<function_calls>"):
            xml_tool_calls = message["text"] + "</function_calls>"
            tool_calls = xmltodict.parse(xml_tool_calls)
            if tool_calls["function_calls"]["invoke"] is list:
                for key, value in tool_calls["function_calls"]["invoke"].items():
                    parsed_tool_calls.append({
                        "index": 0,
                        "id": value['tool_name'],
                        "type": "function",
                        "function": {
                            "name": value["tool_name"],
                            "arguments": str(value["parameters"]),
                        },
                    })
            else:
                parsed_tool_calls.append({
                    "index": 0,
                    "id": tool_calls["function_calls"]["invoke"]["tool_name"],
                    "type": "function",
                    "function": {
                        "name": tool_calls["function_calls"]["invoke"]["tool_name"],
                        "arguments": json.dumps(tool_calls["function_calls"]["invoke"]["parameters"]),
                    },
                })

            print("PRINT PARSED TOOL_CALLS: ", parsed_tool_calls)
            message.pop("text", None)
            message.pop("type", None)
            message["tool_calls"] = parsed_tool_calls
            message["content"] = None
            message["role"] = "assistant"

        try:
            message["content"] = message["text"]
        except KeyError:
            print("No text field on this message: ", message)

    if "stop_reason" in data.keys() and data["stop_reason"] == "stop_sequence":
        finish_reason = "tool_calls"

    if "stop_reason" in data.keys() and data["stop_reason"] == "end_turn":
        finish_reason = "stop"

    print("RESPONSE: ", data)

    translated = {
        "id": data["id"],
        "object": "chat.completion.chunk",
        "created": 0,
        "model": data["model"],
        "system_fingerprint": "TEMP",
        "choices": [
            {
                "index": 0,
                "delta": data["content"][0],
            },
        ],
        "finish_reason": finish_reason,
    }

    print("TRANSLATED RESPONSE: ", translated)

    return json.dumps(translated)
