import os
import xmltodict
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from prompt_constructors import *

debug = os.environ.get("GPTSCRIPT_DEBUG", "false") == "true"
app = FastAPI()


def log(*args):
    if debug:
        print(*args)


@app.middleware("http")
async def log_body(request: Request, call_next):
    body = await request.body()
    log("HTTP REQUEST BODY: ", body)
    return await call_next(request)


@app.post("/")
async def post_root():
    return 'ok'


@app.get("/")
async def get_root():
    return 'ok'


# The anthropic API does not have a method to list models, so we are hard coding the models here
@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(content={"data": [
        {"id": "claude-3-opus-20240229", "name": "Anthropic Claude 3 Opus"},
        {"id": "claude-3-sonnet-20240229", "name": "Anthropic Claude 3 Sonnet"},
        {"id": "claude-3-haiku-20240307", "name": "Anthropic Claude 3 Haiku"},
        {"id": "anthropic.claude-3-sonnet-20240229-v1:0", "name": "AWS Bedrock Anthropic Claude 3 Sonnet"},
        {"id": "anthropic.claude-3-haiku-20240307-v1:0", "name": "AWS Bedrock Anthropic Claude 3 Haiku"},
    ]})


def map_req(req: dict) -> dict:
    system: str | None = ""

    max_tokens = 4096
    if 'max_tokens' in req.keys():
        max_tokens = req["max_tokens"]

    if "tools" in req.keys():
        system += construct_tool_use_system_prompt(req["tools"])

    messages = req["messages"]

    tool_inputs_xml: list[str] = []
    tool_outputs_xml: list[str] = []
    for message in messages:
        if 'role' in message.keys() and message["role"] == "system":
            system += message["content"] + "\n"

        if 'role' in message.keys() and message["role"] == "tool":
            tool_outputs_xml.append(construct_tool_outputs_message([message], None))

        if 'role' in message.keys() and message["role"] == "assistant":
            tool_inputs = []
            for tool_call in message["tool_calls"]:
                tool_inputs.append({
                    "tool_name": tool_call["function"]["name"],
                    "tool_arguments": tool_call["function"]["arguments"],
                })
            tool_inputs_xml.append(construct_tool_inputs_message(message["content"], tool_inputs))

    messages = [d for d in messages if d.get('role') != 'system']
    messages = [d for d in messages if d.get('role') != 'assistant']
    messages = [d for d in messages if d.get('role') != 'tool']

    if tool_inputs_xml != []:
        content: str = '\n'.join(tool_inputs_xml)
        messages.append({
            "role": "assistant",
            "content": content,
        })
        if tool_outputs_xml != []:
            content: str = '\n'.join(tool_outputs_xml)
            messages.append({
                "role": "user",
                "content": content,
            })

    mapped_req = {
        "messages": messages,
        "max_tokens": max_tokens,
        "system": system,
        "model": req["model"],
        "temperature": req["temperature"],
    }
    return mapped_req


@app.post("/v1/chat/completions")
async def completions(request: Request) -> StreamingResponse:
    data = await request.body()
    req = map_req(json.loads(data))

    if req["model"].startswith("anthropic."):
        log('here')
        client = AsyncAnthropicBedrock()
    else:
        client = AsyncAnthropic()

    log("MESSAGES: ", req["messages"])
    async with client.messages.stream(
            max_tokens=req["max_tokens"],
            system=req["system"],
            messages=req["messages"],
            model=req["model"],
            temperature=req["temperature"],
            stop_sequences=["</function_calls>"],
    ) as stream:
        accumulated = await stream.get_final_message()

    resp = "data: " + map_resp(accumulated.json()) + "\n\n"
    return StreamingResponse(resp, media_type="application/x-ndjson")


def map_resp(response) -> str:
    data = json.loads(response)
    finish_reason = None
    parsed_tool_calls = []

    log("INITIAL DATA: ", data)

    for message in data["content"]:
        if 'text' in message.keys() and message["text"].lstrip().startswith("<function_calls>"):
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

            message.pop("text", None)
            message.pop("type", None)
            message["tool_calls"] = parsed_tool_calls
            message["content"] = None
            message["role"] = "assistant"

        if 'text' in message.keys():
            message["content"] = message["text"]

    if "stop_reason" in data.keys() and data["stop_reason"] == "stop_sequence":
        finish_reason = "tool_calls"

    if "stop_reason" in data.keys() and data["stop_reason"] == "end_turn":
        finish_reason = "stop"
    log("DATA: ", data)
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

    return json.dumps(translated)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
                log_level="debug" if debug else "critical", reload=debug, access_log=debug)
