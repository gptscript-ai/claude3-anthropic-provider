import asyncio
import json
import os

from gptscript.gptscript import GPTScript
from gptscript.opts import Options

gptscript = GPTScript()


async def prompt(tool_input) -> dict:
    run = gptscript.run(
        tool_path="sys.prompt",
        opts=Options(
            input=json.dumps(tool_input),
        )
    )
    output = await run.text()
    gptscript.close()
    return json.loads(output)


def main():
    if 'ANTHROPIC_API_KEY' in os.environ:
        token = os.environ['ANTHROPIC_API_KEY']
    else:
        tool_input = {
            "message": "Please enter your Anthropic API token.",
            "fields": "token",
            "sensitive": "true",
        }
        result = asyncio.run(prompt(tool_input))
        token = result["token"]

    output = {
        "env": {
            "ANTHROPIC_API_KEY": token,
        }
    }
    print(json.dumps(output))


if __name__ == '__main__':
    main()
