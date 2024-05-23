import json
import os
import subprocess
import sys

if 'ANTHROPIC_API_KEY' not in os.environ:
    tool_input = {
        "message": "Please enter your Anthropic API token.",
        "fields": "token",
        "sensitive": "true",
    }
    command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt", json.dumps(tool_input)]
    result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Failed to run sys.prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        resp = json.loads(result.stdout.strip())
        token = resp["token"]

    except json.JSONDecodeError:
        print("Failed to decode JSON.", file=sys.stderr)
        sys.exit(1)
else:
    token = os.environ['ANTHROPIC_API_KEY']

output = {
    "env": {
        "ANTHROPIC_API_KEY": token,
    }
}

print(json.dumps(output))
