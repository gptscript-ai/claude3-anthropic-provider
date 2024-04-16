import json
import os
import subprocess
import sys

import boto3

tool_input = {
    "message": "Please pick a model provider: AWS or Anthropic.",
    "fields": "provider",
}
command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt", json.dumps(tool_input)]
result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)
if result.returncode != 0:
    print("Failed to run sys.prompt.", file=sys.stderr)
    sys.exit(1)

try:
    resp = json.loads(result.stdout.strip())
    provider = resp["provider"].lower()

except json.JSONDecodeError:
    print("Failed to decode JSON.", file=sys.stderr)
    sys.exit(1)

if provider == 'aws':
    client = boto3.client('sts')
    try:
        response = client.get_caller_identity()
        output = {
            "env": {
                "AWS": 'true',
            }
        }
        print(json.dumps(output))
        sys.exit(0)

    except:
        print("Please authenticate with AWS.", file=sys.stderr)
        sys.exit(1)

elif provider == 'gcp':
    # TODO: some command to check if GCP creds are available
    output = {
        "env": {
            "GCP": 'true',
        }
    }
    print(json.dumps(output))
    sys.exit(0)

elif provider == 'anthropic':
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
