* You need either ANTHROPIC_API_KEY or AWS credentials configured, depending on which model you want to use.

```
export ANTHROPIC_API_KEY=<your-api-key>
```

Run using the following commands

```
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
./run.sh
```

```
export OPENAI_BASE_URL=http://127.0.0.1:8000

# one of the anthropic-hosted models
gptscript --default-model=claude-3-opus-20240229 examples/bob.gpt

# Claude on Bedrock
gptscript --default-model=anthropic.claude-3-sonnet-20240229-v1:0 examples/bob.gpt
```