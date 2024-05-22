# Claude3 Anthropic Provider

## Usage Example

```
gptscript --default-model='claude-3-haiku-20240307 from github.com/gptscript-ai/claude3-anthropic-provider' examples/helloworld.gpt
```

The Claude3 Bedrock provider can be found [here](https://github.com/gptscript-ai/claude3-bedrock-provider)

## Development

* You need an ANTHROPIC_API_KEY set in your environment.

```
export ANTHROPIC_API_KEY=<your-api-key>
```

Run using the following commands

```
python -m venv .venv
source ./.venv/bin/activate
pip install --upgrade -r requirements.txt
./run.sh
```

```
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export GPTSCRIPT_DEBUG=true

gptscript --default-model=claude-3-opus-20240229 examples/bob.gpt
```
