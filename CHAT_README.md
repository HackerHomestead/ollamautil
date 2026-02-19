# Turn-Based Chat

A feature for **ollama-model-mgr** that lets two AI models debate or converse with each other in a split-view UI.

## Quick Start

```bash
python -m ollama_mgr.cli chat
```

You'll be prompted to:
1. Select Player 1 (left panel)
2. Select Player 2 (right panel)
3. Enter an initial prompt

Then watch them go back and forth!

## CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--player1` | `-1` | (prompted) | Player 1 model name |
| `--player2` | `-2` | (prompted) | Player 2 model name |
| `--prompt` | | (prompted) | Initial prompt to start the conversation |
| `--exchanges` | `-e` | 5 | Number of turns per model |
| `--timeout` | `-t` | 3600 | Timeout in seconds (1 hour) |

## Examples

### Same model vs itself

Compare how a model argues with itself:

```bash
python -m ollama_mgr.cli chat -1 llama3 -2 llama3 -e 3
```

### Two different models debate

```bash
python -m ollama_mgr.cli chat -1 llama3 -2 mistral -e 3
```

### With a controversial topic

```bash
python -m ollama_mgr.cli chat -1 deepseek-r1 -2 llama3 -p "Should artificial intelligence be regulated?"
```

### Longer conversation

```bash
python -m ollama_mgr.cli chat -1 llama3 -2 codellama -e 10 -t 7200
```

### Custom prompt

```bash
python -m ollama_mgr.cli chat -1 phi3 -2 mistral -p "What is the best programming language and why?"
```

## Controls

During the chat:

- **p** - Pause/resume the display
- **s** - Skip to the next turn
- **q** - Quit the session

## Output

Chat logs are saved to `/tmp` with timestamps:

```
/tmp/chat_20260219_143052.md
```

The log includes:
- Session start time
- Player model names
- Initial prompt
- Each exchange with relative timestamps (MM:SS)

Example log content:

```markdown
# Turn-Based Chat Session

**Start Time:** 2026-02-19T14:30:52.123456
**Player 1:** llama3
**Player 2:** mistral

---

## Initial Prompt (00:00)

What is better, Python or JavaScript?

---

### Exchange 1 — llama3 (00:03)

Python is better because...

### Exchange 1 — mistral (00:08)

JavaScript is better because...
```

## Post-Session Summary

When the chat ends, you'll be asked:

```
Would you like a summary of this conversation using another model?
```

If yes, select a model and enter a prompt like:

- "Who won this debate?"
- "Summarize each side's arguments"
- "Which model made better points?"

The summary model will analyze the full conversation and provide insights.

## How It Works

1. **Initial prompt** is sent to Player 1
2. Player 1's response is captured and sent to Player 2
3. Player 2's response is captured and sent back to Player 1
4. Repeat for the specified number of exchanges
5. Optionally, a third model summarizes the conversation

The conversation is built up as a message history, so each model sees the full context.
