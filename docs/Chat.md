# Local Chat

Chat sits between **Live Metrics** and **Upload to HF**. It uses the Python environment selected in Settings and runs text-model inference on this Mac.

1. Choose a downloaded **Model**. The list includes the Hugging Face cache, saved custom paths and models referenced by training runs. **Browse…** starts in the local model cache; select a model folder containing `config.json` or a Hugging Face model cache directory.
2. Choose a **LoRA**. The three most recent completed runs for that base model are listed newest first. Older adapters remain available through Browse… if retained in an archive. **Browse…** starts in the configured run output folder; select the `adapters` folder containing `adapter_config.json` and `adapters.safetensors`. Choose **None — base model** to compare the original model.
3. Click **Load model**. A model not already cached may download from Hugging Face. The worker restores Studio's recorded training quantization when loading its adapter; the base-model comparison uses the latest matching run's quantization too. A known mismatched base model is rejected.
4. Enable **Thinking** if the loaded tokenizer supports a real on/off switch. Some reasoning models do not provide one. Thinking appears in a collapsible section. The switch permits thinking; it cannot force a model to produce it or improve factual accuracy.
5. Set **Max output**, then enter a message and click **Send** or press **Command–Return**. The limit includes both thinking and the answer, up to 32,768 output tokens and within the model's context size. Chat checks conversation length before generation and explains when a new chat or smaller limit is needed.
6. **Stop** cancels after the current inference step. **Clear chat** clears the current conversation, saved history, draft, and worker context cache. **Unload** releases the model. Changing the model or adapter requires **Reload**, which preserves the conversation and rebuilds the model cache. Switching sidebar pages or restarting the app preserves the current chat. History is saved locally at `~/.mlxlorastudio/chat/history.json`; it is removed from the active conversation only when explicitly cleared.

The model receives the conversation history on each turn. Compatible models reuse the exact shared token prefix in memory, trimming divergent cache entries; models with non-trimmable caches reprocess the full history. The page reports the last prompt token count and reused token count. Reloading a model resets only this inference cache, not the saved text.

Use the same new question with and without LoRA to compare results. Clear the chat between independent comparisons so previous answers do not influence the next model. A successful run or lower loss is not evidence that every answer is correct. Chat does not insert training answers or retrieval context into prompts.

## Recovery

- If Python cannot import MLX, select the intended environment in Settings and install dependencies via **Package Updates → Update**.
- Loading failures appear above the conversation, with expandable diagnostics.
- Failed or stopped requests with neither answer nor thinking content remove the incomplete turn. If the composer is empty, the unanswered prompt is restored there for retry; a newly typed draft is preserved.
- Training and Chat share GPU memory. Starting a training run unloads Chat; loading or sending in Chat is disabled while training runs.
- Manual adapters without a Studio run specification lack reliable base-model/quantization metadata. Select the correct base model; tensor shape validation still runs, but cannot prove that identically shaped models share the same weights.

## Implementation and checks

`Backend/chat_runner.py` maintains one model per subprocess and exchanges JSONL over stdin/stdout; library logs go to stderr. MLX operations stay on its main thread, while a reader thread handles stop requests. Native state lives in `ChatSession`, independent of view navigation. Responses stream to SwiftUI without adding a network listener.

Checks: `python -m unittest discover -s Tests -p test_chat_runner.py` and `swift test`. On machines whose default macOS SDK is missing SwiftUI compiler macros, select an installed compatible SDK with `SDKROOT` when building, as with other Studio pages.
