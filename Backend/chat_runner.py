"""Local chat worker. JSONL over stdio; all MLX work stays on the main thread."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import queue
import sys
import threading

# Library output must never corrupt the protocol (including during imports).
PROTOCOL = sys.stdout


def emit(kind, **values):
    PROTOCOL.write(json.dumps(dict(type=kind, **values), ensure_ascii=False) + '\n')
    PROTOCOL.flush()


def canonical_model(value):
    path = Path(value).expanduser()
    if path.exists():
        for parent in (path, *path.parents):
            if parent.name.startswith('models--'):
                return parent.name.removeprefix('models--').replace('--', '/')
        return str(path.resolve())
    return value.strip().rstrip('/')


def read_run(adapter):
    path = Path(adapter).expanduser().parent / 'run_spec.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def quantization_for(spec):
    for key, bits, group, mode in [
        ('load_in_4bits', 4, 128, 'affine'),
        ('load_in_6bits', 6, 128, 'affine'),
        ('load_in_8bits', 8, 128, 'affine'),
        ('load_in_mxfp4', 4, 32, 'mxfp4'),
    ]:
        if spec.get(key):
            return dict(bits=bits, group_size=group, mode=mode)
    return None


def validate_adapter(model, adapter):
    folder = Path(adapter).expanduser()
    for name in ('adapter_config.json', 'adapters.safetensors'):
        if not (folder / name).is_file():
            raise ValueError(f'LoRA folder is missing {name}. Select the adapters folder of a completed run.')
    spec = read_run(adapter)
    if spec.get('model') and canonical_model(spec['model']) != canonical_model(model):
        raise ValueError(f"This LoRA was trained on {spec['model']}. Select that base model first.")
    if spec.get('model_family', 'text') != 'text':
        raise ValueError('Chat currently supports text models, not vision-language adapters.')
    return spec


def context_limit(config, tokenizer):
    candidates = [config.get('max_position_embeddings'), config.get('text_config', {}).get('max_position_embeddings'), getattr(tokenizer, 'model_max_length', None)]
    limits = [n for n in candidates if isinstance(n, int) and 0 < n < 10_000_000]
    return min(limits) if limits else 4096


def thinking_switch(tokenizer):
    # Only offer a switch if the tokenizer actually changes its prompt for it.
    template = getattr(tokenizer, 'chat_template', '') or ''
    if 'enable_thinking' not in str(template):
        return False
    messages = [{'role': 'user', 'content': 'Hello'}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True) != tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return False


def shared_prefix_length(previous, current):
    # Always evaluate at least one prompt token to obtain next-token logits.
    limit = min(len(previous), max(0, len(current) - 1))
    for i in range(limit):
        if previous[i] != current[i]:
            return i
    return limit


class Worker:
    def __init__(self):
        self.model = self.tokenizer = None
        self.limit = 4096
        self.can_think = False
        self.prompt_cache = None
        self.cache_tokens = []

    def reset_cache(self):
        self.prompt_cache = None
        self.cache_tokens = []

    def prepare_cache(self, tokens):
        from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache
        reused = 0
        if self.prompt_cache is not None:
            offsets = [getattr(c, 'offset', None) for c in self.prompt_cache]
            if offsets and all(n == len(self.cache_tokens) for n in offsets) and can_trim_prompt_cache(self.prompt_cache):
                reused = shared_prefix_length(self.cache_tokens, tokens)
                trim_prompt_cache(self.prompt_cache, len(self.cache_tokens) - reused)
            else:
                self.reset_cache()
        if self.prompt_cache is None or reused == 0:
            self.prompt_cache = make_prompt_cache(self.model)
            reused = 0
        self.cache_tokens = tokens[:reused]
        return reused

    def load(self, request):
        import mlx.core as mx
        import mlx.nn as nn
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        from mlx_lm.tuner.utils import load_adapters

        self.model = self.tokenizer = None
        self.reset_cache()
        gc.collect()
        mx.clear_cache()
        model_id = request['model'].strip()
        adapter = request.get('adapter', '')
        spec = validate_adapter(model_id, adapter) if adapter else {}
        quant = quantization_for(spec) if adapter else request.get('quantization')
        local = Path(model_id).expanduser()
        if not local.is_dir():
            # Prefer the actual local snapshot, avoiding needless HF requests.
            try:
                local = Path(snapshot_download(model_id, local_files_only=True))
            except Exception:
                emit('status', message='Downloading model from Hugging Face…')
                local = Path(snapshot_download(model_id, allow_patterns=['*.json', '*.safetensors', '*.model', '*.tiktoken', '*.txt', '*.jinja']))
        elif (local / 'snapshots').is_dir():
            ref = local / 'refs/main'
            if not ref.is_file():
                raise ValueError('Select a model snapshot containing config.json.')
            local = local / 'snapshots' / ref.read_text().strip()
        emit('status', message='Loading model into memory…')
        model, tokenizer, config = load(str(local), return_config=True, lazy=True)
        if quant and not config.get('quantization') and not config.get('quantization_config'):
            nn.quantize(model, **quant)
        if adapter:
            emit('status', message='Applying LoRA…')
            model = load_adapters(model, str(Path(adapter).expanduser()))
        model.eval()
        mx.eval(model.parameters())
        self.model, self.tokenizer = model, tokenizer
        self.limit = context_limit(config, tokenizer)
        self.can_think = thinking_switch(tokenizer)
        emit('loaded', thinking_supported=self.can_think, context_limit=self.limit,
             model_path=str(local), quantization=quant,
             message='Ready · LoRA loaded' if adapter else 'Ready · Base model')

    def generate(self, request, cancel):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if self.model is None:
            raise ValueError('Load a model before sending a message.')
        messages = request.get('messages', [])
        if not messages or any(m.get('role') not in ('user', 'assistant') or not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Messages must contain user and assistant text.')
        maximum = request.get('max_tokens', 1024)
        if not isinstance(maximum, int) or not 1 <= maximum <= 32768:
            raise ValueError('Maximum output must be between 1 and 32,768 tokens.')
        kwargs = {'enable_thinking': bool(request.get('thinking'))} if self.can_think else {}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) + maximum > self.limit:
            available = max(0, self.limit - len(tokens))
            raise ValueError(f'Conversation has {len(tokens):,} tokens; model context is {self.limit:,}. Lower Max output to {available:,} or start a new chat.')
        # Some templates put the opening reasoning tag in the prompt itself.
        without_generation = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kwargs)
        suffix = prompt[len(without_generation):] if prompt.startswith(without_generation) else ''
        open_thought = suffix.rfind('<think>') > suffix.rfind('</think>')
        reused = self.prepare_cache(tokens)
        emit('started', prefix='<think>' if open_thought else '', prompt_tokens=len(tokens), cached_tokens=reused)
        mx.random.seed(0)
        last = None
        thinking = bool(request.get('thinking')) and self.can_think
        sampler = make_sampler(temp=0.6 if thinking else 0.7,
                               top_p=0.95 if thinking else 0.8, top_k=20)
        generated_tokens = []
        iterator = stream_generate(self.model, self.tokenizer, tokens[reused:], max_tokens=maximum,
                                   sampler=sampler, prefill_step_size=256, prompt_cache=self.prompt_cache)
        try:
            for response in iterator:
                generated_tokens.append(response.token)
                if cancel.is_set():
                    break
                last = response
                emit('delta', text=response.text)
        finally:
            iterator.close()
            from mlx_lm.models.cache import can_trim_prompt_cache
            known_tokens = tokens + generated_tokens
            offsets = [getattr(c, 'offset', None) for c in (self.prompt_cache or [])]
            if offsets and all(n == len(known_tokens) for n in offsets) and can_trim_prompt_cache(self.prompt_cache):
                self.cache_tokens = known_tokens
            else:
                # Recurrent / non-trimmable models still receive full history;
                # only reuse the KV state when its exact token boundary is known.
                self.reset_cache()
            mx.clear_cache()
        emit('done', stopped=cancel.is_set(), tokens=last.generation_tokens if last else 0,
             finish_reason='stopped' if cancel.is_set() else getattr(last, 'finish_reason', 'stop'))


def main():
    sys.stdout = sys.stderr
    pending = queue.Queue()
    cancel = threading.Event()
    def read_commands():
        nonlocal cancel
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get('command') == 'stop':
                    cancel.set()
                else:
                    if request.get('command') == 'generate':
                        cancel = threading.Event()
                        request['_cancel'] = cancel
                    pending.put(request)
            except Exception as exc:
                emit('error', message=f'Invalid command: {exc}')
        cancel.set()
        pending.put({'command': 'shutdown'})
    threading.Thread(target=read_commands, daemon=True).start()
    worker = Worker()
    while True:
        request = pending.get()
        command = request.get('command')
        if command == 'shutdown':
            break
        try:
            if command == 'load':
                worker.load(request)
            elif command == 'clear_cache':
                worker.reset_cache()
                import mlx.core as mx
                mx.clear_cache()
                emit('cache_cleared')
            elif command == 'generate':
                worker.generate(request, request['_cancel'])
            else:
                raise ValueError(f'Unknown command: {command}')
        except Exception as exc:
            if command == 'generate':
                worker.reset_cache()
            emit('error', message=f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
