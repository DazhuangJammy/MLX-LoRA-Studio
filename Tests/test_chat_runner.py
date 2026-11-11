import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('chat_runner', Path(__file__).parents[1] / 'Backend/chat_runner.py')
chat = importlib.util.module_from_spec(spec)
# Import does not load MLX, but preserves stdout for the worker protocol.
import sys
stdout = sys.stdout
spec.loader.exec_module(chat)
sys.stdout = stdout


class ChatValidationTests(unittest.TestCase):
    def test_cache_prefix_never_reuses_changed_or_final_input_token(self):
        self.assertEqual(chat.shared_prefix_length([1, 2, 3, 4], [1, 2, 9]), 2)
        self.assertEqual(chat.shared_prefix_length([1, 2], [1, 2]), 1)
        self.assertEqual(chat.shared_prefix_length([1, 2], [1, 2, 3]), 2)
        self.assertEqual(chat.shared_prefix_length([1, 2], [9]), 0)

    def test_clearing_cache_discards_tokens_without_unloading_model(self):
        worker = chat.Worker()
        worker.model = object()
        model = worker.model
        worker.prompt_cache = ['old cache']
        worker.cache_tokens = [1, 2, 3]
        worker.reset_cache()
        self.assertIs(worker.model, model)
        self.assertIsNone(worker.prompt_cache)
        self.assertEqual(worker.cache_tokens, [])

    def test_hf_snapshot_matches_repository(self):
        self.assertEqual(chat.canonical_model('Qwen/Qwen3-0.6B'), 'Qwen/Qwen3-0.6B')
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder) / 'models--Qwen--Qwen3-0.6B' / 'snapshots' / 'abc'
            snapshot.mkdir(parents=True)
            self.assertEqual(chat.canonical_model(str(snapshot)), 'Qwen/Qwen3-0.6B')

    def test_wrong_base_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            adapters = Path(folder) / 'adapters'
            adapters.mkdir()
            (adapters / 'adapter_config.json').write_text('{}')
            (adapters / 'adapters.safetensors').touch()
            (Path(folder) / 'run_spec.json').write_text(json.dumps({'model': 'Qwen/Qwen3-0.6B', 'load_in_4bits': True}))
            with self.assertRaisesRegex(ValueError, 'trained on'):
                chat.validate_adapter('other/model', str(adapters))
            loaded = chat.validate_adapter('Qwen/Qwen3-0.6B', str(adapters))
            self.assertEqual(chat.quantization_for(loaded), {'bits': 4, 'group_size': 128, 'mode': 'affine'})

    def test_incomplete_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'adapter_config.json'):
                chat.validate_adapter('Qwen/Qwen3-0.6B', folder)

    def test_context_limit_ignores_tokenizer_infinity(self):
        class Tokenizer:
            model_max_length = 10**30
        self.assertEqual(chat.context_limit({'max_position_embeddings': 40960}, Tokenizer()), 40960)
        self.assertEqual(chat.context_limit({}, Tokenizer()), 4096)

    def test_thinking_switch_must_change_prompt(self):
        class Ignored:
            chat_template = 'enable_thinking'
            def apply_chat_template(self, *args, **kwargs):
                return 'same'
        class Supported(Ignored):
            def apply_chat_template(self, *args, **kwargs):
                return str(kwargs['enable_thinking'])
        self.assertFalse(chat.thinking_switch(Ignored()))
        self.assertTrue(chat.thinking_switch(Supported()))


if __name__ == '__main__':
    unittest.main()
