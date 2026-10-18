import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interaction_models.inference import (
    InteractionEngine,
    call_llama_cli,
    call_ollama,
    call_ollama_chat,
)
from interaction_models.parser import TargetParseError
from interaction_models.schema import Event


class LlamaInferenceTest(unittest.TestCase):
    @patch('subprocess.run')
    def test_call_llama_cli_extracts_decision(self, run):
        run.return_value = SimpleNamespace(stdout='loading\n<act>wait</act>\n<done/>\nexiting\n')

        raw = call_llama_cli(
            executable=Path('llama-cli'),
            model_path=Path('base.gguf'),
            adapter_path=Path('adapter.gguf'),
            prompt='events',
            max_new_tokens=99,
        )

        command = run.call_args.args[0]
        self.assertIn('--lora', command)
        self.assertIn('adapter.gguf', command)
        self.assertIn('4096', command)
        self.assertEqual(raw, '<act>wait</act>\n<done/>')

    @patch('interaction_models.inference.call_llama_cli')
    def test_engine_uses_llama_cli_without_loading_transformers(self, llama_cli):
        llama_cli.return_value = '<act>interject</act>\n<msg>Switching now.</msg>\n<done/>'
        engine = InteractionEngine(
            llama_cli_path=Path('llama-cli'),
            llama_model_path=Path('base.gguf'),
            llama_adapter_path=Path('adapter.gguf'),
        )

        decision = engine.decide(
            (
                Event(role='assistant', text='Starting the first version.', dt_ms=0),
                Event(role='user', text='Wait, make it shorter.', dt_ms=400),
            )
        )

        self.assertEqual(decision.mode, 'llama.cpp')
        self.assertEqual(decision.target.action, 'interject')
        self.assertEqual(decision.target.messages, ('Switching now.',))

    @patch('urllib.request.urlopen')
    def test_call_ollama_requests_raw_decision(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = b'{"response":"<act>wait</act>\\n<done/>"}'

        raw = call_ollama(
            url='http://127.0.0.1:11434',
            model='tim',
            prompt='events',
            max_new_tokens=99,
        )

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertTrue(body['raw'])
        self.assertFalse(body['think'])
        self.assertEqual(body['model'], 'tim')
        self.assertEqual(raw, '<act>wait</act>\n<done/>')

    @patch('urllib.request.urlopen')
    def test_call_ollama_chat_combines_consecutive_user_messages(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = b'{"message":{"content":"Here is the answer."}}'

        content = call_ollama_chat(
            url='http://127.0.0.1:11434',
            model='tim',
            events=(
                Event(role='user', text='Can you explain', dt_ms=0),
                Event(role='user', text='how this works?', dt_ms=500),
            ),
        )

        request = urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, 'http://127.0.0.1:11434/api/chat')
        self.assertEqual(
            body['messages'],
            [
                {
                    'role': 'system',
                    'content': (
                        "Answer the user's request directly in at most three sentences "
                        'unless the user explicitly asks for a longer format.'
                    ),
                },
                {'role': 'user', 'content': 'Can you explain\nhow this works?'},
            ],
        )
        self.assertEqual(content, 'Here is the answer.')

    @patch('interaction_models.inference.call_ollama')
    def test_engine_uses_ollama_without_loading_transformers(self, ollama):
        ollama.return_value = '<act>respond</act>\n<msg>Hello.</msg>\n<done/>'
        engine = InteractionEngine(ollama_model='tim')

        decision = engine.decide((Event(role='user', text='Hello', dt_ms=0),))

        self.assertEqual(decision.mode, 'ollama')
        self.assertEqual(decision.target.action, 'respond')
        self.assertEqual(decision.target.messages, ('Hello.',))

    @patch('interaction_models.inference.call_ollama')
    def test_engine_does_not_disguise_invalid_ollama_output_as_wait(self, ollama):
        ollama.return_value = 'ordinary chatbot output'
        engine = InteractionEngine(ollama_model='tim')

        with self.assertRaises(TargetParseError):
            engine.decide((Event(role='user', text='Hello', dt_ms=0),))


if __name__ == '__main__':
    unittest.main()
