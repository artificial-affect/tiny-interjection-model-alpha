from __future__ import annotations

import argparse
import subprocess
import time
import urllib.request
from pathlib import Path

from .inference import InteractionEngine, call_ollama_chat
from .schema import Event
from .train import DEFAULT_MODEL


def print_decision(engine: InteractionEngine, events: tuple[Event, ...]) -> None:
    decision = engine.decide(events)
    print(f'\n[{decision.mode}] action={decision.target.action}')
    for message in decision.target.messages:
        print(f'assistant> {message}')
    print('')


def ensure_ollama(url: str, executable: Path | None):
    for attempt in range(61):
        try:
            with urllib.request.urlopen(f'{url}/api/tags', timeout=1):
                return
        except OSError:
            if attempt == 0:
                if executable is None:
                    raise RuntimeError('Ollama is not running and no CLI was provided.')
                subprocess.Popen(
                    [str(executable), 'serve'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            time.sleep(0.5)
    raise RuntimeError('Ollama did not become ready within 30 seconds.')


def print_demo_decision(decision, response: str | None):
    if decision.target.action == 'wait':
        print('TIM: …')
        return
    if response:
        print(f'TIM: {response}')
        return
    if decision.target.messages:
        for message in decision.target.messages:
            print(f'TIM: {message}')
        return
    print(f'TIM: {decision.target.action.capitalize()}.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Interactive typed-chat demo for the interaction model.')
    parser.add_argument('--model-name', default=DEFAULT_MODEL)
    parser.add_argument('--adapter-dir', type=Path, default=Path('adapters/qwen3-4b-instruct-2507'))
    parser.add_argument('--base-only', action='store_true')
    parser.add_argument('--heuristic', action='store_true')
    parser.add_argument('--no-4bit', action='store_true')
    parser.add_argument('--llama-cli-path', type=Path, default=None)
    parser.add_argument('--llama-model-path', type=Path, default=None)
    parser.add_argument('--llama-adapter-path', type=Path, default=None)
    parser.add_argument('--ollama-model', default=None)
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11434')
    parser.add_argument('--ollama-cli-path', type=Path, default=None)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.ollama_model:
        ensure_ollama(args.ollama_url, args.ollama_cli_path)

    engine = InteractionEngine(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        base_only=args.base_only,
        heuristic=args.heuristic,
        use_4bit=not args.no_4bit,
        llama_cli_path=args.llama_cli_path,
        llama_model_path=args.llama_model_path,
        llama_adapter_path=args.llama_adapter_path,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
    )

    events: list[Event] = []
    last_at = time.monotonic()
    if args.demo:
        print('\033[2J\033[H', end='', flush=True)
    else:
        print('Type user messages. Blank line forces a decision. Use /assistant text, /reset, or /quit.')

    while True:
        raw = input('You: ' if args.demo else 'you> ')
        now = time.monotonic()
        dt_ms = max(0, int((now - last_at) * 1000))
        last_at = now

        if raw.strip() == '':
            if events and not args.demo:
                print_decision(engine, tuple(events))
            continue
        if raw.strip() == '/quit':
            break
        if raw.strip() == '/reset':
            events.clear()
            print('reset')
            continue
        if raw.startswith('/assistant '):
            events.append(
                Event(
                    role='assistant',
                    text=raw[len('/assistant ') :].strip(),
                    dt_ms=dt_ms,
                )
            )
        else:
            events.append(Event(role='user', text=raw.strip(), dt_ms=dt_ms))

        if args.demo:
            decision = engine.decide(tuple(events))
            response = None
            if decision.target.action != 'wait' and args.ollama_model:
                response = call_ollama_chat(
                    url=args.ollama_url,
                    model=args.ollama_model,
                    events=tuple(events),
                )
            print_demo_decision(decision, response)
            if response:
                events.append(Event(role='assistant', text=response, dt_ms=0))
            else:
                for message in decision.target.messages:
                    events.append(Event(role='assistant', text=message, dt_ms=0))
            last_at = time.monotonic()


if __name__ == '__main__':
    main()
