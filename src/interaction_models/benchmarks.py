from __future__ import annotations

from dataclasses import dataclass

from .schema import Event, Target


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    events: tuple[Event, ...]
    target: Target
    expected_contains: tuple[str, ...] = ()


def _user(text: str, dt_ms: int = 0, *, partial: bool = False) -> Event:
    return Event(role='user', text=text, dt_ms=dt_ms, partial=partial)


def _assistant(text: str, dt_ms: int) -> Event:
    return Event(role='assistant', text=text, dt_ms=dt_ms)


def _case(
    case_id: str, action: str, *events: Event, messages: tuple[str, ...] = (), expected_contains: tuple[str, ...] = ()
) -> BenchCase:
    return BenchCase(
        case_id=case_id,
        events=events,
        target=Target(action=action, messages=messages),  # type: ignore[arg-type]
        expected_contains=expected_contains,
    )


DOUBLE_TEXT_BENCH: tuple[BenchCase, ...] = (
    _case(
        'warm-revision',
        'respond',
        _user('can you help write this'),
        _user('make it warmer actually', 1300),
        messages=("Got it - warmer. Here's a version that keeps the ask clear.",),
        expected_contains=('warmer',),
    ),
    _case(
        'second-detail',
        'respond',
        _user('write a follow up'),
        _user("mention that I'm free Thursday afternoon", 1800),
        messages=("Absolutely - I'll include Thursday afternoon as the availability window.",),
        expected_contains=('Thursday',),
    ),
    _case(
        'customer-email-beta',
        'respond',
        _user('draft a customer email'),
        _user('include the beta waitlist', 900),
        messages=("I'll include the beta waitlist in the customer email.",),
        expected_contains=('beta',),
    ),
    _case(
        'launch-note-pricing',
        'respond',
        _user('draft a launch note'),
        _user('say the pricing is unchanged', 2200),
        messages=("I'll keep the pricing unchanged in the launch note.",),
        expected_contains=('pricing',),
    ),
    _case(
        'support-reply-fix',
        'respond',
        _user('make a support reply'),
        _user('say we already shipped the fix', 650),
        messages=("I'll mention that the fix already shipped.",),
        expected_contains=('fix',),
    ),
    _case(
        'status-update-dashboard',
        'respond',
        _user('make a crisp status update'),
        _user('mention the new dashboard', 2900),
        messages=("I'll include the new dashboard.",),
        expected_contains=('dashboard',),
    ),
    _case(
        'recruiting-note-warmer',
        'respond',
        _user('write a recruiting note'),
        _user('keep the ask very clear, actually make it warmer', 1200),
        messages=("I'll make it warmer while keeping the ask clear.",),
        expected_contains=('warmer',),
    ),
    _case(
        'meeting-summary-apology',
        'respond',
        _user('summarize the meeting'),
        _user('keep the apology brief', 1700),
        messages=("I'll keep the apology brief in the summary.",),
        expected_contains=('apology',),
    ),
    _case(
        'opening-line-calendar',
        'respond',
        _user('write the opening line'),
        _user('make the next step a calendar hold', 1000),
        messages=("I'll make the next step a calendar hold.",),
        expected_contains=('calendar',),
    ),
    _case(
        'onboarding-link',
        'respond',
        _user('turn this into a text message'),
        _user('include the onboarding link', 850),
        messages=("I'll include the onboarding link.",),
        expected_contains=('onboarding',),
    ),
    _case('mid-thought-wait', 'wait', _user('wait I mean the part where she says')),
    _case('fragment-section-about', 'wait', _user('can you write the section about')),
    _case('fragment-before-answer', 'wait', _user('actually before you answer')),
    _case('fragment-sounds-like', 'wait', _user('I want it to sound like')),
    _case('fragment-include-with', 'wait', _user('include the part with')),
    _case('fragment-main-thing-that', 'wait', _user('the main thing is that')),
    _case('partial-typed-fragment', 'wait', _user('for the paragraph where', partial=True)),
    _case('colon-fragment', 'wait', _user('send this to:')),
    _case(
        'correction-after-assistant',
        'interject',
        _user('draft a short invite'),
        _assistant("Sure - here's a crisp invite:", 900),
        _user('actually make it for investors, not friends', 700),
        messages=("Got it - investor-facing, not casual. Here's the revised invite:",),
        expected_contains=('investor',),
    ),
    _case(
        'nevermind-interrupt',
        'interject',
        _user('write a long email to Sam'),
        _assistant('Subject: Following up on our discussion', 900),
        _user('never mind, make it a two-line text', 600),
        messages=('Switching to a two-line text instead:',),
        expected_contains=('two-line',),
    ),
    _case(
        'scratch-that-customers',
        'interject',
        _user('prepare a product update'),
        _assistant("I'll frame it for the internal team.", 750),
        _user('scratch that, make it for customers', 500),
        messages=('Switching the audience to customers.',),
        expected_contains=('customers',),
    ),
    _case(
        'wait-make-calmer',
        'interject',
        _user('rewrite this paragraph'),
        _assistant('Here is a stronger version:', 650),
        _user('wait, make it calmer', 550),
        messages=("I'll make it calmer.",),
        expected_contains=('calmer',),
    ),
    _case(
        'actually-support-lead',
        'interject',
        _user('make a short invite'),
        _assistant("Sure - here's a friendly invite.", 700),
        _user('actually make it for the support lead', 800),
        messages=("I'll adjust it for the support lead.",),
        expected_contains=('support',),
    ),
    _case(
        'not-formal',
        'interject',
        _user('draft a customer email'),
        _assistant('Dear customer,', 600),
        _user('not formal, make it friendlier', 700),
        messages=("I'll make it friendlier and less formal.",),
        expected_contains=('friendlier',),
    ),
    _case(
        'instead-investor-update',
        'interject',
        _user('make a status update'),
        _assistant("Here's a short internal update:", 900),
        _user('instead mention the investor update', 650),
        messages=("I'll mention the investor update instead.",),
        expected_contains=('investor',),
    ),
    _case(
        'i-mean-migration',
        'interject',
        _user('write a support reply'),
        _assistant("Here's a concise reply:", 500),
        _user('I mean include the migration deadline', 500),
        messages=("I'll include the migration deadline.",),
        expected_contains=('migration',),
    ),
    _case(
        'assistant-continues',
        'continue',
        _user('give me the three steps'),
        _assistant('First, define the decision the model has to make.', 800),
        messages=('Second, collect examples where waiting is better than replying immediately.',),
        expected_contains=('Second',),
    ),
    _case(
        'assistant-second-bullet',
        'continue',
        _user('list the rollout plan'),
        _assistant('First, finish the local verifier.', 900),
        messages=('Second, run the adapter against the holdout set.',),
        expected_contains=('Second',),
    ),
    _case(
        'assistant-next-paragraph',
        'continue',
        _user('explain the model timeline'),
        _assistant('The first phase proves data and eval plumbing.', 850),
        messages=('Second, train the neural LoRA on the same contract.',),
        expected_contains=('Second',),
    ),
    _case(
        'assistant-continues-after-email',
        'continue',
        _user('draft the email in two parts'),
        _assistant('Subject: Quick update', 700),
        messages=('Second, add the body copy.',),
        expected_contains=('Second',),
    ),
    _case(
        'assistant-continues-analysis',
        'continue',
        _user('compare the two options'),
        _assistant('Option one is faster to test.', 650),
        messages=('Second, option two gives stronger evidence.',),
        expected_contains=('Second',),
    ),
    _case(
        'assistant-continues-recipe',
        'continue',
        _user('give me the process'),
        _assistant('First, collect clean examples.', 650),
        messages=('Second, train a small adapter.',),
        expected_contains=('Second',),
    ),
    _case(
        'rapid-backchannel',
        'continue',
        _user('explain the plan'),
        _assistant('The first stage is an offline harness.', 700),
        _user('yeah', 400),
        messages=('The second stage is a small LoRA run against fixed double-text scenarios.',),
        expected_contains=('second',),
    ),
    _case(
        'okay-backchannel',
        'continue',
        _user('walk me through the eval'),
        _assistant('The first check is schema validity.', 750),
        _user('okay', 500),
        messages=('Second, check the action label.',),
        expected_contains=('Second',),
    ),
    _case(
        'mhm-backchannel',
        'continue',
        _user('tell me the migration plan'),
        _assistant('First, freeze the current artifact.', 800),
        _user('mhm', 300),
        messages=('Second, widen the benchmark.',),
        expected_contains=('Second',),
    ),
    _case(
        'right-backchannel',
        'continue',
        _user('explain training stages'),
        _assistant('Stage one is local verification.', 700),
        _user('right', 450),
        messages=('Second, move to a neural LoRA smoke run.',),
        expected_contains=('Second',),
    ),
    _case(
        'complete-question',
        'respond',
        _user('what should the first benchmark measure?'),
        messages=(
            'It should measure whether the model waits through double texts and incorporates the latest user correction.',
        ),
        expected_contains=('double', 'latest'),
    ),
    _case(
        'summarize-product-update',
        'respond',
        _user('can you summarize the product update?'),
        messages=('Yes - I can summarize the product update.',),
        expected_contains=('product',),
    ),
    _case(
        'clear-next-step',
        'respond',
        _user('what is the clearest next step?'),
        messages=('The clearest next step is to run the verification gate.',),
        expected_contains=('next',),
    ),
    _case(
        'customer-reply-question',
        'respond',
        _user('what should the customer reply say?'),
        messages=('The reply should be concise and useful for the customer.',),
        expected_contains=('customer',),
    ),
)


SEMANTIC_TURN_ACTION_MAP = {
    'start_speaking': 'respond',
    '<|start_speaking|>': 'respond',
    'continue_listening': 'wait',
    '<|continue_listening|>': 'wait',
    'start_listening': 'interject',
    '<|start_listening|>': 'interject',
    'continue_speaking': 'continue',
    '<|continue_speaking|>': 'continue',
}


def map_semantic_turn_action(action: str) -> str:
    key = action.strip().lower()
    if key not in SEMANTIC_TURN_ACTION_MAP:
        raise ValueError(f'Unsupported semantic turn-taking action: {action!r}')
    return SEMANTIC_TURN_ACTION_MAP[key]
