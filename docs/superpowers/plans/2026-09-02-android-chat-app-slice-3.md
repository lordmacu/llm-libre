# Android chat app — slice 3: knowing who answered

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The app says which route answered each message, and lets you pick a
model — or a profile — instead of always taking `auto`.

**Architecture:** The gateway's `/v1/ranking` is the only place that says what a
model can do, so a catalogue client fetches it, a `routes_cache` table holds the
last good copy, and the model picker reads that. Attribution comes from the
`x_route` chunk the gateway now emits before `data: [DONE]`, recorded into the
`routeUsed` column that has existed unused since slice 2. Nothing here blocks
the chat: with no catalogue, `auto` still works.

**Tech Stack:** Flutter 3.38.3 via fvm, Dart ^3.10.1, `drift` `>=2.30.0 <2.31.0`,
`drift_flutter` `>=0.2.4 <0.3.0`, `http` `^1.6.0`, `gpt_markdown`. No new
dependencies.

**Spec:** `docs/superpowers/specs/2026-09-01-android-chat-app-design.md`
(slices 1-2 were built by `docs/superpowers/plans/2026-09-01-android-chat-app-slices-1-2.md`,
whose "Follow-ups carried into the next slice" section this plan discharges).

## Global Constraints

- All code — identifiers, comments, strings — in English. No Spanish in source.
- The API key is a build-time constant from `--dart-define-from-file=.env`;
  never a literal in source, and `.env` is never committed.
- `lib/api/` must not import `package:flutter/*`.
- Decode response bodies with `utf8.decode(response.bodyBytes)`, never
  `response.body`. The gateway declares no charset and Dart's `http` falls back
  to latin1.
- `x_requires` accepts only `tools`, `vision`, `search`. Any other name is a 400
  from the gateway. The capabilities `/v1/ranking` publishes are a SUPERSET of
  that: `images`, `audio_speech`, `audio_transcription` and `translate` are
  displayable but not requirable from a chat body.
- A route key from `/v1/ranking` is `provider/model_id` and **model ids contain
  slashes** (`kilo/nvidia/nemotron-3-super-120b-a12b:free`). Split on the FIRST
  slash only; `split('/')` is a bug.
- Never let `pub add` resolve a pinned dependency to latest.
- Run `fvm dart format lib test` before every commit.
- To run one test by name the flag is `--plain-name`, not `-n`; `flutter test`
  has no `-n`.
- App repo is `~/llm-libre-chat`; run Flutter as `fvm flutter <cmd>` from inside
  it. Commits authored `lordmacu <10134930+lordmacu@users.noreply.github.com>`,
  no `Co-Authored-By` trailer.
- Widget tests must not wait on a drift live query: drift resolves those on
  timers `flutter_test`'s fake clock never fires, which deadlocks the harness.
  Inject a fake through the `controllerFactory` seam instead, or use a bounded
  `tester.runAsync` and say why next to it.
- **Every widget test that renders a screen backed by the database must unmount
  before teardown:**

  ```dart
    await tester.pumpWidget(const SizedBox());
    await tester.pumpAndSettle();
  ```

  Closing the database with a `StreamBuilder` still subscribed waits on the same
  drift timer, and the symptom looks nothing like the cause: no assertion fails,
  the whole suite hangs for five minutes and then the harness crashes. Measured
  on this plan's Task 3 — the two tests went from hanging to 5 passing in 2
  seconds with those two lines added. The pre-existing tests in
  `chat_screen_test.dart` already do this; copy them rather than trusting a
  single `pump()`.

---

## File Structure

| File | Responsibility |
|---|---|
| `lib/api/types.dart` | Modify: `ChatDelta` gains `route`/`tier`. |
| `lib/api/route_info.dart` | Create: one route's capabilities and state, parsed from `/v1/ranking`. |
| `lib/api/catalog.dart` | Create: fetch and parse `/v1/ranking`. Nothing about storage or UI. |
| `lib/api/llm_client.dart` | Modify: parse `x_route`/`x_tier` from the stream. |
| `lib/data/db.dart` | Modify: `RoutesCache` table, schema v2, catalogue DAO, `updateModelOverride`. |
| `lib/features/chat/chat_controller.dart` | Modify: supersession guard, persist `routeUsed`, honour the override. |
| `lib/features/chat/chat_screen.dart` | Modify: attribution chip, picker entry point. |
| `lib/features/catalog/catalog_store.dart` | Create: the refresh policy over client + database. |
| `lib/features/catalog/model_picker.dart` | Create: the bottom sheet and its capability chips. |

---

## Task 1: The turn guard cannot be an ABA

**Files:**
- Modify: `lib/features/chat/chat_controller.dart`
- Test: `test/features/chat_controller_test.dart`

**Interfaces:**
- Consumes: nothing new.
- Produces: `bool _superseded(Completer<void> done)` replacing the `_disposed || !_sending` checks. No public API change.

**Why first.** Slice 2 left this as a known follow-up, unreachable only because
nothing rebuilt `ChatScreen.build` during `send()`'s awaits. The picker in Task 7
rebuilds. Fixing it after the picker means shipping the door open.

- [ ] **Step 1: Write the failing test**

```dart
  test('a stopped turn does not resume when a new one has started', () async {
    // `!_sending` was an ABA: stop() clears the flag and the next send() sets
    // it again, so the abandoned turn resuming inside its own awaits saw
    // `_sending == true` and carried on -- stranding its placeholder at
    // 'streaming' and leaving two live streams appending into one buffer.
    final id = await db.createConversation();
    final held = Completer<void>();
    var streams = 0;
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((_, __) async {
          streams++;
          return http.StreamedResponse(
            Stream.multi((c) async {
              c.add(utf8.encode(
                  'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'));
              await held.future;
              await c.close();
            }),
            200,
          );
        }),
      ),
      conversationId: id,
    );

    final first = controller.send('first');
    await controller.stop();
    final second = controller.send('second');
    // `held` is completed BEFORE either future is awaited. `send()` does not
    // resolve until its own stream closes, and this mock's stream closes only
    // when `held` does, so awaiting first would deadlock on a completer nothing
    // has reached yet -- a fixed ordering bug, not a race.
    held.complete();
    await first;
    await second;

    // The sharpest statement of the bug: the abandoned turn must never
    // subscribe. Before the fix it resumed and opened a SECOND stream, because
    // `_close` had cleared `_sending` and the new turn had set it again.
    expect(streams, 1, reason: 'the abandoned turn subscribed a second stream');
    final rows = await db.watchMessages(id).first;
    expect(rows.where((m) => m.status == 'streaming'), isEmpty,
        reason: 'the abandoned turn stranded its placeholder');
    expect(controller.streamingText.contains('oneone'), isFalse,
        reason: 'two live streams appended into one buffer');
  });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/features/chat_controller_test.dart --plain-name "does not resume"`
Expected: FAIL — a row left at `streaming`, or a doubled buffer.

- [ ] **Step 3: Replace the flag check with an identity check**

Add beside `busy`:

```dart
  /// True when the turn that owns [done] is no longer the one in flight — it
  /// was stopped, the conversation was left, or another turn has since started.
  ///
  /// Checking `!_sending` instead was an ABA: `_close` clears that flag and the
  /// next `send()` sets it again, so an abandoned turn resuming inside its own
  /// awaits saw it true and carried on. A completer's identity cannot be reused
  /// that way, and `_close` sets `_done = null` synchronously, so this is true
  /// the instant the turn stops being current.
  bool _superseded(Completer<void> done) => _disposed || _done != done;
```

The file has THREE guard sites, not four: two plain and one carrying a row.
Replace the two plain ones,

```dart
    if (_disposed || !_sending) return _abandon(done);
```

with

```dart
    if (_superseded(done)) return _abandon(done);
```

and the one that carries a row:

```dart
    if (_superseded(done)) return _abandon(done, strandedRow: row);
```

Leave the comment above the first one, updating its `!_sending` sentence to
name supersession instead.

- [ ] **Step 4: Run the test to verify it passes**

Run: `fvm flutter test test/features/chat_controller_test.dart`
Expected: PASS, every test in the file.

- [ ] **Step 5: Run the whole suite and commit**

```bash
fvm flutter test
fvm dart format lib test
git add -A
git commit -m "fix(chat): a stopped turn cannot resume under the next one"
```

---

## Task 2: The route reaches the row

**Files:**
- Modify: `lib/api/types.dart`, `lib/api/llm_client.dart`, `lib/features/chat/chat_controller.dart`
- Test: `test/api/llm_client_stream_test.dart`, `test/features/chat_controller_test.dart`

**Interfaces:**
- Consumes: `ChatDelta`, `LlmClient.stream`, `AppDb.finishMessage`.
- Produces: `ChatDelta({String? text, String? model, String? route, String? tier, bool done})`; `ChatController` persists `routeUsed`.

The gateway emits one chunk immediately before `data: [DONE]`:

```json
{"object": "chat.completion.chunk", "choices": [],
 "x_route": "grok/imagine-agent-mode-dev", "x_tier": "free"}
```

`routeUsed` has existed on the `Messages` table since slice 2 and has never been
written, because until the gateway shipped that chunk there was nothing to put
in it. The per-chunk `model` is not a substitute: the same model id exists at
several providers.

- [ ] **Step 1: Write the failing client test**

```dart
  test('the final chunk carries the route that served', () async {
    final client = streamingClient(
      'data: {"model":"turbo","choices":[{"delta":{"content":"hi"}}]}\n\n'
      'data: {"object":"chat.completion.chunk","choices":[],'
      '"x_route":"perplexity/turbo","x_tier":"free"}\n\n'
      'data: [DONE]\n\n',
    );
    final deltas = await client
        .stream(messages: const [ChatMessage(role: 'user', content: 'hi')])
        .toList();
    expect(deltas.map((d) => d.route).whereType<String>().single,
        'perplexity/turbo');
    expect(deltas.map((d) => d.tier).whereType<String>().single, 'free');
    // The attribution chunk carries no text, and must not be mistaken for any.
    expect(deltas.map((d) => d.text).whereType<String>().join(), 'hi');
  });
```

- [ ] **Step 2: Run it to verify it fails**

Run: `fvm flutter test test/api/llm_client_stream_test.dart --plain-name "carries the route"`
Expected: FAIL — `The getter 'route' isn't defined for the type 'ChatDelta'`.

- [ ] **Step 3: Extend `ChatDelta`**

```dart
class ChatDelta {
  const ChatDelta({
    this.text,
    this.model,
    this.route,
    this.tier,
    this.done = false,
  });

  final String? text;
  final String? model;

  /// `provider/model_id` of the route that served, from the gateway's own
  /// chunk before `[DONE]`. Null on every other chunk.
  ///
  /// `X-Route-Used` cannot travel on a stream — headers go out before the
  /// failover chain resolves — and [model] alone cannot stand in for this: the
  /// same model id exists at several providers, which is why the gateway routes
  /// by model rather than by provider in the first place.
  final String? route;

  /// `free` or `paid`, from the same chunk.
  final String? tier;

  final bool done;
}
```

- [ ] **Step 4: Parse it in `stream()`**

In the `await for` loop, replace the final `yield` with:

```dart
      yield ChatDelta(
        text: delta?['content'] as String?,
        model: json['model'] as String?,
        route: json['x_route'] as String?,
        tier: json['x_tier'] as String?,
      );
```

Nothing else changes: the attribution chunk has `choices: []`, which the
existing `choices.isNotEmpty` check already turns into a null `delta`.

- [ ] **Step 5: Run it to verify it passes**

Run: `fvm flutter test test/api/llm_client_stream_test.dart`
Expected: PASS, every test in the file.

- [ ] **Step 6: Write the failing controller test**

```dart
  test('the row records which route served it', () async {
    final id = await db.createConversation();
    final controller = ChatController(
      db: db,
      client: clientStreaming([
        'data: {"model":"turbo","choices":[{"delta":{"content":"hi"}}]}\n\n',
        'data: {"object":"chat.completion.chunk","choices":[],'
            '"x_route":"perplexity/turbo","x_tier":"free"}\n\n',
        'data: [DONE]\n\n',
      ]),
      conversationId: id,
    );
    await controller.send('hi');
    final row = (await db.watchMessages(id).first).last;
    expect(row.routeUsed, 'perplexity/turbo');
    expect(row.modelUsed, 'turbo');
  });
```

- [ ] **Step 7: Run it, then record the route**

Run: `fvm flutter test test/features/chat_controller_test.dart --plain-name "which route served"`
Expected: FAIL — `routeUsed` is null.

Add a field beside `_model`:

```dart
  String? _route;
```

Reset it in `send()` beside `_model = null;`:

```dart
    _route = null;
```

Capture it in the `listen` callback, beside the `model` line:

```dart
            if (delta.route != null) _route = delta.route;
```

Claim it synchronously in `_close`, beside `final model = _model;`:

```dart
    final route = _route;
```

and pass it on:

```dart
      await db.finishMessage(
        row,
        content: text,
        status: status,
        modelUsed: model,
        routeUsed: route,
      );
```

`finishMessage` does not accept `routeUsed` yet. Add the parameter and write it
alongside `modelUsed` — and **change nothing else in that method**. It is not
the one-liner it looks like: it reads the row first to learn which conversation
to touch, and calls `touchConversation` afterwards so a conversation used today
does not sort by its creation date. Replacing the body with a bare `write`
would silently delete that, which is a bug the previous plan's final review
had to find once already. Add only:

```dart
    String? routeUsed,
```

to the parameter list, and

```dart
        routeUsed: Value(routeUsed),
```

to the `MessagesCompanion`.

Any test double that overrides `finishMessage` needs the new named parameter
too — Dart requires an override to declare every named parameter of the base
method.

- [ ] **Step 8: Run the whole suite and commit**

```bash
fvm flutter test
fvm dart format lib test
git add -A
git commit -m "feat(chat): record which route served each answer"
```

---

## Task 3: The attribution chip says who answered

**Files:**
- Modify: `lib/features/chat/chat_screen.dart`
- Test: `test/features/chat_screen_test.dart`

**Interfaces:**
- Consumes: `Message.routeUsed`, `Message.modelUsed`, `Message.status`.
- Produces: no new API. `_Turn` renders the chip from `routeUsed`.

Also discharges a slice-2 follow-up: `_Turn` gates its "interrupted" label on
`modelUsed != null`, so a row that never received a delta — the one a process
death leaves behind — renders as a blank turn with no explanation.

- [ ] **Step 1: Write the failing tests**

```dart
  testWidgets('an answer says which provider and model served it',
      (tester) async {
    final db = AppDb.forTesting(NativeDatabase.memory());
    addTearDown(db.close);
    final id = await db.createConversation();
    final row = await db.addMessage(
        conversationId: id, role: 'assistant', content: 'hi',
        status: 'streaming');
    await db.finishMessage(row,
        content: 'hi', status: 'ok', modelUsed: 'turbo',
        routeUsed: 'perplexity/turbo');

    await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
            db: db,
            client: LlmClient(baseUrl: 'https://gw.test', apiKey: 'k'),
            conversationId: id,
            controllerFactory: (cid) => _FakeChatController(
                db: db,
                client: LlmClient(baseUrl: 'https://gw.test', apiKey: 'k'),
                conversationId: cid))));
    await tester.pump();

    expect(find.text('perplexity · turbo'), findsOneWidget);

    // Unmount before teardown: closing the database while a StreamBuilder is
    // still subscribed waits on a drift timer the fake clock never fires, and
    // the suite hangs rather than failing.
    await tester.pumpWidget(const SizedBox());
    await tester.pumpAndSettle();
  });

  testWidgets('an interrupted answer says so even with no route recorded',
      (tester) async {
    // What a process death leaves behind: repaired to 'partial', but with no
    // model and no route, so a label gated on those said nothing at all.
    final db = AppDb.forTesting(NativeDatabase.memory());
    addTearDown(db.close);
    final id = await db.createConversation();
    final row = await db.addMessage(
        conversationId: id, role: 'assistant', content: '',
        status: 'streaming');
    await db.finishMessage(row, content: '', status: 'partial');

    await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
            db: db,
            client: LlmClient(baseUrl: 'https://gw.test', apiKey: 'k'),
            conversationId: id,
            controllerFactory: (cid) => _FakeChatController(
                db: db,
                client: LlmClient(baseUrl: 'https://gw.test', apiKey: 'k'),
                conversationId: cid))));
    await tester.pump();

    expect(find.textContaining('interrupted'), findsOneWidget);

    await tester.pumpWidget(const SizedBox());
    await tester.pumpAndSettle();
  });
```

`_FakeChatController` already exists at the bottom of
`test/features/chat_screen_test.dart` — it extends `ChatController`, overrides
`send`/`stop` to reach nothing, and exposes `emit({row, text})`. Reuse it; do
not write a second fake. Its constructor takes `db`, `client` and
`conversationId` as `super` parameters.

- [ ] **Step 2: Run them to verify they fail**

Run: `fvm flutter test test/features/chat_screen_test.dart`
Expected: FAIL — no `perplexity · turbo`, and no `interrupted` on the second.

- [ ] **Step 3: Rewrite the chip**

Replace `_Turn`'s trailing block with:

```dart
          if (_label(message) != null)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                _label(message)!,
                style: Theme.of(context).textTheme.labelSmall,
              ),
            ),
```

and add to `_Turn`:

```dart
  /// What to show under an assistant turn, or null when there is nothing to
  /// say.
  ///
  /// Reads `routeUsed` in preference to `modelUsed`: the model id alone is
  /// ambiguous because the same one exists at several providers. The status
  /// comes first because an interrupted answer has to say so even when no
  /// route was ever recorded — which is exactly the state a process death
  /// leaves behind, and the case a label gated on `modelUsed` stayed silent for.
  static String? _label(Message message) {
    final route = message.routeUsed;
    final served = route != null
        ? route.replaceFirst('/', ' · ')
        : message.modelUsed;
    if (message.status == 'partial') {
      return served == null ? 'interrupted' : '$served · interrupted';
    }
    if (message.status == 'error') return 'failed';
    return served;
  }
```

`replaceFirst` and not `replace`: a model id contains slashes of its own.

- [ ] **Step 4: Run them to verify they pass**

Run: `fvm flutter test test/features/chat_screen_test.dart`
Expected: PASS

- [ ] **Step 5: Run the whole suite and commit**

```bash
fvm flutter test
fvm dart format lib test
git add -A
git commit -m "feat(chat): each answer names the route that served it"
```

---

## Shared test helper

Tasks 6 to 9 all build `RouteInfo` values in tests. Define this once, in each
test file that needs it (they are separate libraries, so it cannot be shared
without a new file, and a `test/support/` file for four lines is not worth the
indirection):

```dart
RouteInfo _route(String key, {String tier = 'free', int context = 32000}) {
  final cut = key.indexOf('/');
  return RouteInfo(
    key: key,
    provider: key.substring(0, cut),
    modelId: key.substring(cut + 1),
    tier: tier,
    context: context,
  );
}
```

---

## Task 4: One route's capabilities, parsed

**Files:**
- Create: `lib/api/route_info.dart`
- Test: `test/api/route_info_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `class RouteInfo` with `final String key, provider, modelId, tier; final bool tools, vision, images, search, audioSpeech, audioTranscription, translate; final int context; final double? quality, latencyP50Ms; final double cooldownUntil;` plus `factory RouteInfo.fromRanking(Map<String, dynamic> row)`, `bool inCooldown(DateTime now)`, and `Set<String> get capabilities`.

- [ ] **Step 1: Write the failing test**

Every field name below is copied from a real `/v1/ranking` row.

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/api/route_info.dart';

void main() {
  Map<String, dynamic> row({String key = 'kilo/nvidia/nemotron-3:free'}) => {
        'key': key,
        'tier': 'free',
        'tools': true,
        'vision': false,
        'images': false,
        'search': true,
        'audio_speech': true,
        'audio_transcription': false,
        'translate': false,
        'context': 1000000,
        'quality': 0.87,
        'latency_p50_ms': 420.5,
        'cooldown_until': 0.0,
      };

  test('a model id containing slashes keeps them', () {
    // The key is provider/model_id and model ids contain slashes of their own.
    // split('/') loses everything after the second one, and asking the gateway
    // for the truncated id is a 404.
    final r = RouteInfo.fromRanking(row());
    expect(r.provider, 'kilo');
    expect(r.modelId, 'nvidia/nemotron-3:free');
  });

  test('capabilities are the ones the row actually claims', () {
    expect(RouteInfo.fromRanking(row()).capabilities,
        {'tools', 'search', 'audio_speech'});
  });

  test('a missing capability is absent, not an error', () {
    // /v1/ranking has gained fields before and will again. A row without one
    // must read as "cannot", never throw.
    final sparse = {'key': 'p/m', 'tier': 'free', 'context': 32000};
    final r = RouteInfo.fromRanking(sparse);
    expect(r.tools, isFalse);
    expect(r.capabilities, isEmpty);
    expect(r.quality, isNull);
  });

  test('cooldown is read against a supplied clock', () {
    final future = DateTime.utc(2030).millisecondsSinceEpoch / 1000;
    final r = RouteInfo.fromRanking({...row(), 'cooldown_until': future});
    expect(r.inCooldown(DateTime.utc(2029)), isTrue);
    expect(r.inCooldown(DateTime.utc(2031)), isFalse);
  });
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `fvm flutter test test/api/route_info_test.dart`
Expected: FAIL — `Target of URI doesn't exist`.

- [ ] **Step 3: Write the implementation**

```dart
/// One route as `/v1/ranking` describes it.
///
/// The capability booleans are a SUPERSET of what a chat body may require:
/// only `tools`, `vision` and `search` are accepted in `x_requires`, while
/// `images`, `audio_speech`, `audio_transcription` and `translate` belong to
/// endpoints of their own and are shown here but cannot be asked for.
class RouteInfo {
  const RouteInfo({
    required this.key,
    required this.provider,
    required this.modelId,
    required this.tier,
    required this.context,
    this.tools = false,
    this.vision = false,
    this.images = false,
    this.search = false,
    this.audioSpeech = false,
    this.audioTranscription = false,
    this.translate = false,
    this.quality,
    this.latencyP50Ms,
    this.cooldownUntil = 0,
  });

  /// `provider/model_id`, as the gateway reports it and as `X-Route-Used` and
  /// the streamed `x_route` chunk both spell it.
  final String key;
  final String provider;

  /// What to put in a request's `model` field: the model's own id, WITHOUT the
  /// provider. It can contain slashes.
  final String modelId;

  final String tier; // 'free' | 'paid'
  final int context;
  final bool tools, vision, images, search;
  final bool audioSpeech, audioTranscription, translate;
  final double? quality, latencyP50Ms;

  /// Unix seconds until which the router will not pick this route. 0 = free.
  final double cooldownUntil;

  factory RouteInfo.fromRanking(Map<String, dynamic> row) {
    final key = (row['key'] as String?) ?? '';
    // FIRST slash only: the provider is one segment, the model id is all the
    // rest, and model ids legitimately contain slashes.
    final cut = key.indexOf('/');
    bool flag(String name) => row[name] == true;
    return RouteInfo(
      key: key,
      provider: cut < 0 ? key : key.substring(0, cut),
      modelId: cut < 0 ? '' : key.substring(cut + 1),
      tier: (row['tier'] as String?) ?? 'free',
      context: (row['context'] as num?)?.toInt() ?? 0,
      tools: flag('tools'),
      vision: flag('vision'),
      images: flag('images'),
      search: flag('search'),
      audioSpeech: flag('audio_speech'),
      audioTranscription: flag('audio_transcription'),
      translate: flag('translate'),
      quality: (row['quality'] as num?)?.toDouble(),
      latencyP50Ms: (row['latency_p50_ms'] as num?)?.toDouble(),
      cooldownUntil: (row['cooldown_until'] as num?)?.toDouble() ?? 0,
    );
  }

  bool inCooldown(DateTime now) =>
      cooldownUntil > now.millisecondsSinceEpoch / 1000;

  /// The wire names of everything this route claims, for display.
  Set<String> get capabilities => {
        if (tools) 'tools',
        if (vision) 'vision',
        if (images) 'images',
        if (search) 'search',
        if (audioSpeech) 'audio_speech',
        if (audioTranscription) 'audio_transcription',
        if (translate) 'translate',
      };
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `fvm flutter test test/api/route_info_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
fvm dart format lib test
git add lib/api/route_info.dart test/api/route_info_test.dart
git commit -m "feat(api): one route's capabilities, as /v1/ranking reports them"
```

---

## Task 5: Fetching the catalogue

**Files:**
- Create: `lib/api/catalog.dart`
- Test: `test/api/catalog_test.dart`

**Interfaces:**
- Consumes: `RouteInfo.fromRanking`, `errorFromResponse` and the `LlmError` hierarchy from `lib/api/errors.dart`.
- Produces: `class CatalogClient({required String baseUrl, required String apiKey, http.Client? httpClient})` with `Future<List<RouteInfo>> fetch()`.

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:llm_libre_chat/api/catalog.dart';
import 'package:llm_libre_chat/api/errors.dart';

CatalogClient clientReturning(http.Response response,
        {List<http.Request>? seen}) =>
    CatalogClient(
      baseUrl: 'https://gw.test',
      apiKey: 'k',
      httpClient: MockClient((req) async {
        seen?.add(req);
        return response;
      }),
    );

void main() {
  test('it reads the routes and hits /v1/ranking with the key', () async {
    final seen = <http.Request>[];
    final client = clientReturning(
      http.Response.bytes(
        utf8.encode(jsonEncode({
          'routes': [
            {'key': 'perplexity/turbo', 'tier': 'free', 'search': true,
             'context': 32000},
            {'key': 'kilo/a/b:free', 'tier': 'free', 'tools': true,
             'context': 128000},
          ]
        })),
        200,
      ),
      seen: seen,
    );
    final routes = await client.fetch();
    expect(routes.map((r) => r.key), ['perplexity/turbo', 'kilo/a/b:free']);
    expect(routes.last.modelId, 'a/b:free');
    expect(seen.single.url.path, '/v1/ranking');
    expect(seen.single.headers['Authorization'], 'Bearer k');
  });

  test('accented content survives a body with no charset declared', () async {
    // Same trap as the chat client: Dart's http falls back to latin1 when the
    // server declares no charset, and this gateway declares none.
    final client = clientReturning(http.Response.bytes(
      utf8.encode(jsonEncode({
        'routes': [
          {'key': 'p/modelo-español', 'tier': 'free', 'context': 1}
        ]
      })),
      200,
    ));
    expect((await client.fetch()).single.modelId, 'modelo-español');
  });

  test('a bad key is the same typed error the chat path raises', () async {
    final client =
        clientReturning(http.Response('{"detail":"invalid api key"}', 401));
    expect(client.fetch(), throwsA(isA<Unauthorized>()));
  });

  test('a body without routes is empty, not a crash', () async {
    final client = clientReturning(http.Response('{}', 200));
    expect(await client.fetch(), isEmpty);
  });

  test('one malformed row does not cost the whole catalogue', () async {
    // `fromRanking` tolerates a missing field but not one of the wrong type,
    // and the picker going blank because a single row went strange is a worse
    // failure than that row being absent from it.
    final client = clientReturning(http.Response.bytes(
      utf8.encode(jsonEncode({
        'routes': [
          {'key': 'p/good', 'tier': 'free', 'context': 32000},
          {'key': 12345, 'tier': 'free', 'context': 32000}, // key is not a String
          {'key': 'p/alsogood', 'tier': 'free', 'context': 'lots'}, // context is not a num
          {'key': 'q/good', 'tier': 'free', 'context': 8000},
        ]
      })),
      200,
    ));
    expect((await client.fetch()).map((r) => r.key), ['p/good', 'q/good']);
  });
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `fvm flutter test test/api/catalog_test.dart`
Expected: FAIL — `Target of URI doesn't exist`.

- [ ] **Step 3: Write the implementation**

```dart
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'errors.dart';
import 'route_info.dart';

/// Reads `/v1/ranking`, which is the only endpoint that says what each model
/// can do: `/v1/models` returns `{id, object, owned_by}` and nothing else.
///
/// Separate from `LlmClient` on purpose — this one answers "what exists", that
/// one answers "serve this". They share only the base URL and the key.
class CatalogClient {
  CatalogClient({
    required this.baseUrl,
    required this.apiKey,
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client();

  final String baseUrl;
  final String apiKey;
  final http.Client _http;

  Future<List<RouteInfo>> fetch() async {
    final response = await _http.get(
      Uri.parse('$baseUrl/v1/ranking'),
      headers: {'Authorization': 'Bearer $apiKey'},
    );
    // Not `response.body`: http decodes as latin1 unless the server declares a
    // charset, and the gateway does not.
    final text = utf8.decode(response.bodyBytes);
    if (response.statusCode != 200) {
      throw errorFromResponse(response.statusCode, text);
    }
    final json = jsonDecode(text) as Map<String, dynamic>;
    final rows = json['routes'] as List? ?? const [];
    final routes = <RouteInfo>[];
    for (final row in rows) {
      if (row is! Map<String, dynamic>) continue;
      try {
        routes.add(RouteInfo.fromRanking(row));
      } on TypeError {
        // One malformed row costs one model in the picker, not the whole
        // picker. `fromRanking` tolerates a MISSING field by design -- this
        // endpoint has gained fields before -- but a field present with the
        // wrong type throws, and hardening fifteen casts individually would
        // trade a crash for fifteen silent defaults. Skipping the row keeps
        // the failure proportionate and visible in what is offered.
        continue;
      }
    }
    return routes;
  }
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `fvm flutter test test/api/catalog_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
fvm dart format lib test
git add lib/api/catalog.dart test/api/catalog_test.dart
git commit -m "feat(api): read the route catalogue from /v1/ranking"
```

---

## Task 6: The catalogue survives a restart

**Files:**
- Modify: `lib/data/db.dart`
- Test: `test/data/db_test.dart`

**Interfaces:**
- Consumes: `RouteInfo`.
- Produces: table `RoutesCache`; `schemaVersion` 2 with an `onUpgrade`; `Future<void> saveRoutes(List<RouteInfo>, DateTime fetchedAt)`, `Future<List<RouteInfo>> cachedRoutes()`, `Future<DateTime?> routesFetchedAt()`, `Future<void> updateModelOverride(String conversationId, String? model)`.

This is the project's first schema upgrade. `migration` currently carries only
`beforeOpen`; the `onUpgrade` goes beside it, and `beforeOpen` must keep working.

- [ ] **Step 1: Write the failing test**

```dart
  test('the catalogue round-trips, newest fetch wins', () async {
    const a = RouteInfo(
        key: 'perplexity/turbo', provider: 'perplexity', modelId: 'turbo',
        tier: 'free', context: 32000, search: true);
    const b = RouteInfo(
        key: 'kilo/a/b:free', provider: 'kilo', modelId: 'a/b:free',
        tier: 'free', context: 128000, tools: true);

    final t1 = DateTime.utc(2026, 9, 2, 10);
    await db.saveRoutes([a, b], t1);
    expect((await db.cachedRoutes()).map((r) => r.key),
        containsAll(<String>['perplexity/turbo', 'kilo/a/b:free']));
    expect(await db.routesFetchedAt(), t1);

    // A later sweep REPLACES the catalogue rather than merging into it: a route
    // the gateway stopped serving must disappear, not linger as a pickable
    // model that 404s.
    final t2 = DateTime.utc(2026, 9, 2, 11);
    await db.saveRoutes([a], t2);
    expect((await db.cachedRoutes()).map((r) => r.key), ['perplexity/turbo']);
    expect(await db.routesFetchedAt(), t2);
  });

  test('a cached route keeps its capabilities and its slashes', () async {
    const r = RouteInfo(
        key: 'kilo/nvidia/nemotron-3:free', provider: 'kilo',
        modelId: 'nvidia/nemotron-3:free', tier: 'free', context: 1000000,
        tools: true, search: true, quality: 0.9);
    await db.saveRoutes([r], DateTime.utc(2026));
    final back = (await db.cachedRoutes()).single;
    expect(back.modelId, 'nvidia/nemotron-3:free');
    expect(back.capabilities, {'tools', 'search'});
    expect(back.quality, 0.9);
  });

  test('an empty cache is empty, with no fetch time', () async {
    expect(await db.cachedRoutes(), isEmpty);
    expect(await db.routesFetchedAt(), isNull);
  });

  test('the model override persists per conversation', () async {
    final id = await db.createConversation();
    await db.updateModelOverride(id, 'turbo');
    var row = await (db.select(db.conversations)
          ..where((c) => c.id.equals(id)))
        .getSingle();
    expect(row.modelOverride, 'turbo');
    // Clearing it returns the conversation to auto.
    await db.updateModelOverride(id, null);
    row = await (db.select(db.conversations)..where((c) => c.id.equals(id)))
        .getSingle();
    expect(row.modelOverride, isNull);
  });
```

- [ ] **Step 2: Run it to verify it fails**

Run: `fvm flutter test test/data/db_test.dart`
Expected: FAIL — `saveRoutes` is not defined.

- [ ] **Step 3: Add the table**

```dart
/// The last catalogue read from `/v1/ranking`.
///
/// Cached so the model picker opens instantly and still works with no network.
/// It is a CACHE and never a source of truth: the gateway can drop a route at
/// any time, and asking for one it no longer serves is a 404 (with suggestions).
class RoutesCache extends Table {
  /// `provider/model_id`.
  TextColumn get key => text()();
  TextColumn get provider => text()();
  TextColumn get modelId => text()();
  TextColumn get tier => text()();
  IntColumn get context => integer()();

  /// The wire names of the capabilities this route claims, comma-separated.
  ///
  /// One column rather than seven booleans because nothing queries by
  /// capability — the picker loads the whole catalogue and filters in memory —
  /// and `/v1/ranking` has gained capability fields before. A new one costs a
  /// string here instead of a migration.
  TextColumn get capabilities => text().withDefault(const Constant(''))();

  RealColumn get quality => real().nullable()();
  RealColumn get latencyP50Ms => real().nullable()();
  RealColumn get cooldownUntil => real().withDefault(const Constant(0))();
  DateTimeColumn get fetchedAt => dateTime()();

  @override
  Set<Column> get primaryKey => {key};
}
```

Register it: `@DriftDatabase(tables: [Conversations, Messages, RoutesCache])`.

- [ ] **Step 4: Bump the schema and upgrade**

```dart
  @override
  int get schemaVersion => 2;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onCreate: (m) => m.createAll(),
        // v2 adds the route cache. A device upgrading from v1 has conversations
        // worth keeping, so the table is created and nothing else is touched;
        // the catalogue refills itself on the next sweep.
        onUpgrade: (m, from, to) async {
          if (from < 2) await m.createTable(routesCache);
        },
        beforeOpen: (_) => repairInterruptedMessages(),
      );
```

`onCreate` is spelled out even though it only restates drift's default.
`MigrationStrategy`'s parameters are independent named defaults — verified in
drift 2.30.1's source — so supplying `onUpgrade` alone would still leave
`onCreate` as `m.createAll()`, and a fresh install would be fine either way. An
earlier revision of this plan claimed the opposite and was wrong. It is written
out because a migration getter that names all three hooks is one a future reader
can reason about without going to drift's source, which is exactly what the
false claim cost.

- [ ] **Step 5: Add the DAO methods**

```dart
  /// Replaces the cached catalogue wholesale.
  ///
  /// Not a merge: a route the gateway stopped serving has to disappear, or the
  /// picker keeps offering a model that answers 404.
  Future<void> saveRoutes(List<RouteInfo> routes, DateTime fetchedAt) =>
      transaction(() async {
        await delete(routesCache).go();
        await batch((b) => b.insertAll(routesCache, [
              for (final r in routes)
                RoutesCacheCompanion.insert(
                  key: r.key,
                  provider: r.provider,
                  modelId: r.modelId,
                  tier: r.tier,
                  context: r.context,
                  capabilities: Value(r.capabilities.join(',')),
                  quality: Value(r.quality),
                  latencyP50Ms: Value(r.latencyP50Ms),
                  cooldownUntil: Value(r.cooldownUntil),
                  fetchedAt: fetchedAt,
                )
            ]));
      });

  Future<List<RouteInfo>> cachedRoutes() async {
    final rows = await select(routesCache).get();
    return [
      for (final row in rows)
        RouteInfo(
          key: row.key,
          provider: row.provider,
          modelId: row.modelId,
          tier: row.tier,
          context: row.context,
          tools: row.capabilities.split(',').contains('tools'),
          vision: row.capabilities.split(',').contains('vision'),
          images: row.capabilities.split(',').contains('images'),
          search: row.capabilities.split(',').contains('search'),
          audioSpeech: row.capabilities.split(',').contains('audio_speech'),
          audioTranscription:
              row.capabilities.split(',').contains('audio_transcription'),
          translate: row.capabilities.split(',').contains('translate'),
          quality: row.quality,
          latencyP50Ms: row.latencyP50Ms,
          cooldownUntil: row.cooldownUntil,
        ),
    ];
  }

  Future<DateTime?> routesFetchedAt() async {
    final row = await (select(routesCache)..limit(1)).getSingleOrNull();
    return row?.fetchedAt;
  }

  Future<void> updateModelOverride(String conversationId, String? model) =>
      (update(conversations)..where((c) => c.id.equals(conversationId))).write(
        ConversationsCompanion(modelOverride: Value(model)),
      );
```

Import `RouteInfo` at the top of `db.dart`:
`import '../api/route_info.dart';`

- [ ] **Step 6: Regenerate and run**

```bash
fvm dart run build_runner build --delete-conflicting-outputs
fvm flutter test test/data/db_test.dart
```
Expected: PASS

- [ ] **Step 7: Prove the upgrade path, not just the fresh one**

The tests above all run against a fresh in-memory database, which exercises
`onCreate` and never `onUpgrade`. Add:

```dart
  test('a v1 database upgrades without losing its conversations', () async {
    // The tests above only ever exercise onCreate. Deleting the onUpgrade
    // branch would fail nothing without this one.
    final file = File(
        '${Directory.systemTemp.createTempSync('llmv1').path}/app.sqlite');
    addTearDown(() => file.parent.deleteSync(recursive: true));

    // A v1 database: conversations and messages, no route cache.
    final raw = sqlite3.sqlite3.open(file.path);
    raw
      ..execute('CREATE TABLE conversations (id TEXT NOT NULL, '
          'title TEXT NOT NULL DEFAULT \'\', created_at TEXT NOT NULL, '
          'updated_at TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0, '
          'model_override TEXT NULL, system_prompt TEXT NULL, '
          'PRIMARY KEY (id))')
      ..execute('CREATE TABLE messages (id INTEGER NOT NULL PRIMARY KEY '
          'AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL, '
          'content TEXT NOT NULL, reasoning TEXT NULL, model_used TEXT NULL, '
          'route_used TEXT NULL, status TEXT NOT NULL DEFAULT \'ok\', '
          'created_at TEXT NOT NULL)')
      ..execute("INSERT INTO conversations VALUES "
          "('c1','kept','2026-09-01T10:00:00.000','2026-09-01T10:00:00.000',"
          "0,NULL,NULL)")
      ..execute('PRAGMA user_version = 1')
      ..dispose();

    final upgraded = AppDb.forTesting(NativeDatabase(file));
    addTearDown(upgraded.close);
    expect((await upgraded.watchConversations().first).single.title, 'kept');
    expect(await upgraded.cachedRoutes(), isEmpty); // the table now exists
  });
```

That test needs the `sqlite3` package directly, to build the v1 fixture before
drift ever opens the file. It is only a TRANSITIVE dependency of drift today,
and importing one of those trips the `depend_on_referenced_packages` lint, so
add it as a dev dependency first — pinned to the version drift already
resolves, because a different major would load a second native library:

```bash
fvm flutter pub add --dev 'sqlite3:>=2.9.0 <3.0.0'
```

Then import `dart:io` and `package:sqlite3/sqlite3.dart' as sqlite3` in the
test.

If that constraint will not resolve, do NOT loosen it — the upper bound is what
keeps `drift_dev` resolvable at all (`drift_flutter` 0.3.x pulls sqlite3 3.x and
the generator then has no valid version). Report it instead, and fall back to
asserting the upgrade through drift alone: open the file with the current
`AppDb`, close it, delete only the `routes_cache` table with a
`customStatement`, set `PRAGMA user_version = 1`, and reopen. That proves the
same branch without a second sqlite binding.

Run: `fvm flutter test test/data/db_test.dart`
Expected: PASS

- [ ] **Step 8: Run the whole suite and commit**

```bash
fvm flutter test
fvm dart format lib test
git add -A
git commit -m "feat(data): cache the route catalogue, and keep it across upgrades"
```

---

## Task 7: When to refetch

**Files:**
- Create: `lib/features/catalog/catalog_store.dart`
- Test: `test/features/catalog_store_test.dart`

**Interfaces:**
- Consumes: `CatalogClient.fetch`, `AppDb.saveRoutes`, `AppDb.cachedRoutes`, `AppDb.routesFetchedAt`.
- Produces: `class CatalogStore({required AppDb db, required CatalogClient client, Duration floor})` with `Future<List<RouteInfo>> routes({DateTime? now})` and `static const Duration defaultFloor = Duration(minutes: 10)`.

- [ ] **Step 1: Write the failing test**

```dart
  test('it serves the cache and does not refetch inside the floor', () async {
    var fetches = 0;
    final store = CatalogStore(
      db: db,
      client: _clientYielding(() { fetches++; return [_route('p/m')]; }),
    );
    final t = DateTime.utc(2026, 9, 2, 10);
    expect((await store.routes(now: t)).single.key, 'p/m');
    expect(fetches, 1);

    // Cooldowns are measured in minutes, and this endpoint carries the whole
    // telemetry payload; polling it per rebuild is load for nothing.
    await store.routes(now: t.add(const Duration(minutes: 9)));
    expect(fetches, 1);
    await store.routes(now: t.add(const Duration(minutes: 11)));
    expect(fetches, 2);
  });

  test('a failed sweep serves the last good catalogue', () async {
    // The chat works without a catalogue -- `auto` needs no local knowledge --
    // so a refresh failure must never surface as an empty picker.
    final t = DateTime.utc(2026, 9, 2, 10);
    await db.saveRoutes([_route('p/kept')], t);
    final store = CatalogStore(
      db: db,
      client: _clientThrowing(const Unauthorized('invalid api key')),
    );
    final routes = await store.routes(now: t.add(const Duration(hours: 2)));
    expect(routes.single.key, 'p/kept');
    expect(store.lastFailure, isA<Unauthorized>());
  });

  test('a failure with nothing cached is empty, not a throw', () async {
    final store = CatalogStore(
      db: db,
      client: _clientThrowing(const Disconnected('offline')),
    );
    expect(await store.routes(now: DateTime.utc(2026)), isEmpty);
  });

  test('a clean call clears the failure a previous one recorded', () async {
    // lastFailure was only ever assigned in the catch, so it survived a
    // subsequent success: a caller reading it after a healthy cache read would
    // report an error for an attempt that had already been superseded.
    final t = DateTime.utc(2026, 9, 2, 10);
    await db.saveRoutes([_route('p/kept')], t);
    final store = CatalogStore(
      db: db,
      client: _clientThrowing(const Disconnected('offline')),
    );

    // Two hours past the seeded fetchedAt, so this one refetches and fails.
    await store.routes(now: t.add(const Duration(hours: 2)));
    expect(store.lastFailure, isNotNull);

    // Re-seed so fetchedAt is recent relative to the NEXT call. Without this
    // the failing client never lets saveRoutes run, fetchedAt stays at `t`,
    // and a second call at any later time exceeds the floor and refetches --
    // so the fast path this test exists to exercise is never reached, and the
    // clock cannot be moved backwards to fake it either.
    final later = t.add(const Duration(hours: 2));
    await db.saveRoutes([_route('p/kept')], later);

    // Five minutes on: inside the ten-minute floor, so no fetch happens at all.
    final again = await store.routes(
      now: later.add(const Duration(minutes: 5)),
    );
    expect(again.single.key, 'p/kept');
    expect(store.lastFailure, isNull);
  });

  test('a database that cannot be read is empty, not a throw', () async {
    // The property this class exists to guarantee, tested adversarially rather
    // than confirmed: with only the network call guarded, BOTH the fetched-at
    // read and the cache read inside the catch escaped uncaught. Keep the
    // failing-executor harness from that investigation.
    final broken = AppDb.forTesting(_FailingExecutor(NativeDatabase.memory()));
    addTearDown(() async {
      try {
        await broken.close();
      } on Object {
        // Closing a database whose executor throws is not the subject here.
      }
    });
    final store = CatalogStore(
      db: broken,
      client: _clientYielding(() => [_route('p/m')]),
    );
    expect(await store.routes(now: DateTime.utc(2026)), isEmpty);
    expect(store.lastFailure, isNotNull);
  });

  test('a non-JSON 200 does not escape as a raw FormatException', () async {
    // `fetch()` decodes the body without guarding the shape, so a proxy or an
    // error page answering 200 with HTML raises FormatException, not LlmError.
    // This class promises it never throws; `on LlmError` alone would break that
    // promise, which is precisely how a swallow-too-narrow shipped a hang once.
    final t = DateTime.utc(2026, 9, 2, 10);
    await db.saveRoutes([_route('p/kept')], t);
    final store = CatalogStore(
      db: db,
      client: _clientThrowing(const FormatException('not json')),
    );
    final routes = await store.routes(now: t.add(const Duration(hours: 2)));
    expect(routes.single.key, 'p/kept');
    expect(store.lastFailure, isA<Disconnected>());
  });
```

Write `_clientYielding` and `_clientThrowing` as `CatalogClient` subclasses
overriding `fetch()` — note `_clientThrowing` must accept any `Object`, not just
an `LlmError`, since one test hands it a `FormatException`. `_FailingExecutor`
wraps a real `QueryExecutor` and throws from `runSelect`, which is what proves
the database reads are guarded; a `MockClient` would test the transport again rather than
the policy, which Task 5 already covers.

- [ ] **Step 2: Run it to verify it fails**

Run: `fvm flutter test test/features/catalog_store_test.dart`
Expected: FAIL — `Target of URI doesn't exist`.

- [ ] **Step 3: Write the implementation**

```dart
import '../../api/catalog.dart';
import '../../api/errors.dart';
import '../../api/route_info.dart';
import '../../data/db.dart';

/// Decides WHEN to read the catalogue. [CatalogClient] decides how.
///
/// Never throws: the chat does not need a catalogue at all — `auto` needs no
/// local knowledge — so a refresh failure has to degrade to the last good copy
/// rather than reach the user as a broken picker. The failure is kept in
/// [lastFailure] so a screen can say so if it wants to.
class CatalogStore {
  CatalogStore({
    required this.db,
    required this.client,
    this.floor = defaultFloor,
  });

  /// Cooldowns are measured in minutes, and `/v1/ranking` carries the whole
  /// telemetry payload. Refetching per rebuild would be load on the gateway
  /// for information that has not changed.
  static const Duration defaultFloor = Duration(minutes: 10);

  final AppDb db;
  final CatalogClient client;
  final Duration floor;

  /// Why the MOST RECENT call to [routes] failed, or null when it did not.
  ///
  /// Reset on entry rather than only assigned in the catch: a consumer showing
  /// "could not refresh" needs it to describe the call it just made, not some
  /// earlier one that has since been superseded by a clean read.
  LlmError? lastFailure;

  Future<List<RouteInfo>> routes({DateTime? now}) async {
    final at = now ?? DateTime.now();
    // The WHOLE body is guarded, not just the network call. The database reads
    // fail too -- a corrupt or locked file, a full disk -- and on Android none
    // of those is exotic. Verified by injecting a QueryExecutor whose runSelect
    // always throws: with only the fetch guarded, both `routesFetchedAt` and
    // the `cachedRoutes` inside the catch escaped uncaught.
    //
    // Note every return inside the try is `return await`. `return someFuture;`
    // leaves the try block before the future completes, so its error reaches
    // the CALLER instead of the catch below -- which would quietly undo this
    // whole guard.
    try {
      // Cleared on every entry, so `lastFailure` means "the most recent call to
      // routes() failed" and nothing subtler. Setting it only in the catch left
      // it stale: a transient failure, then a clean call served from cache, and
      // a caller reading it would still show an error for an attempt that had
      // since succeeded.
      lastFailure = null;
      final fetchedAt = await db.routesFetchedAt();
      if (fetchedAt != null && at.difference(fetchedAt) < floor) {
        return await db.cachedRoutes();
      }
      final fresh = await client.fetch();
      await db.saveRoutes(fresh, at);
      return fresh;
    } on Object catch (e) {
      // Catches EVERYTHING, not just LlmError, because the promise in this
      // class's doc comment is that it never throws — and `on LlmError` does
      // not keep that promise. `fetch()` raises a bare FormatException if the
      // gateway answers 200 with a body that is not JSON, which is exactly what
      // a proxy or an error page in front of it produces. The same mistake in
      // `_titleIfUnnamed` shipped a hang: a cosmetic feature able to wedge the
      // path that calls it. Wrapped the way the chat controller already wraps
      // its own stream errors, so `lastFailure` stays one type.
      lastFailure = e is LlmError ? e : Disconnected(e);
      return _cachedOrEmpty();
    }
  }

  /// The fallback's own fallback.
  ///
  /// Reached only when something has already failed, so if the cache read fails
  /// too there is nothing left to return but an empty catalogue. A picker with
  /// no models in it is survivable; a crash on the way to opening it is not.
  Future<List<RouteInfo>> _cachedOrEmpty() async {
    try {
      return await db.cachedRoutes();
    } on Object {
      return const [];
    }
  }
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `fvm flutter test test/features/catalog_store_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
fvm dart format lib test
git add lib/features/catalog/catalog_store.dart test/features/catalog_store_test.dart
git commit -m "feat(catalog): refresh at most every ten minutes, never block the chat"
```

---

## Task 8: The model picker

**Files:**
- Create: `lib/features/catalog/model_picker.dart`
- Test: `test/features/model_picker_test.dart`

**Interfaces:**
- Consumes: `RouteInfo`, `CatalogStore.routes`.
- Produces: `Future<String?> showModelPicker(BuildContext context, {required List<RouteInfo> routes, required String? current, required DateTime now})` returning the chosen `model` string — a `modelId`, an `auto*` alias, or null when nothing was chosen; and `const List<ModelProfile> autoProfiles`.

The five aliases go on top because they are what the app sends by default, then
models grouped by provider. Forty-eight rows need a search field.

- [ ] **Step 1: Write the failing test**

```dart
  testWidgets('the auto profiles come first and return their alias',
      (tester) async {
    String? chosen;
    await tester.pumpWidget(MaterialApp(
      home: Builder(builder: (context) => TextButton(
        onPressed: () async {
          chosen = await showModelPicker(context,
              routes: [_route('p/m')], current: null,
              now: DateTime.utc(2026));
        },
        child: const Text('open'),
      )),
    ));
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();

    expect(find.text('auto'), findsOneWidget);
    expect(find.text('auto:strong'), findsOneWidget);
    await tester.tap(find.text('auto:fast'));
    await tester.pumpAndSettle();
    expect(chosen, 'auto:fast');
  });

  testWidgets('a model returns its id without the provider', (tester) async {
    // The gateway's `model` field takes the model's own id; sending
    // provider/model is a 404. The id keeps its own slashes.
    String? chosen;
    await tester.pumpWidget(MaterialApp(
      home: Builder(builder: (context) => TextButton(
        onPressed: () async {
          chosen = await showModelPicker(context,
              routes: [_route('kilo/nvidia/nemotron-3:free')],
              current: null, now: DateTime.utc(2026));
        },
        child: const Text('open'),
      )),
    ));
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('nvidia/nemotron-3:free'));
    await tester.pumpAndSettle();
    expect(chosen, 'nvidia/nemotron-3:free');
  });

  testWidgets('a route in cooldown cannot be picked', (tester) async {
    final now = DateTime.utc(2026, 9, 2, 10);
    final cold = RouteInfo(
        key: 'p/cold', provider: 'p', modelId: 'cold', tier: 'free',
        context: 1000,
        cooldownUntil:
            now.add(const Duration(minutes: 5)).millisecondsSinceEpoch / 1000);
    String? chosen = 'untouched';
    await tester.pumpWidget(MaterialApp(
      home: Builder(builder: (context) => TextButton(
        onPressed: () async {
          chosen = await showModelPicker(context,
              routes: [cold], current: null, now: now);
        },
        child: const Text('open'),
      )),
    ));
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('cold'));
    await tester.pumpAndSettle();
    // Still open, nothing chosen: the router will not pick it either.
    expect(chosen, 'untouched');
    expect(find.textContaining('cooldown'), findsOneWidget);
  });

  testWidgets('capabilities and tier are visible per model', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: Builder(builder: (context) => TextButton(
        onPressed: () => showModelPicker(context,
            routes: [
              RouteInfo(key: 'p/m', provider: 'p', modelId: 'm', tier: 'paid',
                  context: 200000, tools: true, vision: true)
            ],
            current: null, now: DateTime.utc(2026)),
        child: const Text('open'),
      )),
    ));
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    expect(find.text('tools'), findsOneWidget);
    expect(find.text('vision'), findsOneWidget);
    expect(find.text('paid'), findsOneWidget);
    expect(find.textContaining('200k'), findsOneWidget);
  });

  testWidgets('the search field narrows the list', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: Builder(builder: (context) => TextButton(
        onPressed: () => showModelPicker(context,
            routes: [_route('p/alpha'), _route('q/beta')],
            current: null, now: DateTime.utc(2026)),
        child: const Text('open'),
      )),
    ));
    await tester.tap(find.text('open'));
    await tester.pumpAndSettle();
    await tester.enterText(find.byType(TextField), 'bet');
    await tester.pumpAndSettle();
    expect(find.text('beta'), findsOneWidget);
    expect(find.text('alpha'), findsNothing);
  });
```

- [ ] **Step 2: Run them to verify they fail**

Run: `fvm flutter test test/features/model_picker_test.dart`
Expected: FAIL — `Target of URI doesn't exist`.

- [ ] **Step 3: Write the implementation**

```dart
import 'package:flutter/material.dart';

import '../../api/route_info.dart';

/// One of the gateway's virtual `auto*` aliases.
class ModelProfile {
  const ModelProfile(this.alias, this.summary);
  final String alias;
  final String summary;
}

/// The five aliases, with what each actually selects. Kept in the order the
/// gateway's README lists them.
const List<ModelProfile> autoProfiles = [
  ModelProfile('auto', 'Balanced: quality, reliability and latency weigh the same'),
  ModelProfile('auto:fast', 'Prioritises low latency, gives up some quality'),
  ModelProfile('auto:strong', 'Prioritises measured quality, gives up some latency'),
  ModelProfile('auto:tools', 'Balanced, and only routes that support tools'),
  ModelProfile('auto:vision', 'Balanced, and only routes that accept images'),
];

/// Opens the picker. Returns the string to put in a request's `model` field —
/// an alias or a model id — or null when the sheet was dismissed.
Future<String?> showModelPicker(
  BuildContext context, {
  required List<RouteInfo> routes,
  required String? current,
  required DateTime now,
}) =>
    showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      builder: (_) => _PickerSheet(routes: routes, current: current, now: now),
    );

class _PickerSheet extends StatefulWidget {
  const _PickerSheet({
    required this.routes,
    required this.current,
    required this.now,
  });

  final List<RouteInfo> routes;
  final String? current;
  final DateTime now;

  @override
  State<_PickerSheet> createState() => _PickerSheetState();
}

class _PickerSheetState extends State<_PickerSheet> {
  final _search = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  bool _matches(String text) =>
      _query.isEmpty || text.toLowerCase().contains(_query.toLowerCase());

  @override
  Widget build(BuildContext context) {
    // Grouped by provider so the same model at two providers reads as two
    // entries rather than a duplicate.
    final byProvider = <String, List<RouteInfo>>{};
    for (final r in widget.routes) {
      if (!_matches(r.modelId) && !_matches(r.provider)) continue;
      byProvider.putIfAbsent(r.provider, () => []).add(r);
    }
    final providers = byProvider.keys.toList()..sort();

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: _search,
              decoration: const InputDecoration(
                prefixIcon: Icon(Icons.search),
                hintText: 'Search models',
                border: OutlineInputBorder(),
              ),
              onChanged: (v) => setState(() => _query = v),
            ),
            const SizedBox(height: 8),
            Flexible(
              child: ListView(
                shrinkWrap: true,
                children: [
                  for (final p in autoProfiles)
                    if (_matches(p.alias))
                      ListTile(
                        title: Text(p.alias),
                        subtitle: Text(p.summary),
                        selected: widget.current == p.alias,
                        onTap: () => Navigator.of(context).pop(p.alias),
                      ),
                  for (final provider in providers) ...[
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
                      child: Text(provider,
                          style: Theme.of(context).textTheme.labelSmall),
                    ),
                    for (final r in byProvider[provider]!)
                      _RouteTile(
                        route: r,
                        selected: widget.current == r.modelId,
                        cold: r.inCooldown(widget.now),
                      ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _RouteTile extends StatelessWidget {
  const _RouteTile({
    required this.route,
    required this.selected,
    required this.cold,
  });

  final RouteInfo route;
  final bool selected;
  final bool cold;

  @override
  Widget build(BuildContext context) {
    final chips = [
      ...route.capabilities.map((c) => c),
      route.tier,
      '${(route.context / 1000).round()}k context',
    ];
    return ListTile(
      enabled: !cold,
      selected: selected,
      title: Text(route.modelId),
      subtitle: Wrap(
        spacing: 6,
        children: [
          for (final c in chips)
            Text(c, style: Theme.of(context).textTheme.labelSmall),
          if (cold)
            Text('in cooldown',
                style: Theme.of(context).textTheme.labelSmall),
        ],
      ),
      // A route the router will not pick is not offered: choosing it would
      // produce a 503 the user cannot act on.
      onTap: cold ? null : () => Navigator.of(context).pop(route.modelId),
    );
  }
}
```

- [ ] **Step 4: Run them to verify they pass**

Run: `fvm flutter test test/features/model_picker_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
fvm dart format lib test
git add lib/features/catalog/model_picker.dart test/features/model_picker_test.dart
git commit -m "feat(catalog): pick a model, or a profile, from what the gateway serves"
```

---

## Task 9: The choice reaches the request

**Files:**
- Modify: `lib/features/chat/chat_controller.dart`, `lib/features/chat/chat_screen.dart`, `lib/main.dart`
- Test: `test/features/chat_controller_test.dart`, `test/features/chat_screen_test.dart`

**Interfaces:**
- Consumes: `AppDb.updateModelOverride`, `showModelPicker`, `CatalogStore.routes`.
- Produces: `ChatController.send` sends the conversation's `modelOverride` when set; `ChatScreen` opens the picker from its title.

- [ ] **Step 1: Write the failing controller test**

```dart
  test('the conversation override decides the model asked for', () async {
    final id = await db.createConversation();
    await db.updateModelOverride(id, 'turbo');
    Map<String, dynamic>? sent;
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((request, bodyStream) async {
          final body = jsonDecode(await bodyStream.bytesToString())
              as Map<String, dynamic>;
          if (body['stream'] == true) sent = body;
          return http.StreamedResponse(
              Stream.fromIterable(['data: [DONE]\n\n'].map(utf8.encode)), 200);
        }),
      ),
      conversationId: id,
    );
    await controller.send('hi');
    expect(sent!['model'], 'turbo');
  });

  test('with no override it still asks for auto', () async {
    final id = await db.createConversation();
    Map<String, dynamic>? sent;
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((request, bodyStream) async {
          final body = jsonDecode(await bodyStream.bytesToString())
              as Map<String, dynamic>;
          if (body['stream'] == true) sent = body;
          return http.StreamedResponse(
              Stream.fromIterable(['data: [DONE]\n\n'].map(utf8.encode)), 200);
        }),
      ),
      conversationId: id,
    );
    await controller.send('hi');
    expect(sent!['model'], 'auto');
  });
```

- [ ] **Step 2: Run them, then read the override**

Run: `fvm flutter test test/features/chat_controller_test.dart --plain-name "override"`
Expected: FAIL — `model` is `auto` when an override is set.

In `send()`, after the history query and its supersession check:

```dart
    // The conversation's own choice, or `auto`. Read per turn rather than held
    // in a field: the picker writes it straight to the database, and a stale
    // copy here would keep asking for the model the user just changed away
    // from.
    final row = await (db.select(db.conversations)
          ..where((c) => c.id.equals(conversationId)))
        .getSingleOrNull();
    if (_superseded(done)) return _abandon(done);
    final model = row?.modelOverride ?? 'auto';
```

and pass it to the stream:

```dart
    _subscription = client
        .stream(messages: turns, model: model)
        .listen(
```

Careful: `_close` already has a local named `model` for the served model id, and
these are NOT in different scopes — an earlier revision of this plan said they
were and was wrong; it is a genuine redeclaration error. Name this one
`requested`, and name the query result `conversationRow` rather than `row`,
which collides the same way.

- [ ] **Step 3: Run them to verify they pass**

Run: `fvm flutter test test/features/chat_controller_test.dart`
Expected: PASS, every test in the file.

- [ ] **Step 4: Write the failing screen test**

```dart
  testWidgets('the title opens the picker and stores the choice',
      (tester) async {
    final db = AppDb.forTesting(NativeDatabase.memory());
    addTearDown(db.close);
    final id = await db.createConversation();
    await db.saveRoutes(
        [_route('perplexity/turbo')], DateTime.utc(2026, 9, 2, 10));

    await tester.pumpWidget(MaterialApp(
        home: ChatScreen(
            db: db,
            client: LlmClient(baseUrl: 'https://gw.test', apiKey: 'k'),
            conversationId: id,
            controllerFactory: (cid) => _FakeChatController(
                db: db,
                client: LlmClient(baseUrl: 'https://gw.test', apiKey: 'k'),
                conversationId: cid),
            // Injected so the sheet opens from the cache and never waits on a
            // drift live query, which the fake clock cannot resolve.
            catalogRoutes: () async => db.cachedRoutes())));
    await tester.pump();

    await tester.tap(find.byKey(const Key('model-title')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('turbo'));
    await tester.pumpAndSettle();

    final conversation = await (db.select(db.conversations)
          ..where((c) => c.id.equals(id)))
        .getSingle();
    expect(conversation.modelOverride, 'turbo');
    expect(find.text('turbo'), findsWidgets); // the title now shows it
  });
```

- [ ] **Step 5: Run it, then wire the screen**

Run: `fvm flutter test test/features/chat_screen_test.dart --plain-name "opens the picker"`
Expected: FAIL — no `model-title` key.

`ChatScreen` gains an injectable source, so a widget test never waits on drift:

```dart
    this.catalogRoutes,
```

```dart
  /// Where the picker's routes come from. Injectable because the real source is
  /// a drift-backed cache, and a widget test waiting on a drift live query
  /// deadlocks: those resolve on timers `flutter_test`'s fake clock never fires.
  final Future<List<RouteInfo>> Function()? catalogRoutes;
```

Replace the app bar title with a tappable one:

```dart
      appBar: AppBar(
        title: InkWell(
          key: const Key('model-title'),
          onTap: _pickModel,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(_model ?? 'auto'),
              const Icon(Icons.expand_more, size: 20),
            ],
          ),
        ),
      ),
```

with the state and handler:

```dart
  String? _model;

  Future<void> _loadModel() async {
    final row = await (widget.db.select(widget.db.conversations)
          ..where((c) => c.id.equals(_conversationId)))
        .getSingleOrNull();
    if (!mounted) return;
    setState(() => _model = row?.modelOverride);
  }

  Future<void> _pickModel() async {
    final routes = await (widget.catalogRoutes?.call() ??
        widget.db.cachedRoutes());
    if (!mounted) return;
    final chosen = await showModelPicker(
      context,
      routes: routes,
      current: _model,
      now: DateTime.now(),
    );
    if (chosen == null || !mounted) return;
    // 'auto' is stored as no override, so a conversation that was never
    // touched and one explicitly set back to auto behave identically.
    await widget.db
        .updateModelOverride(_conversationId, chosen == 'auto' ? null : chosen);
    if (!mounted) return;
    setState(() => _model = chosen == 'auto' ? null : chosen);
  }
```

Call `_loadModel()` from `initState` and from `_open` after the conversation
changes.

- [ ] **Step 6: Wire the real store in `main.dart`**

`_ChatHome` builds the `CatalogStore` once and passes its `routes` as
`catalogRoutes`, so the app refreshes from the gateway while tests inject the
cache:

```dart
    final catalog = CatalogStore(
      db: widget.db,
      client: CatalogClient(baseUrl: Config.baseUrl, apiKey: Config.apiKey),
    );
```

```dart
      catalogRoutes: () => catalog.routes(),
```

- [ ] **Step 7: Run the whole suite and commit**

```bash
fvm flutter test
fvm dart analyze
fvm dart format lib test
git add -A
git commit -m "feat(chat): the model you pick is the model that is asked for"
```

---

## Task 10: Close the two gaps that pass for the wrong reason

**Files:**
- Test: `test/features/chat_controller_test.dart`

**Interfaces:**
- Consumes: `ChatController.dispose`, `AppDb.finishMessage`.
- Produces: no production change. Two tests that fail when the code they claim to cover is deleted.

Slice 2 left these named: `_abandon`'s stranded-row arm is untested because the
existing test disposes at the FIRST await, where no placeholder exists yet, so
its "no row stranded" assertion passes trivially. The `beforeOpen` wiring gap
was closed by Task 6 Step 7.

- [ ] **Step 1: Write the failing test**

```dart
  test('a placeholder inserted just before the turn ended is not stranded',
      () async {
    // The existing dispose test disposes at the FIRST await, before any
    // placeholder exists, so its "nothing stranded" assertion cannot fail.
    // This one disposes AFTER the placeholder insert, which is the only window
    // where _abandon's strandedRow arm runs. Deleting that arm must fail here.
    final id = await db.createConversation();
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((_, __) async => http.StreamedResponse(
              const Stream<List<int>>.empty(), 200)),
      ),
      conversationId: id,
    );

    // Dispose once the assistant placeholder exists but before the stream is
    // subscribed: poll the rows rather than guessing a duration.
    final sending = controller.send('hi');
    for (var i = 0; i < 200; i++) {
      final rows = await db.watchMessages(id).first;
      if (rows.any((m) => m.role == 'assistant')) break;
      await Future<void>.delayed(const Duration(milliseconds: 1));
    }
    controller.dispose();
    await sending;

    final rows = await db.watchMessages(id).first;
    expect(rows.where((m) => m.status == 'streaming'), isEmpty);
    expect(rows.last.status, 'partial');
  });
```

- [ ] **Step 2: Prove it covers what it claims**

Delete the `strandedRow` handling from `_abandon`:

```dart
    if (strandedRow != null) {
      await db.finishMessage(strandedRow, content: '', status: 'partial');
    }
```

Run: `fvm flutter test test/features/chat_controller_test.dart --plain-name "just before the turn ended"`
Expected: FAIL — a row left at `streaming`.

Record that output, then put the code back and confirm it passes. A test that
stays green with the code removed is covering nothing; report it rather than
keeping it.

- [ ] **Step 3: Run the whole suite and commit**

```bash
fvm flutter test
fvm dart format lib test
git add -A
git commit -m "test(chat): the stranded-row arm is actually covered now"
```

---

## Follow-up carried into the slice that first DISPLAYS `lastFailure`

`CatalogStore.lastFailure` describes the caller's own awaited call only under
NON-OVERLAPPING use of one instance. Resetting it on entry — which is what stops
it going stale after a clean sweep — also made every call touch it before any
`await`, so two genuinely interleaved `routes()` calls on the same store can
clobber each other: call A records its failure, suspends inside
`_cachedOrEmpty`, and call B clears the field at its own entry before A's future
resolves. A's caller then reads null for a call that failed.

Latent today: nothing reads `lastFailure`, and Task 9 wires the store in without
displaying it. It becomes reachable the moment a screen shows "could not
refresh" AND something can invoke `routes()` twice concurrently — a retry
button, a poll timer, or a double tap on the picker title. Whoever adds that
either serialises the calls, or gives the failure to the caller as a return
value instead of reading it off shared state. The second is the better shape and
was not chosen here only because no consumer existed to shape it for.

## Follow-up: the picker can write to the wrong conversation

`_pickModel` reads `_conversationId` AFTER awaiting the sheet, so a conversation
switch while the sheet is open writes the chosen override onto whichever
conversation is current when the sheet closes — not the one it was opened from.

Verified unreachable today, empirically rather than by argument: the modal
barrier consumes the first tap outside the sheet and dismisses it, so the drawer
cannot be reached while it is open. It becomes reachable the moment anything can
change the conversation without dismissing the sheet — a deep link, a
notification tap, a non-modal presentation, or a wider layout showing the drawer
permanently.

The fix is one line: capture the id before the await and use the captured value.

```dart
    final target = _conversationId;
    final chosen = await showModelPicker(...);
    if (chosen == null || !mounted) return;
    await widget.db.updateModelOverride(target, chosen == 'auto' ? null : chosen);
```

Worth distinguishing from the other deferred items in this plan: those are
cosmetic or inert, and this one silently writes a setting onto the wrong record.
It is parked only because nothing can currently reach it.

## What this plan does not cover

Slices 4 to 6 of the spec — vision attachments and image generation, audio
(dictation and read-aloud), and the routes/traffic/usage panel — get their own
plans. The catalogue this slice builds is what they filter on: `x_requires`
comes from `Capability` (three values), while the picker's chips come from
`RouteInfo.capabilities` (seven), and slice 4 is where that distinction starts
to matter for real.

The remaining slice-2 follow-up not addressed here is `\r\r` as an SSE frame
separator, which is spec-legal and which no gateway in use sends.
