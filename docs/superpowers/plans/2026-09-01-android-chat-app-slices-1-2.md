# Android chat app — slices 1 and 2 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Flutter Android app that holds a real streaming conversation with the
llm-libre gateway, persists it on the device, and lets you pick between
conversations.

**Architecture:** A pure-Dart `api/` package with no Flutter imports holds the
transport — request building, SSE parsing, error mapping — so its tests run under
`flutter test` with no emulator. A drift database holds conversations and
messages. Riverpod carries state to a ChatGPT-shaped UI. There is no backend and
no repository layer: `api/` is the boundary.

**Tech Stack:** Flutter 3.38.3 via fvm, Dart ^3.10.1. Versions are pinned with
hard upper bounds where a major bump breaks the toolchain:
`drift` and `drift_dev` `>=2.30.0 <2.31.0`, `drift_flutter` `>=0.2.4 <0.3.0`
(0.3.x pulls sqlite3 3.x and drift_dev then cannot resolve at all),
`flutter_riverpod` `^2.6.1`, `go_router` `^17.3.0`,
`http` `^1.6.0`, `build_runner` `^2.15.0`,
plus `gpt_markdown` and `uuid` unpinned. This is the set `~/testenglish`
resolves with on the same Dart.

**Spec:** `docs/superpowers/specs/2026-09-01-android-chat-app-design.md`

## Global Constraints

- All code — identifiers, comments, strings — in English. No Spanish in source.
- Gateway base URL default: `https://llm.comparadorinternet.co`. The API key is
  a BUILD-TIME constant from `--dart-define-from-file=.env` — never a literal in
  source, and `.env` is never committed. The owner has accepted that the key
  ships inside the APK; it is rotatable at the gateway.
- `lib/api/` must not import `package:flutter/*`. A Flutter import there is a
  review rejection.
- `x_requires` accepts only `tools`, `vision`, `search`. Any other name is a 400
  from the gateway (fixed in llm-libre commit `08cbc46`).
- Decode response bodies with `utf8.decode(response.bodyBytes)`, never
  `response.body`. Dart's `http` falls back to latin1 when the server does not
  declare a charset, which turns every accented character into mojibake.
- The app repository is `~/llm-libre-chat`, separate from the llm-libre repo this
  plan lives in.
- Commit after every task.
- Never let `pub add` resolve a pinned dependency to latest. The constraints in
  Tech Stack are load-bearing, not preferences.
- Run `fvm dart format lib test` before every commit. The code blocks in this
  plan are hand-written and are NOT format-clean; transcribing them verbatim
  leaves the tree dirty, and the drift compounds across tasks.

---

## File Structure

| File | Responsibility |
|---|---|
| `lib/api/types.dart` | `Capability`, `ChatMessage`, `ChatAnswer`, `ChatDelta`. No behaviour. |
| `lib/api/errors.dart` | The `LlmError` hierarchy and `errorFromResponse`. |
| `lib/api/sse.dart` | Byte stream to SSE payload strings. Nothing gateway-specific. |
| `lib/api/llm_client.dart` | Request building and the two chat calls. |
| `lib/config.dart` | API key and base URL, compiled in from `.env`. |
| `lib/data/db.dart` | drift tables, DAO methods. |
| `lib/features/chat/chat_controller.dart` | `ChangeNotifier` driving one conversation. |
| `lib/features/chat/chat_screen.dart` | Message list, composer, stop button. |
| `lib/features/chat/conversation_drawer.dart` | Conversation list grouped by date. |
| `lib/app.dart` | Not written in these slices — see the note below. |

**On Riverpod and go_router:** both are declared dependencies because slices 3
to 6 need them, and neither is wired here. These two slices have one screen and
one drawer; a router for a single route, and a provider graph for a single
controller passed by constructor, would be scaffolding with no user. The
controller is a plain `ChangeNotifier` and the screen reads the database through
`StreamBuilder`. Introduce them in the slice that first needs a second route.

---

## Task 1: Project scaffold and toolchain

**Files:**
- Create: `~/llm-libre-chat/` (whole Flutter project)
- Modify: `~/llm-libre-chat/pubspec.yaml`

**Interfaces:**
- Consumes: nothing.
- Produces: a project where `fvm flutter test` runs.

- [ ] **Step 1: Install the pinned Flutter SDK**

The fvm cache is empty (verified 2026-09-01: `Directory Size: 0 B`), so this
downloads roughly 1–2 GB and takes several minutes.

```bash
fvm install 3.38.3
```

- [ ] **Step 2: Create the project**

```bash
cd ~ && fvm spawn 3.38.3 create --org co.cristiangarcia --platforms android \
  --project-name llm_libre_chat llm-libre-chat
cd ~/llm-libre-chat && fvm use 3.38.3 --force
```

- [ ] **Step 3: Add the dependencies**

Versions are PINNED, not resolved to latest. A bare `pub add` takes
`drift_flutter` to 0.3.x, which pulls `sqlite3` 3.x, and `drift_dev` then cannot
resolve at all — verified on this machine. The upper bound on `drift_flutter` is
what holds sqlite3 at 2.x. These exact constraints are the ones `~/testenglish`
resolves with on this same Dart 3.10.1.

(An earlier revision of this plan also added `flutter_secure_storage` here, and
Task 1 did install it. The API key moved to a build-time `.env` before Task 5
shipped, so the dependency has no user left; Task 5 removes it.)

```bash
cd ~/llm-libre-chat
fvm flutter pub add \
  flutter_riverpod:^2.6.1 \
  go_router:^17.3.0 \
  'drift:>=2.30.0 <2.31.0' \
  'drift_flutter:>=0.2.4 <0.3.0' \
  http:^1.6.0 \
  gpt_markdown \
  uuid
fvm flutter pub add --dev 'drift_dev:>=2.30.0 <2.31.0' build_runner:^2.15.0
```

- [ ] **Step 4: Verify the toolchain**

Run: `cd ~/llm-libre-chat && fvm flutter test`
Expected: PASS — the generated widget test runs.

- [ ] **Step 5: Commit**

```bash
cd ~/llm-libre-chat && git init && git add -A
git commit -m "chore: scaffold the Flutter project on the pinned SDK"
```

---

## Task 2: Wire types

**Files:**
- Create: `lib/api/types.dart`
- Test: `test/api/types_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `enum Capability { tools, vision, search }`;
  `String capabilityWire(Capability)`;
  `class ChatMessage { final String role; final String content;
  Map<String, dynamic> toJson(); }`;
  `class ChatAnswer { final String content; final String? reasoning;
  final String? model; final String? route; final String? tier;
  final int? attempts; }`.

- [ ] **Step 1: Write the failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/api/types.dart';

void main() {
  test('capability wire names match what the gateway accepts', () {
    // The gateway rejects anything outside this set with a 400 (llm-libre
    // commit 08cbc46), so a rename here is a runtime failure, not a typo.
    expect(Capability.values.map(capabilityWire).toList(),
        ['tools', 'vision', 'search']);
  });

  test('a chat message serialises to the OpenAI shape', () {
    expect(const ChatMessage(role: 'user', content: 'hola').toJson(),
        {'role': 'user', 'content': 'hola'});
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/api/types_test.dart`
Expected: FAIL — `Target of URI doesn't exist: 'package:llm_libre_chat/api/types.dart'`

- [ ] **Step 3: Write the implementation**

```dart
/// Capabilities a chat body may require through `x_requires`.
///
/// Deliberately not every capability the gateway knows: `images`,
/// `audio_speech`, `audio_transcription` and `translate` are decided by the
/// endpoint that was called, and naming one of them in a chat body is a 400.
enum Capability { tools, vision, search }

String capabilityWire(Capability c) => switch (c) {
      Capability.tools => 'tools',
      Capability.vision => 'vision',
      Capability.search => 'search',
    };

class ChatMessage {
  const ChatMessage({required this.role, required this.content});

  final String role; // 'user' | 'assistant' | 'system'
  final String content;

  Map<String, dynamic> toJson() => {'role': role, 'content': content};
}

/// A completed answer. [route] and [tier] come from response headers and are
/// null on the streaming path, where the headers cannot travel; [model] comes
/// from the body and is present either way.
class ChatAnswer {
  const ChatAnswer({
    required this.content,
    this.reasoning,
    this.model,
    this.route,
    this.tier,
    this.attempts,
  });

  final String content;
  final String? reasoning;
  final String? model;
  final String? route;
  final String? tier;
  final int? attempts;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/api/types_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/api/types.dart test/api/types_test.dart
git commit -m "feat(api): wire types for the chat contract"
```

---

## Task 3: Error mapping

**Files:**
- Create: `lib/api/errors.dart`
- Test: `test/api/errors_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `sealed class LlmError implements Exception` with subclasses
  `ModelGone(String message, List<String> suggestions)`,
  `Unsatisfiable(String message, int activeRoutes)`,
  `AllRoutesDown(String message, int compatibleRoutes, double? nextRelease, bool paidCapReached)`,
  `Unauthorized(String message)`, `RateLimited(String message)`,
  `UnexpectedStatus(int status, String body)`; and
  `LlmError errorFromResponse(int status, String body)`.

- [ ] **Step 1: Write the failing test**

Every body below was captured from the live gateway on 2026-09-01.

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/api/errors.dart';

void main() {
  test('404 carries the suggestions the gateway offers', () {
    final e = errorFromResponse(404,
        '{"detail":{"message":"the model \'x\' no longer exists",'
        '"suggestions":["gpt-5-4-t-mini","gpt-5-6-mini"]}}') as ModelGone;
    expect(e.suggestions, ['gpt-5-4-t-mini', 'gpt-5-6-mini']);
  });

  test('400 and 503 are different failures and must not be merged', () {
    // 400: no route can EVER serve this. 503: routes exist and are down.
    final bad = errorFromResponse(400,
        '{"detail":{"message":"no route satisfies the request",'
        '"active_routes":48}}');
    final down = errorFromResponse(503,
        '{"detail":{"message":"every route that could serve is down or in '
        'cooldown","compatible_routes":30,"next_release":1788275330.0,'
        '"paid_cap_reached":false}}');
    expect(bad, isA<Unsatisfiable>());
    expect((bad as Unsatisfiable).activeRoutes, 48);
    expect(down, isA<AllRoutesDown>());
    expect((down as AllRoutesDown).nextRelease, 1788275330.0);
  });

  test('401 and 429 return detail as a bare string, not an object', () {
    // The shape differs by status; assuming a Map here throws at runtime.
    expect((errorFromResponse(401, '{"detail":"invalid api key"}')
        as Unauthorized).message, 'invalid api key');
    expect((errorFromResponse(429, '{"detail":"too many requests for this key"}')
        as RateLimited).message, 'too many requests for this key');
  });

  test('a body that is not JSON at all still produces a typed error', () {
    // Cloudflare can return an HTML error page before the gateway is reached.
    expect(errorFromResponse(502, '<html>bad gateway</html>'),
        isA<UnexpectedStatus>());
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/api/errors_test.dart`
Expected: FAIL — `Target of URI doesn't exist`

- [ ] **Step 3: Write the implementation**

```dart
import 'dart:convert';

sealed class LlmError implements Exception {
  const LlmError();
}

/// 404: the requested model is not in the catalogue. Reaches streaming
/// requests too — the check runs before the SSE headers go out.
class ModelGone extends LlmError {
  const ModelGone(this.message, this.suggestions);
  final String message;
  final List<String> suggestions;
}

/// 400: no route can ever satisfy these requirements.
class Unsatisfiable extends LlmError {
  const Unsatisfiable(this.message, this.activeRoutes);
  final String message;
  final int activeRoutes;
}

/// 503: routes exist that could serve, but all are down or in cooldown.
/// [nextRelease] is a unix timestamp, or null when none is being punished.
class AllRoutesDown extends LlmError {
  const AllRoutesDown(
      this.message, this.compatibleRoutes, this.nextRelease, this.paidCapReached);
  final String message;
  final int compatibleRoutes;
  final double? nextRelease;
  final bool paidCapReached;
}

class Unauthorized extends LlmError {
  const Unauthorized(this.message);
  final String message;
}

class RateLimited extends LlmError {
  const RateLimited(this.message);
  final String message;
}

class UnexpectedStatus extends LlmError {
  const UnexpectedStatus(this.status, this.body);
  final int status;
  final String body;
}

class Disconnected extends LlmError {
  const Disconnected(this.cause);
  final Object cause;
}

LlmError errorFromResponse(int status, String body) {
  Object? detail;
  try {
    final decoded = jsonDecode(body);
    if (decoded is Map<String, dynamic>) detail = decoded['detail'];
  } on FormatException {
    detail = null;
  }
  final map = detail is Map<String, dynamic> ? detail : const <String, dynamic>{};
  final message = detail is String
      ? detail
      : (map['message'] as String?) ?? 'HTTP $status';

  return switch (status) {
    404 => ModelGone(
        message, (map['suggestions'] as List?)?.cast<String>() ?? const []),
    400 => Unsatisfiable(message, (map['active_routes'] as int?) ?? 0),
    503 => AllRoutesDown(
        message,
        (map['compatible_routes'] as int?) ?? 0,
        (map['next_release'] as num?)?.toDouble(),
        map['paid_cap_reached'] == true),
    401 => Unauthorized(message),
    429 => RateLimited(message),
    _ => UnexpectedStatus(status, body),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/api/errors_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/api/errors.dart test/api/errors_test.dart
git commit -m "feat(api): map every gateway failure to its own type"
```

---

## Task 4: Non-streaming chat

**Files:**
- Create: `lib/api/llm_client.dart`
- Test: `test/api/llm_client_test.dart`

**Interfaces:**
- Consumes: `Capability`, `capabilityWire`, `ChatMessage`, `ChatAnswer` from
  `types.dart`; `errorFromResponse` and the `LlmError` subclasses from
  `errors.dart`.
- Produces: `class LlmClient({required String baseUrl, required String apiKey, http.Client? httpClient})`
  with `Map<String, dynamic> buildBody({required List<ChatMessage> messages, String model, Set<Capability> requires, int? minContext, bool raw, int? maxTokens, bool stream})`
  and `Future<ChatAnswer> complete({required List<ChatMessage> messages, String model, Set<Capability> requires, int? minContext, bool raw, int? maxTokens})`.

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:llm_libre_chat/api/errors.dart';
import 'package:llm_libre_chat/api/llm_client.dart';
import 'package:llm_libre_chat/api/types.dart';

LlmClient clientReturning(http.Response response, {List<http.Request>? seen}) =>
    LlmClient(
      baseUrl: 'https://gw.test',
      apiKey: 'k',
      httpClient: MockClient((req) async {
        seen?.add(req);
        return response;
      }),
    );

void main() {
  test('the body carries only the extensions that were asked for', () {
    final client = clientReturning(http.Response('{}', 200));
    final body = client.buildBody(
      messages: const [ChatMessage(role: 'user', content: 'hi')],
      requires: {Capability.vision},
      minContext: 200000,
    );
    expect(body['model'], 'auto');
    expect(body['x_requires'], ['vision']);
    expect(body['x_min_context'], 200000);
    // Absent, not false: sending x_raw: false is noise on every request.
    expect(body.containsKey('x_raw'), isFalse);
    expect(body.containsKey('stream'), isFalse);
  });

  test('it reads the answer, the reasoning and the route headers', () async {
    final client = clientReturning(http.Response(
      jsonEncode({
        'model': 'turbo',
        'choices': [
          {'message': {'role': 'assistant', 'content': 'the answer is 4'}}
        ],
        'x_reasoning': '2+2 is 4',
      }),
      200,
      headers: {
        'x-route-used': 'perplexity/turbo',
        'x-tier': 'free',
        'x-attempts': '2',
      },
    ));
    final answer = await client.complete(
        messages: const [ChatMessage(role: 'user', content: 'hi')]);
    expect(answer.content, 'the answer is 4');
    expect(answer.reasoning, '2+2 is 4');
    expect(answer.route, 'perplexity/turbo');
    expect(answer.attempts, 2);
  });

  test('accented content survives a body with no charset declared', () async {
    // Dart's http falls back to latin1 when the server declares no charset, so
    // reading `response.body` here yields mojibake. The client must decode
    // bodyBytes as utf8 itself.
    final client = clientReturning(http.Response.bytes(
      utf8.encode(jsonEncode({
        'choices': [{'message': {'content': 'el niño comió'}}]
      })),
      200,
    ));
    final answer = await client.complete(
        messages: const [ChatMessage(role: 'user', content: 'hi')]);
    expect(answer.content, 'el niño comió');
  });

  test('a non-200 is thrown as its typed error', () async {
    final client = clientReturning(http.Response(
        '{"detail":{"message":"gone","suggestions":["a"]}}', 404));
    expect(
      () => client.complete(
          messages: const [ChatMessage(role: 'user', content: 'hi')]),
      throwsA(isA<ModelGone>()),
    );
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/api/llm_client_test.dart`
Expected: FAIL — `Target of URI doesn't exist: 'package:llm_libre_chat/api/llm_client.dart'`

- [ ] **Step 3: Write the implementation**

```dart
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'errors.dart';
import 'types.dart';

class LlmClient {
  LlmClient({
    required this.baseUrl,
    required this.apiKey,
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client();

  final String baseUrl;
  final String apiKey;
  final http.Client _http;

  Map<String, String> get _headers => {
        'Authorization': 'Bearer $apiKey',
        'Content-Type': 'application/json',
      };

  Uri get _chatUrl => Uri.parse('$baseUrl/v1/chat/completions');

  /// Every `x_*` extension is omitted rather than sent false or empty: a third
  /// party reading these requests should see only what was actually asked for.
  Map<String, dynamic> buildBody({
    required List<ChatMessage> messages,
    String model = 'auto',
    Set<Capability> requires = const {},
    int? minContext,
    bool raw = false,
    int? maxTokens,
    bool stream = false,
  }) =>
      {
        'model': model,
        'messages': [for (final m in messages) m.toJson()],
        if (requires.isNotEmpty)
          'x_requires': [for (final c in requires) capabilityWire(c)],
        if (minContext != null) 'x_min_context': minContext,
        if (raw) 'x_raw': true,
        if (maxTokens != null) 'max_tokens': maxTokens,
        if (stream) 'stream': true,
      };

  Future<ChatAnswer> complete({
    required List<ChatMessage> messages,
    String model = 'auto',
    Set<Capability> requires = const {},
    int? minContext,
    bool raw = false,
    int? maxTokens,
  }) async {
    final response = await _http.post(
      _chatUrl,
      headers: _headers,
      body: jsonEncode(buildBody(
        messages: messages,
        model: model,
        requires: requires,
        minContext: minContext,
        raw: raw,
        maxTokens: maxTokens,
      )),
    );
    // Not `response.body`: http decodes as latin1 unless the server declares a
    // charset, and the gateway does not.
    final text = utf8.decode(response.bodyBytes);
    if (response.statusCode != 200) {
      throw errorFromResponse(response.statusCode, text);
    }
    final json = jsonDecode(text) as Map<String, dynamic>;
    final choices = json['choices'] as List?;
    final message = choices != null && choices.isNotEmpty
        ? (choices.first as Map<String, dynamic>)['message']
            as Map<String, dynamic>?
        : null;
    return ChatAnswer(
      content: (message?['content'] as String?) ?? '',
      reasoning: json['x_reasoning'] as String?,
      model: json['model'] as String?,
      route: response.headers['x-route-used'],
      tier: response.headers['x-tier'],
      attempts: int.tryParse(response.headers['x-attempts'] ?? ''),
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/api/llm_client_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/api/llm_client.dart test/api/llm_client_test.dart
git commit -m "feat(api): non-streaming chat with typed failures"
```

---

## Task 5: Build-time configuration from `.env`

**Files:**
- Create: `lib/config.dart`
- Create: `.env.example`
- Create: `.env` (git-ignored, holds the real key)
- Modify: `.gitignore`
- Modify: `pubspec.yaml` (remove `flutter_secure_storage`)
- Test: `test/config_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `abstract final class Config` with
  `static const String apiKey`, `static const String baseUrl`, and
  `static bool get isConfigured`.

The app is single-user and its owner has decided the key ships inside the build.
That makes the key a compile-time constant, not a stored secret:
`--dart-define-from-file=.env` is native to Flutter 3.38.3 and reads a plain
`.env`, so this needs no package at all. `flutter_secure_storage` is removed
because nothing is left for it to hold.

**The key is in the binary and an APK is decompilable.** That is the accepted
trade for a personal app. It means the key must be rotatable at the gateway and
must never be committed: `.env` is git-ignored and `.env.example` carries the
shape with no value.

- [ ] **Step 1: Write the failing test**

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/config.dart';

void main() {
  test('the base URL falls back to the production gateway', () {
    // Tests run without --dart-define-from-file, so this exercises the default.
    expect(Config.baseUrl, 'https://llm.comparadorinternet.co');
  });

  test('a build with no key reports itself unconfigured', () {
    // Without this the app would 401 on every message and read as a broken
    // gateway rather than a build that was never given a key.
    expect(Config.apiKey, isEmpty);
    expect(Config.isConfigured, isFalse);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/config_test.dart`
Expected: FAIL — `Target of URI doesn't exist: 'package:llm_libre_chat/config.dart'`

- [ ] **Step 3: Write the implementation**

```dart
/// Build-time configuration, supplied by `--dart-define-from-file=.env`.
///
/// These MUST be `const`: `String.fromEnvironment` only reads the build's
/// defines in a const context, and silently yields the default otherwise.
abstract final class Config {
  static const String apiKey = String.fromEnvironment('LLM_LIBRE_API_KEY');

  static const String baseUrl = String.fromEnvironment(
    'LLM_LIBRE_BASE_URL',
    defaultValue: 'https://llm.comparadorinternet.co',
  );

  static bool get isConfigured => apiKey.isNotEmpty;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/config_test.dart`
Expected: PASS

- [ ] **Step 5: Create the env files and ignore the real one**

`.env.example` — committed, no value:

```
LLM_LIBRE_API_KEY=
LLM_LIBRE_BASE_URL=https://llm.comparadorinternet.co
```

`.env` — NOT committed, same keys with the real value filled in.

Append to `.gitignore`:

```
# Holds the real API key, which is compiled into the build.
.env
```

- [ ] **Step 6: Remove the now-unused dependency**

```bash
fvm flutter pub remove flutter_secure_storage
```

- [ ] **Step 7: Prove the define actually reaches the constant**

Run: `fvm flutter test --dart-define=LLM_LIBRE_API_KEY=probe test/config_test.dart`
Expected: FAIL on the second test, which asserts the key is empty — that failure
is the proof the define is wired. Record the output in your report, then move on;
do not change the test to accommodate it.

- [ ] **Step 8: Commit**

```bash
fvm dart format lib test
git add -A
git commit -m "feat(config): API key and base URL from a build-time .env"
```

---

## Task 6: First screen — send one message, see one answer

**Files:**
- Create: `lib/features/chat/simple_chat_screen.dart`
- Modify: `lib/main.dart`
- Test: `test/features/simple_chat_screen_test.dart`

**Interfaces:**
- Consumes: `LlmClient.complete`, `Config`.
- Produces: `class SimpleChatScreen extends StatefulWidget` taking a
  `LlmClient client`. Replaced in Task 10; it exists so slice 1 is a running app.

**Startup is synchronous.** The key is a compile-time constant, so there is no
async read and no loading state: `main.dart` builds
`LlmClient(baseUrl: Config.baseUrl, apiKey: Config.apiKey)` and shows the chat.
When `Config.isConfigured` is false it shows a single line naming the fix
(`rebuild with --dart-define-from-file=.env`) instead — otherwise a build with no
key 401s on every message and reads as a broken gateway.

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:llm_libre_chat/api/llm_client.dart';
import 'package:llm_libre_chat/features/chat/simple_chat_screen.dart';

void main() {
  testWidgets('typing and sending shows the answer', (tester) async {
    final client = LlmClient(
      baseUrl: 'https://gw.test',
      apiKey: 'k',
      httpClient: MockClient((_) async => http.Response.bytes(
            utf8.encode(jsonEncode({
              'choices': [{'message': {'content': 'pong'}}]
            })),
            200,
          )),
    );
    await tester.pumpWidget(
        MaterialApp(home: SimpleChatScreen(client: client)));
    await tester.enterText(find.byType(TextField), 'ping');
    await tester.tap(find.byIcon(Icons.send));
    await tester.pumpAndSettle();
    expect(find.text('pong'), findsOneWidget);
  });

  testWidgets('a failure is shown instead of an empty screen', (tester) async {
    final client = LlmClient(
      baseUrl: 'https://gw.test',
      apiKey: 'k',
      httpClient: MockClient((_) async =>
          http.Response('{"detail":"invalid api key"}', 401)),
    );
    await tester.pumpWidget(
        MaterialApp(home: SimpleChatScreen(client: client)));
    await tester.enterText(find.byType(TextField), 'ping');
    await tester.tap(find.byIcon(Icons.send));
    await tester.pumpAndSettle();
    expect(find.textContaining('invalid api key'), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/features/simple_chat_screen_test.dart`
Expected: FAIL — `Target of URI doesn't exist`

- [ ] **Step 3: Write the implementation**

```dart
import 'package:flutter/material.dart';

import '../../api/errors.dart';
import '../../api/llm_client.dart';
import '../../api/types.dart';

/// The smallest thing that proves the transport works end to end. Replaced by
/// the real chat screen in Task 10.
class SimpleChatScreen extends StatefulWidget {
  const SimpleChatScreen({super.key, required this.client});

  final LlmClient client;

  @override
  State<SimpleChatScreen> createState() => _SimpleChatScreenState();
}

class _SimpleChatScreenState extends State<SimpleChatScreen> {
  final _input = TextEditingController();
  final _turns = <ChatMessage>[];
  String? _failure;
  bool _busy = false;

  @override
  void dispose() {
    _input.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final text = _input.text.trim();
    if (text.isEmpty || _busy) return;
    setState(() {
      _turns.add(ChatMessage(role: 'user', content: text));
      _input.clear();
      _failure = null;
      _busy = true;
    });
    try {
      final answer = await widget.client.complete(messages: _turns);
      setState(() =>
          _turns.add(ChatMessage(role: 'assistant', content: answer.content)));
    } on LlmError catch (e) {
      setState(() => _failure = _describe(e));
    } finally {
      setState(() => _busy = false);
    }
  }

  String _describe(LlmError e) => switch (e) {
        Unauthorized(:final message) => message,
        RateLimited(:final message) => message,
        ModelGone(:final message) => message,
        Unsatisfiable(:final message) => message,
        AllRoutesDown(:final message) => message,
        UnexpectedStatus(:final status) => 'unexpected status $status',
        Disconnected() => 'connection lost',
      };

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('llm-libre')),
        body: Column(
          children: [
            if (_failure != null)
              Padding(
                padding: const EdgeInsets.all(12),
                child: Text(_failure!,
                    style: TextStyle(color: Theme.of(context).colorScheme.error)),
              ),
            Expanded(
              child: ListView.builder(
                itemCount: _turns.length,
                itemBuilder: (_, i) => ListTile(
                  title: Text(_turns[i].content),
                  subtitle: Text(_turns[i].role),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8),
              child: Row(
                children: [
                  Expanded(child: TextField(controller: _input)),
                  IconButton(
                      onPressed: _busy ? null : _send,
                      icon: const Icon(Icons.send)),
                ],
              ),
            ),
          ],
        ),
      );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/features/simple_chat_screen_test.dart`
Expected: PASS

- [ ] **Step 5: Wire `main.dart` and commit**

`main.dart` builds the client from `Config` and shows `SimpleChatScreen`, or a
single centred line naming the missing define when `Config.isConfigured` is
false. No `FutureBuilder`, no loading state — the values are compile-time
constants.

Run: `fvm flutter test`
Expected: PASS, every test.

```bash
fvm dart format lib test
git add -A
git commit -m "feat(chat): a screen that sends one message and shows the answer"
```

---

## Task 7: SSE parser

**Files:**
- Create: `lib/api/sse.dart`
- Test: `test/api/sse_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `Stream<String> sseEvents(Stream<List<int>> bytes)` — yields each
  `data:` payload as a string, `[DONE]` included, leaving interpretation to the
  caller.

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/api/sse.dart';

void main() {
  test('it yields one payload per data line', () async {
    final bytes = Stream.fromIterable(
        [utf8.encode('data: {"a":1}\n\ndata: [DONE]\n\n')]);
    expect(await sseEvents(bytes).toList(), ['{"a":1}', '[DONE]']);
  });

  test('an event split across two chunks is reassembled', () async {
    final whole = utf8.encode('data: {"a":1}\n\n');
    final bytes = Stream.fromIterable(
        [whole.sublist(0, 7), whole.sublist(7)]);
    expect(await sseEvents(bytes).toList(), ['{"a":1}']);
  });

  test('a chunk cut inside a multi-byte character does not corrupt it', () async {
    // The gateway answers in Spanish constantly, so this is the common case,
    // not an exotic one. 'ñ' is 0xC3 0xB1: cutting between them and decoding
    // each chunk on its own yields two replacement characters.
    final whole = utf8.encode('data: {"c":"el niño"}\n\n');
    final cut = whole.indexOf(0xC3) + 1;
    final bytes =
        Stream.fromIterable([whole.sublist(0, cut), whole.sublist(cut)]);
    expect(await sseEvents(bytes).toList(), ['{"c":"el niño"}']);
  });

  test('lines that are not data are ignored', () async {
    final bytes = Stream.fromIterable(
        [utf8.encode(': keep-alive\n\nevent: ping\ndata: {"a":1}\n\n')]);
    expect(await sseEvents(bytes).toList(), ['{"a":1}']);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/api/sse_test.dart`
Expected: FAIL — `Target of URI doesn't exist`

- [ ] **Step 3: Write the implementation**

```dart
import 'dart:convert';

/// Yields the payload of each `data:` line of a Server-Sent Events stream.
///
/// Both stages are stateful on purpose. `utf8.decoder` used as a stream
/// transformer holds an incomplete multi-byte sequence until its continuation
/// arrives; decoding each chunk on its own would corrupt any character that
/// straddles a chunk boundary. The buffer below does the same for events split
/// across chunks.
Stream<String> sseEvents(Stream<List<int>> bytes) async* {
  var buffer = '';
  await for (final chunk in bytes.transform(utf8.decoder)) {
    buffer += chunk;
    while (true) {
      final end = buffer.indexOf('\n\n');
      if (end < 0) break;
      final block = buffer.substring(0, end);
      buffer = buffer.substring(end + 2);
      for (final line in const LineSplitter().convert(block)) {
        if (line.startsWith('data:')) {
          yield line.substring(5).trim();
        }
      }
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/api/sse_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/api/sse.dart test/api/sse_test.dart
git commit -m "feat(api): SSE parser that survives chunk boundaries"
```

---

## Task 8: Streaming chat

**Files:**
- Modify: `lib/api/llm_client.dart`
- Modify: `lib/api/types.dart`
- Test: `test/api/llm_client_stream_test.dart`

**Interfaces:**
- Consumes: `sseEvents` from `sse.dart`.
- Produces: `class ChatDelta { final String? text; final String? model; final bool done; }`
  in `types.dart`, and
  `Stream<ChatDelta> stream({required List<ChatMessage> messages, String model, Set<Capability> requires, int? minContext, bool raw, int? maxTokens})`
  on `LlmClient`.

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:llm_libre_chat/api/errors.dart';
import 'package:llm_libre_chat/api/llm_client.dart';
import 'package:llm_libre_chat/api/types.dart';

LlmClient streamingClient(String body, {int status = 200}) => LlmClient(
      baseUrl: 'https://gw.test',
      apiKey: 'k',
      httpClient: MockClient.streaming((request, _) async =>
          http.StreamedResponse(
              Stream.fromIterable([utf8.encode(body)]), status)),
    );

void main() {
  test('deltas carry the text and the model that served', () async {
    // Captured from the live gateway: the model is in every chunk even though
    // X-Route-Used cannot travel on a stream.
    final client = streamingClient(
      'data: {"model":"turbo","choices":[{"delta":{"content":"ho"}}]}\n\n'
      'data: {"model":"turbo","choices":[{"delta":{"content":"la"}}]}\n\n'
      'data: [DONE]\n\n',
    );
    final deltas = await client
        .stream(messages: const [ChatMessage(role: 'user', content: 'hi')])
        .toList();
    expect(deltas.map((d) => d.text).whereType<String>().join(), 'hola');
    expect(deltas.first.model, 'turbo');
    expect(deltas.last.done, isTrue);
  });

  test('unknown fields such as _pplx do not break parsing', () async {
    // perplexity leaks its own metadata into the final chunk.
    final client = streamingClient(
      'data: {"model":"turbo","choices":[{"delta":{},"finish_reason":"stop"}],'
      '"_pplx":{"display_model":"turbo"}}\n\n'
      'data: [DONE]\n\n',
    );
    final deltas = await client
        .stream(messages: const [ChatMessage(role: 'user', content: 'hi')])
        .toList();
    expect(deltas.last.done, isTrue);
  });

  test('a 404 on the streaming path still arrives as ModelGone', () async {
    // Verified live 2026-09-01: the catalogue check runs before the SSE
    // headers, so stream: true does receive the suggestions.
    final client = streamingClient(
        '{"detail":{"message":"gone","suggestions":["a","b"]}}',
        status: 404);
    expect(
      client.stream(messages: const [ChatMessage(role: 'user', content: 'hi')]).toList(),
      throwsA(isA<ModelGone>()),
    );
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/api/llm_client_stream_test.dart`
Expected: FAIL — `The method 'stream' isn't defined for the type 'LlmClient'`

- [ ] **Step 3: Add ChatDelta to `types.dart`**

```dart
/// One step of a streamed answer. [model] repeats on every chunk; the app keeps
/// the last non-null one to attribute the answer, because `X-Route-Used` cannot
/// travel on a stream.
class ChatDelta {
  const ChatDelta({this.text, this.model, this.done = false});

  final String? text;
  final String? model;
  final bool done;
}
```

- [ ] **Step 4: Add `stream` to `LlmClient`**

Add the import `import 'sse.dart';` and this method:

```dart
  Stream<ChatDelta> stream({
    required List<ChatMessage> messages,
    String model = 'auto',
    Set<Capability> requires = const {},
    int? minContext,
    bool raw = false,
    int? maxTokens,
  }) async* {
    final request = http.Request('POST', _chatUrl)
      ..headers.addAll(_headers)
      ..body = jsonEncode(buildBody(
        messages: messages,
        model: model,
        requires: requires,
        minContext: minContext,
        raw: raw,
        maxTokens: maxTokens,
        stream: true,
      ));
    final response = await _http.send(request);
    if (response.statusCode != 200) {
      throw errorFromResponse(
          response.statusCode, utf8.decode(await response.stream.toBytes()));
    }
    await for (final payload in sseEvents(response.stream)) {
      if (payload == '[DONE]') {
        yield const ChatDelta(done: true);
        return;
      }
      final Map<String, dynamic> json;
      try {
        json = jsonDecode(payload) as Map<String, dynamic>;
      } on FormatException {
        continue; // a keep-alive or a frame this version does not understand
      }
      final choices = json['choices'] as List?;
      final delta = choices != null && choices.isNotEmpty
          ? (choices.first as Map<String, dynamic>)['delta']
              as Map<String, dynamic>?
          : null;
      yield ChatDelta(
        text: delta?['content'] as String?,
        model: json['model'] as String?,
      );
    }
  }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `fvm flutter test test/api/llm_client_stream_test.dart`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lib/api/llm_client.dart lib/api/types.dart \
  test/api/llm_client_stream_test.dart
git commit -m "feat(api): streaming chat with per-chunk attribution"
```

---

## Task 9: Persistence

**Files:**
- Create: `lib/data/db.dart`
- Test: `test/data/db_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `class AppDb extends _$AppDb` with
  `Future<String> createConversation()`,
  `Future<void> renameConversation(String id, String title)`,
  `Future<int> addMessage({required String conversationId, required String role, required String content, String? reasoning, String? modelUsed, String? routeUsed, String status})`,
  `Future<void> finishMessage(int id, {required String content, required String status, String? modelUsed})`,
  `Stream<List<Conversation>> watchConversations()`,
  `Stream<List<Message>> watchMessages(String conversationId)`.

- [ ] **Step 1: Write the failing test**

```dart
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/data/db.dart';

void main() {
  late AppDb db;

  setUp(() => db = AppDb.forTesting(NativeDatabase.memory()));
  tearDown(() => db.close());

  test('a conversation holds its messages in order', () async {
    final id = await db.createConversation();
    await db.addMessage(conversationId: id, role: 'user', content: 'hi');
    await db.addMessage(conversationId: id, role: 'assistant', content: 'hello');
    final messages = await db.watchMessages(id).first;
    expect(messages.map((m) => m.content), ['hi', 'hello']);
  });

  test('a streaming message can be finished in place', () async {
    // The row is written before the answer exists so the UI has something to
    // render while it arrives; a lost connection leaves it as 'partial'.
    final id = await db.createConversation();
    final row = await db.addMessage(
        conversationId: id, role: 'assistant', content: '', status: 'streaming');
    await db.finishMessage(row,
        content: 'done', status: 'ok', modelUsed: 'turbo');
    final messages = await db.watchMessages(id).first;
    expect(messages.single.status, 'ok');
    expect(messages.single.modelUsed, 'turbo');
  });

  test('conversations come back newest first', () async {
    // This also pins `storeDateTimeAsText: true`. Under drift's default of
    // whole-second unix timestamps these two rows share an `updatedAt` and the
    // order becomes whatever SQLite returns, so a silent revert of that option
    // fails here rather than in the drawer.
    final older = await db.createConversation();
    await db.renameConversation(older, 'older');
    final newer = await db.createConversation();
    await db.renameConversation(newer, 'newer');
    final list = await db.watchConversations().first;
    expect(list.first.title, 'newer');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/data/db_test.dart`
Expected: FAIL — `Target of URI doesn't exist`

- [ ] **Step 3: Write the schema and DAO**

```dart
import 'package:drift/drift.dart';
import 'package:drift_flutter/drift_flutter.dart';
import 'package:uuid/uuid.dart';

part 'db.g.dart';

class Conversations extends Table {
  TextColumn get id => text()();
  TextColumn get title => text().withDefault(const Constant(''))();
  DateTimeColumn get createdAt => dateTime()();
  DateTimeColumn get updatedAt => dateTime()();
  BoolColumn get pinned => boolean().withDefault(const Constant(false))();
  TextColumn get modelOverride => text().nullable()();
  TextColumn get systemPrompt => text().nullable()();

  @override
  Set<Column> get primaryKey => {id};
}

class Messages extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get conversationId =>
      text().references(Conversations, #id, onDelete: KeyAction.cascade)();
  TextColumn get role => text()();
  TextColumn get content => text()();
  TextColumn get reasoning => text().nullable()();
  TextColumn get modelUsed => text().nullable()();
  TextColumn get routeUsed => text().nullable()();
  /// 'ok' | 'streaming' | 'partial' | 'error'
  TextColumn get status => text().withDefault(const Constant('ok'))();
  DateTimeColumn get createdAt => dateTime()();
}

@DriftDatabase(tables: [Conversations, Messages])
class AppDb extends _$AppDb {
  AppDb() : super(driftDatabase(name: 'llm_libre_chat'));
  AppDb.forTesting(super.executor);

  static const _uuid = Uuid();

  @override
  int get schemaVersion => 1;

  /// Datetimes are stored as ISO-8601 text, not as unix timestamps.
  ///
  /// drift's default is integer seconds, and it says so only to stay
  /// backwards-compatible with existing databases. This one is new, so it costs
  /// nothing to choose — and the default is lossy in a way that shows: two
  /// conversations created in the same second get identical `updatedAt` values,
  /// `ORDER BY updatedAt DESC` cannot separate them, and the drawer lists them
  /// in whatever order SQLite happens to return. Measured, not theorised: with
  /// the default, the "newest first" test below fails 5 times out of 5.
  @override
  DriftDatabaseOptions get options =>
      const DriftDatabaseOptions(storeDateTimeAsText: true);

  Future<String> createConversation() async {
    final now = DateTime.now();
    final id = _uuid.v4();
    await into(conversations).insert(ConversationsCompanion.insert(
        id: id, createdAt: now, updatedAt: now));
    return id;
  }

  Future<void> renameConversation(String id, String title) =>
      (update(conversations)..where((c) => c.id.equals(id)))
          .write(ConversationsCompanion(
              title: Value(title), updatedAt: Value(DateTime.now())));

  Future<int> addMessage({
    required String conversationId,
    required String role,
    required String content,
    String? reasoning,
    String? modelUsed,
    String? routeUsed,
    String status = 'ok',
  }) =>
      into(messages).insert(MessagesCompanion.insert(
        conversationId: conversationId,
        role: role,
        content: content,
        reasoning: Value(reasoning),
        modelUsed: Value(modelUsed),
        routeUsed: Value(routeUsed),
        status: Value(status),
        createdAt: DateTime.now(),
      ));

  Future<void> finishMessage(
    int id, {
    required String content,
    required String status,
    String? modelUsed,
  }) =>
      (update(messages)..where((m) => m.id.equals(id))).write(MessagesCompanion(
        content: Value(content),
        status: Value(status),
        modelUsed: Value(modelUsed),
      ));

  Stream<List<Conversation>> watchConversations() =>
      (select(conversations)
            ..orderBy([
              (c) => OrderingTerm.desc(c.pinned),
              (c) => OrderingTerm.desc(c.updatedAt),
            ]))
          .watch();

  Stream<List<Message>> watchMessages(String conversationId) =>
      (select(messages)
            ..where((m) => m.conversationId.equals(conversationId))
            ..orderBy([(m) => OrderingTerm.asc(m.id)]))
          .watch();
}
```

- [ ] **Step 4: Generate the drift code**

Run: `fvm dart run build_runner build --delete-conflicting-outputs`
Expected: `db.g.dart` is written with no errors.

- [ ] **Step 5: Run test to verify it passes**

Run: `fvm flutter test test/data/db_test.dart`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lib/data/db.dart lib/data/db.g.dart test/data/db_test.dart
git commit -m "feat(data): on-device conversations and messages"
```

---

## Task 10: The real chat screen

**Files:**
- Create: `lib/features/chat/chat_controller.dart`
- Create: `lib/features/chat/chat_screen.dart`
- Create: `lib/features/chat/conversation_drawer.dart`
- Delete: `lib/features/chat/simple_chat_screen.dart` and its test
- Modify: `lib/main.dart`
- Test: `test/features/chat_controller_test.dart`

**Interfaces:**
- Consumes: `LlmClient.stream`, `AppDb`, `Config`.
- Produces: `class ChatController` with
  `Future<void> send(String text)`, `Future<void> stop()`, and a `bool get busy`;
  `class ChatScreen` and `class ConversationDrawer` widgets.

- [ ] **Step 1: Write the failing test**

```dart
import 'dart:convert';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:llm_libre_chat/api/llm_client.dart';
import 'package:llm_libre_chat/data/db.dart';
import 'package:llm_libre_chat/features/chat/chat_controller.dart';

LlmClient clientStreaming(List<String> frames) => LlmClient(
      baseUrl: 'https://gw.test',
      apiKey: 'k',
      httpClient: MockClient.streaming((_, __) async => http.StreamedResponse(
          Stream.fromIterable(frames.map(utf8.encode)), 200)),
    );

void main() {
  late AppDb db;
  setUp(() => db = AppDb.forTesting(NativeDatabase.memory()));
  tearDown(() => db.close());

  test('a streamed answer is persisted as it arrives and then finished',
      () async {
    final id = await db.createConversation();
    final controller = ChatController(
      db: db,
      client: clientStreaming([
        'data: {"model":"turbo","choices":[{"delta":{"content":"ho"}}]}\n\n',
        'data: {"model":"turbo","choices":[{"delta":{"content":"la"}}]}\n\n',
        'data: [DONE]\n\n',
      ]),
      conversationId: id,
    );
    await controller.send('hi');
    final stored = await db.watchMessages(id).first;
    expect(stored.map((m) => m.content), ['hi', 'hola']);
    expect(stored.last.status, 'ok');
    expect(stored.last.modelUsed, 'turbo');
  });

  test('an empty row from an earlier turn is not sent as context', () async {
    // watchMessages returns every row, placeholder and all. Task 9's review
    // flagged this as a trap baked into the shared interface: without the
    // filter, the second message of a conversation carries an empty assistant
    // turn to the gateway, which is meaningless as context and which some
    // providers reject outright.
    final id = await db.createConversation();
    await db.addMessage(conversationId: id, role: 'user', content: 'first');
    final dead = await db.addMessage(
        conversationId: id, role: 'assistant', content: '', status: 'partial');
    await db.finishMessage(dead, content: '', status: 'partial');

    List<dynamic>? sentMessages;
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        // MockClient.streaming hands the request body over as its second
        // argument, so the outgoing turns can be read directly.
        httpClient: MockClient.streaming((request, bodyStream) async {
          final body = jsonDecode(await bodyStream.bytesToString())
              as Map<String, dynamic>;
          // A send makes TWO requests through this same client: the streamed
          // chat turn, and then a plain completion that names the conversation.
          // This test is about the chat turn, so it captures that one by its
          // `stream` flag; capturing unconditionally lets the titling request
          // overwrite the answer before the assertions run.
          if (body['stream'] == true) {
            sentMessages = body['messages'] as List<dynamic>;
          }
          // Must emit real content: a stream of nothing but [DONE] finishes the
          // new placeholder empty too, leaving TWO empty rows and making the
          // assertion below about the pre-seeded one meaningless.
          return http.StreamedResponse(
              Stream.fromIterable([
                'data: {"model":"turbo","choices":[{"delta":{"content":"ok"}}]}\n\n',
                'data: [DONE]\n\n',
              ].map(utf8.encode)),
              200);
        }),
      ),
      conversationId: id,
    );
    await controller.send('second');

    expect(sentMessages, isNotNull);
    expect(sentMessages!.any((m) => (m as Map)['content'] == ''), isFalse,
        reason: 'an empty assistant turn reached the gateway');
    expect(sentMessages!.map((m) => (m as Map)['content']),
        containsAll(<String>['first', 'second']));

    // The empty row still exists for the UI to render as an interrupted answer.
    final stored = await db.watchMessages(id).first;
    expect(stored.where((m) => m.content.isEmpty).length, 1);
  });

  test('the answer is readable while it is still arriving', () async {
    // The database row stays empty until the answer completes, so the streamed
    // text has to be readable from the controller or the screen shows nothing
    // until the end -- one rebuild per token, each displaying no new text.
    // That is the difference between streaming and a slow reveal.
    final id = await db.createConversation();
    final held = Completer<void>();
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((_, __) async => http.StreamedResponse(
              Stream.multi((c) async {
                c.add(utf8.encode(
                    'data: {"choices":[{"delta":{"content":"partial "}}]}\n\n'));
                await held.future;
                await c.close();
              }),
              200,
            )),
      ),
      conversationId: id,
    );

    final arrived = Completer<void>();
    controller.addListener(() {
      if (controller.streamingText.isNotEmpty && !arrived.isCompleted) {
        arrived.complete();
      }
    });
    unawaited(controller.send('hi'));
    await arrived.future;

    expect(controller.streamingText, 'partial ');
    expect(controller.streamingRow, isNotNull);
    // ...and the persisted row is still empty at this moment, which is exactly
    // why the buffer has to be exposed.
    final stored = await db.watchMessages(id).first;
    expect(stored.last.content, isEmpty);

    held.complete();
    await controller.stop();
  });

  test('disposing mid-answer keeps what arrived instead of stranding the row',
      () async {
    // Switching conversations disposes the controller. Without the fallback in
    // dispose(), the buffered text was discarded AND the row stayed
    // 'streaming' forever, so reopening that conversation showed a message
    // frozen half-written. Neither the network nor a timer is waited on here:
    // the first delta is awaited through the notifier, and the row is then
    // polled until it settles, so the test cannot flake on timing.
    final id = await db.createConversation();
    final held = Completer<void>();
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((_, __) async => http.StreamedResponse(
              Stream.multi((c) async {
                c.add(utf8.encode(
                    'data: {"choices":[{"delta":{"content":"half"}}]}\n\n'));
                await held.future; // the answer never finishes arriving
                await c.close();
              }),
              200,
            )),
      ),
      conversationId: id,
    );

    final contentArrived = Completer<void>();
    controller.addListener(() {
      // Waits for real CONTENT, not for a notify count. send() notifies once
      // when it inserts the placeholder, before the stream is even subscribed;
      // treating that first notify as "a delta arrived" disposes the controller
      // before the mock has produced anything.
      if (controller.streamingText.isNotEmpty && !contentArrived.isCompleted) {
        contentArrived.complete();
      }
    });
    unawaited(controller.send('hi'));
    await contentArrived.future;

    controller.dispose();

    // dispose() cannot await its own write, so wait for the row to settle.
    Message? row;
    for (var i = 0; i < 100 && (row == null || row.status == 'streaming'); i++) {
      row = (await db.watchMessages(id).first).last;
      if (row.status != 'streaming') break;
      await Future<void>.delayed(const Duration(milliseconds: 10));
    }
    expect(row!.content, 'half');
    expect(row.status, 'partial');
    held.complete();
  });

  test('a failure leaves the partial text marked, not discarded', () async {
    final id = await db.createConversation();
    final controller = ChatController(
      db: db,
      client: LlmClient(
        baseUrl: 'https://gw.test',
        apiKey: 'k',
        httpClient: MockClient.streaming((_, __) async => http.StreamedResponse(
              Stream.multi((c) {
                c.add(utf8.encode(
                    'data: {"choices":[{"delta":{"content":"par"}}]}\n\n'));
                c.addError(const SocketExceptionStub());
                c.close();
              }),
              200,
            )),
      ),
      conversationId: id,
    );
    await controller.send('hi');
    final stored = await db.watchMessages(id).first;
    expect(stored.last.content, 'par');
    expect(stored.last.status, 'partial');
  });
}

class SocketExceptionStub implements Exception {
  const SocketExceptionStub();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/features/chat_controller_test.dart`
Expected: FAIL — `Target of URI doesn't exist: '.../chat_controller.dart'`

- [ ] **Step 3: Write the controller**

```dart
import 'dart:async';

import 'package:flutter/foundation.dart';

import '../../api/errors.dart';
import '../../api/llm_client.dart';
import '../../api/types.dart';
import '../../data/db.dart';

/// Drives one conversation. The assistant row is inserted empty before the
/// answer exists, so the UI has something to render while it streams and a lost
/// connection leaves visible partial text instead of nothing.
class ChatController extends ChangeNotifier {
  ChatController({
    required this.db,
    required this.client,
    required this.conversationId,
  });

  final AppDb db;
  final LlmClient client;
  final String conversationId;

  StreamSubscription<ChatDelta>? _subscription;
  bool _disposed = false;
  int? _openRow;
  String _buffer = '';
  String? _model;
  LlmError? failure;

  bool get busy => _subscription != null;

  /// The row currently being streamed into, or null when nothing is in flight.
  int? get streamingRow => _openRow;

  /// The answer accumulated so far. The database row is not written until the
  /// answer completes, so while it is arriving THIS is the only place the text
  /// exists — the screen renders it in place of the still-empty row. Without
  /// it, every `notifyListeners()` below rebuilds a view showing nothing new
  /// and the answer appears all at once at the end, which is not streaming.
  String get streamingText => _buffer;

  Future<void> send(String text) async {
    if (busy || text.trim().isEmpty) return;
    failure = null;
    await db.addMessage(
        conversationId: conversationId, role: 'user', content: text.trim());

    // Empty rows never go out. This query returns every message including the
    // assistant placeholder inserted below and any row a stopped or dropped
    // stream left with no text; sending an empty assistant turn as context is
    // meaningless and some providers reject it outright. Partial answers DO go
    // out -- the user saw them on screen and may refer to them.
    final history = await db.watchMessages(conversationId).first;
    final turns = [
      for (final m in history)
        if (m.content.isNotEmpty) ChatMessage(role: m.role, content: m.content),
    ];

    _buffer = '';
    _model = null;
    _openRow = await db.addMessage(
        conversationId: conversationId,
        role: 'assistant',
        content: '',
        status: 'streaming');
    notifyListeners();

    final done = Completer<void>();
    _subscription = client.stream(messages: turns).listen(
      (delta) {
        if (delta.model != null) _model = delta.model;
        if (delta.text != null) {
          _buffer += delta.text!;
          notifyListeners();
        }
      },
      // Both handlers AWAIT _close before completing: `unawaited` here would
      // let send() return before the row was written, which is a flaky test in
      // development and a lost partial answer in production.
      onError: (Object e) async {
        failure = e is LlmError ? e : Disconnected(e);
        await _close('partial');
        if (!done.isCompleted) done.complete();
      },
      onDone: () async {
        await _close('ok');
        if (!done.isCompleted) done.complete();
      },
      cancelOnError: true,
    );
    return done.future;
  }

  /// Cancels the stream and keeps what already arrived.
  Future<void> stop() async {
    if (!busy) return;
    await _close('partial');
  }

  Future<void> _close(String status) async {
    // Claimed synchronously, before any await, so a stop() racing a naturally
    // firing onDone cannot both find a row and write it twice.
    final row = _openRow;
    _openRow = null;
    if (row == null) return;

    await _subscription?.cancel();
    _subscription = null;
    await db.finishMessage(row,
        content: _buffer, status: status, modelUsed: _model);
    if (status == 'ok' && !_disposed) await _titleIfUnnamed();
    // Titling is a full network round trip, and the user can switch away or
    // leave during it. Notifying a disposed ChangeNotifier throws.
    if (!_disposed) notifyListeners();
  }

  @override
  void dispose() {
    _disposed = true;
    // Persist whatever already arrived, exactly as stop() does. dispose() cannot
    // await, but the write does not need this object alive -- only
    // notifyListeners() does, and _close skips it once disposed. Without this,
    // switching conversations mid-answer discarded the buffered text AND left
    // the row stuck at status 'streaming' forever, so reopening that
    // conversation showed a message frozen half-written.
    if (_openRow != null) unawaited(_close('partial'));
    super.dispose();
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/features/chat_controller_test.dart`
Expected: PASS

- [ ] **Step 5: Write the screen**

Create `lib/features/chat/chat_screen.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:gpt_markdown/gpt_markdown.dart';

import '../../api/llm_client.dart';
import '../../data/db.dart';
import 'chat_controller.dart';
import 'conversation_drawer.dart';

/// Owns which conversation is open, because a [ChatController] is bound to one
/// conversation for its lifetime. Switching conversations therefore means
/// replacing the controller, which is why this screen builds it rather than
/// receiving it.
class ChatScreen extends StatefulWidget {
  const ChatScreen({
    super.key,
    required this.db,
    required this.client,
    required this.conversationId,
  });

  final AppDb db;
  final LlmClient client;

  /// The conversation to open first. The drawer changes it from there.
  final String conversationId;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _input = TextEditingController();
  late String _conversationId;
  late ChatController _controller;

  /// Held in state rather than built in `build()`. Every streamed token calls
  /// notifyListeners, which rebuilds; creating the query inline there would
  /// make StreamBuilder tear down and re-subscribe drift's live query on every
  /// delta.
  late Stream<List<Message>> _messages;

  @override
  void initState() {
    super.initState();
    _conversationId = widget.conversationId;
    _controller = _newController();
    _messages = widget.db.watchMessages(_conversationId);
  }

  ChatController _newController() =>
      ChatController(
        db: widget.db,
        client: widget.client,
        conversationId: _conversationId,
      )..addListener(_onControllerChanged);

  /// Disposing the outgoing controller cancels any stream still writing into
  /// the conversation being left, so an answer cannot keep arriving into a
  /// screen nobody is looking at.
  Future<void> _open(String id) async {
    if (id == _conversationId) return;
    final outgoing = _controller;
    outgoing.removeListener(_onControllerChanged);
    // Awaited, so a half-arrived answer is persisted before the screen moves
    // on. dispose() has a fire-and-forget fallback for app teardown, where
    // there is nothing left to await from.
    await outgoing.stop();
    if (!mounted) return;
    setState(() {
      _conversationId = id;
      _controller = _newController();
      _messages = widget.db.watchMessages(_conversationId);
    });
    outgoing.dispose();
  }

  @override
  void dispose() {
    _controller.removeListener(_onControllerChanged);
    _controller.dispose();
    _input.dispose();
    super.dispose();
  }

  void _onControllerChanged() => setState(() {});

  @override
  Widget build(BuildContext context) {
    final busy = _controller.busy;
    return Scaffold(
      appBar: AppBar(title: const Text('llm-libre')),
      drawer: ConversationDrawer(db: widget.db, onOpen: _open),
      body: Column(
        children: [
          Expanded(
            child: StreamBuilder<List<Message>>(
              stream: _messages,
              builder: (context, snapshot) {
                final messages = snapshot.data ?? const <Message>[];
                return ListView.builder(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  itemCount: messages.length,
                  itemBuilder: (context, i) {
                    final message = messages[i];
                    return _Turn(
                      message: message,
                      liveText: message.id == _controller.streamingRow
                          ? _controller.streamingText
                          : null,
                    );
                  },
                );
              },
            ),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Expanded(
                    child: TextField(
                      controller: _input,
                      maxLines: 6,
                      minLines: 1,
                      decoration: const InputDecoration(
                        hintText: 'Message',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  IconButton(
                    icon: Icon(busy ? Icons.stop : Icons.send),
                    onPressed: () {
                      if (busy) {
                        _controller.stop();
                      } else {
                        final text = _input.text;
                        _input.clear();
                        _controller.send(text);
                      }
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// User turns get a bubble; assistant turns run full width, as ChatGPT does.
class _Turn extends StatelessWidget {
  const _Turn({required this.message, this.liveText});

  final Message message;

  /// The text arriving right now, when this row is the one being streamed.
  /// Null for every other row, and for all of them once nothing is in flight.
  final String? liveText;

  @override
  Widget build(BuildContext context) {
    if (message.role == 'user') {
      return Align(
        alignment: Alignment.centerRight,
        child: Container(
          margin: const EdgeInsets.fromLTRB(48, 4, 12, 4),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(18),
          ),
          child: Text(message.content),
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          GptMarkdown(liveText ?? message.content),
          if (message.modelUsed != null)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                message.status == 'partial'
                    ? '${message.modelUsed} · interrupted'
                    : message.modelUsed!,
                style: Theme.of(context).textTheme.labelSmall,
              ),
            ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 6: Write the drawer**

Create `lib/features/chat/conversation_drawer.dart`:

```dart
import 'package:flutter/material.dart';

import '../../data/db.dart';

class ConversationDrawer extends StatelessWidget {
  const ConversationDrawer({super.key, required this.db, this.onOpen});

  final AppDb db;
  final void Function(String conversationId)? onOpen;

  /// The buckets ChatGPT uses. Compared against midnight rather than a rolling
  /// 24 hours, so something written last night reads as "Yesterday" all day.
  static String _bucket(DateTime when, DateTime now) {
    final today = DateTime(now.year, now.month, now.day);
    final day = DateTime(when.year, when.month, when.day);
    final days = today.difference(day).inDays;
    if (days <= 0) return 'Today';
    if (days == 1) return 'Yesterday';
    if (days <= 7) return 'Last 7 days';
    return 'Older';
  }

  @override
  Widget build(BuildContext context) => Drawer(
        child: SafeArea(
          child: Column(
            children: [
              ListTile(
                leading: const Icon(Icons.add),
                title: const Text('New conversation'),
                onTap: () async {
                  final id = await db.createConversation();
                  onOpen?.call(id);
                  if (context.mounted) Navigator.of(context).pop();
                },
              ),
              const Divider(height: 1),
              Expanded(
                child: StreamBuilder<List<Conversation>>(
                  stream: db.watchConversations(),
                  builder: (context, snapshot) {
                    final rows = snapshot.data ?? const <Conversation>[];
                    final now = DateTime.now();
                    String? lastBucket;
                    final tiles = <Widget>[];
                    for (final c in rows) {
                      final bucket = _bucket(c.updatedAt, now);
                      if (bucket != lastBucket) {
                        lastBucket = bucket;
                        tiles.add(Padding(
                          padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
                          child: Text(bucket,
                              style: Theme.of(context).textTheme.labelSmall),
                        ));
                      }
                      tiles.add(ListTile(
                        title: Text(
                          c.title.isEmpty ? 'New conversation' : c.title,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        onTap: () {
                          onOpen?.call(c.id);
                          Navigator.of(context).pop();
                        },
                      ));
                    }
                    return ListView(children: tiles);
                  },
                ),
              ),
            ],
          ),
        ),
      );
}
```

Add its bucketing test to `test/features/conversation_drawer_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/features/chat/conversation_drawer.dart';

void main() {
  test('buckets are calendar days, not rolling 24-hour windows', () {
    // 23:00 yesterday against 01:00 today is 2 hours apart but belongs in
    // 'Yesterday', which a Duration comparison would call 'Today'.
    final now = DateTime(2026, 9, 2, 1);
    expect(ConversationDrawer.bucketFor(DateTime(2026, 9, 2, 0, 30), now),
        'Today');
    expect(ConversationDrawer.bucketFor(DateTime(2026, 9, 1, 23), now),
        'Yesterday');
    expect(ConversationDrawer.bucketFor(DateTime(2026, 8, 28), now),
        'Last 7 days');
    expect(ConversationDrawer.bucketFor(DateTime(2026, 7, 1), now), 'Older');
  });
}
```

Rename `_bucket` to `bucketFor` so the test can reach it — a private helper with
a real rule in it is worth exposing rather than testing through the widget.

- [ ] **Step 6b: Title the conversation after the first exchange**

Add to `ChatController`, and call it from `_close` when `status == 'ok'`:

```dart
  /// ChatGPT names a conversation after the first exchange. A failure here is a
  /// cosmetic loss and must never surface as a chat error, hence the swallow.
  Future<void> _titleIfUnnamed() async {
    final row = await (db.select(db.conversations)
          ..where((c) => c.id.equals(conversationId)))
        .getSingleOrNull();
    if (row == null || row.title.isNotEmpty) return;
    try {
      final answer = await client.complete(
        model: 'auto:fast',
        maxTokens: 16,
        messages: [
          ChatMessage(
            role: 'user',
            content: 'Title this conversation in three or four words, '
                'no quotes, no final period:\n\n$_buffer',
          ),
        ],
      );
      final title = answer.content.trim();
      if (title.isNotEmpty) await db.renameConversation(conversationId, title);
    } on Object {
      // Deliberately catches EVERYTHING, not just LlmError. The promise this
      // method makes is that a failure to name a conversation never surfaces as
      // a chat failure, and `on LlmError` does not keep it: a malformed body
      // throws FormatException, which escapes, which leaves `_close` unfinished
      // and `send()`'s future hanging forever. A cosmetic feature must not be
      // able to wedge the send path. Leave it untitled and move on.
    }
  }
```

In `_close`, after `db.finishMessage(...)`:

```dart
      if (status == 'ok') await _titleIfUnnamed();
```

- [ ] **Step 6c: Rewire `main.dart`**

`ChatScreen` now owns its controller, so `main.dart` resolves only WHICH
conversation to open and hands the id over:

```dart
    return ChatScreen(
      db: widget.db,
      client: widget.client,
      conversationId: id,
    );
```

Picking that conversation is the one genuinely asynchronous step at startup —
creating a fresh one writes a row — unlike the key and base URL, which are
compile-time constants. So a brief loading state lives here and nowhere else.
Reopen the most recently active conversation, in the same order the drawer
lists them, and create one only when the device has none yet.

`test/widget_test.dart` asserts that an unconfigured build shows
`rebuild with --dart-define-from-file=.env`. That path must keep working: it is
reached before any of this, straight from `Config.isConfigured`.

### Known gap: the streaming wire is not covered by a test

**Parked deliberately, with evidence.** `ChatController` is proven by two tests
to hold the arriving text in `streamingText`, and `ChatScreen`'s `itemBuilder`
is confirmed by reading to pass it as `liveText` to the row being streamed. What
no test covers is the wire between them: **deleting the `liveText:` argument
would fail nothing in this suite.**

That is the exact shape of the defect it guards against — the controller was
correct, the text existed, thirty tests were green, and nothing rendered it.

Two attempts were made to close it, and both failed on the test harness rather
than on the code. What was learned, so a third attempt does not repeat it:

- A widget test that drives a real streamed response through
  `MockClient.streaming` and waits with `tester.runAsync()` plus real delays
  makes `flutter test` die with `Bad state: Cannot close sink while adding
  stream` from `flutter_tools/src/test/flutter_platform.dart`, after running for
  over five minutes. The drawer-switch test in the same file finishes in three
  seconds, so it is specific to driving the stream, not to the screen.
- Replacing the real delays with a `StreamController` the test feeds, pumping
  with no `runAsync` at all, also hangs. So the obstacle is not the waiting
  strategy.

A third attempt should start by finding out WHY the harness deadlocks — most
likely how drift's live query emits under `flutter_test`'s fake clock — rather
than trying another synchronisation idiom. Until then the wire is verified by
reading, which is weaker than a test and is recorded here as such.

- [ ] **Step 7: Delete the placeholder screen**

```bash
git rm lib/features/chat/simple_chat_screen.dart \
  test/features/simple_chat_screen_test.dart
```

- [ ] **Step 8: Run the whole suite**

Run: `fvm flutter test`
Expected: PASS, every test.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat(chat): streaming conversation persisted on the device"
```

---

## What this plan does not cover

Slices 3 to 6 of the spec — the route catalogue and model picker, vision and
image generation, audio, and the routes panel — get their own plans, written
against the code these ten tasks produce rather than guessed at now.
