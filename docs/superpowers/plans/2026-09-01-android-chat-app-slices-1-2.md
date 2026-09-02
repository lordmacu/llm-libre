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

**Tech Stack:** Flutter 3.38.3 via fvm, Dart ^3.10.1, `flutter_riverpod` 2.6.1,
`go_router` 17.3.0, `drift` 2.30.1 + `drift_flutter` 0.2.8,
`flutter_secure_storage` 10.3.1, `http` 1.6.0, `gpt_markdown`, `uuid`.

**Spec:** `docs/superpowers/specs/2026-09-01-android-chat-app-design.md`

## Global Constraints

- All code — identifiers, comments, strings — in English. No Spanish in source.
- Gateway base URL default: `https://llm.comparadorinternet.co`. Never hardcode
  the API key; it lives in `flutter_secure_storage`.
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

---

## File Structure

| File | Responsibility |
|---|---|
| `lib/api/types.dart` | `Capability`, `ChatMessage`, `ChatAnswer`, `ChatDelta`. No behaviour. |
| `lib/api/errors.dart` | The `LlmError` hierarchy and `errorFromResponse`. |
| `lib/api/sse.dart` | Byte stream to SSE payload strings. Nothing gateway-specific. |
| `lib/api/llm_client.dart` | Request building and the two chat calls. |
| `lib/settings/settings_store.dart` | API key and base URL in secure storage. |
| `lib/data/db.dart` | drift tables, DAO methods. |
| `lib/features/chat/chat_controller.dart` | Riverpod notifier driving one conversation. |
| `lib/features/chat/chat_screen.dart` | Message list, composer, stop button. |
| `lib/features/chat/conversation_drawer.dart` | Conversation list grouped by date. |
| `lib/app.dart` | go_router wiring. |

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

```bash
cd ~/llm-libre-chat
fvm flutter pub add flutter_riverpod go_router drift drift_flutter \
  flutter_secure_storage http gpt_markdown uuid
fvm flutter pub add --dev drift_dev build_runner
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

## Task 5: Settings storage

**Files:**
- Create: `lib/settings/settings_store.dart`
- Test: `test/settings/settings_store_test.dart`

**Interfaces:**
- Consumes: nothing.
- Produces: `class SettingsStore(FlutterSecureStorage storage)` with
  `Future<String?> readApiKey()`, `Future<void> writeApiKey(String)`,
  `Future<String> readBaseUrl()`, `Future<void> writeBaseUrl(String)`, and
  `const String defaultBaseUrl`.

- [ ] **Step 1: Write the failing test**

```dart
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llm_libre_chat/settings/settings_store.dart';

void main() {
  setUp(() {
    // The plugin has no Android side in a unit test; this in-memory map stands
    // in for the Keystore.
    FlutterSecureStorage.setMockInitialValues({});
  });

  test('the base URL falls back to the production gateway', () async {
    final store = SettingsStore(const FlutterSecureStorage());
    expect(await store.readBaseUrl(), 'https://llm.comparadorinternet.co');
  });

  test('a stored key comes back', () async {
    final store = SettingsStore(const FlutterSecureStorage());
    await store.writeApiKey('llmlibre_abc');
    expect(await store.readApiKey(), 'llmlibre_abc');
  });

  test('a base URL with a trailing slash is normalised', () async {
    // '$baseUrl/v1/chat/completions' would otherwise produce a double slash,
    // which Cloudflare answers with a redirect the POST does not survive.
    final store = SettingsStore(const FlutterSecureStorage());
    await store.writeBaseUrl('https://gw.test/');
    expect(await store.readBaseUrl(), 'https://gw.test');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `fvm flutter test test/settings/settings_store_test.dart`
Expected: FAIL — `Target of URI doesn't exist`

- [ ] **Step 3: Write the implementation**

```dart
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

const String defaultBaseUrl = 'https://llm.comparadorinternet.co';

const _apiKeyKey = 'api_key';
const _baseUrlKey = 'base_url';

/// The API key never reaches the source: an APK is trivially decompiled, and a
/// key held here can be rotated without a rebuild.
class SettingsStore {
  const SettingsStore(this._storage);

  final FlutterSecureStorage _storage;

  Future<String?> readApiKey() => _storage.read(key: _apiKeyKey);

  Future<void> writeApiKey(String value) =>
      _storage.write(key: _apiKeyKey, value: value.trim());

  Future<String> readBaseUrl() async =>
      await _storage.read(key: _baseUrlKey) ?? defaultBaseUrl;

  Future<void> writeBaseUrl(String value) => _storage.write(
      key: _baseUrlKey, value: _normalise(value));

  static String _normalise(String url) {
    var out = url.trim();
    while (out.endsWith('/')) {
      out = out.substring(0, out.length - 1);
    }
    return out;
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `fvm flutter test test/settings/settings_store_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/settings/settings_store.dart test/settings/settings_store_test.dart
git commit -m "feat(settings): key and base URL in secure storage"
```

---

## Task 6: First screen — send one message, see one answer

**Files:**
- Create: `lib/features/chat/simple_chat_screen.dart`
- Modify: `lib/main.dart`
- Test: `test/features/simple_chat_screen_test.dart`

**Interfaces:**
- Consumes: `LlmClient.complete`, `SettingsStore`.
- Produces: `class SimpleChatScreen extends StatefulWidget` taking a
  `LlmClient client`. Replaced in Task 10; it exists so slice 1 is a running app.

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

- [ ] **Step 5: Commit**

```bash
git add lib/features/chat/simple_chat_screen.dart lib/main.dart \
  test/features/simple_chat_screen_test.dart
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
- Modify: `lib/app.dart`, `lib/main.dart`
- Test: `test/features/chat_controller_test.dart`

**Interfaces:**
- Consumes: `LlmClient.stream`, `AppDb`, `SettingsStore`.
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
  int? _openRow;
  String _buffer = '';
  String? _model;
  LlmError? failure;

  bool get busy => _subscription != null;

  Future<void> send(String text) async {
    if (busy || text.trim().isEmpty) return;
    failure = null;
    await db.addMessage(
        conversationId: conversationId, role: 'user', content: text.trim());

    final history = await db.watchMessages(conversationId).first;
    final turns = [
      for (final m in history)
        ChatMessage(role: m.role, content: m.content),
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
    final row = _openRow;
    await _subscription?.cancel();
    _subscription = null;
    _openRow = null;
    if (row != null) {
      await db.finishMessage(row,
          content: _buffer, status: status, modelUsed: _model);
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _subscription?.cancel();
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

import '../../data/db.dart';
import 'chat_controller.dart';
import 'conversation_drawer.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.db, required this.controller});

  final AppDb db;
  final ChatController controller;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _input = TextEditingController();

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onControllerChanged);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onControllerChanged);
    _input.dispose();
    super.dispose();
  }

  void _onControllerChanged() => setState(() {});

  @override
  Widget build(BuildContext context) {
    final busy = widget.controller.busy;
    return Scaffold(
      appBar: AppBar(title: const Text('llm-libre')),
      drawer: ConversationDrawer(db: widget.db),
      body: Column(
        children: [
          Expanded(
            child: StreamBuilder<List<Message>>(
              stream: widget.db.watchMessages(widget.controller.conversationId),
              builder: (context, snapshot) {
                final messages = snapshot.data ?? const <Message>[];
                return ListView.builder(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  itemCount: messages.length,
                  itemBuilder: (context, i) => _Turn(message: messages[i]),
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
                        widget.controller.stop();
                      } else {
                        final text = _input.text;
                        _input.clear();
                        widget.controller.send(text);
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
  const _Turn({required this.message});

  final Message message;

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
          GptMarkdown(message.content),
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
    } on LlmError {
      // Leave it untitled.
    }
  }
```

In `_close`, after `db.finishMessage(...)`:

```dart
      if (status == 'ok') await _titleIfUnnamed();
```

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
