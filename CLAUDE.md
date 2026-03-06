# CLAUDE.md — llm-stream

## Build
```bash
cmake -B build && cmake --build build
```

## Test
```bash
cd build && ctest
```

## Run Examples
```bash
export OPENAI_API_KEY=sk-...
./build/basic_stream

export ANTHROPIC_API_KEY=sk-ant-...
./build/chat_loop
```

## THE ONE RULE: SINGLE HEADER
`include/llm_stream.hpp` is the entire library. Never refactor it into multiple files.
Never create a `src/` directory. Never split into `.cpp` files.

## API Surface to Maintain
These signatures must remain stable:

```cpp
namespace llm {
    void stream_openai(const std::string& prompt, const Config& config,
                       TokenCallback on_token,
                       DoneCallback on_done = nullptr,
                       ErrorCallback on_error = nullptr);

    void stream_anthropic(const std::string& prompt, const Config& config,
                          TokenCallback on_token,
                          DoneCallback on_done = nullptr,
                          ErrorCallback on_error = nullptr);

    void stream(const std::string& prompt, const Config& config,
                TokenCallback on_token,
                DoneCallback on_done = nullptr,
                ErrorCallback on_error = nullptr);
}
```

`stream()` auto-detects provider: `gpt-*` → OpenAI, `claude-*` → Anthropic.

## Common Mistakes to Avoid
1. **Adding dependencies** — libcurl is the only allowed dep. No JSON libs, no HTTP libs.
2. **Throwing exceptions in callbacks** — callbacks must be noexcept-friendly. Use `on_error` instead.
3. **Blocking the stream** — never do heavy work inside `on_token`. It blocks the curl write callback.
4. **Forgetting `#ifdef LLM_STREAM_IMPLEMENTATION`** — implementation must be inside this guard.
5. **Skipping null checks on callbacks** — `on_done` and `on_error` may be nullptr; always check before calling.
6. **Using `curl_global_init` inside the RAII handle** — call it once at stream start, cleanup at end.

## JSON Parsing
Use only the hand-rolled minimal JSON parser inside `llm_stream.hpp`.
Never add nlohmann/json, rapidjson, simdjson, or any other JSON library.

## Implementation Guard Pattern
```cpp
#pragma once
// ... declarations ...

#ifdef LLM_STREAM_IMPLEMENTATION
// ... full implementation ...
#endif // LLM_STREAM_IMPLEMENTATION
```

Users put `#define LLM_STREAM_IMPLEMENTATION` in exactly ONE .cpp file before including.
