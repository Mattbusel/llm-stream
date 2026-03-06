# AGENTS.md — llm-stream

## Purpose

`llm-stream` is a zero-dependency single-header C++ library for streaming LLM responses
from OpenAI and Anthropic APIs. The entire library lives in one file: `include/llm_stream.hpp`.

## Architecture

```
llm-stream/
  include/
    llm_stream.hpp      <- THE ENTIRE LIBRARY. Do not split this.
  examples/
    basic_stream.cpp    <- Minimal OpenAI streaming example
    chat_loop.cpp       <- Multi-turn interactive REPL (OpenAI + Anthropic)
  CMakeLists.txt        <- Builds examples only (library is header-only)
  README.md
  AGENTS.md
  CLAUDE.md
  LICENSE
```

## Build & Test

```bash
cmake -B build && cmake --build build
cd build && ctest
```

## Rules for All Agents

### The One Absolute Constraint
**The library MUST remain a single header.** `include/llm_stream.hpp` is the entire library.
Never create additional `.cpp` or `.hpp` files for library code. Never split the implementation.

### Dependencies
- `libcurl` is the **only** allowed external dependency.
- Do NOT add nlohmann/json, rapidjson, boost, or any other library.
- JSON parsing must use the hand-rolled minimal parser inside `llm_stream.hpp`.

### Namespace
All public API must live in namespace `llm`.

### API Surface
Callback-based streaming:
- `on_token(std::string_view token)` - called for each streamed token
- `on_done(const StreamStats&)` - called when stream ends successfully
- `on_error(std::string_view error)` - called on failure

### Style
- C++17, no exceptions in the hot path / callbacks
- RAII for all curl handles (`curl_easy_cleanup` via destructor)
- Each `stream_*()` call is fully self-contained and thread-safe
- Compile clean with `-Wall -Wextra -std=c++17`

### PR Order (Agent 1 merges)
1. Agent 3: branch `feat/core-library` -> PR to `main`
2. Agent 4: branch `feat/examples` -> PR to `main`
3. Agent 2: docs -> PR to `main`
4. Agent 5: README -> PR to `main`
