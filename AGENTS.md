# AGENTS.md — llm-stream

## Purpose
`llm-stream` is a zero-dependency, single-header C++ library for streaming LLM responses
from OpenAI and Anthropic APIs. The only external dependency is libcurl.

## Architecture
- Everything lives in `include/llm_stream.hpp` — this is the ENTIRE library.
- Examples live in `examples/`.
- There are no .cpp source files for the library itself.
- Implementation is guarded by `#ifdef LLM_STREAM_IMPLEMENTATION` so users control where it compiles.

## Build & Test
```bash
cmake -B build && cmake --build build
cd build && ctest
```

## Critical Constraints
- **SINGLE HEADER**: Never split the library into multiple files. `include/llm_stream.hpp` is everything.
- **libcurl ONLY**: No additional dependencies under any circumstance. No nlohmann, no rapidjson, no boost.
- **Namespace**: All public API must be in namespace `llm`.
- **C++17**: Compile with `-std=c++17`. No C++20 features.
- **No exceptions in hot path**: Use error callbacks or std::optional/error codes instead.
- **RAII**: Wrap all CURL* handles in RAII wrappers — never call curl_easy_cleanup manually.

## Public API Surface
```cpp
namespace llm {
    struct Config { std::string api_key, model, system_prompt; int max_tokens; double temperature; };
    struct StreamStats { size_t token_count; double elapsed_ms, tokens_per_sec; };
    using TokenCallback = std::function<void(std::string_view)>;
    using DoneCallback  = std::function<void(const StreamStats&)>;
    using ErrorCallback = std::function<void(std::string_view)>;

    void stream_openai(const std::string& prompt, const Config&, TokenCallback, DoneCallback, ErrorCallback);
    void stream_anthropic(const std::string& prompt, const Config&, TokenCallback, DoneCallback, ErrorCallback);
    void stream(const std::string& prompt, const Config&, TokenCallback, DoneCallback, ErrorCallback);
}
```

## Callback Rules
- `on_token`: called for every text delta received from the stream
- `on_done`: called once when stream ends successfully, with stats
- `on_error`: called on any error (network, API, parse); stream stops after this

## Agent Workflow
- Agent 1: repo setup (main branch)
- Agent 2: AGENTS.md + CLAUDE.md (main branch)
- Agent 3: core library (`feat/core-library` branch → PR to main)
- Agent 4: examples + CMakeLists.txt (`feat/examples` branch → PR to main)
- Agent 5: README.md (main branch)
- Merge order: PR 3 first, then PR 4, then README
- No agent merges their own PR — coordinate through Agent 1

## Style Guide
- Prefer `std::string_view` over `const std::string&` for read-only string params
- Use `using` aliases over `typedef`
- snake_case for everything
- Minimal includes — only what is strictly needed
