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

## Key Constraint: SINGLE HEADER

`include/llm_stream.hpp` is the entire library. Never refactor it into multiple files.

## Public API (maintain exactly)

```cpp
namespace llm {

struct Config {
    std::string api_key;
    std::string model;
    int max_tokens = 1024;
    double temperature = 0.7;
    std::string system_prompt;
};

struct StreamStats {
    size_t token_count;
    double elapsed_ms;
    double tokens_per_sec;
};

using TokenCallback = std::function<void(std::string_view token)>;
using DoneCallback  = std::function<void(const StreamStats&)>;
using ErrorCallback = std::function<void(std::string_view error)>;

void stream_openai(const std::string& prompt, const Config& config,
                   TokenCallback on_token,
                   DoneCallback on_done = nullptr,
                   ErrorCallback on_error = nullptr);

void stream_anthropic(const std::string& prompt, const Config& config,
                      TokenCallback on_token,
                      DoneCallback on_done = nullptr,
                      ErrorCallback on_error = nullptr);

// Auto-detects provider: "gpt-*" -> OpenAI, "claude-*" -> Anthropic
void stream(const std::string& prompt, const Config& config,
            TokenCallback on_token,
            DoneCallback on_done = nullptr,
            ErrorCallback on_error = nullptr);

} // namespace llm
```

## Implementation Guard

Define in exactly one translation unit:

```cpp
#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"
```

## Common Mistakes to Avoid

- Adding dependencies — libcurl only, no JSON libraries
- Throwing exceptions in callbacks
- Blocking inside callbacks — they run on the curl write thread
- Splitting the header into multiple files
- Raw `curl_easy_cleanup` at function exit — use RAII
