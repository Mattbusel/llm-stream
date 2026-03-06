# llm-stream

Stream OpenAI and Anthropic responses in C++. Drop in one header. No deps.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Single Header](https://img.shields.io/badge/single-header-orange.svg)
![libcurl](https://img.shields.io/badge/dep-libcurl-lightgrey.svg)

## Quickstart

Copy `include/llm_stream.hpp` into your project. That's the whole library.

```cpp
#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"
#include <iostream>
#include <cstdlib>

int main() {
    const char* key = std::getenv("OPENAI_API_KEY");
    if (!key) { std::cerr << "OPENAI_API_KEY not set\n"; return 1; }

    llm::Config cfg;
    cfg.api_key = key;
    cfg.model   = "gpt-4o-mini";

    llm::stream_openai("Explain recursion in one paragraph.", cfg,
        [](std::string_view token) { std::cout << token << std::flush; },
        [](const llm::StreamStats& s) {
            std::cout << "\n[" << s.token_count << " tokens, " << s.tokens_per_sec << " tok/s]\n";
        }
    );
}
```

Compile:
```bash
g++ -std=c++17 main.cpp -lcurl -o stream
./stream
```

## Installation

Copy `include/llm_stream.hpp` into your project. No build system changes required beyond linking libcurl.

Put `#define LLM_STREAM_IMPLEMENTATION` in exactly one `.cpp` file before the include. All other files just `#include "llm_stream.hpp"`.

## API Reference

### Config

```cpp
struct Config {
    std::string api_key;
    std::string model        = "gpt-4o-mini";  // or "claude-3-5-haiku-20241022"
    int         max_tokens   = 1024;
    double      temperature  = 0.7;
    std::string system_prompt;
};
```

### StreamStats

```cpp
struct StreamStats {
    size_t token_count;    // number of tokens received
    double elapsed_ms;     // total wall time in milliseconds
    double tokens_per_sec; // throughput
};
```

### Callbacks

```cpp
using TokenCallback = std::function<void(std::string_view token)>;
using DoneCallback  = std::function<void(const StreamStats&)>;
using ErrorCallback = std::function<void(std::string_view error)>;
```

### Functions

```cpp
// Stream from OpenAI
void llm::stream_openai(
    const std::string& prompt,
    const Config& config,
    TokenCallback on_token,
    DoneCallback  on_done  = nullptr,   // optional
    ErrorCallback on_error = nullptr    // optional
);

// Stream from Anthropic
void llm::stream_anthropic(
    const std::string& prompt,
    const Config& config,
    TokenCallback on_token,
    DoneCallback  on_done  = nullptr,
    ErrorCallback on_error = nullptr
);

// Auto-detect provider from model name: "gpt-*" → OpenAI, "claude-*" → Anthropic
void llm::stream(
    const std::string& prompt,
    const Config& config,
    TokenCallback on_token,
    DoneCallback  on_done  = nullptr,
    ErrorCallback on_error = nullptr
);
```

## Examples

- [`examples/basic_stream.cpp`](examples/basic_stream.cpp) — single prompt, print tokens as they arrive
- [`examples/chat_loop.cpp`](examples/chat_loop.cpp) — interactive multi-turn REPL with conversation history

## Building the Examples

```bash
cmake -B build && cmake --build build

# Run basic stream (OpenAI)
export OPENAI_API_KEY=sk-...
./build/basic_stream

# Run chat loop (Anthropic preferred, falls back to OpenAI)
export ANTHROPIC_API_KEY=sk-ant-...
./build/chat_loop
```

## Why

- **No Python runtime.** Ship LLM features in an existing C++ binary — games, CLIs, embedded apps, servers.
- **No build complexity.** One header, one link flag (`-lcurl`). Works with any build system: CMake, Make, Bazel, MSVC, whatever.
- **Drop into any project.** No SDK to install, no package manager, no versioning hell. Copy one file and you're streaming.

## Requirements

- C++17 (`-std=c++17`)
- libcurl — ships by default on macOS and most Linux distros. On Windows, grab it from [curl.se](https://curl.se/windows/).

## See Also

The C++ LLM header suite — each is a single `.hpp`, no extra deps:

| Repo | What it does |
|------|-------------|
| **llm-stream** *(this repo)* | Streaming responses from OpenAI & Anthropic |
| [llm-cache](https://github.com/Mattbusel/llm-cache) | Response caching — skip redundant API calls |
| [llm-cost](https://github.com/Mattbusel/llm-cost) | Token counting + per-model cost estimation |
| [llm-retry](https://github.com/Mattbusel/llm-retry) | Retry logic + circuit breaker |
| [llm-format](https://github.com/Mattbusel/llm-format) | Structured output / JSON schema enforcement |

## License

MIT — see [LICENSE](LICENSE).
