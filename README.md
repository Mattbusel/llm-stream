# llm-stream

[![CI](https://github.com/Mattbusel/llm-stream/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-stream/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stream OpenAI and Anthropic responses token by token from C++.** One header, `llm_stream.hpp`. Needs libcurl (`apt install libcurl4-openssl-dev`, preinstalled on macOS, `vcpkg install curl` on Windows).

Streaming is what makes an LLM feature feel fast: text appears as it is generated instead of after a long pause. Doing it from C++ means an HTTP client, server-sent events parsing and two different provider formats. llm-stream wraps all of that in three functions and a callback, built on libcurl.

## Features

- `stream_openai()` for the Chat Completions API, `stream_anthropic()` for the Messages API
- `stream()` picks the provider from the model name: `claude-*` goes to Anthropic, everything else to OpenAI
- Token callback receives each text fragment as it arrives
- Done callback with token count, elapsed time and tokens per second
- Error callback for network and HTTP errors
- System prompt, max tokens and temperature in `Config`

## Quick start

Copy the header into your project:

```bash
curl -fsSLO https://raw.githubusercontent.com/Mattbusel/llm-stream/main/include/llm_stream.hpp
```

Define `LLM_STREAM_IMPLEMENTATION` in exactly one `.cpp` file before including it; every other file just includes the header. Save this as `main.cpp` next to the header:

```cpp
#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    const char* key = std::getenv("OPENAI_API_KEY");
    if (!key) { std::cerr << "set OPENAI_API_KEY\n"; return 1; }

    llm::Config cfg;
    cfg.api_key       = key;
    cfg.model         = "gpt-4o-mini";   // a "claude-..." model goes to Anthropic
    cfg.system_prompt = "Answer in one short paragraph.";

    llm::stream("Explain recursion.", cfg,
        [](std::string_view tok) { std::cout << tok << std::flush; },
        [](const llm::StreamStats& s) {
            std::cout << "\n[" << s.token_count << " tokens, "
                      << s.tokens_per_sec << " tok/s]\n";
        },
        [](std::string_view err) { std::cerr << "error: " << err << "\n"; });
}
```

```bash
g++ -std=c++17 -O2 main.cpp -lcurl -o demo
```

## API at a glance

| Call | Purpose |
|---|---|
| `stream(prompt, cfg, on_token, on_done, on_error)` | Auto-select provider by model name |
| `stream_openai(...)` / `stream_anthropic(...)` | Call a specific provider |
| `Config{api_key, model, max_tokens, temperature, system_prompt}` | Request settings |
| `StreamStats{token_count, elapsed_ms, tokens_per_sec}` | Passed to `on_done` |

## Notes and limitations

- Each call is a single-turn request (one user message plus optional system prompt). For multi-turn history see [llm-chat](https://github.com/Mattbusel/llm-chat).
- Calls block until the stream ends; run them on a worker thread (or via [llm-pool](https://github.com/Mattbusel/llm-pool)) if you need your UI to stay responsive.
- `<curl/curl.h>` is included by the public header, so every file that includes `llm_stream.hpp` needs the libcurl headers on its include path.

## Build the examples

The repo builds `examples/basic_stream.cpp`, `examples/chat_loop.cpp` with CMake (requires libcurl):

```bash
cmake -B build
cmake --build build
```

## Part of llm-cpp

llm-stream is one of 26 single-header C++ libraries in [llm-cpp](https://github.com/Mattbusel/llm-cpp), a toolkit for building LLM features into native code. Each library stands alone; combine them by giving each `*_IMPLEMENTATION` define its own `.cpp` file. See the [llm-cpp README](https://github.com/Mattbusel/llm-cpp#using-several-together) for the full list and examples of using several together.

## License

MIT. See [LICENSE](LICENSE).
