# llm-stream

Stream OpenAI and Anthropic responses in C++. Drop in one header. No deps.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Single Header](https://img.shields.io/badge/single-header-orange)
![libcurl](https://img.shields.io/badge/dep-libcurl-lightgrey)

---

## 30-second quickstart

```cpp
#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    llm::Config cfg;
    cfg.api_key = std::getenv("OPENAI_API_KEY");
    cfg.model   = "gpt-4o-mini";

    llm::stream_openai("Explain recursion in one paragraph.", cfg,
        [](std::string_view token) { std::cout << token << std::flush; },
        [](const llm::StreamStats& s) {
            std::cout << "\n\n[" << s.token_count << " tokens, "
                      << static_cast<int>(s.tokens_per_sec) << " tok/s]\n";
        }
    );
}
```

```bash
export OPENAI_API_KEY=sk-...
cmake -B build && cmake --build build
./build/basic_stream
```

---

## Installation

Copy `include/llm_stream.hpp` into your project. That's it.

```bash
cp llm_stream.hpp /your/project/include/
```

In **one** `.cpp` file:
```cpp
#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"
```

In all other files:
```cpp
#include "llm_stream.hpp"
```

Link against libcurl (`-lcurl` or `CURL::libcurl` in CMake).

---

## API reference

### `llm::Config`

```cpp
struct Config {
    std::string api_key;          // Required: your API key
    std::string model;            // Required: e.g. "gpt-4o-mini", "claude-3-5-haiku-20241022"
    int         max_tokens  = 1024;
    double      temperature = 0.7;
    std::string system_prompt;    // Optional system/instruction message
};
```

### `llm::StreamStats`

```cpp
struct StreamStats {
    size_t token_count;    // Number of token fragments received
    double elapsed_ms;     // Total wall-clock time in milliseconds
    double tokens_per_sec; // Throughput
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
// Stream from OpenAI chat/completions endpoint
void llm::stream_openai(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,   // optional
    ErrorCallback      on_error = nullptr    // optional
);

// Stream from Anthropic messages endpoint
void llm::stream_anthropic(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

// Auto-detect provider: "claude-*" → Anthropic, everything else → OpenAI
void llm::stream(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);
```

---

## Examples

| File | Description |
|------|-------------|
| [`examples/basic_stream.cpp`](examples/basic_stream.cpp) | Minimal OpenAI streaming hello-world |
| [`examples/chat_loop.cpp`](examples/chat_loop.cpp) | Multi-turn interactive REPL, auto-detects provider |

---

## Why

- **No Python runtime.** Ship LLM features inside your existing C++ binary — games, embedded apps, CLIs, servers.
- **No build complexity.** One header, one dependency (libcurl), done. No CMake package hunting for JSON libraries.
- **Drop into any project.** Copy one file. Works with any build system: CMake, Meson, Bazel, hand-written Makefiles.

---

## Building the examples

```bash
# Configure and build
cmake -B build
cmake --build build

# OpenAI example
export OPENAI_API_KEY=sk-...
./build/basic_stream

# Anthropic or OpenAI interactive REPL (set either key)
export ANTHROPIC_API_KEY=sk-ant-...
./build/chat_loop
```

---

## Requirements

- C++17 or later
- libcurl (ships by default on macOS and most Linux distros; install via `apt install libcurl4-openssl-dev` on Ubuntu, or [vcpkg](https://vcpkg.io) on Windows)

---

## License

MIT — see [LICENSE](LICENSE).
