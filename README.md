# llm-stream

Stream OpenAI and Anthropic responses in C++. Drop in one header. No deps.

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Single Header](https://img.shields.io/badge/library-single--header-orange)
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

Compile and run:

```bash
g++ -std=c++17 -O2 basic_stream.cpp -lcurl -o basic_stream
export OPENAI_API_KEY=sk-...
./basic_stream
```

---

## Installation

Copy one file into your project:

```bash
cp include/llm_stream.hpp your-project/
```

That's it. No package manager, no build system changes beyond adding `-lcurl`.

---

## API Reference

### Types

```cpp
namespace llm {

// Configuration for an LLM request
struct Config {
    std::string api_key;
    std::string model;
    int         max_tokens  = 1024;   // token budget
    double      temperature = 0.7;    // 0.0-2.0
    std::string system_prompt;        // optional system/instruction message
};

// Statistics reported at end of stream
struct StreamStats {
    size_t token_count;    // total tokens received
    double elapsed_ms;     // wall time from first byte to [DONE]
    double tokens_per_sec; // throughput
};

// Callback types
using TokenCallback = std::function<void(std::string_view token)>;
using DoneCallback  = std::function<void(const StreamStats&)>;
using ErrorCallback = std::function<void(std::string_view error)>;

} // namespace llm
```

### Functions

```cpp
// Stream from OpenAI chat/completions
void llm::stream_openai(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,   // optional
    ErrorCallback      on_error = nullptr    // optional
);

// Stream from Anthropic messages API
void llm::stream_anthropic(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

// Auto-detect provider from model name:
//   "claude-*"  -> Anthropic
//   everything else -> OpenAI
void llm::stream(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);
```

### Implementation guard

In **exactly one** `.cpp` file, define `LLM_STREAM_IMPLEMENTATION` before the include:

```cpp
// my_app.cpp
#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"
```

In all other files, just include it normally:

```cpp
// other_file.cpp
#include "llm_stream.hpp"
```

---

## Examples

| File | Description |
|------|-------------|
| [`examples/basic_stream.cpp`](examples/basic_stream.cpp) | Stream one OpenAI response, print tokens and stats |
| [`examples/chat_loop.cpp`](examples/chat_loop.cpp) | Interactive multi-turn REPL, auto-picks OpenAI or Anthropic |

### Chat loop (Anthropic)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./build/chat_loop

# [Using Anthropic: claude-3-5-haiku-20241022]
#
# Chat loop started. Type 'exit' to quit.
# ----------------------------------------
#
# You: What is the capital of France?
# Assistant: Paris is the capital of France...
# [23 tokens | 87 tok/s | 264 ms]
```

---

## Why

- **No Python runtime.** Deploy a static binary. LLM calls in a game engine, CLI tool, embedded app, or server -- no interpreter required.
- **No build complexity.** One header + `-lcurl`. Works with any existing C++ build system: Make, CMake, Bazel, Meson, or plain `g++`.
- **Drop into any C++ project.** No namespace pollution, no global state, no init/shutdown -- just call `llm::stream()` and go.

---

## Building the examples

```bash
cmake -B build
cmake --build build

# OpenAI example
export OPENAI_API_KEY=sk-...
./build/basic_stream

# Multi-turn REPL (uses ANTHROPIC_API_KEY if set, else OPENAI_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
./build/chat_loop
```

---

## Requirements

- C++17 or later
- libcurl (ships by default on macOS and most Linux distros; `apt install libcurl4-openssl-dev` on Debian/Ubuntu; `vcpkg install curl` on Windows)

---

## License

MIT -- see [LICENSE](LICENSE).
