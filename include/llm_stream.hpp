#pragma once

// llm_stream.hpp — Zero-dependency single-header C++ library for streaming LLM responses.
// Supports OpenAI and Anthropic APIs via libcurl SSE streaming.
//
// USAGE:
//   In exactly ONE .cpp file before including:
//     #define LLM_STREAM_IMPLEMENTATION
//     #include "llm_stream.hpp"
//
//   In all other files just:
//     #include "llm_stream.hpp"
//
// Requires: libcurl (ships on macOS, most Linux distros; vcpkg/apt on Windows)

#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <curl/curl.h>

namespace llm {

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Statistics surfaced at the end of a stream.
struct StreamStats {
    size_t token_count    = 0;
    double elapsed_ms     = 0.0;
    double tokens_per_sec = 0.0;
};

/// Configuration for an LLM request.
struct Config {
    std::string api_key;
    std::string model;
    int         max_tokens  = 1024;
    double      temperature = 0.7;
    std::string system_prompt; ///< Optional system/instruction prompt
};

using TokenCallback = std::function<void(std::string_view token)>;
using DoneCallback  = std::function<void(const StreamStats&)>;
using ErrorCallback = std::function<void(std::string_view error)>;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Stream a response from OpenAI (chat/completions) via SSE.
///
/// # Arguments
/// * `prompt`   — User message
/// * `config`   — API key, model, generation parameters
/// * `on_token` — Invoked for every streamed token fragment
/// * `on_done`  — Invoked once when the stream ends successfully (optional)
/// * `on_error` — Invoked on network/HTTP error (optional)
///
/// # Panics
/// This function never panics.
void stream_openai(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

/// Stream a response from Anthropic (messages) via SSE.
///
/// # Arguments
/// * `prompt`   — User message
/// * `config`   — API key, model, generation parameters
/// * `on_token` — Invoked for every streamed token fragment
/// * `on_done`  — Invoked once when the stream ends successfully (optional)
/// * `on_error` — Invoked on network/HTTP error (optional)
///
/// # Panics
/// This function never panics.
void stream_anthropic(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

/// Auto-detect provider from model name prefix and stream.
/// Models starting with "claude-" → Anthropic; everything else → OpenAI.
///
/// # Example
/// ```cpp
/// llm::Config cfg;
/// cfg.api_key = std::getenv("OPENAI_API_KEY");
/// cfg.model   = "gpt-4o-mini";
/// llm::stream("Hello!", cfg, [](std::string_view tok){ std::cout << tok; });
/// ```
///
/// # Panics
/// This function never panics.
void stream(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

} // namespace llm

// ---------------------------------------------------------------------------
// Implementation — compiled in exactly one translation unit.
// ---------------------------------------------------------------------------
#ifdef LLM_STREAM_IMPLEMENTATION

namespace llm {
namespace detail {

// ---------------------------------------------------------------------------
// Minimal hand-rolled JSON string extractor. No external deps.
// Finds `"key": "value"` (whitespace-tolerant) and returns the unescaped
// string value, or empty string if not found.
// ---------------------------------------------------------------------------
static std::string json_string_value(std::string_view json, std::string_view key) {
    // Build search token: "key"
    std::string pattern;
    pattern.reserve(key.size() + 2);
    pattern += '"';
    pattern += key;
    pattern += '"';

    auto pos = json.find(pattern);
    if (pos == std::string_view::npos) return {};

    pos += pattern.size();
    // Skip optional whitespace and the colon
    while (pos < json.size() &&
           (json[pos] == ' ' || json[pos] == '\t' || json[pos] == ':'))
        ++pos;

    if (pos >= json.size() || json[pos] != '"') return {};
    ++pos; // skip opening quote

    std::string result;
    result.reserve(64);
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            switch (json[pos + 1]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += json[pos + 1]; break;
            }
            pos += 2;
        } else {
            result += json[pos++];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// RAII wrappers around libcurl types
// ---------------------------------------------------------------------------
struct CurlHandle {
    CURL* handle = nullptr;
    CurlHandle() : handle(curl_easy_init()) {}
    ~CurlHandle() { if (handle) curl_easy_cleanup(handle); }
    CurlHandle(const CurlHandle&)            = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    explicit operator CURL*() const { return handle; }
    bool ok() const { return handle != nullptr; }
};

struct CurlSlist {
    curl_slist* list = nullptr;
    ~CurlSlist() { if (list) curl_slist_free_all(list); }
    CurlSlist(const CurlSlist&)            = delete;
    CurlSlist& operator=(const CurlSlist&) = delete;
    CurlSlist() = default;
    void append(const char* s) { list = curl_slist_append(list, s); }
};

// ---------------------------------------------------------------------------
// SSE streaming write-callback context
// ---------------------------------------------------------------------------
enum class Provider { OpenAI, Anthropic };

struct StreamCtx {
    Provider      provider;
    TokenCallback on_token;
    std::string   buffer;       // incomplete SSE line accumulator
    size_t        token_count = 0;
};

// Extract the content delta from an OpenAI SSE data line.
// Payload format: {"id":...,"choices":[{"delta":{"content":"<tok>"},...}],...}
static std::string openai_extract_delta(std::string_view payload) {
    auto choices = payload.find("\"choices\"");
    if (choices == std::string_view::npos) return {};
    auto delta = payload.find("\"delta\"", choices);
    if (delta == std::string_view::npos) return {};
    return json_string_value(payload.substr(delta), "content");
}

// Extract the text delta from an Anthropic SSE data line.
// Payload format: {"type":"content_block_delta","delta":{"type":"text_delta","text":"<tok>"}}
static std::string anthropic_extract_delta(std::string_view payload) {
    auto delta = payload.find("\"delta\"");
    if (delta == std::string_view::npos) return {};
    return json_string_value(payload.substr(delta), "text");
}

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t total = size * nmemb;
    auto* ctx = static_cast<StreamCtx*>(userdata);

    ctx->buffer.append(ptr, total);

    // Process every complete newline-terminated SSE line
    size_t start = 0;
    while (true) {
        auto nl = ctx->buffer.find('\n', start);
        if (nl == std::string::npos) break;

        std::string_view line(ctx->buffer.data() + start, nl - start);
        if (!line.empty() && line.back() == '\r')
            line = line.substr(0, line.size() - 1);

        if (line.substr(0, 6) == "data: ") {
            std::string_view payload = line.substr(6);
            if (payload != "[DONE]") {
                std::string token;
                if (ctx->provider == Provider::OpenAI)
                    token = openai_extract_delta(payload);
                else
                    token = anthropic_extract_delta(payload);

                if (!token.empty() && ctx->on_token) {
                    ctx->on_token(token);
                    ++ctx->token_count;
                }
            }
        }
        start = nl + 1;
    }

    // Keep any incomplete line
    ctx->buffer.erase(0, start);
    return total;
}

// ---------------------------------------------------------------------------
// Hand-rolled JSON body builders
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static std::string build_openai_body(const std::string& prompt, const Config& cfg) {
    std::string b;
    b  = "{\"model\":\""      + json_escape(cfg.model) + "\","
         "\"stream\":true,"
         "\"max_tokens\":"    + std::to_string(cfg.max_tokens) + ","
         "\"temperature\":"   + std::to_string(cfg.temperature) + ","
         "\"messages\":[";
    if (!cfg.system_prompt.empty())
        b += "{\"role\":\"system\",\"content\":\"" + json_escape(cfg.system_prompt) + "\"},";
    b += "{\"role\":\"user\",\"content\":\"" + json_escape(prompt) + "\"}]}";
    return b;
}

static std::string build_anthropic_body(const std::string& prompt, const Config& cfg) {
    std::string b;
    b  = "{\"model\":\""    + json_escape(cfg.model) + "\","
         "\"stream\":true,"
         "\"max_tokens\":"  + std::to_string(cfg.max_tokens) + ",";
    if (!cfg.system_prompt.empty())
        b += "\"system\":\"" + json_escape(cfg.system_prompt) + "\",";
    b += "\"messages\":[{\"role\":\"user\",\"content\":\"" + json_escape(prompt) + "\"}]}";
    return b;
}

// ---------------------------------------------------------------------------
// Core SSE driver — each call is fully self-contained and thread-safe.
// ---------------------------------------------------------------------------
static void do_stream(
    const std::string& url,
    const std::string& json_body,
    const CurlSlist&   headers,
    Provider           provider,
    TokenCallback      on_token,
    DoneCallback       on_done,
    ErrorCallback      on_error)
{
    CurlHandle curl;
    if (!curl.ok()) {
        if (on_error) on_error("Failed to initialize curl handle");
        return;
    }

    StreamCtx ctx;
    ctx.provider = provider;
    ctx.on_token = on_token;

    const auto t0 = std::chrono::steady_clock::now();

    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_URL,           url.c_str());
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_HTTPHEADER,    headers.list);
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_POSTFIELDS,    json_body.c_str());
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_WRITEDATA,     &ctx);
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_FOLLOWLOCATION,1L);
    curl_easy_setopt(static_cast<CURL*>(curl), CURLOPT_TIMEOUT,       120L);

    const CURLcode res = curl_easy_perform(static_cast<CURL*>(curl));

    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    if (res != CURLE_OK) {
        if (on_error) on_error(curl_easy_strerror(res));
        return;
    }

    long http_code = 0;
    curl_easy_getinfo(static_cast<CURL*>(curl), CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code >= 400) {
        if (on_error) on_error("HTTP error " + std::to_string(http_code));
        return;
    }

    if (on_done) {
        StreamStats stats;
        stats.token_count    = ctx.token_count;
        stats.elapsed_ms     = ms;
        stats.tokens_per_sec = (ms > 0.0) ? (ctx.token_count / (ms / 1000.0)) : 0.0;
        on_done(stats);
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public function definitions
// ---------------------------------------------------------------------------

void stream_openai(
    const std::string& prompt, const Config& config,
    TokenCallback on_token, DoneCallback on_done, ErrorCallback on_error)
{
    detail::CurlSlist headers;
    headers.append(("Authorization: Bearer " + config.api_key).c_str());
    headers.append("Content-Type: application/json");
    detail::do_stream(
        "https://api.openai.com/v1/chat/completions",
        detail::build_openai_body(prompt, config),
        headers, detail::Provider::OpenAI,
        on_token, on_done, on_error);
}

void stream_anthropic(
    const std::string& prompt, const Config& config,
    TokenCallback on_token, DoneCallback on_done, ErrorCallback on_error)
{
    detail::CurlSlist headers;
    headers.append(("x-api-key: " + config.api_key).c_str());
    headers.append("anthropic-version: 2023-06-01");
    headers.append("Content-Type: application/json");
    detail::do_stream(
        "https://api.anthropic.com/v1/messages",
        detail::build_anthropic_body(prompt, config),
        headers, detail::Provider::Anthropic,
        on_token, on_done, on_error);
}

void stream(
    const std::string& prompt, const Config& config,
    TokenCallback on_token, DoneCallback on_done, ErrorCallback on_error)
{
    if (config.model.rfind("claude-", 0) == 0)
        stream_anthropic(prompt, config, on_token, on_done, on_error);
    else
        stream_openai(prompt, config, on_token, on_done, on_error);
}

} // namespace llm

#endif // LLM_STREAM_IMPLEMENTATION
