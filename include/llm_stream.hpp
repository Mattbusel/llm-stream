#pragma once

// llm_stream.hpp -- Zero-dependency single-header C++ library for streaming
// OpenAI and Anthropic LLM responses via libcurl.
//
// USAGE:
//   In exactly ONE translation unit:
//     #define LLM_STREAM_IMPLEMENTATION
//     #include "llm_stream.hpp"
//
//   In all other translation units:
//     #include "llm_stream.hpp"

#include <functional>
#include <string>
#include <string_view>
#include <chrono>

namespace llm {

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

struct Config {
    std::string api_key;
    std::string model;
    int         max_tokens  = 1024;
    double      temperature = 0.7;
    std::string system_prompt;
};

struct StreamStats {
    size_t token_count    = 0;
    double elapsed_ms     = 0.0;
    double tokens_per_sec = 0.0;
};

using TokenCallback = std::function<void(std::string_view token)>;
using DoneCallback  = std::function<void(const StreamStats&)>;
using ErrorCallback = std::function<void(std::string_view error)>;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Stream a completion from OpenAI.
/// on_token is called for each text delta, on_done when complete, on_error on failure.
void stream_openai(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

/// Stream a completion from Anthropic.
/// on_token is called for each text delta, on_done when complete, on_error on failure.
void stream_anthropic(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

/// Auto-detect provider from model name prefix.
/// "claude-*" -> stream_anthropic, everything else -> stream_openai.
void stream(
    const std::string& prompt,
    const Config&      config,
    TokenCallback      on_token,
    DoneCallback       on_done  = nullptr,
    ErrorCallback      on_error = nullptr
);

} // namespace llm

// ---------------------------------------------------------------------------
// Implementation -- compiled in exactly one translation unit
// ---------------------------------------------------------------------------

#ifdef LLM_STREAM_IMPLEMENTATION

#include <curl/curl.h>
#include <cstdio>
#include <sstream>

namespace llm {
namespace detail {

// ---------------------------------------------------------------------------
// RAII wrappers
// ---------------------------------------------------------------------------

struct CurlHandle {
    CURL* handle = nullptr;
    CurlHandle() : handle(curl_easy_init()) {}
    ~CurlHandle() { if (handle) curl_easy_cleanup(handle); }
    CurlHandle(const CurlHandle&)            = delete;
    CurlHandle& operator=(const CurlHandle&) = delete;
    explicit operator bool() const { return handle != nullptr; }
};

struct CurlSlist {
    curl_slist* list = nullptr;
    CurlSlist() = default;
    ~CurlSlist() { if (list) curl_slist_free_all(list); }
    CurlSlist(const CurlSlist&)            = delete;
    CurlSlist& operator=(const CurlSlist&) = delete;
    void append(const char* h) { list = curl_slist_append(list, h); }
};

// ---------------------------------------------------------------------------
// Minimal hand-rolled JSON string extractor (no external library)
// ---------------------------------------------------------------------------

static std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return {};
    ++pos;
    std::string result;
    while (pos < json.size()) {
        char c = json[pos++];
        if (c == '\\' && pos < json.size()) {
            char e = json[pos++];
            switch (e) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += e;    break;
            }
        } else if (c == '"') {
            break;
        } else {
            result += c;
        }
    }
    return result;
}

// Find sub-object after key1, then extract string key2 from within it.
static std::string extract_nested_string(const std::string& json,
                                          const std::string& key1,
                                          const std::string& key2) {
    std::string needle = "\"" + key1 + "\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    while (pos < json.size() && json[pos] != '{') ++pos;
    if (pos >= json.size()) return {};
    int depth = 0;
    size_t start = pos, end = pos;
    for (size_t i = pos; i < json.size(); ++i) {
        if      (json[i] == '{') ++depth;
        else if (json[i] == '}') { if (--depth == 0) { end = i + 1; break; } }
    }
    return extract_json_string(json.substr(start, end - start), key2);
}

// ---------------------------------------------------------------------------
// SSE streaming context and curl write callback
// ---------------------------------------------------------------------------

struct StreamContext {
    std::string   buffer;
    TokenCallback on_token;
    DoneCallback  on_done;
    ErrorCallback on_error;
    size_t        token_count  = 0;
    bool          done         = false;
    bool          is_anthropic = false;
};

static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    size_t total = size * nmemb;
    auto* ctx = static_cast<StreamContext*>(userdata);
    ctx->buffer.append(ptr, total);

    size_t pos = 0;
    while (true) {
        auto nl = ctx->buffer.find('\n', pos);
        if (nl == std::string::npos) break;
        std::string line = ctx->buffer.substr(pos, nl - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        pos = nl + 1;
        if (line.empty()) continue;

        if (line.rfind("data: ", 0) == 0) {
            std::string data = line.substr(6);
            if (data == "[DONE]") { ctx->done = true; break; }

            std::string token;
            if (ctx->is_anthropic) {
                // Anthropic SSE: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
                token = extract_nested_string(data, "delta", "text");
            } else {
                // OpenAI SSE: {"choices":[{"delta":{"content":"..."}}]}
                token = extract_nested_string(data, "delta", "content");
            }
            if (!token.empty() && ctx->on_token) {
                ctx->on_token(token);
                ++ctx->token_count;
            }
        }
    }
    ctx->buffer = ctx->buffer.substr(pos);
    return total;
}

// ---------------------------------------------------------------------------
// JSON body builders
// ---------------------------------------------------------------------------

static std::string escape_json(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
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
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

static std::string build_openai_body(const std::string& prompt, const Config& cfg) {
    std::ostringstream ss;
    ss << "{\"model\":\"" << escape_json(cfg.model) << "\","
       << "\"stream\":true,"
       << "\"max_tokens\":" << cfg.max_tokens << ","
       << "\"temperature\":" << cfg.temperature << ","
       << "\"messages\":[";
    if (!cfg.system_prompt.empty())
        ss << "{\"role\":\"system\",\"content\":\"" << escape_json(cfg.system_prompt) << "\"},";
    ss << "{\"role\":\"user\",\"content\":\"" << escape_json(prompt) << "\"}]}";
    return ss.str();
}

static std::string build_anthropic_body(const std::string& prompt, const Config& cfg) {
    std::ostringstream ss;
    ss << "{\"model\":\"" << escape_json(cfg.model) << "\","
       << "\"stream\":true,"
       << "\"max_tokens\":" << cfg.max_tokens << ",";
    if (!cfg.system_prompt.empty())
        ss << "\"system\":\"" << escape_json(cfg.system_prompt) << "\",";
    ss << "\"messages\":[{\"role\":\"user\",\"content\":\"" << escape_json(prompt) << "\"}]}";
    return ss.str();
}

// ---------------------------------------------------------------------------
// Core HTTP streaming
// ---------------------------------------------------------------------------

static void do_stream(const std::string& url, const std::string& body,
                       CurlSlist& headers, StreamContext& ctx) {
    CurlHandle curl;
    if (!curl) {
        if (ctx.on_error) ctx.on_error("Failed to initialize curl handle");
        return;
    }
    curl_easy_setopt(curl.handle, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl.handle, CURLOPT_POST,           1L);
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDS,     body.c_str());
    curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDSIZE,  static_cast<long>(body.size()));
    curl_easy_setopt(curl.handle, CURLOPT_HTTPHEADER,     headers.list);
    curl_easy_setopt(curl.handle, CURLOPT_WRITEFUNCTION,  write_callback);
    curl_easy_setopt(curl.handle, CURLOPT_WRITEDATA,      &ctx);
    curl_easy_setopt(curl.handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.handle, CURLOPT_TCP_NODELAY,    1L);

    CURLcode res = curl_easy_perform(curl.handle);
    if (res != CURLE_OK) {
        if (ctx.on_error) ctx.on_error(curl_easy_strerror(res));
        return;
    }
    long http_code = 0;
    curl_easy_getinfo(curl.handle, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code >= 400) {
        std::string err = "HTTP " + std::to_string(http_code);
        if (!ctx.buffer.empty()) err += ": " + ctx.buffer;
        if (ctx.on_error) ctx.on_error(err);
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public implementations
// ---------------------------------------------------------------------------

void stream_openai(const std::string& prompt, const Config& config,
                   TokenCallback on_token, DoneCallback on_done, ErrorCallback on_error) {
    auto t0 = std::chrono::steady_clock::now();

    detail::StreamContext ctx;
    ctx.on_token     = on_token;
    ctx.on_done      = on_done;
    ctx.on_error     = on_error;
    ctx.is_anthropic = false;

    std::string body = detail::build_openai_body(prompt, config);
    detail::CurlSlist headers;
    headers.append("Content-Type: application/json");
    headers.append(("Authorization: Bearer " + config.api_key).c_str());

    detail::do_stream("https://api.openai.com/v1/chat/completions", body, headers, ctx);

    if (on_done) {
        auto t1 = std::chrono::steady_clock::now();
        StreamStats s;
        s.token_count    = ctx.token_count;
        s.elapsed_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
        s.tokens_per_sec = s.elapsed_ms > 0.0 ? s.token_count / (s.elapsed_ms / 1000.0) : 0.0;
        on_done(s);
    }
}

void stream_anthropic(const std::string& prompt, const Config& config,
                      TokenCallback on_token, DoneCallback on_done, ErrorCallback on_error) {
    auto t0 = std::chrono::steady_clock::now();

    detail::StreamContext ctx;
    ctx.on_token     = on_token;
    ctx.on_done      = on_done;
    ctx.on_error     = on_error;
    ctx.is_anthropic = true;

    std::string body = detail::build_anthropic_body(prompt, config);
    detail::CurlSlist headers;
    headers.append("Content-Type: application/json");
    headers.append(("x-api-key: " + config.api_key).c_str());
    headers.append("anthropic-version: 2023-06-01");

    detail::do_stream("https://api.anthropic.com/v1/messages", body, headers, ctx);

    if (on_done) {
        auto t1 = std::chrono::steady_clock::now();
        StreamStats s;
        s.token_count    = ctx.token_count;
        s.elapsed_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
        s.tokens_per_sec = s.elapsed_ms > 0.0 ? s.token_count / (s.elapsed_ms / 1000.0) : 0.0;
        on_done(s);
    }
}

void stream(const std::string& prompt, const Config& config,
            TokenCallback on_token, DoneCallback on_done, ErrorCallback on_error) {
    if (config.model.rfind("claude", 0) == 0)
        stream_anthropic(prompt, config, on_token, on_done, on_error);
    else
        stream_openai(prompt, config, on_token, on_done, on_error);
}

} // namespace llm

#endif // LLM_STREAM_IMPLEMENTATION
