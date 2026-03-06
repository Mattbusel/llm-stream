#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Simple multi-turn message history entry
struct Message {
    std::string role;
    std::string content;
};

int main() {
    // Auto-detect provider from environment
    const char* anthropic_key = std::getenv("ANTHROPIC_API_KEY");
    const char* openai_key    = std::getenv("OPENAI_API_KEY");

    llm::Config cfg;
    bool use_anthropic = false;

    if (anthropic_key) {
        cfg.api_key      = anthropic_key;
        cfg.model        = "claude-3-5-haiku-20241022";
        use_anthropic    = true;
    } else if (openai_key) {
        cfg.api_key = openai_key;
        cfg.model   = "gpt-4o-mini";
    } else {
        std::cerr << "Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY\n";
        return 1;
    }

    std::cout << "[Using " << (use_anthropic ? "Anthropic" : "OpenAI")
              << ": " << cfg.model << "]\n\n"
              << "Chat loop started. Type 'exit' to quit.\n"
              << "----------------------------------------\n\n";

    std::vector<Message> history;
    std::string line;

    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, line) || line == "exit") break;
        if (line.empty()) continue;

        history.push_back({"user", line});

        // Build a single prompt string with conversation history embedded.
        // For a real app you would pass the messages array directly via the API;
        // here we keep it simple and fold history into one text block so we can
        // reuse the same stream_* functions without changing the public API.
        std::string prompt;
        for (const auto& msg : history) {
            prompt += msg.role + ": " + msg.content + "\n";
        }
        prompt += "assistant:";

        std::string assistant_reply;

        std::cout << "Assistant: " << std::flush;

        auto on_token = [&](std::string_view token) {
            std::cout << token << std::flush;
            assistant_reply += token;
        };

        auto on_done = [](const llm::StreamStats& s) {
            std::cout << "\n[" << s.token_count << " tokens | "
                      << static_cast<int>(s.tokens_per_sec) << " tok/s | "
                      << static_cast<int>(s.elapsed_ms) << " ms]\n\n";
        };

        auto on_error = [](std::string_view err) {
            std::cerr << "\nError: " << err << "\n\n";
        };

        if (use_anthropic)
            llm::stream_anthropic(prompt, cfg, on_token, on_done, on_error);
        else
            llm::stream_openai(prompt, cfg, on_token, on_done, on_error);

        if (!assistant_reply.empty())
            history.push_back({"assistant", assistant_reply});
    }

    std::cout << "Goodbye.\n";
}
