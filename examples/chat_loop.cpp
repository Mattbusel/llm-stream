#define LLM_STREAM_IMPLEMENTATION
#include "llm_stream.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Simple multi-turn conversation history (role + content pairs)
struct Message {
    std::string role;
    std::string content;
};

// Manually accumulate the full assistant reply so we can append it to history
// after the stream completes.
static std::string g_current_reply;

int main() {
    // Auto-detect provider: prefer Anthropic if key is set, else OpenAI
    const char* anthropic_key = std::getenv("ANTHROPIC_API_KEY");
    const char* openai_key    = std::getenv("OPENAI_API_KEY");

    bool use_anthropic = (anthropic_key && anthropic_key[0] != '\0');
    bool use_openai    = (openai_key    && openai_key[0]    != '\0');

    if (!use_anthropic && !use_openai) {
        std::cerr << "Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY.\n";
        return 1;
    }

    llm::Config cfg;
    if (use_anthropic) {
        cfg.api_key = anthropic_key;
        cfg.model   = "claude-3-5-haiku-20241022";
        std::cout << "[Using Anthropic: " << cfg.model << "]\n\n";
    } else {
        cfg.api_key = openai_key;
        cfg.model   = "gpt-4o-mini";
        std::cout << "[Using OpenAI: " << cfg.model << "]\n\n";
    }

    cfg.system_prompt = "You are a helpful, concise assistant.";
    cfg.max_tokens    = 1024;

    std::vector<Message> history;
    std::string user_input;

    std::cout << "Chat loop started. Type 'exit' to quit.\n";
    std::cout << "----------------------------------------\n";

    while (true) {
        std::cout << "\nYou: ";
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "exit" || user_input == "quit") break;
        if (user_input.empty()) continue;

        history.push_back({"user", user_input});
        g_current_reply.clear();

        std::cout << "\nAssistant: ";

        llm::stream(
            user_input,
            cfg,
            [](std::string_view token) {
                std::cout << token << std::flush;
                g_current_reply += token;
            },
            [](const llm::StreamStats& s) {
                std::cout << "\n\n["
                          << s.token_count << " tokens | "
                          << static_cast<int>(s.tokens_per_sec) << " tok/s | "
                          << static_cast<int>(s.elapsed_ms) << " ms]\n";
            },
            [](std::string_view err) {
                std::cerr << "\nError: " << err << '\n';
            }
        );

        if (!g_current_reply.empty())
            history.push_back({"assistant", g_current_reply});
    }

    std::cout << "\nGoodbye!\n";
    return 0;
}
