// Local, single-process evaluation bridge. stdout is exclusively JSON lines.
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {
constexpr size_t max_line_bytes = 16 * 1024 * 1024;
constexpr uint32_t batch_capacity = 512;
constexpr int32_t cpu_threads = 4;

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

struct Batch {
    llama_batch value;
    explicit Batch(int32_t size) : value(llama_batch_init(size, 0, 1)) {
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.logits) {
            llama_batch_free(value);
            throw std::runtime_error("Unable to allocate decode batch");
        }
    }
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};

void log_to_stderr(ggml_log_level level, const char * text, void *) {
    if (level != GGML_LOG_LEVEL_DEBUG) {
        std::fputs(text, stderr);
    }
}

void respond(const json & value) {
    // A length-limited generation can end halfway through a UTF-8 byte sequence.
    // Retain IDs and produce valid JSON; the caller can also decode those IDs with HF.
    std::cout << value.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
    std::cout.flush();
    if (!std::cout) {
        throw std::runtime_error("Unable to write JSON response");
    }
}

int32_t parse_token_id(const std::string & text) {
    int32_t value = -1;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || value < 0) {
        throw std::runtime_error("--eos-token-id must be a nonnegative int32 token ID");
    }
    return value;
}

int64_t bounded_integer(const json & value, int64_t lower, int64_t upper, const char * name) {
    if (!value.is_number_integer()) {
        throw std::runtime_error(std::string(name) + " must be an integer");
    }
    if (value.is_number_unsigned()) {
        const uint64_t integer = value.get<uint64_t>();
        if (integer > static_cast<uint64_t>(upper)) {
            throw std::runtime_error(std::string(name) + " is outside the permitted range");
        }
    }
    const int64_t integer = value.get<int64_t>();
    if (integer < lower || integer > upper) {
        throw std::runtime_error(std::string(name) + " is outside the permitted range");
    }
    return integer;
}

// Drain oversized requests before reporting an error, so the next line is still
// a fresh request. No line or token array is allowed to allocate without a bound.
bool read_request(std::string & line, bool & oversized) {
    line.clear();
    oversized = false;
    char character;
    bool received = false;
    while (std::cin.get(character)) {
        received = true;
        if (character == '\n') {
            break;
        }
        if (line.size() < max_line_bytes) {
            line.push_back(character);
        } else {
            oversized = true;
        }
    }
    return received;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & prompt,
                                  int32_t model_context_length) {
    if (prompt.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Prompt exceeds the tokenizer byte limit");
    }
    const int32_t required = llama_tokenize(
        vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (required == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Prompt tokenization overflow");
    }
    if (required >= 0) {
        throw std::runtime_error("Prompt produced no tokens");
    }
    if (-required > model_context_length) {
        throw std::runtime_error("Prompt token count exceeds the trained model context");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-required));
    const int32_t count = llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                                        tokens.data(), static_cast<int32_t>(tokens.size()), true, true);
    if (count != -required) {
        throw std::runtime_error("Prompt tokenization returned inconsistent token counts");
    }
    return tokens;
}

void decode(llama_context * context, Batch & batch, const llama_token * tokens,
            int32_t count, llama_pos first_position, bool output_last) {
    batch.value.n_tokens = count;
    for (int32_t index = 0; index < count; ++index) {
        batch.value.token[index] = tokens[index];
        batch.value.pos[index] = first_position + index;
        batch.value.n_seq_id[index] = 1;
        batch.value.seq_id[index][0] = 0;
        batch.value.logits[index] = output_last && index == count - 1;
    }
    const int32_t result = llama_decode(context, batch.value);
    if (result != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(result) +
                                 "; context shifting and truncation are disabled");
    }
}

llama_token argmax(llama_context * context, int32_t vocabulary_size) {
    const float * logits = llama_get_logits_ith(context, -1);
    if (!logits) {
        throw std::runtime_error("Decode produced no logits");
    }
    llama_token selected = 0;
    for (int32_t token = 0; token < vocabulary_size; ++token) {
        if (std::isnan(logits[token])) {
            throw std::runtime_error("Decode produced NaN logits");
        }
        // Lowest token ID wins a tie, like torch.argmax on the vocabulary axis.
        if (logits[token] > logits[selected]) {
            selected = token;
        }
    }
    if (!std::isfinite(logits[selected])) {
        throw std::runtime_error("Decode produced no finite winning logit");
    }
    return selected;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::array<char, 256> small_buffer{};
    int32_t size = llama_token_to_piece(vocab, token, small_buffer.data(),
                                       static_cast<int32_t>(small_buffer.size()), 0, false);
    if (size >= 0) {
        return std::string(small_buffer.data(), static_cast<size_t>(size));
    }
    if (size == std::numeric_limits<int32_t>::min() || -size > static_cast<int32_t>(max_line_bytes)) {
        throw std::runtime_error("Generated token piece exceeds the text limit");
    }
    std::string piece(static_cast<size_t>(-size), '\0');
    const int32_t actual = llama_token_to_piece(vocab, token, piece.data(), -size, 0, false);
    if (actual != -size) {
        throw std::runtime_error("Generated token piece returned inconsistent byte counts");
    }
    return piece;
}

json evaluate(llama_model * model, llama_token eos, int32_t model_context_length,
              const json & request) {
    if (!request.is_object() || !request.contains("prompt") || !request["prompt"].is_string() ||
        !request.contains("max_new_tokens") || !request.contains("expected_prompt_ids") ||
        !request["expected_prompt_ids"].is_array()) {
        throw std::runtime_error("Request requires prompt, max_new_tokens and expected_prompt_ids");
    }
    const auto & prompt = request["prompt"].get_ref<const std::string &>();
    const int32_t maximum = static_cast<int32_t>(bounded_integer(
        request["max_new_tokens"], 1, model_context_length, "max_new_tokens"));
    const auto * vocab = llama_model_get_vocab(model);
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
    const auto & expected = request["expected_prompt_ids"];
    if (expected.empty() || expected.size() > static_cast<size_t>(model_context_length)) {
        throw std::runtime_error("Expected prompt token count exceeds the model context or is empty");
    }
    std::vector<llama_token> prompt_tokens = tokenize(vocab, prompt, model_context_length);
    if (expected.size() != prompt_tokens.size()) {
        throw std::runtime_error("Prompt tokenizer mismatch: native and HF token counts differ");
    }
    for (size_t index = 0; index < expected.size(); ++index) {
        const int64_t id = bounded_integer(expected[index], 0, vocabulary_size - 1, "expected_prompt_ids");
        if (id != prompt_tokens[index]) {
            throw std::runtime_error("Prompt tokenizer mismatch at index " + std::to_string(index));
        }
    }
    // The last generated token is returned without another decode, so no extra
    // KV slot beyond the shared HF prompt+generation budget is needed.
    const uint64_t required_context = prompt_tokens.size() + static_cast<uint64_t>(maximum);
    if (required_context > static_cast<uint64_t>(model_context_length)) {
        throw std::runtime_error("Prompt plus generation budget exceeds the trained model context; "
                                 "context shifting and truncation are disabled");
    }

    auto parameters = llama_context_default_params();
    parameters.n_ctx = static_cast<uint32_t>(required_context);
    parameters.n_batch = std::min(batch_capacity, static_cast<uint32_t>(prompt_tokens.size()));
    parameters.n_ubatch = parameters.n_batch;
    parameters.n_seq_max = 1;
    parameters.n_outputs_max = 1;
    parameters.n_outputs_max_per_seq = 1;
    parameters.n_threads = cpu_threads;
    parameters.n_threads_batch = cpu_threads;
    parameters.offload_kqv = false;
    parameters.op_offload = false;
    parameters.no_perf = true;
    parameters.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    Context context(llama_init_from_model(model, parameters), llama_free);
    if (!context || llama_n_ctx(context.get()) < required_context) {
        throw std::runtime_error("Unable to allocate the complete requested context");
    }
    Batch batch(static_cast<int32_t>(parameters.n_batch));
    for (size_t position = 0; position < prompt_tokens.size(); position += parameters.n_batch) {
        const int32_t count = static_cast<int32_t>(std::min<size_t>(
            parameters.n_batch, prompt_tokens.size() - position));
        decode(context.get(), batch, prompt_tokens.data() + position, count,
               static_cast<llama_pos>(position), position + count == prompt_tokens.size());
    }

    std::vector<llama_token> generated;
    std::string text;
    std::string finish_reason = "length";
    for (int32_t step = 0; step < maximum; ++step) {
        const llama_token next = argmax(context.get(), vocabulary_size);
        generated.push_back(next);
        if (next == eos) {
            finish_reason = "eos";
            break;
        }
        text += token_piece(vocab, next);
        if (text.size() > max_line_bytes) {
            throw std::runtime_error("Generated text exceeds the response byte limit");
        }
        if (step + 1 < maximum) {
            decode(context.get(), batch, &next, 1,
                   static_cast<llama_pos>(prompt_tokens.size() + step), true);
        }
    }
    return {{"status", "completed"}, {"text", text}, {"generated_token_ids", generated},
            {"prompt_token_ids", prompt_tokens}, {"finish_reason", finish_reason}};
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            respond({{"status", "ready"}, {"protocol", "tunesmith_gguf_eval_v1"},
                     {"revision", TUNESMITH_GGUF_REVISION}});
            return 0;
        }
        std::string model_path;
        llama_token eos = -1;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if ((argument == "--model" || argument == "--eos-token-id") && index + 1 < argc) {
                if (argument == "--model") {
                    if (!model_path.empty()) throw std::runtime_error("Duplicate --model");
                    model_path = argv[++index];
                } else {
                    if (eos >= 0) throw std::runtime_error("Duplicate --eos-token-id");
                    eos = parse_token_id(argv[++index]);
                }
            } else {
                throw std::runtime_error("Usage: tunesmith-gguf-eval --model GGUF_PATH --eos-token-id N");
            }
        }
        if (model_path.empty() || eos < 0) {
            throw std::runtime_error("Both --model and --eos-token-id are required");
        }
        llama_log_set(log_to_stderr, nullptr);
        ggml_log_set(log_to_stderr, nullptr);
        Backend backend;
        auto parameters = llama_model_default_params();
        parameters.n_gpu_layers = 0;
        parameters.split_mode = LLAMA_SPLIT_MODE_NONE;
        parameters.check_tensors = true;
        Model model(llama_model_load_from_file(model_path.c_str(), parameters), llama_model_free);
        if (!model) {
            throw std::runtime_error("Unable to load GGUF model");
        }
        const auto * vocab = llama_model_get_vocab(model.get());
        const int32_t model_context_length = llama_model_n_ctx_train(model.get());
        if (!vocab || llama_vocab_n_tokens(vocab) <= 0 || eos >= llama_vocab_n_tokens(vocab) ||
            model_context_length <= 0 || llama_model_has_encoder(model.get()) ||
            !llama_model_has_decoder(model.get())) {
            throw std::runtime_error("Bridge requires a decoder-only model, valid vocabulary/EOS and trained context");
        }
        respond({{"status", "ready"}, {"model_context_length", model_context_length}});
        std::string line;
        bool oversized = false;
        while (read_request(line, oversized)) {
            try {
                if (oversized) throw std::runtime_error("Request exceeds the JSON line byte limit");
                respond(evaluate(model.get(), eos, model_context_length, json::parse(line)));
            } catch (const std::exception & error) {
                respond({{"status", "error"}, {"error", error.what()}});
            }
        }
        if (std::cin.bad()) throw std::runtime_error("Unable to read stdin");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { respond({{"status", "error"}, {"error", error.what()}}); } catch (...) {}
        return 1;
    }
}
