// llama-greedy — llama.cpp's greedy walk on the reference model, as the
// reference run for cajeta's GreedyAgreement (xpu-kernel-adaptor plan
// 8.1.3, spec §7.1 "correctness first").
//
// Prints the prompt's token ids (llama.cpp's tokenizer, BOS added), then
// one line per generated position with the greedy token and the top-2
// logit gap at that position — the tie clause GreedyAgreement weighs: a
// divergence at a gap under 1e-3 is a tie broken differently, not a bug.
//
//   llama-greedy --model M.gguf --prompt "..." --n 32 [--ngl 99] [--ctx 512]
//
//   prompt-ids <n> <id,id,...>
//   greedy <pos> <id> <top2gap>
//
// Deterministic by construction: no sampler, argmax over the raw logits.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model_path, prompt = "The measured baseline for the parity gate is";
    int n = 32, ngl = 99, n_ctx = 512;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](void) -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model_path = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--n") n = atoi(next());
        else if (a == "--ngl") ngl = atoi(next());
        else if (a == "--ctx") n_ctx = atoi(next());
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (model_path.empty()) { fprintf(stderr, "--model is required\n"); return 2; }

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed: %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    std::vector<llama_token> toks(prompt.size() + 16);
    int nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
    if (nt < 0) {
        toks.resize(-nt);
        nt = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
    }
    toks.resize(nt);
    printf("prompt-ids\t%d\t", nt);
    for (int i = 0; i < nt; i++) printf("%s%d", i ? "," : "", toks[i]);
    printf("\n");

    llama_batch batch = llama_batch_get_one(toks.data(), nt);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }
    const int nv = llama_vocab_n_tokens(vocab);
    llama_token cur = 0;
    for (int pos = 0; pos < n; pos++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        int best = 0, second = -1;
        for (int t = 1; t < nv; t++) {
            if (logits[t] > logits[best]) { second = best; best = t; }
            else if (second < 0 || logits[t] > logits[second]) { second = t; }
        }
        float gap = second >= 0 ? logits[best] - logits[second] : 0.0f;
        printf("greedy\t%d\t%d\t%.6g\n", pos, best, gap);
        if (llama_vocab_is_eog(vocab, best)) break;
        cur = best;
        batch = llama_batch_get_one(&cur, 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed at %d\n", pos); return 1; }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
