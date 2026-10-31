"""GRPO training of a theory writer on planted worlds, on Modal GPUs.

The writer (Qwen3-4B + LoRA) reads a planted world's lab notebook and writes a report: up to three hypotheses with
credences. GRPO optimizes one of three rewards, computed on *reward questions* about new crystals of the same world:

* ``latent``: S_tau (tau = 0.9) of the report on sampled outcomes, per question, through a frozen reader;
* ``judge``: a frozen judge's 1-10 rating of the report given the notebook (rescaled to [-1, 0]);
* ``oracle``: the expected log score of the report's predictions under the world's true conditionals (uses the
  truth, so it is the upper bound of what a reward can tell the writer).

Reader and judge are the same frozen model (Qwen3-8B) served by vLLM in a separate GPU container. Training worlds are
8..8+N; worlds 0..7 are held out. After training, the trained writer samples reports for the held-out worlds, which
are returned for evaluation with API readers (experiments/arena_rl_eval.py).

    modal run lvpm/rl/grpo_modal.py --reward latent --steps 150
"""

from __future__ import annotations

import json
import math
import re

import modal

APP = "lvpm-essay-grpo"
POLICY = "Qwen/Qwen3-4B"
READER = "Qwen/Qwen3-8B"
# Qwen3's thinking mode: without it the 8B reader barely separates the true planted theory from a wrong one
# (true -0.626 vs swapped -0.645 nats/q against Bayes -0.536 on 40 questions)
READER_THINKING = True
READER_BUDGET = 10000 if READER_THINKING else 2500
READER_GPUS = 4  # reader containers the reward fans out to (a thinking reading takes ~0.6 s of one H100)

image = (
    # a CUDA *devel* image: vLLM's FlashInfer sampler compiles kernels at start-up and needs nvcc
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("vllm==0.32.0", "trl==1.17.1", "peft==0.21.2", "transformers>=5.16.1,<5.20", "datasets", "accelerate",
                 "numpy", "pydantic", "pyyaml", "scipy")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("cvtestbed", "lvpm")
)
cache = modal.Volume.from_name("lvpm-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("lvpm-grpo-runs", create_if_missing=True)
app = modal.App(APP, image=image)

READER_SYSTEM = ("You answer questions by applying a theory someone else wrote about the domain. Assume the theory is "
                 "correct and answer as it implies. Where it does not settle the question, use your best judgment in "
                 "the spirit of the theory.")


def parse_report(text: str):
    """(texts, weights) from a report's JSON, or None."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    for cand in reversed(re.findall(r"\{.*\}", text, flags=re.S)):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        hyps = []
        for h in obj.get("hypotheses", []) if isinstance(obj, dict) else []:
            try:
                hyps.append((str(h["text"]).strip(), max(0.0, float(h.get("weight", 1.0)))))
            except (KeyError, TypeError, ValueError):
                continue
        hyps = [h for h in hyps if h[0]][:4]
        if hyps and sum(w for _, w in hyps) > 0:
            tot = sum(w for _, w in hyps)
            return [h for h, _ in hyps], [w / tot for _, w in hyps]
    return None


@app.cls(gpu="H100", volumes={"/cache": cache}, timeout=60 * 60 * 24, scaledown_window=600,
         max_containers=READER_GPUS)
class Reader:
    @modal.enter()
    def load(self):
        from vllm import LLM

        self.llm = LLM(READER, max_model_len=12288, gpu_memory_utilization=0.9, enable_prefix_caching=True)

    def _chat(self, convs, max_tokens, thinking=False):
        from vllm import SamplingParams

        # Qwen3 recommends sampling (not greedy) in thinking mode; greedy thinking can loop
        sp = (SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=0) if thinking else
              SamplingParams(temperature=0.0, max_tokens=max_tokens))
        outs = self.llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": thinking}, use_tqdm=False)
        cut = sum(o.outputs[0].finish_reason == "length" for o in outs)
        print(f"reader: {len(outs)} generations, {cut / max(1, len(outs)):.3%} truncated", flush=True)
        return [o.outputs[0].text for o in outs]

    @modal.method()
    def probs(self, items: list[tuple[str, str]]) -> list[float]:
        """P(yes) for each (hypothesis, question): brief reasoning, then a stated probability; both answer orders."""
        convs = []
        for hyp, q in items:
            for yes_first in (True, False):
                opts = ("(A) Yes\n(B) No" if yes_first else "(A) No\n(B) Yes")
                convs.append([{"role": "system", "content": READER_SYSTEM},
                              {"role": "user", "content": f"THEORY:\n{hyp}\n\nQUESTION:\n{q}\n\n{opts}\n\nWork it out "
                               "step by step, briefly, according to the theory. Then end with a line 'P(A) = x' giving "
                               "the probability that answer A is correct."}])
        texts = self._chat(convs, READER_BUDGET, READER_THINKING)  # truncated readings guess: keep the budget ample
        texts = [t.split("</think>")[-1] for t in texts]
        out = []
        for i in range(0, len(texts), 2):
            ps = []
            for j, yes_first in ((i, True), (i + 1, False)):
                m = re.findall(r"P\(A\)\s*=\s*([0-9]*\.?[0-9]+)", texts[j])
                pa = min(1.0, max(0.0, float(m[-1]))) if m else 0.5
                ps.append(pa if yes_first else 1 - pa)
            out.append(min(0.99, max(0.01, sum(ps) / 2)))
        return out

    @modal.method()
    def ratings(self, items: list[tuple[str, str]]) -> list[float]:
        """1-10 ratings of (notebook, report) pairs."""
        convs = [[{"role": "user", "content": f"LAB NOTEBOOK:\n{nb}\n\nREPORT:\n{rep}\n\nRate how good this report is "
                   "as a scientific account of the world: correct, insightful, consistent with the notebook, useful "
                   "for predicting new crystals. End with a line 'Score: N' (1-10)."}] for nb, rep in items]
        texts = self._chat(convs, 500)
        out = []
        for t in texts:
            m = re.findall(r"score\W{0,3}(\d+(?:\.\d+)?)", t, re.I)
            out.append(min(10.0, max(1.0, float(m[-1]))) if m else 5.0)
        return out


def make_reward(kind: str, n_questions: int, tau: float = 0.9):
    from lvpm.rl import worlds as W

    reader = Reader()

    def reward(prompts, completions, world, **kw):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        reps = [parse_report(t) for t in texts]
        if kind == "judge":
            items = [(W.P.notebook_text(k), "\n\n".join(f"Hypothesis {i + 1} (credence {w:.2f}): {h}" for i, (h, w)
                                                          in enumerate(zip(*r)))) for k, r in zip(world, reps) if r]
            rs = iter(reader.ratings.remote(items))
            return [(next(rs) - 10) / 9 if r else -1.0 for r in reps]
        items, index = [], []
        qs_by_world = {k: W.questions(k, n_questions) for k in set(world)}
        for ci, (k, r) in enumerate(zip(world, reps)):
            if not r:
                continue
            for hi, h in enumerate(r[0]):
                for qi, q in enumerate(qs_by_world[k]):
                    items.append((h, q["text"]))
                    index.append((ci, hi, qi))
        size = max(1, -(-len(items) // READER_GPUS))
        chunks = [items[i:i + size] for i in range(0, len(items), size)]
        ps = [p for part in reader.probs.map(chunks) for p in part] if items else []
        table = {}
        for (ci, hi, qi), p in zip(index, ps):
            table[(ci, hi, qi)] = p
        out = []
        for ci, (k, r) in enumerate(zip(world, reps)):
            if not r:
                out.append(-1.5)
                continue
            texts_, ws = r
            qs = qs_by_world[k]
            if kind == "oracle":  # expected log score of the mixture's marginal under the true conditional
                tot = 0.0
                for qi, q in enumerate(qs):
                    m = sum(w * table[(ci, hi, qi)] for hi, w in enumerate(ws))
                    m = min(0.999, max(0.001, m))
                    tot += q["p_yes"] * math.log(m) + (1 - q["p_yes"]) * math.log(1 - m)
                out.append(tot / len(qs))
            else:  # latent: S_tau on the sampled outcomes
                logs = []
                for hi, w in enumerate(ws):
                    s = sum(math.log(table[(ci, hi, qi)] if q["yes"] else 1 - table[(ci, hi, qi)])
                            for qi, q in enumerate(qs))
                    logs.append(math.log(max(w, 1e-9)) + tau * s)
                mx = max(logs)
                out.append((mx + math.log(sum(math.exp(v - mx) for v in logs))) / len(qs))
        return out

    reward.__name__ = f"reward_{kind}"
    return reward


@app.function(gpu="H100", volumes={"/cache": cache, "/runs": runs}, timeout=60 * 60 * 24)
def train(reward: str = "latent", steps: int = 150, n_worlds: int = 256, n_questions: int = 32, seed: int = 0,
          lr: float = 1e-5, num_generations: int = 8, prompts_per_step: int = 16) -> str:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from lvpm.rl import worlds as W

    ds = Dataset.from_list([{"prompt": W.prompt(k), "world": k} for k in range(8, 8 + n_worlds)])
    out = f"/runs/{reward}-s{seed}"
    cfg = GRPOConfig(
        output_dir=out, learning_rate=lr, max_steps=steps, num_generations=num_generations,
        per_device_train_batch_size=num_generations * prompts_per_step // 4, gradient_accumulation_steps=4,
        max_completion_length=1024, temperature=1.0, beta=0.0, loss_type="dapo", logging_steps=1, save_steps=50,
        bf16=True, use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.35,
        vllm_max_model_length=6144, chat_template_kwargs={"enable_thinking": False}, report_to="none", seed=seed,
        log_completions=False)
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=POLICY, reward_funcs=make_reward(reward, n_questions), args=cfg, train_dataset=ds,
                          peft_config=lora)
    trainer.train()
    trainer.save_model(out + "/final")
    with open(out + "/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    runs.commit()
    return out


@app.function(gpu="H100", volumes={"/cache": cache, "/runs": runs}, timeout=60 * 60 * 4)
def sample_reports(adapter: str | None, worlds: list[int], n: int = 16, seed: int = 0) -> list[dict]:
    """Reports from the base policy (adapter None) or a trained LoRA adapter, for the given worlds."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from lvpm.rl import worlds as W

    llm = LLM(POLICY, enable_lora=adapter is not None, max_lora_rank=32, max_model_len=6144, seed=seed)
    sp = SamplingParams(temperature=1.0, max_tokens=1024, n=n, seed=seed)
    lora = LoRARequest("trained", 1, adapter) if adapter else None
    outs = llm.chat([W.prompt(k) for k in worlds], sp, lora_request=lora,
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    rows = []
    for k, o in zip(worlds, outs):
        for i, c in enumerate(o.outputs):
            r = parse_report(c.text)
            rows.append({"world": k, "i": i, "raw": c.text, "texts": r[0] if r else None, "weights": r[1] if r else None})
    return rows


@app.function(volumes={"/runs": runs})
def read_log(path: str) -> list:
    with open(path + "/log_history.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main(reward: str = "latent", steps: int = 150, n_worlds: int = 256, seed: int = 0, sample_only: bool = False,
         n_questions: int = 64, prompts_per_step: int = 16, num_generations: int = 8, reader_check: bool = False):
    import pathlib
    import time

    out_dir = pathlib.Path("results/arena-rl")
    out_dir.mkdir(parents=True, exist_ok=True)
    if reader_check:  # the frozen reader on the true and empty theories of held-out world 0, vs the Bayes optimum
        from lvpm.essays.theories import variants
        from lvpm.rl import worlds as W

        qs = W.questions(0, 80, seed="reader-check")
        t0 = time.time()
        rows = {}
        for name in ("true", "qualitative", "merged", "swapped", "vague"):
            ps = Reader().probs.remote([(variants(0)[name], q["text"]) for q in qs])
            rows[name] = sum(q["p_yes"] * math.log(p) + (1 - q["p_yes"]) * math.log(1 - p) for q, p in zip(qs, ps)) / len(qs)
        bayes = sum(q["p_yes"] * math.log(q["p_yes"]) + (1 - q["p_yes"]) * math.log(1 - q["p_yes"]) for q in qs) / len(qs)
        print({k: round(v, 3) for k, v in rows.items()}, "bayes", round(bayes, 3), f"{time.time() - t0:.0f}s")
        return
    if sample_only:
        rows = sample_reports.remote(None, list(range(8)))
        (out_dir / "reports-base.json").write_text(json.dumps(rows))
        return
    path = train.remote(reward=reward, steps=steps, n_worlds=n_worlds, seed=seed, n_questions=n_questions,
                        prompts_per_step=prompts_per_step, num_generations=num_generations)
    (out_dir / f"log-{reward}-s{seed}.json").write_text(json.dumps(read_log.remote(path)))
    rows = sample_reports.remote(path + "/final", list(range(8)), seed=seed)
    (out_dir / f"reports-{reward}-s{seed}.json").write_text(json.dumps(rows))
