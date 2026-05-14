"""Full BFCL v4 eval h13_ep1 — все subsets, greedy + liberal parser, no directions.

Subsets:
- simple_python, live_simple, simple_java, simple_javascript
- multiple, live_multiple
- parallel, live_parallel, parallel_multiple, live_parallel_multiple
- irrelevance, live_irrelevance, live_relevance
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, json, re, time
from pathlib import Path
import torch
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, "code")
sys.path.insert(0, ".")

from integrated_gpt2_torch import load_gpt2_torch_weights, encode, decode
from steering_v2 import FullSteeringGPT2, load_classifier_from_npz
from long_context_pi_chunk import interpolate_wpe

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

BFCL_DATA = Path(os.environ.get("BFCL_DATA", "./data/bfcl_v4"))

SUBSETS = [
    "BFCL_v4_simple_python.json",
    "BFCL_v4_live_simple.json",
    "BFCL_v4_simple_java.json",
    "BFCL_v4_simple_javascript.json",
    "BFCL_v4_multiple.json",
    "BFCL_v4_live_multiple.json",
    "BFCL_v4_parallel.json",
    "BFCL_v4_live_parallel.json",
    "BFCL_v4_parallel_multiple.json",
    "BFCL_v4_live_parallel_multiple.json",
    "BFCL_v4_irrelevance.json",
    "BFCL_v4_live_irrelevance.json",
    "BFCL_v4_live_relevance.json",
]


def build_prompt(content, function_specs):
    if isinstance(function_specs, dict):
        function_specs = [function_specs]
    fn_json = json.dumps(function_specs[0] if function_specs else {}, indent=2)[:600]
    return (
        f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        f"{fn_json}\n\n\n"
        f"USER: {content}\n\n\n"
        f"ASSISTANT: <functioncall> "
    )


def parse_call(text):
    text = text.strip()
    name_match = re.search(r'["\'`]?name["\'`]?\s*:\s*["\']([^"\'(\s,]+)', text)
    if name_match:
        return name_match.group(1)
    return None


def gen_greedy(model, prompt, max_new=40, max_input=1500):
    all_ids = encode(prompt)
    ids = all_ids[-max_input:] if len(all_ids) > max_input else list(all_ids)
    L = len(ids)
    for _ in range(max_new):
        if len(ids) >= model.gpt.wpe.weight.shape[0]: break
        ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        mm = torch.zeros((1, len(ids)), dtype=torch.float32, device=DEVICE)
        mm[:, :L] = 1.0
        with torch.no_grad():
            logits, _, _ = model(ii, mm)
        nxt = int(logits[0, -1, :].argmax().item())
        ids.append(nxt)
        if nxt == encode("}")[0] or nxt == encode("\n")[0]: break
    return decode(ids[L:]).strip()


def eval_subset(model, fname, max_n=40):
    questions_path = BFCL_DATA / fname
    answers_path = BFCL_DATA / "possible_answer" / fname
    if not questions_path.exists():
        return None

    has_answers = answers_path.exists()
    is_irrelevance = "irrelevance" in fname or "relevance" in fname

    with open(questions_path, encoding='utf-8') as f:
        questions = [json.loads(l) for l in f if l.strip()][:max_n]

    answers = {}
    if has_answers:
        with open(answers_path, encoding='utf-8') as f:
            answers = {json.loads(l)["id"]: json.loads(l).get("ground_truth", []) for l in f if l.strip()}

    n = 0; correct = 0
    for q in questions:
        qid = q["id"]
        qq = q["question"][0] if isinstance(q["question"], list) and q["question"] else q["question"]
        content = qq[0].get("content", "") if isinstance(qq, list) and qq else str(qq)
        if not content: continue
        func_specs = q.get("function", [])
        if isinstance(func_specs, dict): func_specs = [func_specs]
        if not func_specs: continue

        prompt = build_prompt(content, func_specs)
        gen = gen_greedy(model, prompt, max_new=40)
        pred_name = parse_call(gen)

        if is_irrelevance:
            # For irrelevance: NO function should be called → pred should be empty/None
            ok = (pred_name is None) or (pred_name not in [f.get("name", "") for f in func_specs])
        elif has_answers and qid in answers:
            gold = answers[qid]
            if isinstance(gold, list) and gold:
                # Check if pred matches ANY of gold names (для parallel — несколько gold)
                gold_names = []
                for g_dict in gold:
                    if isinstance(g_dict, dict):
                        gold_names.extend(g_dict.keys())
                ok = pred_name in gold_names if pred_name else False
            else:
                ok = False
        else:
            # No gold — skip
            continue

        if ok: correct += 1
        n += 1
    return {"n": n, "acc": correct/max(n,1)}


def main():
    print("[Full BFCL v4 eval — h13_ep1, greedy, liberal parser]")
    base = FullSteeringGPT2(adapter_layer=6, alpha=1.0)
    load_gpt2_torch_weights(base.gpt)
    load_classifier_from_npz(base, str(Path(__file__).resolve().parent.parent / "weights" / "adapter_torch_EN_BFCL.npz"))
    base.adapter.load_state_dict(torch.load(str(Path(__file__).resolve().parent.parent / "weights" / "adapter_h13_bfcl_ep1.pt"), map_location='cpu'))
    interpolate_wpe(base.gpt, 2048)
    base.freeze_gpt(); base.to(DEVICE); base.eval()

    rows = []
    for sub in SUBSETS:
        t0 = time.time()
        r = eval_subset(base, sub, max_n=40)
        if r is None:
            print(f"\n  [{sub}] SKIP (file missing)")
            continue
        elapsed = time.time() - t0
        rows.append({"subset": sub.replace("BFCL_v4_", "").replace(".json", ""), **r})
        print(f"  [{sub:<45}] n={r['n']}  acc={r['acc']*100:>5.1f}%  ({elapsed:.0f}s)")

    print(f"\n{'='*70}\n  FULL BFCL v4 SUMMARY (h13_ep1)\n{'='*70}")
    total_n = sum(r["n"] for r in rows)
    total_correct = sum(r["n"] * r["acc"] for r in rows)
    print(f"  {'subset':<35} {'n':>4} {'acc':>8}")
    for r in rows:
        print(f"  {r['subset']:<35} {r['n']:>4} {r['acc']*100:>7.1f}%")
    print(f"  {'OVERALL':<35} {total_n:>4} {total_correct/max(total_n,1)*100:>7.1f}%")

    Path("bfcl_v4_full_results.json").write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f"\nSaved → bfcl_v4_full_results.json")


if __name__ == "__main__":
    main()
