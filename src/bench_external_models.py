"""Run external models (via ollama) on a slice of our fresh bench.

Usage: ollama must be running with target model pulled.
    python bench_external_models.py --model qwen3:4b --n 20

Produces results to compare with our FT GPT-2 92%.
"""
import os, sys, json, re, time, subprocess
from pathlib import Path
import argparse

FRESH = Path(__file__).resolve().parent.parent / "bench"


def parse_name(text):
    if not text: return None
    text = text.strip()
    # try JSON
    m = re.search(r'["\'`]?name["\'`]?\s*:\s*["\']([^"\'(\s,]+)', text)
    if m: return m.group(1)
    # try `function_call(args)` style
    m = re.match(r'^([\w\.]+)\s*\(', text)
    if m: return m.group(1)
    # try tool_call style
    m = re.search(r'tool_call[^a-zA-Z]+([\w_]+)', text)
    if m: return m.group(1)
    return None


def build_prompt(item):
    fn_json = json.dumps(item["function"], indent=2)
    return (
        f"You have access to one function:\n{fn_json}\n\n"
        f"User request: {item['prompt']}\n\n"
        f"Output ONLY the function call as JSON: {{\"name\": \"<function_name>\", \"arguments\": {{...}}}}"
    )


def ollama_call(model, prompt, timeout=120):
    try:
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=timeout,
            encoding='utf-8', errors='replace',
        )
        return (proc.stdout or "").strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"ERR:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ollama model id e.g. qwen3:4b")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--file", default="fresh_bench_opus.json")
    args = ap.parse_args()

    items = json.load(open(FRESH / args.file))[:args.n]
    print(f"[{args.model}] {len(items)} items from {args.file}")
    correct = 0
    t_all = time.time()
    for i, it in enumerate(items):
        prompt = build_prompt(it)
        t0 = time.time()
        out = ollama_call(args.model, prompt)
        pred = parse_name(out)
        ok = (pred == it["gold_name"])
        if ok: correct += 1
        print(f"  [{i+1}/{len(items)}] gold={it['gold_name']:<35} pred={str(pred):<35} ok={ok}  ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / len(items)
    print(f"\n[{args.model}] acc={acc*100:.1f}% ({correct}/{len(items)})  total={time.time()-t_all:.0f}s")
    out = {"model": args.model, "n": len(items), "correct": correct, "acc": acc, "total_s": time.time()-t_all}
    Path(f"../results/external_{args.model.replace(':','_')}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
