"""Bench full-FT GPT-2 124M on 4 open tool-call benches.

No adapter — just FT'd GPT-2.
Compare against h13_ep1 numbers (held in head).
"""
import os, sys, json, re, time, glob
from pathlib import Path
import torch
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
sys.path.insert(0, ".")

from integrated_gpt2_torch import GPT2, encode, decode

DEVICE = torch.device('cpu')
torch.set_num_threads(4)

BFCL_DATA = Path(os.environ.get("BFCL_DATA", "./data/bfcl_v4"))
GLAIVE_FILE = Path(os.environ.get("GLAIVE_FILE", "./data/glaive/glaive-function-calling-v2.json"))
XLAM_FILE = os.environ.get("XLAM_FILE", "./data/xlam/train.parquet")
API_BANK_FILE = Path(os.environ.get("API_BANK_FILE", "./data/api_bank/level-1-api.json"))

FT_WEIGHTS = Path(str(Path(__file__).resolve().parent.parent / "weights" / "gpt2_ft_final.pt"))
N_PER_BENCH = 50

# BFCL v4 — все 13 subsets, чтобы получить OVERALL (как у h13)
BFCL_SUBSETS = [
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


def parse_name(text):
    if not text: return None
    text = text.strip()
    m = re.search(r'["\'`]?name["\'`]?\s*:\s*["\']([^"\'(\s,]+)', text)
    if m: return m.group(1)
    m = re.search(r'\[?\s*([A-Za-z]\w*(?:\.\w+)*)\s*\(', text)
    if m: return m.group(1)
    return None


def build_prompt(system_text, user_text):
    return (
        f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n"
        f"{system_text[:700]}\n\n\n"
        f"USER: {user_text[:400]}\n\n\n"
        f"ASSISTANT: <functioncall> "
    )


@torch.no_grad()
def gen_greedy(model, prompt, max_new=40):
    ids = encode(prompt)
    ids = ids[-512:] if len(ids) > 512 else list(ids)
    L = len(ids)
    for _ in range(max_new):
        if len(ids) >= 1024: break
        ii = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        logits, _ = model(ii)
        nxt = int(logits[0, -1, :].argmax().item())
        ids.append(nxt)
        if nxt == encode("}")[0] or nxt == encode("\n")[0] or nxt == encode(")")[0]: break
    return decode(ids[L:]).strip()


def eval_bfcl_full(model, max_n=40):
    print("\n=== BFCL v4 (13 subsets, n<=40 each) ===")
    rows = []
    total_n = 0; total_c = 0
    for fname in BFCL_SUBSETS:
        q_path = BFCL_DATA / fname
        a_path = BFCL_DATA / "possible_answer" / fname
        if not q_path.exists(): continue
        is_irrel = "irrelevance" in fname or "relevance" in fname
        with open(q_path, encoding='utf-8') as f:
            questions = [json.loads(l) for l in f if l.strip()][:max_n]
        answers = {}
        if a_path.exists():
            with open(a_path, encoding='utf-8') as f:
                answers = {json.loads(l)["id"]: json.loads(l).get("ground_truth", []) for l in f if l.strip()}
        n = 0; correct = 0
        t0 = time.time()
        for q in questions:
            qid = q["id"]
            qq = q["question"][0] if isinstance(q["question"], list) and q["question"] else q["question"]
            content = qq[0].get("content", "") if isinstance(qq, list) and qq else str(qq)
            if not content: continue
            func_specs = q.get("function", [])
            if isinstance(func_specs, dict): func_specs = [func_specs]
            if not func_specs: continue
            fn_json = json.dumps(func_specs[0] if func_specs else {}, indent=2)[:500]
            prompt = build_prompt(fn_json, content)
            gen = gen_greedy(model, prompt)
            pred = parse_name(gen)
            if is_irrel:
                ok = (pred is None) or (pred not in [f.get("name", "") for f in func_specs])
            elif qid in answers:
                gold = answers[qid]
                gold_names = []
                for g in gold:
                    if isinstance(g, dict): gold_names.extend(g.keys())
                ok = pred in gold_names if pred else False
            else:
                continue
            if ok: correct += 1
            n += 1
        rows.append((fname.replace("BFCL_v4_","").replace(".json",""), n, correct))
        total_n += n; total_c += correct
        print(f"  [{fname:<40}] n={n}  acc={correct/max(n,1)*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  BFCL OVERALL: {total_c/max(total_n,1)*100:.1f}%  ({total_c}/{total_n})")
    return rows, total_c/max(total_n,1)


def eval_glaive(model):
    print("\n=== Glaive held-out (skip first 1000 anchor) ===")
    with open(GLAIVE_FILE, encoding='utf-8') as f:
        glaive = json.load(f)
    correct = 0; total = 0
    t0 = time.time()
    for ex in glaive[1000:]:
        if total >= N_PER_BENCH: break
        st = ex.get("system", "")
        chat = ex.get("chat", "")
        um = re.search(r"USER:\s*(.+?)(?:\n\n\nASSISTANT:|$)", chat, re.DOTALL)
        fm = re.search(r"<functioncall>\s*(\{.+?\})\s*<\|endoftext\|>", chat, re.DOTALL)
        if not (um and fm and st): continue
        nm = re.search(r'"name"\s*:\s*"([^"]+)"', fm.group(1))
        if not nm: continue
        gold = nm.group(1)
        prompt = st.strip() + f"\n\nUSER: {um.group(1).strip()[:300]}\n\n\nASSISTANT: <functioncall> "
        gen = gen_greedy(model, prompt)
        pred = parse_name(gen)
        if pred == gold: correct += 1
        total += 1
        if total % 10 == 0:
            print(f"  {total}/{N_PER_BENCH}  acc={correct/total*100:.1f}%  t={time.time()-t0:.0f}s", flush=True)
    print(f"  Glaive: {correct/max(total,1)*100:.1f}%  ({correct}/{total})")
    return correct/max(total,1), total


def eval_xlam(model):
    import pandas as pd
    print("\n=== xLAM tail held-out ===")
    df = pd.read_parquet(XLAM_FILE)
    correct = 0; total = 0
    t0 = time.time()
    for i in range(len(df)-200, len(df)):
        if total >= N_PER_BENCH: break
        msgs = df.iloc[i]['messages']
        sys_t = ""; user_t = ""; gold_name = ""
        for m in msgs:
            if m['role'] == 'system': sys_t = m['content']
            elif m['role'] == 'user': user_t = m['content']
            elif m['role'] == 'assistant':
                mm = re.search(r"<tool_call>\s*\{['\"]tool_name['\"]\s*:\s*['\"]([^'\"]+)['\"]", m['content'])
                if mm: gold_name = mm.group(1)
                break
        if not gold_name or not user_t: continue
        prompt = build_prompt(sys_t, user_t)
        gen = gen_greedy(model, prompt)
        pred = parse_name(gen)
        if pred == gold_name: correct += 1
        total += 1
        if total % 10 == 0:
            print(f"  {total}/{N_PER_BENCH}  acc={correct/total*100:.1f}%  t={time.time()-t0:.0f}s", flush=True)
    print(f"  xLAM: {correct/max(total,1)*100:.1f}%  ({correct}/{total})")
    return correct/max(total,1), total


def eval_api_bank(model):
    print("\n=== API-Bank Level-1 ===")
    with open(API_BANK_FILE, encoding='utf-8') as f:
        items = json.load(f)
    correct = 0; total = 0
    t0 = time.time()
    for item in items[:N_PER_BENCH]:
        instr = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("expected_output", "")
        m = re.search(r'\[([A-Z]\w+)\s*\(', out)
        if not m: continue
        gold = m.group(1)
        prompt = build_prompt(instr, inp)
        gen = gen_greedy(model, prompt)
        pred = parse_name(gen)
        if pred == gold: correct += 1
        total += 1
        if total % 10 == 0:
            print(f"  {total}/{N_PER_BENCH}  acc={correct/total*100:.1f}%  t={time.time()-t0:.0f}s", flush=True)
    print(f"  API-Bank: {correct/max(total,1)*100:.1f}%  ({correct}/{total})")
    return correct/max(total,1), total


def main():
    print("[Full-FT GPT-2 124M — BENCH BATTERY]")
    print(f"Loading FT weights: {FT_WEIGHTS}")
    model = GPT2()
    sd = torch.load(str(FT_WEIGHTS), map_location='cpu')
    model.load_state_dict(sd)
    model.to(DEVICE); model.eval()
    print("  loaded.\n")

    results = {}
    bfcl_rows, bfcl_overall = eval_bfcl_full(model)
    results["BFCL_v4_OVERALL"] = bfcl_overall
    results["BFCL_subsets"] = bfcl_rows
    results["Glaive"], _ = eval_glaive(model)
    results["xLAM"], _ = eval_xlam(model)
    results["API_Bank"], _ = eval_api_bank(model)

    print(f"\n{'='*70}\n  Full-FT GPT-2 SUMMARY  vs  h13_ep1 (adapter)\n{'='*70}")
    print(f"  bench                    full-FT     h13_ep1")
    print(f"  BFCL v4 OVERALL          {bfcl_overall*100:>6.1f}%     50.0%")
    print(f"  Glaive held-out          {results['Glaive']*100:>6.1f}%     54.0%")
    print(f"  xLAM tail held-out       {results['xLAM']*100:>6.1f}%     34.0%")
    print(f"  API-Bank Level-1         {results['API_Bank']*100:>6.1f}%      4.0%")

    Path("ft_gpt2_bench_results.json").write_text(json.dumps({
        "BFCL_OVERALL": bfcl_overall, "BFCL_subsets": bfcl_rows,
        "Glaive": results["Glaive"], "xLAM": results["xLAM"], "API_Bank": results["API_Bank"],
    }, indent=2), encoding='utf-8')
    print(f"\nSaved -> ft_gpt2_bench_results.json")


if __name__ == "__main__":
    main()
