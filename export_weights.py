#!/usr/bin/env python3
"""CPU FP32 export. Linear weights are contiguous [out, in], never transposed.

Order: tok_emb, pos_emb; blocks 0..5 (ln1, qkv, proj, ln2, fc1, fc2;
weight then bias); ln_f weight/bias; untied lm_head weight.
Offsets in the manifest are bytes. Split offsets are relative to that split.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

D, H, FF, L, S, V = 384, 6, 1536, 6, 512, 256


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, eps=1e-5)
        self.qkv = nn.Linear(D, 3 * D, bias=True)
        self.proj = nn.Linear(D, D, bias=True)
        self.ln2 = nn.LayerNorm(D, eps=1e-5)
        self.fc1 = nn.Linear(D, FF, bias=True)
        self.fc2 = nn.Linear(FF, D, bias=True)

    def forward(self, x, cache):
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = [a.reshape(H, D // H) for a in (q, k, v)]
        cache.append((k, v))
        keys = torch.stack([a[0] for a in cache], dim=1)
        vals = torch.stack([a[1] for a in cache], dim=1)
        # Incremental causal attention: only positions 0..current are in cache.
        scores = (keys * q[:, None, :]).sum(-1) * (1 / 8)
        z = (torch.softmax(scores, -1)[:, :, None] * vals).sum(1)
        x = x + self.proj(z.reshape(D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="none"))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, D)
        self.pos_emb = nn.Embedding(S, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.ln_f = nn.LayerNorm(D, eps=1e-5)
        self.lm_head = nn.Linear(D, V, bias=False)

    def step(self, token, pos, caches, trace=False):
        x = self.tok_emb.weight[token] + self.pos_emb.weight[pos]
        first = None
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i])
            if i == 0 and trace:
                first = x.tolist()
        return self.lm_head(self.ln_f(x)), first


def fnv(data):
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return f"{h:08x}"


@torch.inference_mode()
def generate(model, prompt, count, trace=False):
    if not prompt or count < 1 or len(prompt) + count - 1 > S:
        raise ValueError("Require nonempty UTF-8 prompt and 1 <= prompt bytes + N - 1 <= 512")
    caches = [[] for _ in range(L)]
    first = None
    for pos, b in enumerate(prompt):
        logits, a = model.step(b, pos, caches, trace and pos == 0)
        if a is not None:
            first = a
    out, margins = [], []
    for i in range(count):
        top = torch.topk(logits, 2).values
        margins.append(float(top[0] - top[1]))
        b = int(torch.argmax(logits))
        out.append(b)
        if i + 1 < count:
            logits, _ = model.step(b, len(prompt) + i, caches)
    return {"prompt": list(prompt), "bytes": out, "fnv1a": fnv(out),
            "first_layer": first, "min_argmax_margin": min(margins)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=".")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--n", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)  # BEFORE construction; no custom initialization.
    model = GPT().cpu().float().eval()
    total = sum(t.numel() for t in model.parameters())
    assert total == 11_040_768, total
    assert model.lm_head.weight is not model.tok_emb.weight
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    blobs = {k: bytearray() for k in ("weights.bin", "part_a.bin", "part_b.bin")}
    tensors = []
    for name, t in model.named_parameters():
        a = name.startswith(("tok_emb.", "pos_emb.", "blocks.0.", "blocks.1.", "blocks.2."))
        part = "part_a.bin" if a else "part_b.bin"
        values = array("f", t.detach().contiguous().reshape(-1).tolist())
        assert values.itemsize == 4
        if sys.byteorder != "little":
            values.byteswap()
        raw = values.tobytes()
        tensors.append({"name": name, "shape": list(t.shape), "offset": len(blobs["weights.bin"]),
                        "part": part, "part_offset": len(blobs[part]), "nbytes": len(raw)})
        blobs["weights.bin"].extend(raw)
        blobs[part].extend(raw)
    files = {}
    for name, raw in blobs.items():
        (root / name).write_bytes(raw)
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {"format": 1, "dtype": "float32-le", "linear_layout": "out,in",
                "params": total, "model_id": files["weights.bin"]["sha256"],
                "torch_version": torch.__version__, "files": files, "tensors": tensors}
    (root / "weights_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result = generate(model, args.prompt.encode("utf-8"), args.n, True)
    result["model_id"] = manifest["model_id"]
    result["benchmark"] = generate(model, b"Hello", 128)
    (root / "reference.json").write_text(json.dumps(result), encoding="utf-8")
    print("Parameters:", total)
    print("Files:", json.dumps(files, indent=2))
    print("Generated bytes:", result["bytes"])
    print("Generated hex:", bytes(result["bytes"]).hex())
    print("FNV-1a (generated bytes only):", result["fnv1a"])
    print("UTF-8 display (escaped):", ascii(bytes(result["bytes"]).decode("utf-8", errors="replace")))


if __name__ == "__main__":
    main()
