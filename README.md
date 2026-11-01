# HiveGPT

Static, CPU-only distributed inference. Normal Host/Join mode never downloads or
executes the full model. A owns embeddings plus blocks 0–2; B owns blocks 3–5,
final LayerNorm and the untied head. This is a randomly initialized execution
demo, not a trained chatbot.

## Export

Use Python 3.10+ in a fresh virtual environment:

```sh
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`; on macOS/Linux
with `source .venv/bin/activate`. Then:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python export_weights.py --prompt "Hello" --n 32
```

No NumPy dependency. Export produces weights.bin (44,163,072 bytes), part_a.bin
(22,473,216 bytes), part_b.bin (21,689,856 bytes), weights_manifest.json and
reference.json. Every file is headerless little-endian FP32. Tensor registration
order is the export order; each Linear matrix is [out,in], so JS computes each
output from one contiguous weight row. Manifest offsets describe both whole and
split files. Never concatenate tensors alphabetically.

Parameter proof: embeddings 294,912 + 6 blocks × 1,774,464 + final LN 768 +
untied head 98,304 = 11,040,768. The exporter asserts this actual module count.

Keep exactly these web assets together: index.html, app.js, worker.js, qr.js,
sw.js, part_a.bin, part_b.bin, weights_manifest.json, reference.json. weights.bin
is only an archival export; the app never requests it. Do not publish .venv.

## Static HTTPS deployment

GitHub Pages: create a repository, put the web assets in its root, commit/push,
then Settings → Pages → Deploy from a branch → main → /(root) → Save. Use the
published HTTPS URL on both phones. Enable Enforce HTTPS if offered. An empty
.nojekyll file is optional. The 22 MB split files can be committed with Git;
GitHub's browser upload size limits differ from Git push limits.

Netlify: place only the web assets in a folder and drag the folder onto
https://app.netlify.com/drop while signed in. Open the resulting HTTPS URL.
There is no build command, backend, function, or inference server.

Desktop development only: `python -m http.server 8000`, then open
http://localhost:8000. Plain HTTP on a laptop's LAN address is not a secure
context for phone camera/service-worker use; use the deployed HTTPS URL.

## Pair and run

1. Put both phones on the same non-isolated Wi-Fi or hotspot. Open the HTTPS
   site; choose Host on A and Join on B. Wait for each half to load and verify.
2. A taps Create offer. B taps Scan QR with camera and scans all cycling frames,
   or pastes A's complete pairing text and taps Use pasted text.
3. B produces an answer. A scans all answer frames or pastes the complete answer.
4. Wait for Connected / both halves ready. Generate from A. The benchmark uses
   UTF-8 "Hello", temperature 0, and exactly 128 new bytes. It resets all caches.

QR is standard version-5/L byte-mode with Reed–Solomon correction for up to 13
codeword errors. The bundled camera decoder supports this app's QR symbols,
rotation and moderate perspective. It is not an arbitrary QR/URL scanner.
Use bright screens, keep the whole quiet zone visible, and hold nearly parallel.
Text fallback has the identical SDP and does not require any signaling service.

ICE gathering completes before exporting SDP; candidates are not trickled.
The application preserves .local mDNS candidates verbatim. iceServers is always
empty: there is no STUN, TURN, relay, external signaling API, or telemetry.
Inspect ICE transport shows the selected candidate types; a relay is rejected.
Permissions must be granted by a tap. If Safari produces no candidates, grant
camera access using Scan QR, stop the camera, and reload/retry pairing. Also
check local-network permissions, VPNs, access-point client isolation and hotspot
peer policies. No code can bypass a network that blocks local peer traffic.

## Offline test

After both halves say cached/verified and the service worker has installed,
disable the Wi-Fi router's WAN connection while leaving the LAN active. Generate
and benchmark again. Optionally reload both cached pages and repeat pairing
without WAN. Do not disable Wi-Fi itself: the direct peer transport still needs
a network. In browser developer tools confirm no HTTP requests during generation
and only the appropriate half downloaded on each phone. Cache API persists
weights; the service worker persists the app shell. Private browsing, quota
limits or browser eviction can remove these caches; allow storage and reload
online once. After changing exported weights or source, clear site data on both
phones (or deploy to a new path). Cache names deliberately remain fixed.

## Verification

Optional desktop test with Node 20+, requiring no npm dependencies:

```sh
node verify.mjs
```

It checks exact greedy bytes against reference.json, first-block activation
maximum absolute error < 1e-4, split generation, the 128-byte benchmark, half-only
downloads, QR correction/raster detection, signaling compression, and two app
controllers using real workers with an in-memory reliable binary transport.
It does not test real device ICE or cameras.

For browser verification open the site with `?selftest=1`. This explicit TEST
ONLY mode hides Host/Join, loads both halves and displays exact-byte equality,
the first block's 384 output values, max absolute/RMS activation error and the
reference's minimum argmax margin. Normal mode cannot enter this UI. Run it on
each target phone, then reload without the query and pair normally. Compare
generated-byte arrays and FNV-1a on A, B, reference.json and the Python printout.
For the benchmark compare reference.json's benchmark.bytes and benchmark.fnv1a.

The local tested export produced 32-byte checksum 4e97afb2 for "Hello"; first
block maximum error was 7.16e-7 and both reference/128-byte benchmark greedy
sequences matched. The source records the actual PyTorch version and weight
SHA-256, because initialization and numerical kernels may change across releases.
Physical iOS/Android testing remains necessary on the actual phones/network.

## Numerical and performance details

All stored tensors, activations and per-layer K/V caches are Float32Array.
Each phone allocates 3 × 2 × 512 × 384 FP32 cache elements (4.5 MiB). One
activation packet is 1,536 payload bytes plus a 12-byte header; reply is 12 bytes.
Headers carry type, emission flag/byte, little-endian uint16 position, uint32 run
ID and FP32 compute milliseconds. JSON controls are also binary ArrayBuffers.
Ordered reliable delivery plus strict position/run checks keeps caches aligned.
Prompt bytes use the same incremental pipeline; only the final prefill reply
emits the first generated byte. Capacity is prompt_length + new_bytes - 1 ≤ 512;
the last generated byte need not be fed back. Empty prompts are rejected.

The JS matvec has four FP32 accumulators and contiguous [out,in] rows. Stored
results and critical operations use Math.fround. Softmax subtracts its maximum.
GELU uses the erf expression, not tanh; erf uses a positive convergent series
with relative term cutoff 1e-17 and a negligible tail beyond |x|=6. This is more
accurate than the requested ~1e-7 in double precision. No WASM binary is claimed
or required; optimized scalar JS is the only kernel and works without SIMD.

PyTorch can use different reduction trees, fused operations and erf kernels.
FP32 arithmetic does not guarantee identical logits or arbitrary greedy outputs
across implementations; near-tied logits can change argmax. The self-test must
report failure honestly. Do not claim cross-device parity until these tests pass.
Temperature > 0 uses crypto randomness on B only and is intentionally stochastic.

Throughput measures generated bytes divided by decode time, including the first
generated byte's compute and its response; prefill time is separately included
in the completion total. Compute means cover emitting steps. Response RTT
includes B's compute, worker dispatch and network time; the ICE diagnostic also
shows the browser's network RTT when available. Mobile throughput depends on
CPU, thermals, foreground scheduling and LAN latency; no rate is guaranteed.

Keep both tabs foregrounded and phones awake; iOS backgrounding suspends workers
and may break ICE. Generation has a 60-second step timeout. Stop drains the
current step before completing, preserving identical output on both phones.
