# Optimization findings (2026-07-16)

Second-opinion performance review of `__init__.py` (v1.5.0). Hot-path patterns were
micro-benchmarked inside Blender 5.1.0 (numpy 2.3.4) with synthetic data:
N = 100k verts, S = 200 selected seeds, K = 20k weighted overlay points.

**Verdict: the majority of per-frame processing during a transform drag is *not*
unavoidable.** The weight solve (Dijkstra / KD search) is recomputed every draw frame
during a G/R/S spy session, but its inputs — snapshot positions, radius, falloff,
selection — only change on scroll. The genuinely unavoidable per-frame work is just
refreshing live draw positions and re-uploading the batch, which is a small fraction
of the current frame cost.

## Measured baseline (per *frame* at 100k verts / 20k weighted points)

| Pattern | Where | Cost |
|---|---|---|
| `kd.find()` per vert loop | `draw_callback` KD branch (~line 1036) | ~52 ms |
| `_snap_vert_world` per vert | same loop (~line 1037) | ~44 ms |
| Per-edge snapshot math (2× `_snap_vert_world` + length), 200k edge visits | Dijkstra branch (~line 1004) | ~177 ms |
| blake2b per-point fingerprint of `vert_weights` | batch key (~line 1105) — **every redraw, even idle** | ~12 ms |
| Batch list build (4× duplicated Python lists) | ~line 1140 | ~15 ms |
| numpy transform of the whole snapshot array (for comparison) | — | ~0.8 ms |
| Precomputed-list edge math, same 200k visits | — | ~39 ms (4.6×) |

So a drag frame on a dense mesh spends ~100 ms (KD mode) to ~200+ ms (Connected
mode) on work whose result is identical to the previous frame.

## Findings, ranked by impact

### 1. Weight solve reruns every modal frame though it's drag-invariant — HIGH — *implemented 2026-07-16, pending user testing*
`draw_callback` (~line 953): `if RT.modal_radius is not None: rebuild = True` forces a
full Dijkstra/KD weight solve every frame of the spy session. But the solve reads
positions via `_snap_vert_world`, i.e. the **frozen transform snapshot**, and the
radius/falloff/selection cannot change mid-drag except via scroll. The
`(v_index, weight)` result is therefore identical frame-to-frame during a pure drag.

**Fix:** split the rebuild. Recompute `VIZ_CACHE.weights` only when `RT.modal_radius`
(or falloff) actually changed since the last solve; keep refreshing only the second
block (~line 1046) that maps cached `(index, weight)` pairs to live world positions
for drawing. That block is O(weighted verts), not O(all verts)/O(edges).
Caveat: `_snap_vert_world` falls back to live positions for NaN/out-of-range indices,
but topology can't change inside a transform modal, so the invariance holds.

**Expected:** removes ~50–180 ms/frame during drags; the largest single win available.

### 2. Per-point fingerprint hashing on every redraw, even fully idle — HIGH
Batch-key computation (~lines 1100–1130) loops over all `vert_weights` with
`struct.pack` + blake2b on **every** draw callback — including plain viewport
orbit/pan frames where the idle cache fast-path (line 765) already proved nothing
changed. ~12 ms per redraw at 20k points, pure overhead multiplied by every 3D
viewport showing the overlay.

**Fix:** cache `pos_fp` alongside `vert_weights` and recompute it only when
`vert_weights` was rebuilt this frame. During a modal drag, positions change every
frame anyway — skip hashing entirely and rebuild the batch unconditionally (hashing
costs about as much as the list build it tries to avoid).

### 3. MATERIAL mode (always) and VG/SK in object mode have no cache at all — HIGH for those modes
The signature cache at ~line 1067 only stores VG/SK results in edit mode. MATERIAL
mode re-walks **all polygons** and VG/SK-in-object-mode re-walk all verts (with
`vert_world_pos` per vert) on every redraw, forever. Material assignments and weights
rarely change frame-to-frame.

**Fix:** extend the existing `edit_vw_sig` signature-cache pattern to MATERIAL mode
and to VG/SK outside edit mode (signature: object name, matrix fingerprint, material
name/slot or group name, mesh vert/poly counts, `mesh_eval_dirty`).

### 4. Per-vertex Python math instead of per-object precompute — MEDIUM-HIGH
`_snap_vert_world` / `vert_world_pos` are called once per vertex (KD branch) or twice
per edge visit (Dijkstra branch). Each call does dict lookups, bounds checks, a
`Vector` allocation, and a Python-level `Matrix @ Vector`.

**Fix (cheap):** once per solve, precompute a per-object list/array of world
positions, then index it in the loops. Measured 4.6× on the edge-relaxation pattern.
**Fix (better):** the snapshot is already a flat `array('f')` —
`np.frombuffer(...).reshape(-1,3) @ mw3x3.T + t` transforms 100k verts in ~0.8 ms
(vs ~44 ms). For the non-snapshot path, `mesh.vertices.foreach_get("co", ...)` gives
the same bulk read (in edit mode, after `obj.update_from_editmode()`).
Note: naive numpy brute-force distance-to-seeds measured **worse** than
`mathutils.kdtree` (~400 ms vs ~52 ms at 200 seeds) — keep the KD-tree for the
search, feed it precomputed coords.

### 5. `proportional_mirror_world_positions` inverts the matrix per weighted vertex — MEDIUM
Called per weighted vertex (~line 1058); when any mirror flag is set it runs
`mat.inverted()` **per call** (line 460). Hoist the inverse (and the no-mirror check)
per object, outside the vertex loop. Also: the no-mirror early-out returns
`(wp.copy(),)` — an allocation per vertex that callers don't need, since `wp` is
freshly computed each iteration.

### 6. Batch geometry built as 4×-duplicated Python lists — LOW-MEDIUM
~15 ms per batch rebuild at 20k points (lines 1139–1155). Only matters when the batch
rebuilds — but with finding 1/2 unfixed that's every modal frame. numpy
(`np.repeat` for positions/colors, `np.tile` for corners, arange-based index array)
cuts this to ~1 ms, and `batch_for_shader` accepts numpy arrays directly.

### 7. Micro / hygiene — LOW
- `falloff()` re-branches on the mode string per vertex; resolving the falloff mode
  to a local function once per solve shaves a bit in both branches.
- The LUT lookup `lut[min(255, int(w*255))]` and per-point alpha math could be folded
  into a 256-entry premultiplied LUT when `alpha_fade` is constant.
- `softviz_cache_timer` ticks every 0.1 s for the whole session even with the overlay
  off; returning a longer interval when `softviz_running` is false is free.

## What actually is unavoidable

- **Dijkstra along edges in Python** for Connected Only: inherently a per-edge Python
  loop (heapq + neighbors); numpy can't vectorize graph propagation. Finding 1 makes
  it run only on scroll/selection change instead of every frame, and finding 4 cuts
  its constant factor ~4×, but a scroll notch on a dense connected mesh will still
  cost a visible one-off spike. Acceptable.
- **Snapshot capture at G/R/S start** (`_capture_softviz_transform_snapshot`): O(N)
  Python loop over bmesh verts, once per transform start. bmesh has no `foreach_get`,
  so only partially improvable (`update_from_editmode` + `foreach_get` on `obj.data`
  when no cage is needed).
- **One draw callback per 3D viewport region** — Blender's handler model; caches are
  shared, so extra viewports mostly pay only the (post-fix) cheap path.

## Suggested order of work

1. Finding 1 (drag-invariant weights) — biggest win, self-contained in the
   PROPORTIONAL rebuild block.
2. Finding 2 (fingerprint caching) — small diff, fixes idle-orbit cost.
3. Finding 3 (MATERIAL / object-mode caching) — reuses the existing signature pattern.
4. Findings 4–6 (numpy precompute + batch build) — larger diff, do after 1–3 so the
   remaining per-frame path is the only thing left to vectorize.
