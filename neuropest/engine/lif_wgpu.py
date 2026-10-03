"""LIF engine on the GPU through WebGPU (wgpu): same model and interface as `LIFEngine`.

Time-driven instead of event-driven: one dispatch per step. Workgroups [0, nupd) update the neurons
(thread per neuron); the following W workgroups scatter the spikes of the PREVIOUS step into a delay
ring buffer with integer atomics (WGSL has no float atomics, so synaptic input is accumulated in fixed
point, 1/4096 mV). That is safe because a spike only has to arrive t_dly >= 2 steps later: the scatter
writes a ring slot the update does not touch in this step, and the spike lists rotate through three
buffers so nothing reads and writes the same one. Runs on NVIDIA, AMD, Intel and Apple GPUs through
Vulkan / D3D12 / Metal, needs no CUDA install (`uv sync --extra gpu`).

The kernel is memory-bound (full brain: ~90 us/step on a GTX 1650 before trimming traffic), so state is
split into separate arrays and only touched when it changes: a resting neuron with no input costs three
4-byte reads and no write, drive entries are looked up through a bit mask.

Differences from the CPU engine, all small: weights are rounded to 1/4096 mV, the Poisson generator
is a hash (statistically the same), a neuron that spikes and is also driven in one step counts once.
`n_active` is not tracked (every neuron is updated every step).
"""
from __future__ import annotations

import struct

import numpy as np

from .network import Network
from .params import LIFParams

FIX = 4096.0               # fixed-point scale of synaptic input (mV)
MAX_STEPS = 128            # steps encoded per submission
STEP_STRIDE = 256          # min uniform buffer offset alignment
STORAGE_BUFFERS = 12       # the step kernel binds 11; the WebGPU default limit is 8

_MAIN = """
struct Prm { n: u32, D: u32, W: u32, nupd: u32, a: f32, b: f32, c: f32, v0: f32, vth: f32, tref: f32, dt: f32, p1: f32 };
struct Step { pos: u32, slot: u32, w: u32, r: u32, z: u32, seed: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<uniform> prm: Prm;
@group(0) @binding(1) var<storage, read_write> vg: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> rft: array<f32>;
@group(0) @binding(3) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(4) var<storage, read_write> ring: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> drive: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> dmask: array<u32>;
@group(0) @binding(7) var<storage, read> indptr: array<u32>;
@group(0) @binding(8) var<storage, read> indices: array<u32>;
@group(0) @binding(9) var<storage, read> wfix: array<i32>;
@group(0) @binding(10) var<storage, read_write> spk: array<u32>;
@group(0) @binding(11) var<storage, read_write> scount: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> stp: Step;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn urand(i: u32, salt: u32) -> f32 {
    let h = pcg(i ^ pcg(stp.seed * 2u + salt));
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

@compute @workgroup_size(64)
fn step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let grp = wg.x;
    if (grp < prm.nupd) {
        let i = grp * 64u + lid.x;
        if (i < prm.n) {
            let old = vg[i];
            var v = old.x;
            var g = old.y;
            var rf = rft[i];
            let slot = stp.pos * prm.n + i;
            var dirty = false;
            if (atomicLoad(&ring[slot]) != 0) {
                g = g + f32(atomicExchange(&ring[slot], 0)) * (1.0 / 4096.0);
                dirty = true;
            }
            var pforce = 0.0;
            if (((dmask[i >> 5u] >> (i & 31u)) & 1u) != 0u) {
                let d = drive[i];
                if (d.y > 0.0 && urand(i, 1u) < d.y) { g = g + d.z; dirty = true; }
                pforce = d.x;
            }
            var spiked = false;
            if (rf > 0.0) {
                rft[i] = max(rf - prm.dt, 0.0);
            } else {
                let u = (v - prm.v0) * prm.a + g * prm.c * (prm.a - prm.b);
                let gn = g * prm.b;
                if (u + prm.v0 > prm.vth) {
                    v = prm.v0; g = 0.0; rft[i] = prm.tref; spiked = true; dirty = true;
                } else {
                    v = prm.v0 + u;
                    g = gn;
                    if (v != old.x || g != old.y) { dirty = true; }
                }
            }
            if (pforce > 0.0 && urand(i, 2u) < pforce) {
                v = prm.v0; g = 0.0; rft[i] = 0.0; spiked = true; dirty = true;
            }
            if (dirty) { vg[i] = vec2<f32>(v, g); }
            if (spiked) {
                cnt[i] = cnt[i] + 1u;
                let k = atomicAdd(&scount[stp.w], 1u);
                spk[stp.w * prm.n + k] = i;
            }
        }
    } else {
        let count = atomicLoad(&scount[stp.r]);
        var s = grp - prm.nupd;
        while (s < count) {
            let src = spk[stp.r * prm.n + s];
            let hi = indptr[src + 1u];
            var e = indptr[src] + lid.x;
            while (e < hi) {
                atomicAdd(&ring[stp.slot * prm.n + indices[e]], wfix[e]);
                e = e + 64u;
            }
            s = s + prm.W;
        }
        if (grp == prm.nupd && lid.x == 0u) {
            atomicAdd(&scount[3], count);
        }
    }
    if (grp == 0u && lid.x == 0u) {
        atomicStore(&scount[stp.z], 0u);        // the list the next step will write is free again
    }
}
"""

_GATHER = """
@group(0) @binding(0) var<storage, read_write> cnt: array<u32>;
@group(0) @binding(1) var<storage, read> gidx: array<u32>;
@group(0) @binding(2) var<storage, read_write> gout: array<u32>;
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= arrayLength(&gidx)) { return; }
    let i = gidx[q];
    gout[q] = cnt[i];
    cnt[i] = 0u;
}
"""

_APPLY = """
@group(0) @binding(0) var<storage, read_write> drive: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> uidx: array<u32>;
@group(0) @binding(2) var<storage, read> uval: array<vec4<f32>>;
@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= arrayLength(&uidx)) { return; }
    drive[uidx[q]] = uval[q];
}
"""


def list_adapters() -> list[dict]:
    """GPUs WebGPU can use: name, backend ('Vulkan', 'D3D12', ...), type, index into `enumerate_adapters`."""
    import wgpu

    out = []
    for i, a in enumerate(wgpu.gpu.enumerate_adapters_sync()):
        info = a.info
        out.append(dict(index=i, name=info.get("device"), backend=info.get("backend_type"),
                        type=info.get("adapter_type")))
    return out


_DEVICES: dict = {}        # one device per adapter, shared by every engine in the process


def _device(index: int):
    """Shared device of adapter `index` (a new device per engine exhausts memory on small machines)."""
    import wgpu

    if index not in _DEVICES:
        adapter = wgpu.gpu.enumerate_adapters_sync()[index]
        have = adapter.limits.get("max-storage-buffers-per-shader-stage", 0)
        if have < STORAGE_BUFFERS:
            raise RuntimeError(f"{adapter.info.get('device')}: only {have} storage buffers per stage, "
                               f"need {STORAGE_BUFFERS}")
        _DEVICES[index] = adapter.request_device_sync(
            required_limits={"max-storage-buffers-per-shader-stage": STORAGE_BUFFERS})
    return _DEVICES[index]


def gpu_available() -> bool:
    try:
        return any(a["type"] in ("DiscreteGPU", "IntegratedGPU") for a in list_adapters())
    except Exception:           # noqa: BLE001  (wgpu missing or no usable adapter)
        return False


class WGPUEngine:
    """Spiking engine over a `Network` on a GPU. Interface shared with `LIFEngine`."""

    def __init__(self, net: Network, params: LIFParams = LIFParams(), dt: float = 0.5, seed: int = 0,
                 eps: float = 0.0, adapter: int | None = None, scatter_groups: int = 512):
        import wgpu

        self._wgpu = wgpu
        adapters = wgpu.gpu.enumerate_adapters_sync()
        if adapter is None:
            order = [i for i, a in enumerate(adapters) if a.info.get("adapter_type") == "DiscreteGPU"] or \
                    [i for i, a in enumerate(adapters) if a.info.get("adapter_type") == "IntegratedGPU"]
            if not order:
                raise RuntimeError("no GPU adapter available")
            adapter = order[0]
        self.adapter_info = dict(adapters[adapter].info)
        self.dev = _device(adapter)
        self._bufs: list = []
        self.net, self.p, self.dt = net, params, float(dt)
        self.n = n = net.n
        self.dly = max(1, int(round(params.t_dly / dt)))
        if self.dly < 2:
            raise ValueError(f"the GPU engine needs dt <= 1 ms (synaptic delay of at least 2 steps), got {dt}")
        self.D = self.dly + 1
        self._seed = (seed * 7919 + 1) & 0xFFFFFF
        self._rem = 0.0
        self.steps = 0
        self._wg_update = (n + 63) // 64
        self._W = int(min(scatter_groups, max(1, self._wg_update)))
        self._drive_idx = np.zeros(0, np.int64)          # neurons whose drive entry is nonzero on the device
        self._f = (np.zeros(0, np.int64), np.zeros(0, np.float32))
        self._c = (np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32))
        self._gather_cache: dict = {}
        self._build(params)

    def close(self) -> None:
        """Free the GPU buffers now (also done on garbage collection)."""
        for b in getattr(self, "_bufs", []):
            try:
                b.destroy()
            except Exception:       # noqa: BLE001
                pass
        self._bufs = []
        self._gather_cache = {}

    def __del__(self):
        self.close()

    def _new(self, data=None, size=None, usage=None):
        buf = self.dev.create_buffer_with_data(data=data, usage=usage) if data is not None \
            else self.dev.create_buffer(size=size, usage=usage)
        self._bufs.append(buf)
        return buf

    # ------------------------------------------------------------------ setup
    def _build(self, p: LIFParams):
        wgpu, dev, n = self._wgpu, self.dev, self.n
        U = wgpu.BufferUsage
        net = self.net
        vg = np.zeros((n, 2), np.float32)
        vg[:, 0] = p.v_rest
        self.vg = self._new(vg, usage=U.STORAGE | U.COPY_DST)
        self.rft = self._new(np.zeros(n, np.float32), usage=U.STORAGE | U.COPY_DST)
        self.cnt = self._new(np.zeros(n, np.uint32), usage=U.STORAGE | U.COPY_DST)
        self.ring = self._new(np.zeros(self.D * n, np.int32), usage=U.STORAGE | U.COPY_DST)
        self.drive = self._new(np.zeros((n, 4), np.float32), usage=U.STORAGE | U.COPY_DST)
        self.dmask = self._new(np.zeros((n + 31) // 32, np.uint32), usage=U.STORAGE | U.COPY_DST)
        self.indptr = self._new(net.indptr.astype(np.uint32), usage=U.STORAGE)
        # create_buffer_with_data is the cheapest in host commit memory on Windows (write_buffer chunks keep
        # ~2x the size in staging, mapped_at_creation ~4x; measured with a 60 MB buffer)
        self.indices = self._new(net.indices.view(np.uint32), usage=U.STORAGE)    # same bits, no copy (>= 0)
        scaled = net.data * np.float32(FIX)
        np.rint(scaled, out=scaled)
        self.wfix = self._new(scaled.astype(np.int32), usage=U.STORAGE)
        del scaled
        self.spk = self._new(size=3 * n * 4, usage=U.STORAGE)                   # three rotating spike lists
        self.scount = self._new(np.zeros(4, np.uint32), usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
        a = float(np.float32(np.exp(-self.dt / p.tau_m)))
        b = float(np.float32(np.exp(-self.dt / p.tau_syn)))
        c = float(np.float32(p.tau_syn / (p.tau_m - p.tau_syn)))
        prm = struct.pack("<4I8f", n, self.D, self._W, self._wg_update, a, b, c, p.v_rest, p.v_th, p.t_ref,
                          self.dt, 0.0)
        self.prm = self._new(prm, usage=U.UNIFORM)
        self.stepbuf = self._new(size=STEP_STRIDE * MAX_STEPS, usage=U.UNIFORM | U.COPY_DST)

        RO, RW, UB = (wgpu.BufferBindingType.read_only_storage, wgpu.BufferBindingType.storage,
                      wgpu.BufferBindingType.uniform)
        vis = wgpu.ShaderStage.COMPUTE

        def entry(i, kind, dynamic=False):
            e = {"binding": i, "visibility": vis, "buffer": {"type": kind}}
            if dynamic:
                e["buffer"]["has_dynamic_offset"] = True
            return e

        kinds = [UB, RW, RW, RW, RW, RO, RO, RO, RO, RO, RW, RW]
        bgl0 = dev.create_bind_group_layout(entries=[entry(i, k) for i, k in enumerate(kinds)])
        bgl1 = dev.create_bind_group_layout(entries=[entry(0, UB, dynamic=True)])
        layout = dev.create_pipeline_layout(bind_group_layouts=[bgl0, bgl1])
        mod = dev.create_shader_module(code=_MAIN)
        self.p_step = dev.create_compute_pipeline(layout=layout, compute={"module": mod, "entry_point": "step"})
        self.bg0 = dev.create_bind_group(layout=bgl0, entries=[
            {"binding": i, "resource": {"buffer": buf}} for i, buf in enumerate(
                [self.prm, self.vg, self.rft, self.cnt, self.ring, self.drive, self.dmask, self.indptr,
                 self.indices, self.wfix, self.spk, self.scount])])
        self.bg1 = dev.create_bind_group(layout=bgl1, entries=[
            {"binding": 0, "resource": {"buffer": self.stepbuf, "offset": 0, "size": 32}}])

        gl = dev.create_bind_group_layout(entries=[entry(0, RW), entry(1, RO), entry(2, RW)])
        self._gather_layout = gl
        self.p_gather = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[gl]),
            compute={"module": dev.create_shader_module(code=_GATHER), "entry_point": "gather"})
        al = dev.create_bind_group_layout(entries=[entry(0, RW), entry(1, RO), entry(2, RO)])
        self._apply_layout = al
        self.p_apply = dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[al]),
            compute={"module": dev.create_shader_module(code=_APPLY), "entry_point": "apply"})

    # ------------------------------------------------------------------ input
    @staticmethod
    def _p_of(rate_hz, dt, idx):
        rate = np.broadcast_to(np.asarray(rate_hz, np.float32), np.shape(idx))
        return rate, (1.0 - np.exp(-rate * dt / 1000.0)).astype(np.float32)

    def set_drive(self, idx, rate_hz) -> None:
        """Neurons `idx` emit Poisson spikes at `rate_hz` (forced, like Shiu's PoissonInput)."""
        idx = np.asarray(idx, np.int64)
        rate, p = self._p_of(rate_hz, self.dt, idx)
        keep = rate > 0
        self._f = (idx[keep], p[keep])
        self._push_drive()

    def add_drive(self, idx, rate_hz) -> None:
        prev_i, prev_p = self._f
        idx = np.asarray(idx, np.int64)
        rate, p = self._p_of(rate_hz, self.dt, idx)
        keep = rate > 0
        self._f = (np.concatenate([prev_i, idx[keep]]), np.concatenate([prev_p, p[keep]]))
        self._push_drive()

    def set_current_drive(self, idx, rate_hz, w_mv) -> None:
        """Poisson synaptic input: events at `rate_hz`, each adding `w_mv` to g of neurons `idx`."""
        idx = np.asarray(idx, np.int64)
        rate, p = self._p_of(rate_hz, self.dt, idx)
        w = np.broadcast_to(np.asarray(w_mv, np.float32), idx.shape)
        keep = rate > 0
        self._c = (idx[keep], p[keep], np.ascontiguousarray(w[keep]))
        self._push_drive()

    def _push_drive(self) -> None:
        """Send only the drive entries that changed (a few KB) plus the bit mask of driven neurons."""
        fi, fp = self._f
        ci, cp, cw = self._c
        new_idx = np.union1d(fi, ci)
        all_idx = np.union1d(new_idx, self._drive_idx)
        if len(all_idx) == 0:
            return
        vals = np.zeros((len(all_idx), 4), np.float32)
        if len(fi):
            np.maximum.at(vals[:, 0], np.searchsorted(all_idx, fi), fp)
        if len(ci):
            pos = np.searchsorted(all_idx, ci)
            vals[pos, 1] = cp
            vals[pos, 2] = cw
        mask = np.zeros((self.n + 31) // 32, np.uint32)
        np.bitwise_or.at(mask, new_idx >> 5, np.uint32(1) << (new_idx & 31).astype(np.uint32))
        dev, U = self.dev, self._wgpu.BufferUsage
        ibuf = dev.create_buffer_with_data(data=all_idx.astype(np.uint32), usage=U.STORAGE)
        vbuf = dev.create_buffer_with_data(data=vals, usage=U.STORAGE)
        bg = dev.create_bind_group(layout=self._apply_layout, entries=[
            {"binding": 0, "resource": {"buffer": self.drive}}, {"binding": 1, "resource": {"buffer": ibuf}},
            {"binding": 2, "resource": {"buffer": vbuf}}])
        dev.queue.write_buffer(self.dmask, 0, mask)
        enc = dev.create_command_encoder()
        cpass = enc.begin_compute_pass()
        cpass.set_pipeline(self.p_apply)
        cpass.set_bind_group(0, bg)
        cpass.dispatch_workgroups((len(all_idx) + 63) // 64)
        cpass.end()
        dev.queue.submit([enc.finish()])
        ibuf.destroy()                  # destruction waits for the submitted work that uses them
        vbuf.destroy()
        self._drive_idx = new_idx

    # ---------------------------------------------------------------- run
    def advance(self, ms: float) -> int:
        total = ms + self._rem
        n_steps = int(total / self.dt + 1e-9)
        self._rem = total - n_steps * self.dt
        done = 0
        while done < n_steps:
            self._run(min(MAX_STEPS, n_steps - done))
            done += min(MAX_STEPS, n_steps - done)
        return n_steps

    def _run(self, k: int) -> None:
        dev = self.dev
        sp = np.zeros((k, STEP_STRIDE // 4), np.uint32)
        s = self.steps + np.arange(k)                       # global step numbers
        sp[:, 0] = s % self.D                               # ring slot read by this step
        sp[:, 1] = (s - 1 + self.dly) % self.D              # slot the previous step's spikes arrive in
        sp[:, 2] = s % 3                                    # spike list written now
        sp[:, 3] = (s - 1) % 3                              # spike list scattered now (previous step)
        sp[:, 4] = (s + 1) % 3                              # list to clear for the next step
        sp[:, 5] = self._seed + s
        dev.queue.write_buffer(self.stepbuf, 0, sp)
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.p_step)
        cp.set_bind_group(0, self.bg0)
        total = self._wg_update + self._W
        for j in range(k):
            cp.set_bind_group(1, self.bg1, [j * STEP_STRIDE])
            cp.dispatch_workgroups(total)
        cp.end()
        dev.queue.submit([enc.finish()])
        self.steps += k

    def sync(self) -> None:
        """Wait until the GPU has finished everything submitted so far."""
        self.dev.queue.read_buffer(self.scount, 0, 16)

    # ------------------------------------------------------------- output
    def pop_counts(self, idx) -> np.ndarray:
        """Spikes of neurons `idx` since their last pop; resets those counters. Blocks until done."""
        idx = np.ascontiguousarray(idx, np.uint32)
        key = (len(idx), hash(idx.tobytes()))
        hit = self._gather_cache.get(key)
        if hit is None:
            dev, U = self.dev, self._wgpu.BufferUsage
            ibuf = self._new(idx, usage=U.STORAGE)
            obuf = self._new(size=max(4, len(idx) * 4), usage=U.STORAGE | U.COPY_SRC)
            bg = dev.create_bind_group(layout=self._gather_layout, entries=[
                {"binding": 0, "resource": {"buffer": self.cnt}}, {"binding": 1, "resource": {"buffer": ibuf}},
                {"binding": 2, "resource": {"buffer": obuf}}])
            hit = self._gather_cache[key] = (bg, obuf, len(idx))
        bg, obuf, m = hit
        if m == 0:
            return np.zeros(0, np.int32)
        enc = self.dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.p_gather)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((m + 63) // 64)
        cp.end()
        self.dev.queue.submit([enc.finish()])
        return np.frombuffer(self.dev.queue.read_buffer(obuf, 0, m * 4), np.uint32).astype(np.int32)

    @property
    def n_active(self) -> int:
        return -1                                   # every neuron is updated every step

    @property
    def total_spikes(self) -> int:
        return int(np.frombuffer(self.dev.queue.read_buffer(self.scount, 0, 16), np.uint32)[3])

    @property
    def neuron_updates(self) -> int:
        return self.steps * self.n

    @property
    def sim_ms(self) -> float:
        return self.steps * self.dt
