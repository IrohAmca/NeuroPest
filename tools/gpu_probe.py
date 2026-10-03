"""List the GPUs WebGPU (wgpu) can use and check that a compute shader with integer atomics runs.

Run: uv run --with wgpu python tools/gpu_probe.py
"""
from __future__ import annotations

import struct
import time

import wgpu

SHADER = """
@group(0) @binding(0) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&idx)) {
        atomicAdd(&acc[idx[i] % 1024u], 1);
    }
}
"""


def main():
    adapters = wgpu.gpu.enumerate_adapters_sync()
    print(f"wgpu {wgpu.__version__}: {len(adapters)} adapter(s)")
    for a in adapters:
        info = a.info
        print(f"  {info.get('device')!r}  vendor={info.get('vendor')!r}  backend={info.get('backend_type')!r}  "
              f"type={info.get('adapter_type')!r}")
        lim = a.limits
        print(f"    max_storage_buffer_binding_size={lim['max-storage-buffer-binding-size'] / 2**20:.0f} MiB  "
              f"max_buffer_size={lim['max-buffer-size'] / 2**20:.0f} MiB  "
              f"max_compute_invocations={lim['max-compute-invocations-per-workgroup']}")
        try:
            dev = a.request_device_sync()
        except Exception as e:                      # noqa: BLE001
            print("    request_device failed:", e)
            continue
        n = 1 << 20
        data = struct.pack(f"{n}I", *range(n))
        idx_buf = dev.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
        acc_buf = dev.create_buffer(size=1024 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        mod = dev.create_shader_module(code=SHADER)
        pipe = dev.create_compute_pipeline(layout="auto", compute={"module": mod, "entry_point": "main"})
        bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": acc_buf}}, {"binding": 1, "resource": {"buffer": idx_buf}}])
        t = time.perf_counter()
        for _ in range(20):
            enc = dev.create_command_encoder()
            p = enc.begin_compute_pass()
            p.set_pipeline(pipe)
            p.set_bind_group(0, bg)
            p.dispatch_workgroups(n // 64)
            p.end()
            dev.queue.submit([enc.finish()])
        out = bytes(dev.queue.read_buffer(acc_buf))
        dt = time.perf_counter() - t
        vals = struct.unpack("1024i", out)
        ok = all(v == 20 * (n // 1024) for v in vals)
        print(f"    atomic compute test: {'OK' if ok else 'WRONG'}  ({20 * n / dt / 1e6:.0f} M atomic adds/s incl. submit+readback)")


if __name__ == "__main__":
    main()
