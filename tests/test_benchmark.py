import os
import time

import pytest
import torch

lro = pytest.importorskip("optix_lidar_raycast")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for OptiX benchmark", allow_module_level=True)


def _make_plane_batch(batch_size: int, z: float = 0.0):
    vertices = torch.tensor(
        [
            [
                [-1.0, -1.0, z],
                [1.0, -1.0, z],
                [1.0, 1.0, z],
                [-1.0, 1.0, z],
            ]
        ]
        * batch_size,
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
    return vertices, faces


def test_optix_raycast_benchmark_latency_throughput():
    batch_size = int(os.getenv("OPTIX_BENCH_BATCH", "8"))
    warmup_steps = int(os.getenv("OPTIX_BENCH_WARMUP", "5"))
    benchmark_steps = int(os.getenv("OPTIX_BENCH_STEPS", "30"))

    sim = lro.OptiXLidarSimulator(
        num_vertical_beams=64,
        num_horizontal_samples=2560,
        vertical_fov=(-45.0, 45.0),
        horizontal_fov=(0.0, 360.0),
        max_range=20.0,
    )

    vertices, faces = _make_plane_batch(batch_size=batch_size, z=0.0)
    sim.set_mesh_batch(vertices, faces)

    positions = torch.tensor(
        [[0.0, 0.0, 2.0]] * batch_size,
        dtype=torch.float32,
        device="cuda",
    )
    orientations = torch.tensor(
        [[0.0, -90.0, 0.0]] * batch_size,
        dtype=torch.float32,
        device="cuda",
    )

    for _ in range(warmup_steps):
        sim.simulate(positions, orientations)
    torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(benchmark_steps):
        start = time.perf_counter()
        points, distances, face_ids = sim.simulate(positions, orientations)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    throughput = batch_size * 1000.0 / avg_ms

    valid_ratio = (face_ids >= 0).float().mean().item()

    print(
        "benchmark_result "
        f"batch={batch_size} steps={benchmark_steps} "
        f"avg_ms={avg_ms:.3f} min_ms={min_ms:.3f} max_ms={max_ms:.3f} "
        f"samples_per_sec={throughput:.2f} valid_ratio={valid_ratio:.3f}"
    )

    assert torch.isfinite(points).all()
    assert torch.isfinite(distances).all()
    assert avg_ms > 0.0
    assert throughput > 0.0
    assert valid_ratio > 0.01
