import pytest
import torch

lro = pytest.importorskip("optix_lidar_raycast")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for OptiX tests", allow_module_level=True)


def _make_plane_batch(z_values):
    planes = []
    for z in z_values:
        planes.append(
            [
                [-1.0, -1.0, z],
                [1.0, -1.0, z],
                [1.0, 1.0, z],
                [-1.0, 1.0, z],
            ]
        )
    vertices = torch.tensor(planes, dtype=torch.float32, device="cuda")
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device="cuda")
    return vertices, faces


def _make_simulator(v_beams=8, h_samples=64):
    return lro.OptiXLidarSimulator(
        num_vertical_beams=v_beams,
        num_horizontal_samples=h_samples,
        vertical_fov=(-45.0, 45.0),
        horizontal_fov=(0.0, 360.0),
        max_range=20.0,
    )


def test_raycast_plane_correctness():
    sim = _make_simulator(v_beams=8, h_samples=48)
    vertices, faces = _make_plane_batch([0.0])
    sim.set_mesh_batch(vertices, faces)

    positions = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device="cuda")
    orientations = torch.tensor([[0.0, -90.0, 0.0]], dtype=torch.float32, device="cuda")

    points, distances, face_ids = sim.simulate(positions, orientations)

    assert isinstance(points, torch.Tensor)
    assert points.is_cuda
    assert distances.is_cuda
    assert face_ids.is_cuda

    valid = face_ids[0] >= 0
    assert valid.any(), "Expected at least one hit on plane"

    z_vals = points[0, valid, 2]
    assert torch.max(torch.abs(z_vals)).item() < 0.02


def test_multibatch_stability_no_mesh_mixup():
    plane_z = torch.tensor([-1.0, -0.25, 0.5, 1.25], dtype=torch.float32, device="cuda")
    sim = _make_simulator(v_beams=10, h_samples=64)
    vertices, faces = _make_plane_batch(plane_z.tolist())
    sim.set_mesh_batch(vertices, faces)

    positions = torch.stack(
        [
            torch.tensor([0.0, 0.0, z + 2.0], dtype=torch.float32, device="cuda")
            for z in plane_z
        ],
        dim=0,
    )
    orientations = torch.tensor(
        [[0.0, -90.0, 0.0]] * plane_z.numel(),
        dtype=torch.float32,
        device="cuda",
    )

    points, _, face_ids = sim.simulate(positions, orientations)

    for b in range(plane_z.numel()):
        valid = face_ids[b] >= 0
        assert valid.any(), f"Batch {b} has no valid hits"
        z_mean = points[b, valid, 2].mean().item()
        assert abs(z_mean - plane_z[b].item()) < 0.03, (
            f"Batch {b} z mismatch: got {z_mean}, expected {plane_z[b].item()}"
        )


def test_repeatability_same_input_stable():
    sim = _make_simulator(v_beams=8, h_samples=32)
    vertices, faces = _make_plane_batch([0.2, -0.4])
    sim.set_mesh_batch(vertices, faces)

    positions = torch.tensor(
        [[0.0, 0.0, 2.2], [0.0, 0.0, 1.6]],
        dtype=torch.float32,
        device="cuda",
    )
    orientations = torch.tensor(
        [[0.0, -90.0, 0.0], [0.0, -90.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )

    p1, d1, f1 = sim.simulate(positions, orientations)
    p2, d2, f2 = sim.simulate(positions, orientations)

    assert torch.equal(f1, f2)
    assert torch.allclose(d1, d2, atol=1e-5)
    assert torch.allclose(p1, p2, atol=1e-5)


def test_torch_device_path_cuda_and_cpu():
    sim = _make_simulator(v_beams=6, h_samples=24)

    vertices_cuda, faces_cuda = _make_plane_batch([0.0])
    sim.set_mesh_batch(vertices_cuda, faces_cuda)
    positions_cuda = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device="cuda")
    orientations_cuda = torch.tensor([[0.0, -90.0, 0.0]], dtype=torch.float32, device="cuda")
    points_cuda, dist_cuda, face_cuda = sim.simulate(positions_cuda, orientations_cuda)
    assert points_cuda.is_cuda and dist_cuda.is_cuda and face_cuda.is_cuda

    vertices_cpu = vertices_cuda.cpu()
    faces_cpu = faces_cuda.cpu()
    sim.set_mesh_batch(vertices_cpu, faces_cpu)
    positions_cpu = positions_cuda.cpu()
    orientations_cpu = orientations_cuda.cpu()
    points_cpu, dist_cpu, face_cpu = sim.simulate(positions_cpu, orientations_cpu)

    assert isinstance(points_cpu, torch.Tensor)
    assert points_cpu.device.type == "cpu"
    assert dist_cpu.device.type == "cpu"
    assert face_cpu.device.type == "cpu"

    assert points_cpu.shape == points_cuda.shape
    assert dist_cpu.shape == dist_cuda.shape
    assert face_cpu.shape == face_cuda.shape
