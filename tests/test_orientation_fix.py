import pytest
import torch

lro = pytest.importorskip("optix_lidar_raycast")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for OptiX tests", allow_module_level=True)


def _plane_mesh():
    vertices = torch.tensor(
        [[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    return vertices, faces


def _cube_mesh():
    vertices = torch.tensor(
        [[
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ],
        dtype=torch.int32,
    )
    return vertices, faces


def test_plane_with_pitch_minus_90_looks_down():
    vertices, faces = _plane_mesh()
    simulator = lro.OptiXLidarSimulator(
        num_vertical_beams=10,
        num_horizontal_samples=10,
        vertical_fov=(-45, 45),
        horizontal_fov=(0, 360),
        max_range=10.0,
    )
    simulator.set_mesh_batch(vertices, faces)

    positions = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    orientations = torch.tensor([[0.0, -90.0, 0.0]], dtype=torch.float32)

    hit_points, _, hit_face_ids = simulator.simulate(positions, orientations)
    valid_mask = hit_face_ids[0] >= 0
    assert valid_mask.any(), "Expected at least one hit with downward pitch"

    coord = hit_points[0, valid_mask]
    assert torch.abs(coord[:, 2]).max().item() < 0.01


def test_cube_with_yaw_180_looks_toward_negative_x():
    vertices, faces = _cube_mesh()
    simulator = lro.OptiXLidarSimulator(
        num_vertical_beams=20,
        num_horizontal_samples=20,
        vertical_fov=(-30, 30),
        horizontal_fov=(-30, 30),
        max_range=10.0,
    )
    simulator.set_mesh_batch(vertices, faces)

    positions = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    orientations = torch.tensor([[0.0, 0.0, 180.0]], dtype=torch.float32)

    hit_points, _, hit_face_ids = simulator.simulate(positions, orientations)
    valid_mask = hit_face_ids[0] >= 0
    assert valid_mask.any(), "Expected at least one hit with yaw=180"

    coord = hit_points[0, valid_mask]
    abs_coords = torch.abs(coord)
    on_surface = torch.any(torch.abs(abs_coords - 0.5) < 0.01, dim=1)

    assert on_surface.float().mean().item() > 0.95
    assert coord.shape[0] > 100
