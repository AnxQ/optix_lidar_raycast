import numpy as np
import pytest
import torch

lro = pytest.importorskip("optix_lidar_raycast")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for OptiX tests", allow_module_level=True)


def _create_plane_mesh(size=2.0, z=0.0):
    half = size / 2.0
    vertices = np.array(
        [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices, faces


def _create_cube_mesh(size=1.0):
    half = size / 2.0
    vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 5, 1], [0, 4, 5],
            [2, 7, 3], [2, 6, 7],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def test_plane_hits_on_surface():
    vertices, faces = _create_plane_mesh(size=2.0, z=0.0)
    simulator = lro.OptiXLidarSimulator(
        num_vertical_beams=10,
        num_horizontal_samples=10,
        vertical_fov=(-45, 45),
        horizontal_fov=(0, 360),
        max_range=10.0,
    )

    simulator.set_mesh_batch(torch.from_numpy(vertices[None]).float(), torch.from_numpy(faces).int())
    positions = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    orientations = torch.tensor([[0.0, 90.0, 0.0]], dtype=torch.float32)

    hit_points, _, hit_face_ids = simulator.simulate(positions, orientations)
    valid_mask = hit_face_ids[0] >= 0
    assert valid_mask.any(), "Expected at least one plane hit"

    coord = hit_points[0, valid_mask]
    z_error = torch.abs(coord[:, 2])
    assert z_error.max().item() < 0.01


def test_cube_hits_on_surface():
    vertices, faces = _create_cube_mesh(size=1.0)
    simulator = lro.OptiXLidarSimulator(
        num_vertical_beams=20,
        num_horizontal_samples=20,
        vertical_fov=(-30, 30),
        horizontal_fov=(0, 360),
        max_range=10.0,
    )

    simulator.set_mesh_batch(torch.from_numpy(vertices[None]).float(), torch.from_numpy(faces).int())
    positions = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
    orientations = torch.tensor([[0.0, 0.0, 180.0]], dtype=torch.float32)

    hit_points, _, hit_face_ids = simulator.simulate(positions, orientations)
    valid_mask = hit_face_ids[0] >= 0
    assert valid_mask.any(), "Expected at least one cube hit"

    coord = hit_points[0, valid_mask]
    abs_coords = torch.abs(coord)
    on_surface = torch.any(torch.abs(abs_coords - 0.5) < 0.01, dim=1)
    assert (on_surface.float().mean().item()) > 0.95


def test_single_triangle_hits_inside_region():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    simulator = lro.OptiXLidarSimulator(
        num_vertical_beams=5,
        num_horizontal_samples=5,
        vertical_fov=(-45, 45),
        horizontal_fov=(0, 90),
        max_range=5.0,
    )

    simulator.set_mesh_batch(torch.from_numpy(vertices[None]).float(), torch.from_numpy(faces).int())
    positions = torch.tensor([[0.3, 0.3, 1.0]], dtype=torch.float32)
    orientations = torch.tensor([[0.0, 90.0, 0.0]], dtype=torch.float32)

    hit_points, _, hit_face_ids = simulator.simulate(positions, orientations)
    valid_mask = hit_face_ids[0] >= 0
    assert valid_mask.any(), "Expected at least one triangle hit"

    coord = hit_points[0, valid_mask]
    z_error = torch.abs(coord[:, 2])
    inside = (coord[:, 0] >= -0.01) & (coord[:, 1] >= -0.01) & (coord[:, 0] + coord[:, 1] <= 1.01)

    assert z_error.max().item() < 0.01
    assert (inside.float().mean().item()) > 0.9
