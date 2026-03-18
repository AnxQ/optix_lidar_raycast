def test_package_surface_post_fork_safe_removal():
    import optix_lidar_raycast as lro

    assert hasattr(lro, "OptiXLidarSimulator")

    try:
        from optix_lidar_raycast import ForkSafeLidarSimulator  # noqa: F401
    except ImportError:
        pass
    else:
        raise AssertionError("ForkSafeLidarSimulator should not be exported")
