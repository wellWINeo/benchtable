def test_package_exposes_version():
    import benchtable

    assert benchtable.__version__
