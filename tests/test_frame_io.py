import numpy as np

from evomof.core.frame import Frame


def test_save_load_npy(tmp_path):
    """
    Saving a Frame to .npy and loading it back should preserve the vectors exactly.
    """
    rng = np.random.default_rng(123)
    n, d = 5, 3
    frame = Frame.random(n=n, d=d, rng=rng)

    # Save to .npy
    file_path = tmp_path / "frame.npy"
    frame.save_npy(str(file_path))

    # Load back
    loaded = Frame.load_npy(str(file_path))
    assert isinstance(loaded, Frame)
    # Should match (allowing for tiny floating-point differences)
    np.testing.assert_allclose(loaded.vectors, frame.vectors, atol=1e-15, rtol=0)


def test_export_txt(tmp_path):
    """
    Exporting a Frame to text and reading it back should reproduce
    the same vector entries.
    """
    rng = np.random.default_rng(42)
    n, d = 3, 2
    frame = Frame.random(n=n, d=d, rng=rng)

    # Export to txt
    tag = "test"
    filename = f"{d}x{n}_{tag}.txt"
    txt_path = tmp_path / filename
    frame.export_txt(str(txt_path))

    # Read lines
    lines = txt_path.read_text().splitlines()
    expected_lines = 2 * n * d
    assert (
        len(lines) == expected_lines
    ), f"Expected {expected_lines} lines, got {len(lines)}"

    # Parse floats
    vals = np.array([float(line) for line in lines])
    real_vals = vals[: n * d]
    imag_vals = vals[n * d :]

    # Reconstruct in row-major (order='C')
    reconstructed = real_vals.reshape((n, d), order="C") + 1j * imag_vals.reshape(
        (n, d), order="C"
    )

    # Compare
    np.testing.assert_allclose(reconstructed, frame.vectors)
