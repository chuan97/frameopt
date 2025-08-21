import numpy as np

from evomof.core.energy import coherence
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


def test_load_txt_roundtrip(tmp_path):
    """
    Exporting a Frame to txt and loading it back using Frame.load_txt
    should reproduce the same Frame with matching vectors.
    """
    rng = np.random.default_rng(99)
    n, d = 4, 3
    frame = Frame.random(n=n, d=d, rng=rng)

    # Export to txt
    filename = f"frame_{n}x{d}.txt"
    txt_path = tmp_path / filename
    frame.export_txt(str(txt_path))

    # Load back
    loaded = Frame.load_txt(str(txt_path), n=n, d=d)
    assert isinstance(loaded, Frame)
    np.testing.assert_allclose(loaded.vectors, frame.vectors)
    assert np.isclose(coherence(loaded), coherence(frame))


def test_npy_txt_consistency(tmp_path):
    """
    Saving a Frame to both .npy and .txt formats and loading them back
    should produce Frames with consistent vectors and coherence values.
    """
    rng = np.random.default_rng(2024)
    n, d = 6, 3
    frame = Frame.random(n=n, d=d, rng=rng)

    # Save to .npy
    npy_path = tmp_path / "frame.npy"
    frame.save_npy(str(npy_path))

    # Save to .txt
    txt_path = tmp_path / "frame.txt"
    frame.export_txt(str(txt_path))

    # Load back
    loaded_npy = Frame.load_npy(str(npy_path))
    loaded_txt = Frame.load_txt(str(txt_path), n=n, d=d)

    assert isinstance(loaded_npy, Frame)
    assert isinstance(loaded_txt, Frame)

    np.testing.assert_allclose(loaded_npy.vectors, loaded_txt.vectors)
    assert np.isclose(coherence(loaded_npy), coherence(loaded_txt))
