"""camera-specific 入力を共通 InputSpec へ変換する契約を検証する."""

import json

from sphere_reconstruct.colmap import input_workspace


def test_native_fisheye_builder_materializes_images_masks_and_rig(tmp_path):
    project = tmp_path / "project"
    frames_dir = project / "extract_frames"
    frames_dir.mkdir(parents=True)
    lens0 = frames_dir / "lens0.jpg"
    lens1 = frames_dir / "lens1.jpg"
    lens0.write_bytes(b"front")
    lens1.write_bytes(b"back")
    (frames_dir / "manifest_frames.json").write_text(
        json.dumps(
            {
                "kind": "insv_dual",
                "width": 100,
                "height": 100,
                "frames": [
                    {
                        "index": 0,
                        "timestamp_sec": 0.0,
                        "lens0": "extract_frames/lens0.jpg",
                        "lens1": "extract_frames/lens1.jpg",
                    }
                ],
            }
        )
    )
    output = tmp_path / "output"
    output.mkdir()

    spec = input_workspace.build(project, output, "native_fisheye", use_masks=False)

    assert spec.camera_model == "OPENCV_FISHEYE"
    assert spec.image_count == 2
    assert spec.refine_rig is False
    assert spec.multiple_models is False
    assert (output / "images/front/frame_000000.jpg").is_file()
    assert (output / "images/back/frame_000000.jpg").is_file()
    assert (output / "masks/front/frame_000000.jpg.png").is_file()
    assert (output / "masks/back/frame_000000.jpg.png").is_file()
    assert (output / "rig_config.json").is_file()
    assert input_workspace.InputSpec.read(output / "input_spec.json") == spec
