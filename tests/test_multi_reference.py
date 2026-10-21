"""CPU wiring checks with inference/UI dependencies replaced, not model-quality tests."""

import importlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cuda = Mock()
cuda.max_memory_allocated.return_value = 1024**2
cuda.get_device_name.return_value = "test GPU"
torch_stub = SimpleNamespace(
    __version__="test", cuda=cuda, Generator=Mock(), bfloat16="bfloat16"
)
with patch.dict(sys.modules, {
    "torch": torch_stub,
    "diffusers": SimpleNamespace(__version__="test", Flux2KleinPipeline=Mock()),
    "gradio": SimpleNamespace(Error=ValueError),
}):
    flux = importlib.import_module("app.common.flux2")
    editor = importlib.import_module("app.gradio_inference.app")


class MultiReferenceTests(unittest.TestCase):
    def setUp(self):
        self.original = Image.new("RGB", (96, 64), "white")
        self.layer = Image.new("RGBA", self.original.size)
        self.layer.putpixel((20, 20), (255, 0, 0, 255))
        self.overlay = self.original.copy()
        self.overlay.paste(self.layer, mask=self.layer.getchannel("A"))

    def test_editor_composite_and_layer_fallback(self):
        for composite in (self.overlay, None):
            with self.subTest(composite=composite is not None):
                original, overlay = editor._editor_images({
                    "background": self.original, "layers": [self.layer], "composite": composite
                })
                self.assertEqual(original.getpixel((20, 20)), (255, 255, 255))
                self.assertEqual(overlay.getpixel((20, 20)), (255, 0, 0))
                self.assertEqual(original.size, overlay.size)
        original, overlay = editor._editor_images({
            "background": self.original, "composite": self.overlay.resize((48, 32))
        })
        self.assertEqual(original.size, overlay.size)
        with self.assertRaisesRegex(ValueError, "clean original"):
            editor._editor_images({"composite": self.overlay})

    def test_runs_pass_ordered_references_and_save_artifacts(self):
        for variant in ("distill", "base"):
            for mode in ("overlay", "multi_reference"):
                with self.subTest(variant=variant, mode=mode), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                    pipe = Mock(return_value=SimpleNamespace(images=[self.original.copy()]))
                    manager = flux.ModelManager()
                    manager.pipes[variant] = pipe
                    with patch.object(editor, "_MANAGER", manager), patch.object(editor, "OUTPUT_ROOT", Path(tmp) / "runs"):
                        result, visual, config, saved, archive = editor.run_edit(
                            "test", "Add a vase.",
                            {"background": self.original, "composite": self.overlay},
                            variant, 42, 12, 2.0, mode,
                        )
                    args = pipe.call_args.kwargs
                    if mode == "multi_reference":
                        self.assertIsInstance(args["image"], list)
                        self.assertEqual(len(args["image"]), 2)
                        self.assertEqual(args["image"][0].getpixel((20, 20)), (255, 255, 255))
                        self.assertEqual(args["image"][1].getpixel((20, 20)), (255, 0, 0))
                        self.assertIn("Image 1 is the original", args["prompt"])
                        self.assertEqual(config["reference_order"], ["original.png", "overlay.png"])
                    else:
                        self.assertIsInstance(args["image"], Image.Image)
                        self.assertEqual(args["image"].getpixel((20, 20)), (255, 0, 0))
                        self.assertEqual(args["prompt"], "Add a vase.")
                    self.assertEqual((args["width"], args["height"]), self.original.size)
                    self.assertEqual(args["num_inference_steps"], 4 if variant == "distill" else 12)
                    self.assertEqual(args["guidance_scale"], 1.0 if variant == "distill" else 2.0)
                    self.assertEqual(config["effective_prompt"], args["prompt"])
                    self.assertEqual(config["run"]["reference_count"], 2 if mode == "multi_reference" else 1)
                    self.assertEqual(config["run"]["peak_vram_allocated_mb"], 1.0)
                    self.assertEqual(result.size, self.original.size)
                    self.assertEqual(visual.width, 3 * 96 + 4 * 16)
                    filenames = ("original.png", "overlay.png", "result.png", "visual.png", "config.json", "metadata.jsonl")
                    for filename in filenames:
                        self.assertTrue((Path(saved) / filename).is_file())
                    self.assertEqual(json.loads((Path(saved) / "config.json").read_text()), config)
                    row = json.loads((Path(saved) / "metadata.jsonl").read_text(encoding="utf-8"))
                    self.assertEqual(row["reference_file_names"], config["reference_order"])
                    self.assertEqual(row["prompt"], "Add a vase.")
                    self.assertEqual(row["effective_prompt"], args["prompt"])
                    for key in ("original_file_name", "overlay_file_name", "result_file_name", "visual_file_name"):
                        self.assertTrue((Path(saved) / row[key]).is_file())
                    with zipfile.ZipFile(archive) as bundle:
                        self.assertEqual(set(bundle.namelist()), set(filenames))
                        self.assertIsNone(bundle.testzip())
                        self.assertEqual(json.loads(bundle.read("metadata.jsonl")), row)

    def test_size_and_invalid_inputs(self):
        self.assertEqual(flux.model_input_size(Image.new("RGB", (2048, 1024))), (1440, 720))
        self.assertEqual(flux.model_input_size(Image.new("RGB", (101, 67))), (96, 64))
        with self.assertRaises(ValueError):
            flux.ModelManager().edit("distill", [], "test", 42)
        with self.assertRaises(ValueError):
            editor.effective_prompt("test", "unknown")


if __name__ == "__main__":
    unittest.main()
