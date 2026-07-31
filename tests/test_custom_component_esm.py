from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestCustomComponentEsm(unittest.TestCase):
    def test_generated_javascript_is_valid_esm(self):
        node = shutil.which("node")
        self.assertIsNotNone(
            node,
            "Node.js is required for Custom Component ESM validation",
        )

        configs = sorted(ROOT.glob("*/frontend/gradio.config.js"))
        self.assertTrue(configs, "No Gradio Custom Components were discovered")

        checked = []
        for config in configs:
            component_root = config.parent.parent
            template_roots = sorted(
                (component_root / "backend").glob("gradio_*/templates")
            )
            self.assertEqual(
                len(template_roots),
                1,
                f"{component_root.relative_to(ROOT)} must have exactly one "
                "generated templates directory",
            )

            artifacts = sorted(template_roots[0].rglob("*.js"))
            self.assertTrue(
                artifacts,
                f"No generated JavaScript found under "
                f"{template_roots[0].relative_to(ROOT)}",
            )
            for artifact in artifacts:
                result = subprocess.run(
                    [node, "--input-type=module", "--check"],
                    input=artifact.read_bytes(),
                    capture_output=True,
                    check=False,
                    timeout=30,
                )
                checked.append(artifact)
                with self.subTest(artifact=artifact.relative_to(ROOT)):
                    output = (result.stderr or result.stdout).decode(
                        "utf-8",
                        errors="replace",
                    )
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"{artifact.relative_to(ROOT)} is not valid ESM:\n{output}",
                    )

        self.assertTrue(checked, "No generated Custom Component JavaScript was checked")


if __name__ == "__main__":
    unittest.main()
