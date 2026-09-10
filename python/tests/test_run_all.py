"""Exercise batch acceptance exit/log semantics without loading Julia."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[2] / "benchmarks" / "run_all.sh"


class BatchRunnerTests(unittest.TestCase):
    def run_batch(self, root, names=(), fail_float32=False):
        fixtures = root / "fixture inputs"
        fixtures.mkdir()
        for name in names:
            (fixtures / name).touch()
        bindir = root / "bin"
        bindir.mkdir()
        shim = bindir / "julia"
        shim.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$3" >> "$CALL_LOG"\n'
            'echo "Warning: preserve the full diagnostic"\n'
            'echo "  @ failed_location.jl:12"\n'
            'if [ "$FAIL_FLOAT32" = 1 ] && [[ "$3" == *_float32.h5 ]]; then\n'
            '  echo "NOT ACCEPTED"; exit 9\n'
            'fi\n'
            'echo "ACCEPTED"\n'
        )
        shim.chmod(0o755)
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                   CALL_LOG=str(root / "calls.log"), FAIL_FLOAT32=str(int(fail_float32)))
        return subprocess.run(["bash", str(RUNNER), str(fixtures)], cwd=root, env=env,
                              text=True, capture_output=True, timeout=10)

    def test_empty_directory_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_batch(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no acceptance checks ran", result.stderr)
            self.assertFalse((root / "calls.log").exists())

    def test_missing_directory_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(["bash", str(RUNNER), str(Path(tmp) / "missing")],
                                    cwd=tmp, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No eval fixtures", result.stderr)

    def test_success_counts_eval_fixtures_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_batch(root, ("column_mlp_tiny_float64.h5", "column_mlp_tiny_float32.h5",
                                           "column_mlp_tiny_float64_train.h5"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Ran 2 eval fixtures; 0 not accepted", result.stdout)
            self.assertEqual(len((root / "calls.log").read_text().splitlines()), 2)
            log = (root / "benchmarks/results/column_mlp_tiny_float64.log").read_text()
            self.assertIn("Warning: preserve the full diagnostic", log)
            self.assertIn("@ failed_location.jl:12", log)

    def test_failure_propagates_and_remaining_fixtures_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ("column_mlp_tiny_float64.h5", "column_mlp_a_float32.h5", "column_mlp_z_float32.h5")
            result = self.run_batch(root, names, fail_float32=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Ran 3 eval fixtures; 2 not accepted", result.stdout)
            self.assertEqual(len((root / "calls.log").read_text().splitlines()), 3)

    def test_log_write_failure_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmarks/results/column_mlp_tiny_float64.log").mkdir(parents=True)
            result = self.run_batch(root, ("column_mlp_tiny_float64.h5",))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Ran 1 eval fixtures; 1 not accepted", result.stdout)


if __name__ == "__main__":
    unittest.main()
