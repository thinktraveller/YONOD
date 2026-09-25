"""Queue orchestration tests; never launch a model."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from _verify import run_ablation2_batch as batch


class QueueTests(unittest.TestCase):
    def exercise(self, fail=False, existing=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config').mkdir()
            configs, tasks = [], []
            for name in ['one', 'two']:
                path = root / 'config' / (name + '.yaml')
                path.write_text(name)
                tasks.append(dict(task=name, config=str(path.relative_to(root)), sha256=batch.sha(path)))
                configs.append(SimpleNamespace(outputs_root=root / 'result' / name))
            if existing:
                configs[0].outputs_root.mkdir(parents=True)
                (configs[0].outputs_root / 'marker').touch()
            events = []

            class Child:
                pid = 123
                returncode = 0

                def communicate(self, value):
                    events.append(('run', Path(value.strip()).stem))

            def audit(config):
                events.append(('audit', config.outputs_root.name))
                if fail:
                    raise RuntimeError('incomplete predictions')

            with patch.object(batch, 'ROOT', root), patch.object(batch, 'preflight', return_value=(dict(tasks=tasks, inputs={}), configs)), patch.object(batch, 'audit', side_effect=audit), patch.object(batch.subprocess, 'Popen', side_effect=lambda *a, **k: Child()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                        batch.run()
                else:
                    batch.run()
            self.assertFalse((root / 'logs' / 'ablation2_remaining59.lock').exists())
            return events

    def test_strict_serial_order(self):
        self.assertEqual(self.exercise(), [('run', 'one'), ('audit', 'one'), ('run', 'two'), ('audit', 'two')])

    def test_failed_validation_stops_next_task(self):
        self.assertEqual(self.exercise(fail=True), [('run', 'one'), ('audit', 'one')])

    def test_restart_skips_only_audited_completion(self):
        self.assertEqual(self.exercise(existing=True), [('audit', 'one'), ('run', 'two'), ('audit', 'two')])

    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / 'logs' / 'ablation2_remaining59.lock'
            lock.mkdir(parents=True)
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'preflight') as check:
                with self.assertRaises(FileExistsError):
                    batch.run()
                check.assert_not_called()
            self.assertTrue(lock.exists())
