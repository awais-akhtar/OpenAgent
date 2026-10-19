import unittest

from openagent.adaptive_engine.platform_certification.command_adapter import (
    _require_file_backed_runner,
)


class PinnedCommandSourceTests(unittest.TestCase):
    def test_interpreter_inline_and_module_entrypoints_are_rejected(self):
        commands = (
            ['python', '-c', 'print("un-pinned")'],
            ['python3', '-m', 'un-pinned.module'],
            ['node', '--eval', 'process.exit(0)'],
            ['pwsh', '-EncodedCommand', 'un-pinned'],
            ['bash', '-c', 'true'],
        )
        for command in commands:
            with self.subTest(command=command), self.assertRaisesRegex(
                    ValueError, 'inline or module runner code cannot be source-pinned'):
                _require_file_backed_runner(command)

    def test_file_backed_command_is_not_rejected_by_entrypoint_guard(self):
        _require_file_backed_runner(['python', 'tools/run_decoder.py'])
        _require_file_backed_runner(['decoder-runner.exe', '--device', '0'])


if __name__ == '__main__':
    unittest.main()
