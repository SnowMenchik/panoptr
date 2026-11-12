"""PowerShell control-flow tests on Linux (mocks, not Windows acceptance tests)."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PWSH = os.environ.get('PWSH', 'pwsh')
HASH = '0EB85C2DFCCCCF1B17352DE4C397F69194035B7D37149EACC16F1147D93DE3B8'

@unittest.skipUnless(os.name == 'posix' and shutil.which(PWSH), 'Requires PowerShell Core on POSIX')
class InstallerFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'Тест с пробелами'
        self.root.mkdir()
        for name in ('install_environment.ps1', 'install_verify.py', 'requirements.txt'):
            shutil.copy(ROOT / name, self.root / name)
        self.base = self.root / 'base.sh'
        self.base.write_text('''#!/bin/sh
if [ "$1" = "--version" ]; then echo Python-mock; exit 0; fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
 mkdir -p "$3/Scripts"; cp "$PANOPTR_MOCK_VENV" "$3/Scripts/python.exe"; chmod +x "$3/Scripts/python.exe"; exit 0
fi
exit 1
''')
        self.base.chmod(0o755)
        self.venv = self.root / 'venv.sh'
        self.venv.write_text('''#!/bin/sh
if [ "$PANOPTR_MOCK_FAIL" = 1 ]; then exit 1; fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 0; fi
case "$1" in *install_verify.py) exit 0;; esac
exit 1
''')
        self.venv.chmod(0o755)
        self.harness = self.root / 'harness.ps1'
        self.harness.write_text('''param([string]$TestRoot,[string]$BaseMock)
function Invoke-WebRequest { param($Uri,$OutFile,[switch]$UseBasicParsing) [IO.File]::WriteAllText($OutFile,'fixture') }
function Get-FileHash { param($LiteralPath,$Algorithm) [pscustomobject]@{Hash=$env:PANOPTR_MOCK_HASH} }
function Expand-Archive { param($LiteralPath,$DestinationPath) $dir=Join-Path $DestinationPath 'tools'; New-Item -ItemType Directory -Force $dir | Out-Null; Copy-Item $BaseMock (Join-Path $dir 'python.exe'); & chmod +x (Join-Path $dir 'python.exe') }
& (Join-Path $TestRoot 'install_environment.ps1')
exit $LASTEXITCODE
''')

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, fail=False, bad_hash=False):
        env = dict(os.environ, PANOPTR_MOCK_VENV=str(self.venv), PANOPTR_MOCK_FAIL='1' if fail else '0', PANOPTR_MOCK_HASH='BAD' if bad_hash else HASH)
        return subprocess.run([PWSH, '-NoProfile', '-File', str(self.harness), '-TestRoot', str(self.root), '-BaseMock', str(self.base)], env=env, capture_output=True, text=True, timeout=30)

    def test_no_py_required_and_failed_reinstall_preserves_active_environment(self):
        first = self.execute(); self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        pointer = self.root / 'data/python-path.txt'; previous = pointer.read_bytes()
        failed = self.execute(fail=True); self.assertEqual(failed.returncode, 1); self.assertEqual(pointer.read_bytes(), previous)
        second = self.execute(); self.assertEqual(second.returncode, 0, second.stdout + second.stderr); self.assertNotEqual(pointer.read_bytes(), previous)

    def test_bad_download_hash_does_not_activate_environment(self):
        result = self.execute(bad_hash=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse((self.root / 'data/python-path.txt').exists())
        self.assertIn('Контрольная сумма', result.stdout)

if __name__ == '__main__':
    unittest.main()
