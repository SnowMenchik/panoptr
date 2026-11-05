"""Launch profile switches are explicit; opening a window never starts capture."""
from dataset_lab.bootstrap import setup
setup()
import unittest
from unittest.mock import Mock,patch

class LaunchChecks(unittest.TestCase):
    def test_explicit_gpu_launch_and_default_cpu_profile(self):
        import launch_final
        for flag,expected in [([], 'CPU'),(['--gpu'],'GPU')]:
            config={'cameras':[{'device':'Auto'},{'device':'CPU'}]}
            app=Mock();app.exec.return_value=0
            with patch.object(launch_final.sys,'argv',['launch_final.py',*flag]),patch.object(launch_final,'load_config',return_value=config),\
                 patch.object(launch_final,'QApplication',return_value=app),patch.object(launch_final,'configure_font'),\
                 patch.object(launch_final,'MainWindow') as window:
                self.assertEqual(launch_final.launch(),0)
                self.assertTrue(all(c['device']==expected for c in window.call_args.args[0]['cameras']))
                window.return_value.show.assert_called_once()
                window.return_value._start_all.assert_not_called()

if __name__=='__main__':unittest.main()
