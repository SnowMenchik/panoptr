from dataset_lab.bootstrap import setup
setup()
import os,unittest
os.environ['QT_QPA_PLATFORM']='offscreen'
from pathlib import Path
from PySide6.QtWidgets import QApplication
from main import CameraPanel,InferenceScheduler
from trained_models import drone10_config

class SavedProfileCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_partial_legacy_policy_gets_defaults_before_video_and_inference(self):
        root=Path(__file__).resolve().parent;scheduler=InferenceScheduler();panel=CameraPanel(0,drone10_config(root)['cameras'][0],scheduler,[])
        try:
            panel.policy={'confirm_hits':3,'zone':[[0,0],[1,0],[1,1],[0,1]]}
            self.assertFalse(panel.policy['path_enabled']);self.assertEqual(panel.policy['tracker'],'iou')
            self.assertEqual(panel.policy['confirm_hits'],3);self.assertEqual(panel.policy['roles']['drone'],'aerial_object')
            panel.options();panel.refresh()
            before=panel.policy.copy()
            with self.assertRaises(ValueError):panel.policy={'tracker':'invalid'}
            self.assertEqual(panel.policy,before)
        finally:panel.close();scheduler.close()

    def test_every_noncustom_model_preset_resolves_to_existing_weights(self):
        root=Path(__file__).resolve().parent;scheduler=InferenceScheduler();panel=CameraPanel(0,drone10_config(root)['cameras'][0],scheduler,[])
        try:
            for index in range(panel.model_preset.count()):
                value=str(panel.model_preset.itemData(index) or '')
                if not value:continue
                path=Path(value);path=path if path.is_absolute() else root/path
                self.assertTrue(path.is_file(),(panel.model_preset.itemText(index),str(path)))
            row=next(i for i in range(panel.model_preset.count()) if str(panel.model_preset.itemData(i))=='yolo26s-seg.pt')
            panel.model_preset.setCurrentIndex(row);self.assertEqual(panel.image_size.currentText(),'640');self.assertFalse(panel.running)
        finally:panel.close();scheduler.close()

if __name__=='__main__':unittest.main()
