import json
from pathlib import Path
import tempfile
import unittest

import prepare_dataset as prepare


class ConversionTest(unittest.TestCase):
    def test_yolo_uses_actual_image_dimensions_and_rounds_four_places(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.txt'
            path.write_text('1 0.5515625 0.20234375 0.14140625 0.0734375\n')
            rows, empty = prepare.yolo_annotations(path, 588, 441)
            self.assertEqual(rows[0]['bbox'], [282.7453, 73.0406, 83.1469, 32.3859])
            self.assertEqual(rows[0]['area'], round(83.1469 * 32.3859, 4))
            self.assertEqual(empty, 0)

    def test_yolo_rejects_unknown_class(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.txt'; path.write_text('5 0.5 0.5 0.1 0.1')
            with self.assertRaises(ValueError):
                prepare.yolo_annotations(path, 640, 640)

    def test_yolo_rejects_nonfinite_and_polygon(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.txt'
            for line in ('0 nan 0.5 0.1 0.1', '0 0.1 0.2 0.3 0.4 0.5 0.6'):
                path.write_text(line)
                with self.assertRaises(ValueError):
                    prepare.yolo_annotations(path, 640, 640)

    def test_voc_coordinate_convention_and_empty_nodes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.xml'
            path.write_text('<annotation><size><width>100</width><height>80</height></size><object/>'
                            '<object><name>tile</name><difficult>1</difficult><bndbox><xmin>11</xmin><ymin>21</ymin>'
                            '<xmax>41</xmax><ymax>51</ymax></bndbox></object></annotation>')
            rows, empty = prepare.voc_annotations(path, 100, 80)
            self.assertEqual(rows[0]['bbox'], [10.0, 20.0, 30.0, 30.0])
            self.assertEqual(rows[0]['area'], 900.0)
            self.assertEqual(empty, 1)
            self.assertEqual(rows[0]['iscrowd'], 0)
            with self.assertRaises(ValueError):
                prepare.voc_annotations(path, 99, 80)

    def test_voc_incomplete_object_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'example.xml'
            path.write_text('<annotation><size><width>100</width><height>80</height></size>'
                            '<object><name>brick</name></object></annotation>')
            with self.assertRaises(ValueError):
                prepare.voc_annotations(path, 100, 80)

    def test_boxes_not_silently_clipped(self):
        for box in ([-1, 0, 10, 10], [99, 0, 2, 10], [0, 0, 0, 10]):
            with self.assertRaises(ValueError):
                prepare.validate_box(box, 100, 80, 'unit test')

    def test_raw_path_cannot_escape_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with self.assertRaises(ValueError):
                prepare.raw_path(root, '../missing.jpg')

    def test_original_json_serialization(self):
        payload = dict(info={}, licenses=[], images=[], annotations=[], categories=[])
        self.assertEqual(prepare.encoded('cdw1630', payload), json.dumps(payload).encode())
        self.assertEqual(prepare.encoded('btc', payload), (json.dumps(payload, indent=2) + '\n').encode())


if __name__ == '__main__':
    unittest.main()
