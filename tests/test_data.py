from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nephrotox.data import load_data, rows_fingerprint


class SourceTests(unittest.TestCase):
    def test_only_frozen_data_has_expected_records(self):
        rows, info = load_data()
        self.assertEqual((len(rows), len(rows[0])), (1673, 12))
        self.assertEqual(info['label_counts'], {'0': 831, '1': 842})
        self.assertEqual(rows_fingerprint(rows), info['rows_fingerprint'])

    def test_changed_source_file_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'DATA.xlsx'
            path.write_bytes(b'changed-source')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_data(path)

    def test_label_exchange_changes_loaded_record_fingerprint(self):
        records = [{'compound_id': 'a', 'label': 0}, {'compound_id': 'b', 'label': 1}]
        changed = [{'compound_id': 'a', 'label': 1}, {'compound_id': 'b', 'label': 0}]
        self.assertNotEqual(rows_fingerprint(records), rows_fingerprint(changed))


if __name__ == '__main__':
    unittest.main()
