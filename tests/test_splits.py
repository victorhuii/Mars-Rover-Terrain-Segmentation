import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from marsterrain.splits import group_records, scene_key


def row(name, scene, digest, h, split='train', labels=True):
    return {'id': name, 'scene': scene, 'byte_sha256': digest, 'pixel_sha256': digest,
            'dhash': f'{h:016x}', 'source_split': split, 'has_labels': labels}


class SplitsTest(unittest.TestCase):
    def test_scene_key_and_reject_unknown(self):
        self.assertEqual(scene_key('NLA_412496126EDR_F0060000NCAM00320M1.JPG'), 'F0060000')
        with self.assertRaises(ValueError): scene_key('unknown.jpg')

    def test_expert_propagates_through_group_and_near_duplicate(self):
        rows = [row('expert', 'a', 'a', 0, 'test_min3'), row('near', 'b', 'b', 1),
                row('related', 'b', 'c', 0xffffffffffffffff), row('other', 'c', 'd', 0xaaaaaaaaaaaaaaaa)]
        group_records(rows)
        self.assertEqual(rows[0]['assigned_split'], 'test_min3')
        self.assertEqual([r['assigned_split'] for r in rows[1:3]], ['excluded_expert_related'] * 2)
        self.assertIn(rows[3]['assigned_split'], ('train', 'val'))

    def test_corrupt_images_do_not_join_through_missing_hashes(self):
        first = row('broken-a', 'a', 'first', 0)
        second = row('broken-b', 'b', 'second', 1)
        for r in (first, second):
            r['pixel_sha256'] = None
            r['dhash'] = None
            r['image_error'] = 'UnidentifiedImageError'
        group_records([first, second])
        self.assertNotEqual(first['component'], second['component'])
        self.assertEqual(first['assigned_split'], 'excluded_corrupt')
        self.assertEqual(second['assigned_split'], 'excluded_corrupt')

    def test_exact_hash_and_stable_order(self):
        a = [row('a', 'a', 'same', 0), row('b', 'b', 'same', 0xffffffffffffffff),
             row('c', 'c', 'c', 0xaaaaaaaaaaaaaaaa, labels=False)]
        b = [dict(r) for r in reversed(a)]
        group_records(a); group_records(b)
        self.assertEqual(a[0]['component'], a[1]['component'])
        self.assertEqual({r['id']: r['assigned_split'] for r in a}, {r['id']: r['assigned_split'] for r in b})
        self.assertEqual(a[2]['assigned_split'], 'excluded_unlabelled')

if __name__ == '__main__': unittest.main()
