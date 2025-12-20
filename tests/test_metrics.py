import unittest
import numpy as np
import datetime
from src.core import ArtistCareer, Track, StyleEmbedding
from src.metrics import MusicMetrics

class TestMusicMetrics(unittest.TestCase):
    def setUp(self):
        self.career = ArtistCareer(artist_name="Test Artist")
        
        # Create mock tracks
        t1 = Track(file_path="p1.mp3", title="T1", album="A1", release_date=datetime.date(2020, 1, 1))
        t2 = Track(file_path="p2.mp3", title="T2", album="A1", release_date=datetime.date(2020, 1, 2))
        t3 = Track(file_path="p3.mp3", title="T3", album="A2", release_date=datetime.date(2021, 1, 1))
        
        self.career.add_track(t1)
        self.career.add_track(t2)
        self.career.add_track(t3)
        
        # Create mock embeddings (random but stable for test)
        # Use 512-dim as expected
        v1 = np.zeros(512); v1[0] = 1.0
        v2 = np.zeros(512); v2[0] = 0.8; v2[1] = 0.2
        v3 = np.zeros(512); v3[511] = 1.0 # Very different
        
        self.career.add_embedding(StyleEmbedding(vector=v1, track_ref=t1))
        self.career.add_embedding(StyleEmbedding(vector=v2, track_ref=t2))
        self.career.add_embedding(StyleEmbedding(vector=v3, track_ref=t3))

    def test_album_centroids(self):
        names, centroids = MusicMetrics.get_album_centroids(self.career)
        self.assertEqual(len(names), 2)
        self.assertEqual(names[0], "A1")
        self.assertEqual(names[1], "A2")
        
        # A1 centroid should be mean of v1 and v2
        expected_a1 = (np.zeros(512))
        expected_a1[0] = 0.9
        expected_a1[1] = 0.1
        np.testing.assert_array_almost_equal(centroids[0], expected_a1)

    def test_style_velocity(self):
        velocities = MusicMetrics.calculate_style_velocity(self.career)
        self.assertIn("A2", velocities)
        # Velocity between A1 (mostly index 0) and A2 (index 511) should be high
        self.assertGreater(velocities["A2"], 0.5)

    def test_cohesion(self):
        cohesion = MusicMetrics.calculate_cohesion(self.career)
        self.assertIn("A1", cohesion)
        self.assertIn("A2", cohesion)
        # A1 has two different tracks, cohesion should be < 1.0
        self.assertLess(cohesion["A1"], 1.0)
        # A2 has one track, cohesion should be 1.0 (mean dist from centroid is 0)
        self.assertEqual(cohesion["A2"], 1.0)

if __name__ == '__main__':
    unittest.main()
