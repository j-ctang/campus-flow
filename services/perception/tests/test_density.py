import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'concentration'))
from density import kernel_grid, average_window, choose_hit

class DensityTests(unittest.TestCase):
    def test_masked_kernel_preserves_observation_mass(self):
        mask=np.zeros((31,31),bool);mask[:,15:]=True
        g=kernel_grid([[15,15]],mask,3)
        self.assertAlmostEqual(float(g.sum()),1)
        self.assertTrue(np.all(g[~mask]==0))
    def test_empty_frames_remain_in_average(self):
        maps=np.array([[[2.]],[[0.]],[[4.]]])
        np.testing.assert_allclose(average_window(maps,[0,.1,.2],0,.2),[[1.]])
    def test_no_coverage_is_not_zero(self):
        with self.assertRaises(ValueError): average_window(np.zeros((1,2,2)),[0],1,2)
    def test_window_end_is_exclusive(self):
        np.testing.assert_allclose(average_window(np.array([[[2.]],[[8.]]]),[0,1],0,1),[[2.]])
    def test_ambiguous_surfaces_rejected(self):
        point,reason=choose_hit([[1,2,3],[1,2,4]],[[0,0,1],[0,0,1]],[5,6])
        self.assertIsNone(point);self.assertEqual(reason,'multiple_surfaces')
    def test_wall_is_not_replaced_by_hidden_ground(self):
        point,reason=choose_hit([[1,2,3],[1,2,4]],[[1,0,0],[0,0,1]],[5,6])
        self.assertIsNone(point);self.assertEqual(reason,'nonhorizontal_surface')
    def test_grazing_layers_at_same_height_accepted(self):
        # Stacked ground layers hit by a shallow ray: far apart along the ray, same height.
        point,reason=choose_hit([[0,0,.27],[.1,0,.265]],[[0,0,1],[0,0,1]],[18.53,18.6])
        self.assertIsNone(reason);self.assertEqual(point,[0,0,.27])
    def test_no_hit_explicit(self):
        self.assertEqual(choose_hit([],[],[]),(None,'no_surface'))

if __name__=='__main__': unittest.main()
