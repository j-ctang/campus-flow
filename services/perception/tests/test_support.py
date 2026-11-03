import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'concentration'))
from support import raster_support

class SupportTests(unittest.TestCase):
    def test_supported_boundary_does_not_fill_interior_hole(self):
        # Four strips form a ring around an unsupported center.
        vertices=[];faces=[]
        for x0,y0,x1,y1 in [(0,0,4,1),(0,3,4,4),(0,1,1,3),(3,1,4,3)]:
            n=len(vertices);vertices += [[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0]]
            faces += [[n,n+1,n+2],[n,n+2,n+3]]
        mask=raster_support(np.array(vertices),np.array(faces),[[0,0],[4,4]],41)
        self.assertTrue(mask[5,5]);self.assertFalse(mask[20,20])
    def test_multiple_elevations_are_excluded(self):
        v=np.array([[0,0,0],[4,0,0],[0,4,0],[0,0,1],[4,0,1],[0,4,1]])
        mask=raster_support(v,np.array([[0,1,2],[3,4,5]]),[[0,0],[4,4]],41)
        self.assertFalse(mask[5,5])

if __name__=='__main__': unittest.main()
