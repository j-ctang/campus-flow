"""Rasterized mesh-support coverage; holes and multiple elevations stay excluded."""
import numpy as np


def raster_support(vertices, faces, bounds, size, return_height=False):
    low,high=np.asarray(bounds,float)
    points=(vertices[:,:2]-low)/(high-low)*(size-1)
    triangles=points[faces]
    keep=(triangles.max(axis=1)[:,0]>=0)&(triangles.min(axis=1)[:,0]<size)&(triangles.max(axis=1)[:,1]>=0)&(triangles.min(axis=1)[:,1]<size)
    minimum=np.full((size,size),np.inf);maximum=np.full((size,size),-np.inf)
    for face,triangle in zip(faces[keep],triangles[keep]):
        x0,y0=np.maximum(0,np.floor(triangle.min(0)).astype(int));x1,y1=np.minimum(size-1,np.ceil(triangle.max(0)).astype(int))
        if x1<x0 or y1<y0: continue
        a,b,c=triangle
        denominator=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
        if abs(denominator)<1e-8: continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
        u=((xx-a[0])*(c[1]-a[1])-(yy-a[1])*(c[0]-a[0]))/denominator
        v=((b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0]))/denominator
        inside=(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
        z=vertices[face[0],2]+u*(vertices[face[1],2]-vertices[face[0],2])+v*(vertices[face[2],2]-vertices[face[0],2])
        lo=minimum[y0:y1+1,x0:x1+1];hi=maximum[y0:y1+1,x0:x1+1]
        np.minimum(lo,np.where(inside,z,np.inf),out=lo);np.maximum(hi,np.where(inside,z,-np.inf),out=hi)
    present=np.isfinite(minimum)
    span=np.zeros_like(minimum);np.subtract(maximum,minimum,out=span,where=present)
    valid=present&(span<=.05)
    return (valid,minimum) if return_height else valid
