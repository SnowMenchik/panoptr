"""CPU triangle z-buffer. Depth is perspective-correct, independent of mesh order."""
import numpy as np
from stand_geometry import project

def raster(meshes,eye,target,width,height,brightness=1):
    rgb=np.empty((height,width,3),np.uint8); rgb[:]=[17,30,41]; zbuffer=np.full((height,width),np.inf)
    for m in meshes:
        pixels,depth=project(m.vertices,eye,target,width,height)
        if pixels is None: continue
        for face in m.faces:
            world=m.vertices[face]; normal=np.cross(world[1]-world[0],world[2]-world[0]); length=np.linalg.norm(normal)
            if length and normal@(np.asarray(eye)-world.mean(axis=0))<=0: continue
            light=.55+.45*abs(float(normal@[.3,.3,.9]))/length if length else .65
            color=np.clip(np.array(m.color)*light*brightness,0,255).astype(np.uint8)
            for i in range(1,len(face)-1):
                ids=[face[0],face[i],face[i+1]]; pts=pixels[ids]; z=depth[ids]
                x0=max(0,int(np.floor(pts[:,0].min()))); x1=min(width-1,int(np.ceil(pts[:,0].max())))
                y0=max(0,int(np.floor(pts[:,1].min()))); y1=min(height-1,int(np.ceil(pts[:,1].max())))
                if x0>x1 or y0>y1: continue
                ax,ay=pts[0]; bx,by=pts[1]; cx,cy=pts[2]; den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
                if abs(den)<1e-9: continue
                xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
                a=((by-cy)*(xx-cx)+(cx-bx)*(yy-cy))/den; b=((cy-ay)*(xx-cx)+(ax-cx)*(yy-cy))/den; c=1-a-b
                inside=(a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7); inverse=a/z[0]+b/z[1]+c/z[2]
                new_depth=np.divide(1.,inverse,out=np.full_like(inverse,np.inf),where=inverse>0)
                area=zbuffer[y0:y1+1,x0:x1+1]; update=inside&(new_depth<area)
                area[update]=new_depth[update]; rgb[y0:y1+1,x0:x1+1][update]=color
    return rgb
