"""Offline throwaway concentration experiment. No network or inference path."""
import os
# Resource limits come first: thread caps only apply if set before NumPy loads.
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_name,'2')
import argparse
import resource
import sys
import threading
import time
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from density import kernel_grid, average_window, choose_hit
from support import raster_support
from view_crop import faces_in_view


def digest(path):
    h=sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def start_memory_guard(limit_mb):
    """Stop the whole process before it can exhaust the machine's memory."""
    # macOS reports ru_maxrss in bytes; Linux reports kilobytes.
    unit=1 if sys.platform=='darwin' else 1024
    def watch():
        while True:
            used=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*unit/2**20
            if used>limit_mb:
                print(f'memory guard: {used:.0f} MB exceeds {limit_mb} MB limit; stopping',file=sys.stderr,flush=True)
                os._exit(3)
            time.sleep(.25)
    threading.Thread(target=watch,daemon=True).start()


def project_pixels(pixels, calibration, mesh, chunk=256):
    c=calibration
    K,R,T,D=map(np.array,(c['K'],c['R'],c['T'],c['dist']))
    t=T.ravel(); center=-R.T@t
    native=np.array(pixels,float)*[1920/852,1080/480]
    norm=cv2.undistortPoints(native[:,None,:],K,D,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
    rays=np.column_stack([norm,np.ones(len(norm))])@R
    rays/=np.linalg.norm(rays,axis=1)[:,None]
    # Lossless crop: drop triangles no selected ray can reach before building
    # the ray index, so the full campus mesh is never indexed.
    keep=faces_in_view(np.asarray(mesh.vertices),np.asarray(mesh.faces),R,t,[norm.min(0),norm.max(0)])
    full_faces=len(mesh.faces)
    mesh=mesh.submesh([np.flatnonzero(keep)],append=True)
    print(f'view crop: {len(mesh.faces)}/{full_faces} faces kept',flush=True)
    result=[]
    # Small ray batches bound peak memory from multiple-hit intersection lists.
    for begin in range(0,len(rays),chunk):
        part=rays[begin:begin+chunk]
        locations,ids,triangles=mesh.ray.intersects_location(np.tile(center,(len(part),1)),part,multiple_hits=True)
        for i in range(len(part)):
            result.append(choose_ray(locations[ids==i],mesh.face_normals[triangles[ids==i]],R,t))
        print(f'rays {min(begin+chunk,len(rays))}/{len(rays)}',flush=True)
    return result


def choose_ray(hits,normals,R,t):
    depths=(hits@R.T+t)[:,2]
    point,reason=choose_hit(hits,normals,depths)
    return {'world':point,'rejection':reason}


def grid_transform(points,bounds,size):
    points=np.asarray(points,float)
    low,high=np.array(bounds)
    return (points[:,:2]-low)/(high-low)*(size-1)


def write_image(path,heat,mask,scale,points,title,subtitle,bandwidth):
    display=np.clip(heat/scale,0,1)
    color=cv2.applyColorMap(np.uint8(display*255),cv2.COLORMAP_INFERNO)
    color[~mask]=(105,105,105)
    image=cv2.resize(np.flipud(color),(700,700),interpolation=cv2.INTER_NEAREST)
    # Top-down local X increases right; local Y increases upward.
    canvas=np.full((900,1100,3),245,np.uint8);canvas[130:830,30:730]=image
    cv2.putText(canvas,title,(30,35),cv2.FONT_HERSHEY_SIMPLEX,.72,(25,25,25),2)
    cv2.putText(canvas,subtitle,(30,67),cv2.FONT_HERSHEY_SIMPLEX,.5,(45,45,45),1)
    cv2.putText(canvas,'EXPERIMENTAL - alignment not independently validated',(30,99),cv2.FONT_HERSHEY_SIMPLEX,.53,(20,20,190),1)
    for x,y in points:
        px=30+int(x/(mask.shape[1]-1)*699);py=130+699-int(y/(mask.shape[0]-1)*699)
        cv2.circle(canvas,(px,py),5,(255,255,255),1)
    for i,line in enumerate(['Relative observed','concentration','', 'Common color scale', 'across both clips',f'Kernel: {bandwidth:g} grid cells','', 'Gray: excluded /','unobserved region','', 'White: exact-frame','mapped observations','', 'Not people per m2','No unique-person count','No confidence score']):
        cv2.putText(canvas,line,(760,165+i*28),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,40,40),1)
    for j in range(160):
        col=cv2.applyColorMap(np.uint8([[255-j*255/159]]),cv2.COLORMAP_INFERNO)[0,0].tolist()
        cv2.line(canvas,(1045,170+j),(1065,170+j),col,1)
    cv2.putText(canvas,'Higher',(995,150),cv2.FONT_HERSHEY_SIMPLEX,.4,(40,40,40),1)
    cv2.putText(canvas,'Zero',(1010,350),cv2.FONT_HERSHEY_SIMPLEX,.4,(40,40,40),1)
    cv2.putText(canvas,'Local scene X ->     Local scene Y ^     Ground footprint; scale provisional',(30,865),cv2.FONT_HERSHEY_SIMPLEX,.48,(40,40,40),1)
    cv2.imwrite(str(path),canvas)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--region',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/concentration'))
    parser.add_argument('--max-memory-mb',type=int,default=1500)
    parser.add_argument('--ray-chunk',type=int,default=16)
    args=parser.parse_args();os.nice(15);start_memory_guard(args.max_memory_mb);root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=True)
    paths={'observations':out/'cached-observations.json','region':args.region,'calibration':root/'data/scout/calibrations/cam_0.json','mesh':root/'data/scout/meshes/mesh_ground_only.ply','timestamps':root/'data/scout/timestamps/timestamps_sequence_1.json','video':root/'data/scout/videos/cam_0_preview.mp4'}
    hashes={k:digest(v) for k,v in paths.items()}
    loaded=[]
    def load_mesh():
        if not loaded: loaded.append(trimesh.load(paths['mesh'],process=False))
        return loaded[0]
    data=json.loads(paths['observations'].read_text());region=json.loads(paths['region'].read_text());c=json.loads(paths['calibration'].read_text());timestamps=json.loads(paths['timestamps'].read_text())
    polygon=np.array(region['pixel_polygon'],np.float32)
    # Only pixels selected by the conservative region are submitted for mapping.
    candidates=[];samples=[]
    for batch in data['batches']:
        if len(batch['frames'])!=100 or batch['fps']!=10: raise ValueError('Expected complete 100-frame 10fps cached clips')
        for frame in batch['frames']:
            frame['capture_timestamp']=timestamps[str(frame['source_frame'])]['cvlabrpi10']
            for obs in frame['observations']:
                obs['world']=None
                if obs['quality_flags']: obs['rejection']='observation_quality'
                elif cv2.pointPolygonTest(polygon,tuple(obs['foot_pixel']),False)<0: obs['rejection']='outside_region'
                else: candidates.append(obs); samples.append(obs['foot_pixel'])
    # Dense image boundary samples: a curved world footprint cannot be assumed straight.
    boundary=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        for u in np.linspace(0,1,16,endpoint=False): boundary.append((a*(1-u)+b*u).tolist())
    cache=out/'projection-cache.json';key=sha256(json.dumps({'hashes':hashes,'pixels':samples+boundary,'policy':'nearest_positive_horizontal0.9_zspan0.05-v2'},sort_keys=True).encode()).hexdigest()
    if cache.exists() and (cached:=json.loads(cache.read_text())).get('key')==key: mapped=cached['mapped']
    else:
        mesh=load_mesh()
        mapped=project_pixels(samples+boundary,c,mesh,args.ray_chunk)
        tmp=cache.with_suffix('.tmp');tmp.write_text(json.dumps({'key':key,'mapped':mapped}));tmp.replace(cache)
    for obs,position in zip(candidates,mapped): obs.update(position)
    boundary_map=mapped[len(samples):]
    if any(r['world'] is None for r in boundary_map): raise ValueError('Region boundary not uniquely supported by near-horizontal mesh; no heatmap produced')
    world_polygon=np.array([r['world'][:2] for r in boundary_map]);low=world_polygon.min(0);high=world_polygon.max(0)
    center=(low+high)/2; half=float(np.max(high-low))*.58
    bounds=[(center-half).tolist(),(center+half).tolist()];size=160
    mask=np.zeros((size,size),np.uint8);polygrid=grid_transform(world_polygon,bounds,size)
    cv2.fillPoly(mask,[np.rint(polygrid).astype(np.int32)],1);mask=mask.astype(bool)
    # A supported boundary cannot establish interior coverage: rasterize actual
    # horizontal mesh triangles and reject holes/multiple elevations in each cell.
    support_key=sha256(json.dumps({'mesh':hashes['mesh'],'bounds':bounds,'size':size,'policy':'raster-horizontal0.9-span0.05-v1'},sort_keys=True).encode()).hexdigest()
    support_path=out/'support-cache.npz'
    supported=None
    if support_path.exists():
        with np.load(support_path) as stored:
            if str(stored['key'])==support_key: supported=stored['mask'];height=stored['height']
    if supported is None:
        mesh=load_mesh()
        horizontal=np.abs(mesh.face_normals[:,2])>=.9
        supported,height=raster_support(np.asarray(mesh.vertices),np.asarray(mesh.faces)[horizontal],bounds,size,return_height=True)
        np.savez(support_path,key=support_key,mask=supported,height=height)
    mask &= supported
    # Check camera-region inclusion at actual support heights rather than relying
    # only on the straight polygon traced in world XY.
    yy,xx=np.indices(mask.shape);low,high=np.asarray(bounds)
    worldxy=np.stack([xx,yy],axis=-1)/(size-1)*(high-low)+low
    xyz=np.column_stack([worldxy[mask],height[mask]])
    native=cv2.projectPoints(xyz,cv2.Rodrigues(np.array(c['R']))[0],np.array(c['T']).ravel(),np.array(c['K']),np.array(c['dist']))[0].reshape(-1,2)*[852/1920,480/1080]
    included=np.array([cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in native])
    keep=np.zeros_like(mask);keep[mask]=included;mask=keep
    if not np.any(mask): raise ValueError('No eligible interior support cells')
    context=np.full((size,size,3),105,np.uint8)
    present=np.isfinite(height)
    if np.any(present):
        zmin,zmax=np.percentile(height[present],[2,98]);gray=np.zeros_like(height)
        gray[present]=60+160*np.clip((height[present]-zmin)/max(zmax-zmin,1e-6),0,1)
        context[present]=np.repeat(gray[present,None],3,axis=1).astype(np.uint8)
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(context,contours,-1,(0,220,220),1)
    context=cv2.resize(np.flipud(context),(700,700),interpolation=cv2.INTER_NEAREST)
    cv2.putText(context,'Top-down mesh support | yellow: selected upper landing',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,0,200),1)
    cv2.putText(context,'Provisional local XY; shading = relative surface height',(10,680),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,0,200),1)
    cv2.imwrite(str(out/'topdown-scene.png'),context)
    products=[];summary=[];sensitivity=[];capture=cv2.VideoCapture(str(paths['video']))
    for batch in data['batches']:
        points=[];rejections=Counter();accepted=0
        for frame in batch['frames']:
            eligible=[]
            for obs in frame['observations']:
                if obs['world'] is not None:
                    q=grid_transform([obs['world']],bounds,size)[0];x,y=np.rint(q).astype(int)
                    if not(0<=x<size and 0<=y<size and mask[y,x]): obs.update(world=None,rejection='outside_world_footprint')
                    else: eligible.append(q.tolist());accepted+=1
                if obs.get('rejection'): rejections[obs['rejection']]+=1
            points.append(eligible)
        times=[f['clip_frame']/10 for f in batch['frames']]
        for sigma in [3.,5.,8.]:
            grids=np.array([kernel_grid(p,mask,sigma) for p in points])
            for name,start,end in [('exact',5,5.1),('mean1s',5,6),('mean10s',0,10)]:
                heat=average_window(grids,times,start,end)
                products.append({'batch':batch['batch_id'],'name':name,'sigma':sigma,'heat':heat,'points':points[50] if name=='exact' else [],'start':start,'end':end,'source':batch['source_start_frame']/10})
            means=grids.mean(axis=0)
            sensitivity.append({'batch':batch['batch_id'],'sigma_grid_cells':sigma,'peak_grid_xy':list(map(int,np.unravel_index(means.argmax(),means.shape)[::-1])),'mean_accepted_observations':float(means.sum())})
        summary.append({'batch':batch['batch_id'],'frames':len(points),'observations_total':sum(len(f['observations']) for f in batch['frames']),'accepted_observation_frames':accepted,'mean_accepted_per_frame':accepted/100,'rejections':dict(rejections),'preview_interval_s':[batch['source_start_frame']/10,(batch['source_start_frame']+100)/10],'capture_span_s':batch['frames'][-1]['capture_timestamp']-batch['frames'][0]['capture_timestamp']})
        for i in [0,50,90]:
            frame=batch['frames'][i];capture.set(cv2.CAP_PROP_POS_FRAMES,frame['source_frame']);ok,img=capture.read()
            if not ok: raise ValueError('Source frame unavailable')
            cv2.polylines(img,[polygon.astype(np.int32)],True,(0,255,255),1)
            for obs in frame['observations']:
                x,y=map(lambda v:int(round(v)),obs['foot_pixel']);color=(0,255,0) if obs['world'] is not None else (100,100,230)
                cv2.circle(img,(x,y),3,color,-1)
            cv2.putText(img,f"frame {frame['source_frame']} | green: provisional accepted; red: excluded",(10,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,255),1)
            cv2.imwrite(str(out/f"{batch['batch_id']}-frame-{i}.png"),img)
    capture.release()
    scale=max(float(p['heat'].max()) for p in products)
    if scale==0: scale=1.
    for p in products:
        title=f"{p['batch']} | {p['name']} | observed concentration"
        subtitle=f"Preview {p['source']+p['start']:.1f}s" if p['name']=='exact' else f"Preview [{p['source']+p['start']:.1f}, {p['source']+p['end']:.1f})s | mean occupancy"
        write_image(out/f"{p['batch']}-{p['name']}-sigma{p['sigma']:g}.png",p['heat'],mask,scale,p['points'],title,subtitle,p['sigma'])
    report={'status':'experimental_only_not_validated','reason':'Independent metric controls unavailable; heuristic feet and incomplete detections','source_hashes':hashes,'region':region,'bounds_scene_xy':bounds,'grid_size':size,'common_color_scale':scale,'time_basis':'uniform preview frames, 10fps; official capture timestamps retained separately','kernel':'unit mass per accepted observation, normalized within footprint; arbitrary grid units, not people/m2','batches':summary,'smoothing_sensitivity':sensitivity}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'mapped-observations.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
