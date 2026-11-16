import argparse
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1]
LABELS = {2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}

def cli(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default='config/settings.json', help='Settings JSON')
    parser.add_argument('--video', help='Override input with an MP4/video or webcam index 0')
    parser.add_argument('--image', help='Override input with a JPG/PNG image')
    parser.add_argument('--output', help='Optional video output path')
    parser.add_argument('--no-display', action='store_true', help='Run without a video window')
    return parser.parse_args()

def path(p):
    p = Path(p)
    return p if p.is_absolute() else BASE / p

def settings(args):
    with open(path(args.config), encoding='utf-8') as f:
        config = json.load(f)
    if args.video is not None:
        config['video'] = args.video
        config['input'] = args.video
    if args.image is not None:
        config['image'] = args.image
        config['input'] = args.image
    return config

def model(cfg, pose=False):
    return YOLO(cfg['pose_model' if pose else 'model'])

class SingleImageCapture:
    """One real image frame (no fabricated motion or timestamps)."""
    is_image = True
    def __init__(self, filename):
        self.filename = str(filename)
        self.image = cv2.imread(self.filename)
        if self.image is None:
            raise FileNotFoundError(f'Cannot read traffic image: {self.filename}')
        self.done = False
    def isOpened(self): return True
    def read(self):
        if self.done: return False, None
        self.done = True
        return True, self.image.copy()
    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS: return 1
        if prop == cv2.CAP_PROP_FRAME_WIDTH: return self.image.shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self.image.shape[0]
        return 0
    def release(self): pass

def open_video(cfg):
    """Open default single image, or an explicit video/camera source."""
    src = str(cfg.get('input') or cfg.get('image') or 'data/traffic.jpg')
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        candidate = path(src)
        if candidate.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp'):
            return SingleImageCapture(candidate)
        cap = cv2.VideoCapture(str(candidate))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open input: {src}. Use data/traffic.jpg or --video PATH')
    return cap

def image_mode(cap):
    return isinstance(cap, SingleImageCapture)

def image_notice(frame, message='Needs video frames to measure motion'):
    # Dark label strip; avoids inventing measurements from one photograph.
    h,w=frame.shape[:2]
    cv2.rectangle(frame,(0,max(0,h-48)),(w,h),(25,25,25),-1)
    scale=min(0.55, max(0.32,w/850))
    cv2.putText(frame,message[:80],(8,h-17),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)

def frames(cap):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if fps < 1 or fps > 240: fps = 30
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        timestamp = i / fps
        yield i, timestamp, frame
        i += 1

def show(frame, args, delay=None):
    # Always save the annotated result; useful in VS Code and headless terminals.
    import sys
    dest = path(args.output) if args.output else BASE / 'outputs' / (Path(sys.argv[0]).stem+'.jpg')
    dest.parent.mkdir(exist_ok=True, parents=True)
    if dest.suffix.lower() in ('.jpg','.jpeg','.png'):
        cv2.imwrite(str(dest), frame)
    if args.no_display: return True
    if delay is None:
        # On still images, wait until keypress (0); on video, advance frame (1ms)
        delay = 1 if getattr(args, 'video', None) else 0
    try:
        cv2.imshow('VisionPort (Q: quit)',frame)
        return (cv2.waitKey(delay) & 0xff) != ord('q')
    except cv2.error:
        print(f'GUI unavailable; annotated result saved to {dest}. Use --no-display.')
        return False

def writer(args, cap):
    if not args.output or image_mode(cap): return None
    dest = path(args.output)
    if dest.suffix.lower() not in ('.mp4','.avi'):
        return None
    dest.parent.mkdir(exist_ok=True, parents=True)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    if not out.isOpened(): raise RuntimeError(f'Cannot write {dest}')
    return out

def finish(cap, out=None):
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

def detections(result, track=False):
    if result.boxes is None: return []
    output=[]
    ids = result.boxes.id
    for idx, b in enumerate(result.boxes):
        item = {'cls':int(b.cls[0]), 'conf':float(b.conf[0]),
                'xyxy':tuple(map(int,b.xyxy[0].tolist())), 'id':None}
        if track and ids is not None: item['id']=int(ids[idx].item())
        x1,y1,x2,y2=item['xyxy']
        item['center']=((x1+x2)//2,(y1+y2)//2)
        item['foot']=((x1+x2)//2,y2)
        output.append(item)
    return output

def predict(m, frame, cfg, track=False, classes=None):
    kwargs={'conf':cfg['confidence'], 'verbose':False,
            'classes':cfg['vehicle_classes'] if classes is None else classes}
    if track: return m.track(frame, persist=True, tracker='bytetrack.yaml', **kwargs)[0]
    return m.predict(frame, **kwargs)[0]

def draw(frame, item, note=''):
    x1,y1,x2,y2=item['xyxy']
    label=LABELS.get(item['cls'],str(item['cls']))
    if item['id'] is not None: label += f' #{item["id"]}'
    label+=f' {item["conf"]:.2f}'
    if note: label+= ' '+note
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,210,50),2)
    cv2.putText(frame,label,(x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),2)

def text(frame, message, row=0, color=(255,255,255)):
    cv2.putText(frame,message,(15,30+row*32),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,2)

def polygon(frame, vertices, color=(0,220,220)):
    if vertices and len(vertices)>=3:
        cv2.polylines(frame,[np.asarray(vertices,dtype=np.int32)],True,color,2)

def in_zone(point, vertices):
    if len(vertices)<3:return False
    return cv2.pointPolygonTest(np.asarray(vertices,dtype=np.int32),tuple(map(float,point)),False)>=0

def cross_side(point, line):
    (ax,ay),(bx,by)=line
    return (bx-ax)*(point[1]-ay)-(by-ay)*(point[0]-ax)

def crossed(old,new,line,direction=0):
    if old is None or line is None: return False
    a,b=cross_side(old,line),cross_side(new,line)
    if a==0 or b==0: return False
    if a*b>=0:return False
    return direction==0 or ((b-a)>0 if direction>0 else (b-a)<0)

def draw_line(frame,line,color=(0,255,0)):
    if line:cv2.line(frame,tuple(line[0]),tuple(line[1]),color,2)

class Motion:
    """Stores previous points and speed (pixels per second) for track IDs."""
    def __init__(self): self.prev={};self.speeds={}
    def update(self,items,t):
        out={}
        for obj in items:
            k=obj['id']
            if k is None: continue
            old=self.prev.get(k)
            if old:
                pt,t0=old
                dt=t-t0
                if dt>0:
                    dx=obj['foot'][0]-pt[0]; dy=obj['foot'][1]-pt[1]
                    measured=float(np.hypot(dx,dy))/dt
                    self.speeds[k]=.6*self.speeds.get(k,measured)+.4*measured
            out[k]=old[0] if old else None
            self.prev[k]=(obj['foot'],t)
        return out
