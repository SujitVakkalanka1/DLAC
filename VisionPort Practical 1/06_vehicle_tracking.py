from visionport.common import cli,settings,open_video,frames,show,finish,model,predict,detections,draw,text,writer,image_mode,image_notice
args=cli('Task 12: Track vehicle IDs with ByteTrack')
cfg=settings(args);m=model(cfg);cap=open_video(cfg);out=writer(args,cap)
# Image-only mode: preserve actual detections but never fabricate temporal results.
if image_mode(cap):
    print(f"[*] IMAGE MODE: Running detection on single frame using {cfg['model']}.")
    try:
        for _, _, frame in frames(cap):
            objs = detections(predict(m, frame, cfg))
            for obj in objs: draw(frame, obj)
            text(frame, f'Vehicles detected: {len(objs)}')
            image_notice(frame, 'Tracking IDs unavailable from one photo')
            print(f"[*] Detected {len(objs)} vehicles. (Note: Tracking IDs require multi-frame video input via --video).")
            show(frame, args, delay=0)
    finally: finish(cap, out)
    raise SystemExit(0)
print(f"[*] VIDEO MODE: Tracking vehicles across video frames using ByteTrack + {cfg['model']}")
total_frames = int(cap.get(3) or 0) # cv2.CAP_PROP_FRAME_COUNT
try:
    for i, t, frame in frames(cap):
        items = detections(predict(m, frame, cfg, track=True), track=True)
        tracked_count = sum(x["id"] is not None for x in items)
        for obj in items: draw(frame, obj)
        text(frame, f'Currently tracked: {tracked_count}')
        
        if i % 15 == 0 or i < 5:
            ids = [f"#{x['id']}" for x in items if x['id'] is not None]
            print(f"[*] Frame {i:03d} (t={t:.2f}s): {tracked_count} vehicle(s) tracked. Active IDs: {ids}")
            
        if out: out.write(frame)
        if not show(frame, args): break
finally:
    finish(cap, out)
    print(f"[*] Vehicle tracking completed. Processed {i+1} frames.")
