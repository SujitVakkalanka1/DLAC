from visionport.common import cli,settings,open_video,frames,show,finish,model,predict,detections,draw,text,writer,LABELS

args = cli('Tasks 6-9: Vehicle detection, classification, confidence and per-frame count')
cfg = settings(args)
m = model(cfg)
cap = open_video(cfg)
out = writer(args, cap)

print(f"[*] Running Tasks 6-9: Vehicle Detection using {cfg['model']} (Confidence: {cfg['confidence']})")
try:
    for i, t, frame in frames(cap):
        items = detections(predict(m, frame, cfg))
        counts = {}
        for obj in items:
            draw(frame, obj)
            lbl = LABELS.get(obj['cls'], str(obj['cls']))
            counts[lbl] = counts.get(lbl, 0) + 1
        
        breakdown = ", ".join([f"{k.capitalize()}: {v}" for k, v in sorted(counts.items())])
        text(frame, f'Vehicles in frame: {len(items)}')
        if breakdown:
            text(frame, breakdown, row=1, color=(200, 255, 200))
        
        print(f"[*] Frame {i} (t={t:.2f}s): {len(items)} vehicles detected [{breakdown}]")
        for idx, obj in enumerate(items, 1):
            lbl = LABELS.get(obj['cls'], str(obj['cls'])).capitalize()
            print(f"    #{idx:02d}: {lbl:<12} Conf: {obj['conf']:.2f} | Box: {obj['xyxy']}")

        if out: out.write(frame)
        if not show(frame, args): break
finally:
    finish(cap, out)
    print("[*] Vehicle detection complete. Output saved under outputs/05_vehicle_detection.jpg")
