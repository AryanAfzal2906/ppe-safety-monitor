"""
run_video.py — CLI script for PPE violation detection on videos
Usage:
    python src/run_video.py --input path/to/video.mp4 --output output/annotated.mp4
    python src/run_video.py --input path/to/video.mp4  # auto names output
"""

import argparse
import csv
import os
import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


CLASS_NAMES = [
    'hardhat', 'no-hardhat', 'safety-vest', 'no-safety-vest',
    'mask', 'no-mask', 'gloves', 'person'
]

VIOLATION_IDS = [
    i for i, n in enumerate(CLASS_NAMES)
    if any(kw in n.lower() for kw in ['no-', 'no_', 'without', 'missing'])
]
COMPLIANT_IDS = [
    i for i, n in enumerate(CLASS_NAMES)
    if any(kw in n.lower() for kw in ['hardhat', 'helmet', 'vest', 'mask', 'glove'])
    and i not in VIOLATION_IDS
]


def process_video(model_path: str, input_path: str, output_path: str,
                  conf: float = 0.35) -> dict:
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(input_path)

    FPS   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS, (W, H)
    )

    log_path = output_path.replace('.mp4', '_violations.csv')
    log_f    = open(log_path, 'w', newline='')
    log_w    = csv.writer(log_f)
    log_w.writerow(['frame_id', 'time_in_video', 'violation_type',
                    'confidence', 'x1', 'y1', 'x2', 'y2'])

    frame_id         = 0
    total_violations = 0
    violation_frames = 0

    print(f"\n{'='*60}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  FPS    : {FPS}  |  Res: {W}x{H}  |  Frames: {TOTAL}")
    print(f"{'='*60}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preds       = model(frame, conf=conf, verbose=False)[0]
        ann         = frame.copy()
        frame_viols = 0
        mins        = int(frame_id / FPS // 60)
        secs        = int(frame_id / FPS % 60)
        time_str    = f'{mins:02d}:{secs:02d}'

        for box in preds.boxes:
            cls  = int(box.cls[0])
            c    = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)

            if cls in VIOLATION_IDS:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 0, 220), 3)
                label = f'VIOLATION: {name} {c:.2f}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(ann, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 0, 220), -1)
                cv2.putText(ann, label, (x1 + 2, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                frame_viols      += 1
                total_violations += 1
                log_w.writerow([frame_id, time_str, name, f'{c:.3f}', x1, y1, x2, y2])

            elif cls in COMPLIANT_IDS:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(ann, f'OK: {name}', (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 2)
            else:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 200, 200), 2)
                cv2.putText(ann, name, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 2)

        if frame_viols > 0:
            violation_frames += 1

        hud_color = (0, 0, 220) if frame_viols > 0 else (0, 180, 0)
        cv2.rectangle(ann, (0, 0), (W, 50), (0, 0, 0), -1)
        cv2.putText(
            ann,
            f'PPE MONITOR  |  Time: {time_str}  |  Violations: {frame_viols}  |  Total: {total_violations}',
            (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2
        )

        if frame_viols > 0:
            cv2.rectangle(ann, (0, H - 50), (W, H), (0, 0, 180), -1)
            cv2.putText(ann, 'WARNING: PPE VIOLATION DETECTED',
                        (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        out.write(ann)

        if frame_id % (FPS * 10) == 0:
            pct = (frame_id / TOTAL * 100) if TOTAL > 0 else 0
            print(f'  [{pct:5.1f}%]  Time: {time_str}  |  Cumulative violations: {total_violations}')

        frame_id += 1

    cap.release()
    out.release()
    log_f.close()

    print(f'\n✅ Done!')
    print(f'   Annotated video → {output_path}')
    print(f'   Violation log   → {log_path}')
    print(f'   Total violations: {total_violations} across {violation_frames} frames')

    if total_violations > 0:
        df = pd.read_csv(log_path)
        print('\nViolation breakdown:')
        print(df.groupby('violation_type').agg(count=('confidence', 'count')).to_string())

    return {
        'total_violations': total_violations,
        'violation_frames': violation_frames,
        'total_frames':     frame_id,
        'log_path':         log_path,
        'output_path':      output_path,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPE Violation Detector')
    parser.add_argument('--input',  required=True,  help='Input video path')
    parser.add_argument('--output', default=None,   help='Output video path (auto if omitted)')
    parser.add_argument('--model',  default='models/ppe_yolo11n_final.pt', help='Model weights path')
    parser.add_argument('--conf',   type=float, default=0.35, help='Confidence threshold')
    args = parser.parse_args()

    out = args.output or f"output/{Path(args.input).stem}_annotated.mp4"
    process_video(args.model, args.input, out, args.conf)
