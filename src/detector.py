"""
PPE Violation Detector
======================
Production-ready PPE violation detection using YOLO11n.
Auto-detects violation vs. compliant classes from class names.
Logs every violation to a timestamped CSV file.
"""

import csv
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO


@dataclass
class PPEViolation:
    timestamp: str
    violation_type: str
    confidence: float
    bbox: tuple
    frame_id: int = 0
    zone: str = "general"
    severity: str = "HIGH"


class PPEViolationDetector:
    """
    Business-logic layer on top of YOLO11n.

    Parameters
    ----------
    model_path : str
        Path to the trained YOLO .pt weights file.
    class_names : list[str]
        Ordered list of class names matching the model output.
    conf_threshold : float
        Minimum confidence to accept a detection (default 0.35).
    log_path : str | None
        CSV path for the violation log.  Defaults to
        ``logs/violations_<timestamp>.csv`` in the working directory.
    """

    PROXIMITY_THRESHOLD = 0.20  # reserved for future zone logic

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        conf_threshold: float = 0.35,
        log_path: Optional[str] = None,
    ):
        self.model = YOLO(model_path)
        self.class_names = class_names
        self.conf = conf_threshold
        self.violations: List[PPEViolation] = []

        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path or str(
            logs_dir / f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        # Auto-detect violation vs. compliant class indices from names
        self.violation_ids = [
            i
            for i, n in enumerate(class_names)
            if any(kw in n.lower() for kw in ["no-", "no_", "without", "missing", "none"])
        ]
        self.compliant_ids = [
            i
            for i, n in enumerate(class_names)
            if any(
                kw in n.lower()
                for kw in ["hardhat", "helmet", "vest", "mask", "glove", "boot", "ppe"]
            )
            and i not in self.violation_ids
        ]

        self._init_log()

        print("✅ PPEViolationDetector ready")
        print(f"   Violation classes : {[class_names[i] for i in self.violation_ids]}")
        print(f"   Compliant classes : {[class_names[i] for i in self.compliant_ids]}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_log(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "frame_id", "violation_type", "severity",
                 "confidence", "x1", "y1", "x2", "y2", "zone"]
            )

    def _log(self, v: PPEViolation) -> None:
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [v.timestamp, v.frame_id, v.violation_type, v.severity,
                 f"{v.confidence:.3f}", *v.bbox, v.zone]
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        dict with keys:
          - ``annotated_frame``  : BGR numpy array with drawn boxes & HUD
          - ``violations``       : list of PPEViolation objects this frame
          - ``is_compliant``     : True if zero violations detected
          - ``total_detections`` : raw box count (compliant + violation)
        """
        H, W = frame.shape[:2]
        preds = self.model(frame, conf=self.conf, verbose=False)[0]
        ann = frame.copy()
        frame_viols: List[PPEViolation] = []

        for box in preds.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = self.class_names[cls] if cls < len(self.class_names) else str(cls)

            if cls in self.violation_ids:
                color = (0, 0, 220)
                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 3)
                label = f"VIOLATION: {name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(ann, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(ann, label, (x1 + 2, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                v = PPEViolation(
                    timestamp=datetime.now().isoformat(),
                    violation_type=name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    frame_id=frame_id,
                )
                frame_viols.append(v)
                self.violations.append(v)
                self._log(v)

            elif cls in self.compliant_ids:
                color = (0, 200, 0)
                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
                cv2.putText(ann, f"OK: {name}", (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            else:
                cv2.rectangle(ann, (x1, y1), (x2, y2), (180, 180, 0), 2)
                cv2.putText(ann, name, (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 2)

        # HUD overlay
        n = len(frame_viols)
        hud_color = (0, 0, 220) if n > 0 else (0, 180, 0)
        cv2.rectangle(ann, (0, 0), (W, 46), (0, 0, 0), -1)
        cv2.putText(
            ann,
            f"VIOLATIONS: {n}  |  Frame #{frame_id}  |  {datetime.now().strftime('%H:%M:%S')}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            hud_color,
            2,
        )

        # Bottom alert banner
        if n > 0:
            cv2.rectangle(ann, (0, H - 50), (W, H), (0, 0, 180), -1)
            cv2.putText(ann, "WARNING: PPE VIOLATION DETECTED",
                        (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        return {
            "annotated_frame": ann,
            "violations": frame_viols,
            "is_compliant": n == 0,
            "total_detections": len(preds.boxes),
        }

    def report(self) -> pd.DataFrame:
        """Print and return a summary DataFrame of all logged violations."""
        if not self.violations:
            print("No violations recorded.")
            return pd.DataFrame()
        df = pd.DataFrame([vars(v) for v in self.violations])
        print(f"\nViolation Report — {len(df)} total detections")
        print(
            df.groupby("violation_type")
            .agg(count=("confidence", "count"), avg_conf=("confidence", "mean"))
            .round(3)
        )
        return df
