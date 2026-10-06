"""First validation stage: live webcam and skeleton display."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .pose import PoseEstimator, draw_skeleton


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="Probe camera and inference without a window")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--preview", type=Path, help="Save the final annotated frame")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open webcam index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frames = 0
    detected = 0
    start = time.monotonic()
    try:
        with PoseEstimator(Path("models/pose_landmarker_lite.task")) as estimator:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam opened but failed to read a frame")
                landmarks = estimator.detect(frame, round((time.monotonic() - start) * 1000))
                detected += bool(landmarks)
                draw_skeleton(frame, landmarks)
                frames += 1
                cv2.putText(frame, f"Skeleton: {'tracked' if landmarks else 'no person'} | frame {frames}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if not args.headless:
                    cv2.imshow("PTAssist webcam + skeleton (Q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.max_frames and frames >= args.max_frames:
                    break
            if args.preview:
                args.preview.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.preview), frame):
                    raise RuntimeError(f"Could not write preview to {args.preview}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"frames={frames} pose_frames={detected} elapsed_s={time.monotonic()-start:.2f}")


if __name__ == "__main__":
    main()
