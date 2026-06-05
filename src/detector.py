"""Face detection: locate faces in a frame and return their bounding boxes.

Primary detector is RetinaFace (``retina-face``, TensorFlow-based); MTCNN
(``facenet-pytorch``, PyTorch-based) is the explicit fallback used when
RetinaFace cannot be loaded or raises during inference.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.9

# Permissive floor passed to RetinaFace so our own threshold stays authoritative.
_RETINAFACE_SCORE_FLOOR = 0.1

# Canonical landmark ordering we normalise both backends to.
LANDMARK_NAMES = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")


@dataclass
class Detection:
    """A single detected face.

    Coordinates are in pixels relative to the input frame. ``landmarks`` holds
    the five facial points (see :data:`LANDMARK_NAMES`) as ``(x, y)`` tuples when
    the backend provides them, otherwise ``None``. ``backend`` records which
    detector produced the result, which is useful when the fallback engages.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None
    backend: str = "unknown"

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Integer ``(x1, y1, x2, y2)`` suitable for drawing/cropping."""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


class FaceDetector:
    """RetinaFace face detector with an explicit MTCNN fallback.

    Backend selection is a real, inspectable state rather than a blanket
    ``try/except``:

    * If RetinaFace fails to *load*, :attr:`backend` is set to ``"mtcnn"`` at
      construction time and :attr:`fell_back_at_load` is ``True``.
    * If RetinaFace loads but *raises during inference*, the first such failure
      flips :attr:`backend` to ``"mtcnn"`` permanently (sticky fallback) and sets
      :attr:`fell_back_at_inference` to ``True``.

    In both cases the transition is logged so the exact moment and reason the
    fallback engaged can be pointed to.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._retinaface = None
        self._mtcnn = None

        # Observable fallback state.
        self.fell_back_at_load = False
        self.fell_back_at_inference = False

        if self._load_retinaface():
            self.backend = "retinaface"
        else:
            self.fell_back_at_load = True
            self.backend = "mtcnn"

    # -- backend loading ----------------------------------------------------

    def _load_retinaface(self) -> bool:
        """Try to load RetinaFace. Returns ``True`` on success, ``False`` if it
        could not be loaded (the caller then selects the MTCNN backend)."""
        try:
            from retinaface import RetinaFace

            # Build the model now (this also triggers the one-time weights
            # download) so load failures surface here, not mid-stream.
            RetinaFace.build_model()
            self._retinaface = RetinaFace
            logger.info("RetinaFace loaded (serengil/retina-face).")
            return True
        except Exception as exc:  # noqa: BLE001 - load may fail many ways
            logger.warning(
                "RetinaFace failed to load (%s: %s). Falling back to MTCNN.",
                type(exc).__name__,
                exc,
            )
            return False

    def _get_mtcnn(self):
        """Lazily construct the MTCNN backend the first time it is needed."""
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN

            # post_process=False keeps raw boxes/probs; keep_all returns every face.
            self._mtcnn = MTCNN(
                keep_all=True, post_process=False, device=self.device
            )
            logger.info("MTCNN loaded on %s.", self.device)
        return self._mtcnn

    # -- public API ---------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Detect faces in a single BGR frame (as produced by OpenCV).

        Returns a list of :class:`Detection`, filtered by
        :attr:`confidence_threshold`. Routes to RetinaFace when active, with an
        explicit, sticky fallback to MTCNN on inference failure.
        """
        if self.backend == "retinaface":
            try:
                return self._detect_retinaface(frame_bgr)
            except Exception as exc:  # noqa: BLE001 - any inference error triggers fallback
                # Named fallback path: RetinaFace raised at inference time.
                logger.warning(
                    "RetinaFace raised during inference (%s: %s). "
                    "Permanently falling back to MTCNN.",
                    type(exc).__name__,
                    exc,
                )
                self.fell_back_at_inference = True
                self.backend = "mtcnn"
                # fall through to MTCNN below

        return self._detect_mtcnn(frame_bgr)

    # -- backend implementations -------------------------------------------

    def _detect_retinaface(self, frame_bgr: np.ndarray) -> List[Detection]:
        # serengil RetinaFace consumes BGR numpy arrays directly (cv2 layout),
        # so no colour conversion is needed here.
        faces = self._retinaface.detect_faces(
            frame_bgr, threshold=_RETINAFACE_SCORE_FLOOR
        )

        detections: List[Detection] = []
        # Returns a dict keyed per face; anything else (e.g. () or []) means none.
        if not isinstance(faces, dict):
            return detections

        for info in faces.values():
            score = float(info.get("score", 0.0))
            if score < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = info["facial_area"]
            landmarks = self._normalise_retinaface_landmarks(info.get("landmarks"))
            detections.append(
                Detection(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    score,
                    landmarks,
                    backend="retinaface",
                )
            )
        return detections

    @staticmethod
    def _normalise_retinaface_landmarks(
        raw,
    ) -> Optional[List[Tuple[float, float]]]:
        """Reorder RetinaFace's named landmarks into :data:`LANDMARK_NAMES`."""
        if not raw:
            return None
        try:
            return [
                (float(raw["left_eye"][0]), float(raw["left_eye"][1])),
                (float(raw["right_eye"][0]), float(raw["right_eye"][1])),
                (float(raw["nose"][0]), float(raw["nose"][1])),
                (float(raw["mouth_left"][0]), float(raw["mouth_left"][1])),
                (float(raw["mouth_right"][0]), float(raw["mouth_right"][1])),
            ]
        except (KeyError, TypeError, IndexError):
            return None

    def _detect_mtcnn(self, frame_bgr: np.ndarray) -> List[Detection]:
        mtcnn = self._get_mtcnn()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs, points = mtcnn.detect(rgb, landmarks=True)

        detections: List[Detection] = []
        if boxes is None:
            return detections

        for box, prob, lm in zip(boxes, probs, points):
            if prob is None or float(prob) < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box
            landmarks = (
                [(float(x), float(y)) for x, y in lm] if lm is not None else None
            )
            detections.append(
                Detection(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(prob),
                    landmarks,
                    backend="mtcnn",
                )
            )
        return detections


# -- visual smoke test ------------------------------------------------------


def _draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw boxes, landmarks, and confidence onto ``frame`` in place."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{det.backend} {det.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if det.landmarks:
            for lx, ly in det.landmarks:
                cv2.circle(frame, (int(lx), int(ly)), 2, (0, 0, 255), -1)
    return frame


def _parse_source(source: str):
    """A bare integer is treated as a webcam index; anything else is a path/URL."""
    return int(source) if source.isdigit() else source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual smoke test for the face detector."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: a file path, stream URL, or webcam index (default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE_THRESHOLD}).",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(_parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source!r}")

    detector = FaceDetector(confidence_threshold=args.threshold)
    logger.info("Detecting with backend=%s. Press 'q' to quit.", detector.backend)

    window = "face-recognition-watchlist :: detector"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of file or camera read failure

            detections = detector.detect(frame)
            _draw_detections(frame, detections)
            cv2.imshow(window, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    main()
