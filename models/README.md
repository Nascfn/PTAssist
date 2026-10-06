# Local pose model

`pose_landmarker_lite.task` is Google's pretrained MediaPipe Pose Landmarker Lite float16 model, downloaded from:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task

SHA-256: `59929e1d1ee95287735ddd833b19cf4ac46d29bbf6753c459690d574a`

The application reads this local file and makes no network request during use. See Google's [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) for the supported task API and model information.

`pose_landmarker_full.task` is the optional official Full float16 bundle, downloaded from:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

SHA-256: `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1`

Select it with `--model models\pose_landmarker_full.task`. It runs fully offline after download. The default remains Lite until both variants can be compared on the same full-body video for tracking quality and speed.
