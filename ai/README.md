# AI service

Python service that turns an exercise video into a rep count, a form score, and feedback. It is internal only: the backend is the only caller, and it cannot be reached from the internet.

## Stack

FastAPI, OpenCV, ffmpeg, MediaPipe Pose, pytest. Dependencies in `requirements.txt`.

## Pipeline

```
video -> normalize (ffmpeg) -> keypoints (MediaPipe) -> analysis (rules per exercise) -> result
```

| Step | Module | Does |
| --- | --- | --- |
| Normalize | `app/keypoints/` | Fix rotation, frame rate, and size |
| Keypoints | `app/keypoints/` | Body landmarks for each frame |
| Analysis | `app/analysis/` | Shared logic: joint angles, smoothing, rep splitting, scoring |
| Exercise rules | `app/analysis/exercises/` | One file per exercise: what counts as a rep, form checks, feedback |

**Keep keypoints and analysis separate.** Analysis takes keypoint frames, never video. That lets live feedback (a Spring goal) reuse the same analysis code on frames from a camera.

## API

`POST /analyze`

Request:

```json
{ "session_id": "abc123", "exercise": "exercise_1", "video_url": "<temporary read link>" }
```

Response (draft, final schema set in task SCRUM-28):

```json
{
  "status": "complete",
  "rep_count": 10,
  "form_score": 82,
  "reps": [{ "index": 1, "score": 90, "issues": [] }],
  "feedback": ["Slow down on the way back up."],
  "error": null,
  "analyzer_version": "0.1.0"
}
```

On bad input (no person found, clip too short or too long, unknown exercise), return `status: "failed"` with a clear `error`.

## Adding an exercise

1. Write its spec in `docs/`: camera angle, key joints, what counts as one rep, form rules, feedback wording.
2. Add a file in `app/analysis/exercises/` that follows the shared analyzer interface.
3. Add labeled test clips and check it against the accuracy targets.

## Scoring approach

Rule-based checks (joint angles and thresholds), not a trained model. No training data is needed, and every flag can be explained to a therapist.

## Data rules

- Synthetic or open-source data only. Never real patient data.
- Test videos live in the team OneDrive, never in git.
- The repo keeps only keypoint JSON (in `tests/fixtures/`) and at most one small open-source clip whose license allows it.

## Limits

Clips up to 60 seconds. Target: a 60-second clip analyzed in 90 seconds or less on Azure.
