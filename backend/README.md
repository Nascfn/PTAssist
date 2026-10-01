# Backend

FastAPI service. It is the only entry point into PTAssist: the frontend talks only to this API, and only this API talks to the database, storage, and AI service.

## Stack

FastAPI, SQLAlchemy, Alembic, PostgreSQL, pytest. Dependencies in `requirements.txt`.

## Layout

```
app/
  routes/    HTTP endpoints, grouped by area
  schemas/   Request and response models (Pydantic)
  models/    Database tables (SQLAlchemy)
  auth/      Clerk token check, current user, role and ownership checks
  storage/   One module for video storage (MinIO locally, Azure Blob in the cloud)
  jobs/      Background task that calls the AI service
migrations/  Alembic migrations
tests/
```

## Every request

1. Verify the Clerk token. No valid token means `401`.
2. Load the user from our `User` table. First call from a new user creates them as a patient.
3. Check the role (patient or therapist). Wrong role means `403`.
4. Check ownership: patients see only their own data, therapists see only linked patients. Not yours means `404`.

Roles live in our database, not in Clerk. Therapists are promoted by the team with a script.

## Data model

| Table | Holds |
| --- | --- |
| `User` | Clerk id, email, role (`patient` or `therapist`) |
| `TherapistPatient` | Which therapist is linked to which patient |
| `Exercise` | The exercise library (5 exercises for the proof of concept) |
| `Plan` | A therapist's plan for one patient |
| `PlanExercise` | An exercise in a plan, with targets (reps, sets, frequency) |
| `Session` | One uploaded video for a plan exercise, with status |
| `SessionResult` | The AI result for a session (JSONB) |

The schema changes only through Alembic migrations. Never edit the database by hand, and never change a migration after it is merged.

## Session lifecycle

```
upload -> processing -> complete
                     -> failed (with a reason)
```

1. `POST` upload: validate the file, store it, create the `Session` as `processing`.
2. A FastAPI background task creates a 15-minute read link and calls the AI service `POST /analyze`.
3. The result is saved to `SessionResult` and the status becomes `complete`, or `failed` if anything goes wrong.
4. Sessions stuck in `processing` too long are marked `failed`.

The frontend polls the session status. Later, a job queue can replace the background task without changing the API.

## Endpoints (draft)

Final paths are decided in each task.

| Area | Endpoint | Who |
| --- | --- | --- |
| Me | `GET /me` | Anyone signed in |
| Exercises | `GET /exercises` | Anyone signed in |
| Patients | `GET/POST/DELETE /therapist/patients` | Therapist |
| Plans | `POST/PUT /plans`, `GET /plans/me` | Therapist, patient |
| Sessions | `POST /sessions`, `GET /sessions/{id}`, `GET /sessions` | Patient (therapist can read linked patients') |
| Progress | `GET /patients/{id}/progress` | Therapist |

## Storage

All video access goes through `app/storage/`. Buckets are private. Other code never talks to MinIO or Azure Blob directly, and never makes a file public. Raw videos are deleted after 30 days.

## Config

Read from environment variables (see `/.env.example`): database URL, Clerk settings, storage settings, and the AI service URL. No secrets in code.

## Testing

pytest. Every route needs a test that rejects the wrong role and the wrong owner.
