# Backend

FastAPI service. It is the only entry point into PTAssist: the frontend talks only to this API, and only this API talks to the database, storage, and AI service.

## Stack

FastAPI, SQLAlchemy, Alembic, PostgreSQL (psycopg 3 driver), Clerk's Python SDK (`clerk-backend-api`), pytest, ruff. Runtime dependencies in `requirements.txt`; test and lint tools in `requirements-dev.txt`.

## Layout

```
app/
  main.py    Creates the FastAPI app and includes the routers
  config.py  Settings from environment variables
  db.py      SQLAlchemy engine and the get_db session dependency
  routes/    HTTP endpoints, grouped by area
  schemas/   Request and response models (Pydantic)
  models/    Database tables (SQLAlchemy). Base class in base.py
  auth/      Clerk token check, current user, role and ownership checks
  storage/   One module for video storage (MinIO locally, Azure Blob in the cloud)
  jobs/      Background task that calls the AI service
migrations/  Alembic migrations (versions/ holds one file per migration)
tests/
alembic.ini  Alembic settings
Dockerfile   API image, also used to run migrations
```

## Run locally

Needs Python 3.12 and a Postgres database. Run these from `backend/`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp ../.env.example ../.env       # once, then fill in the Clerk values (see Config)
set -a; source ../.env; set +a   # load the settings into this shell (macOS/Linux)
alembic upgrade head
uvicorn app.main:app --reload    # http://localhost:8000/health and /docs

pytest                           # needs the test database, see Testing
ruff check . && ruff format --check .
```

`GET /health` returns `{"status": "ok"}`. It needs no token and does not touch the database, so Docker and Azure can use it as a liveness probe.

To try `GET /me` in `/docs`, click **Authorize** and paste a Clerk session token. In the browser console of the signed-in web app, `await window.Clerk.session.getToken()` prints one. It expires after 60 seconds.

## Docker

```bash
docker build -t ptassist-api backend      # from the repo root
docker run --rm -p 8000:8000 -e DATABASE_URL=<url> -e CLERK_SECRET_KEY=<key> \
  -e CLERK_AUTHORIZED_PARTIES=<origins> ptassist-api
```

The same image runs migrations: `docker run --rm -e DATABASE_URL=<url> ptassist-api alembic upgrade head`.

## Migrations

Alembic reads `DATABASE_URL` from the environment (it is not stored in `alembic.ini`).

```bash
alembic upgrade head                                   # apply all migrations
alembic revision --autogenerate -m "add user table"    # create a new one
```

To add a table: create the model in `app/models/`, import it in `app/models/__init__.py` (so autogenerate can see it), then run `alembic revision --autogenerate`. Always read the generated file before committing. New migrations are linted and formatted with ruff automatically. Constraint and index names follow the naming convention in `app/models/base.py`.

## Every request

1. Verify the Clerk token. No valid token means `401`.
2. Load the user from our `User` table. First call from a new user creates them as a patient.
3. Check the role (patient or therapist). Wrong role means `403`.
4. Check ownership: patients see only their own data, therapists see only linked patients. Not yours means `404`.

Roles live in our database, not in Clerk. Therapists are promoted by the team with a script.

## Sign-in (Clerk)

Steps 1 and 2 are the `CurrentUser` dependency in `app/auth/current_user.py`. A route that takes it only runs for a signed-in user:

```python
@router.get("/me")
def get_me(user: CurrentUser) -> MeResponse: ...
```

The web app sends the Clerk session token as `Authorization: Bearer <token>`. `app/auth/clerk.py` checks it with Clerk's Python SDK, and it is the only code that talks to Clerk:

- Only session tokens are accepted, not Clerk API keys or machine tokens.
- The signature must match our Clerk instance's public keys. The SDK fetches them with `CLERK_SECRET_KEY` and caches them for 5 minutes.
- The token must not be expired, and its `azp` claim (the origin of the page that asked for it) must be in `CLERK_AUTHORIZED_PARTIES`.

On a user's first request, the API asks Clerk's Backend API for their primary email and creates their `users` row as a patient. Later requests only read the row. The email is copied once: if the user changes it in Clerk, our copy keeps the old one.

| Status | When |
| --- | --- |
| `401` | No bearer token, or the token is invalid, expired, or from another origin. `detail` says why |
| `403` | First request from a Clerk account that has no email address. Nothing is saved |
| `503` | Clerk could not be reached to check the token or look up the email. Nothing is saved; try again |

Never log tokens or emails. The database engine hides query values (`hide_parameters`), so they also stay out of SQL logs and error messages.

### Clerk settings

- `CLERK_SECRET_KEY` is the development instance's secret key (`sk_test_...`) from the **API keys** page of the Clerk Dashboard. It gives full access to our Clerk instance: keep it in `/.env` locally and in an Azure secret in the cloud, never in code.
- Users must sign up with an email address (Clerk Dashboard, **User & authentication**). The API needs it to create their row.
- `CLERK_AUTHORIZED_PARTIES` lists the web app's origins: `http://localhost:5173` (the Vite dev server) locally, and the Static Web Apps URL in Azure.

Until the promote script exists (SCRUM-65), promote a local test account by hand: `UPDATE users SET role = 'therapist' WHERE email = '<email>';`

## Data model

| Table | Holds |
| --- | --- |
| `User` (`users`) | Clerk id, email, role (`patient` or `therapist`), created time |
| `TherapistPatient` | Which therapist is linked to which patient |
| `Exercise` | The exercise library (5 exercises for the proof of concept) |
| `Plan` | A therapist's plan for one patient |
| `PlanExercise` | An exercise in a plan, with targets (reps, sets, frequency) |
| `Session` | One uploaded video for a plan exercise, with status |
| `SessionResult` | The AI result for a session (JSONB) |

Table names are plural snake_case (`users`), and primary keys are UUIDs (see `app/models/user.py`). The schema changes only through Alembic migrations. Never edit the database by hand, and never change a migration after it is merged.

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

Read from environment variables (see `/.env.example`): database URL, Clerk settings, storage settings, and the AI service URL. No secrets in code. Settings are defined in `app/config.py`. Add new ones there and in `/.env.example`.

| Variable | Read by | What |
| --- | --- | --- |
| `DATABASE_URL` | API, Alembic | Postgres URL. Must use the `postgresql+psycopg://` scheme so SQLAlchemy picks the psycopg 3 driver |
| `CLERK_SECRET_KEY` | API | Clerk secret key, to check tokens and look up new users' emails. Secret. See Clerk settings |
| `CLERK_AUTHORIZED_PARTIES` | API | Web app origins allowed to call the API, comma-separated. Empty skips the origin check, so set it everywhere |
| `TEST_DATABASE_URL` | pytest | The test database. See Testing |

Alembic only needs `DATABASE_URL`. Without `CLERK_SECRET_KEY` the API still starts, but every route that needs sign-in fails with `500`.

## Testing

pytest, against a real Postgres database: `TEST_DATABASE_URL`, by default `postgresql+psycopg://ptassist:ptassist@localhost:5433/ptassist_test`. Tests always use it in place of `DATABASE_URL`. Each run builds the schema with `alembic upgrade head`, empties the tables before every test, and drops everything at the end with `alembic downgrade base`, so every run also checks that the migrations work both ways. Never point it at a database you want to keep.

Start the test database once with Docker. It uses port 5433 so it does not clash with a dev database on 5432:

```bash
docker run -d --name ptassist-test-db -p 5433:5432 \
  -e POSTGRES_USER=ptassist -e POSTGRES_PASSWORD=ptassist -e POSTGRES_DB=ptassist_test \
  postgres:17
```

After a reboot, run `docker start ptassist-test-db`. Without Docker, create an empty database in any local Postgres 17 and set `TEST_DATABASE_URL` to it.

Tests never call Clerk. `tests/conftest.py` signs tokens with its own key and fakes the SDK's two network calls (fetching Clerk's public keys and looking up a user), so the SDK still checks every token. Use the `client` and `clerk` fixtures:

```python
def test_example(client, clerk) -> None:
    clerk.add_user("user_1", "pat@example.com")  # a Clerk user and their email
    response = client.get("/me", headers=clerk.headers("user_1"))
```

Every route needs a test that rejects the wrong role and the wrong owner.

## Pull request CI

The shared workflow in `/.github/workflows/pr-ci.yml` runs on every pull request,
including documentation-only changes, so required checks are never skipped by a
path filter. It provides three checks:

- `API lint`: ruff lint and formatting checks.
- `API tests`: pytest with Python 3.12 and `requirements-dev.txt`.
- `API image build`: builds `backend/Dockerfile` without pushing an image.

The workflow uses read-only repository permissions and needs no repository secrets
(tests fake Clerk). The `API tests` job runs a `postgres:17` service container on
port 5433 and points `TEST_DATABASE_URL` at it, the same setup as the local test
database. The AI team can add its own job to this
workflow in SCRUM-26 without renaming the API checks.

### Required checks on main

`main` is protected by the `trunk-based` repository ruleset (**Settings > Rules >
Rulesets**), not a classic branch protection rule. It requires a pull request with
1 approving review and allows squash merges only. Its **Require status checks to
pass** rule lists `API lint`, `API tests`, and `API image build` from GitHub
Actions, so a failing check blocks merging. Branches do not need to be up to date
with `main` before merging.

Keep the job names stable: the ruleset matches checks by name, so a renamed or
removed job leaves every pull request waiting for a check that never reports. When
a team adds a job to this workflow, a repository admin must also add its name to
the ruleset for it to block merges.
