# Indian Equity Sentiment Analyzer

Production-style web application that will analyze Indian equity sentiment using
the public Hugging Face model
[`alexcruse07/indian-equity-sentiment-model`](https://huggingface.co/alexcruse07/indian-equity-sentiment-model).

## Project structure

```text
indian-equity-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   ├── services/
│   │   ├── config.py
│   │   └── main.py
│   ├── tests/
│   ├── .env.example
│   └── requirements.txt
├── frontend/
├── .gitignore
└── README.md
```

### `backend/`

This directory will contain the Python backend:

- FastAPI application and API routes
- Pydantic request and response models
- Hugging Face Inference API integration
- Backend configuration, validation, and tests

The backend contains the FastAPI application, environment-based configuration,
health endpoint, Pydantic models, service layer, and backend tests. Hugging Face
integration will be added in a later stage.

### `frontend/`

This directory will contain the React and TypeScript frontend built with Vite:

- User interface components and pages
- Client-side API integration with the FastAPI backend
- Frontend state and presentation logic
- Frontend tests and static assets

No frontend business logic has been implemented yet.

## Planned architecture

- **Frontend:** React, TypeScript, and Vite
- **Backend:** Python, FastAPI, and Pydantic
- **Model integration:** Hugging Face Inference API
- **Model:** `alexcruse07/indian-equity-sentiment-model`

Deployment to AWS S3/CloudFront and AWS backend infrastructure is intentionally
out of scope for this initial scaffold.

## Run the backend locally

From the project root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`. Check
`http://localhost:8000/api/v1/health` for:

```json
{"status": "UP"}
```

## Running the tests

### Backend (pytest)

All Hugging Face calls (`InferenceClient` and the local `transformers`
pipeline) are mocked in these tests, so `pytest` never makes real network or
model calls and runs in a few seconds.

```bash
cd backend
source .venv/bin/activate   # create it first if you haven't (see above)
pytest
```

Covers, across `backend/tests/`:

- `test_health_api.py` — `GET /api/v1/health`.
- `test_sentiment_api.py` — the `POST /api/v1/sentiment` HTTP layer: success,
  empty/missing text, text exceeding the max length, and every service
  exception mapped to its HTTP status (400/502/504). The service itself is
  faked via a FastAPI dependency override, so no Hugging Face code runs here.
- `test_sentiment_service.py` — the `SentimentService` business logic in
  isolation: input validation, successful remote inference, authentication
  failure, timeout, remote-unavailable → local pipeline fallback (400/403/404
  and unexpected errors), local pipeline failures, and malformed responses
  from both the remote client and the local pipeline. `InferenceClient` and
  `transformers.pipelines.pipeline` are both patched with mocks.

### Frontend (Vitest + React Testing Library)

```bash
cd frontend
npm install     # first time only
npm test
```

`src/App.test.tsx` mocks `services/api.ts` (no real HTTP calls) and covers:
rendering the header/subtitle/textarea/button, typing into the textarea, the
Analyze button being disabled with empty input and while a request is in
flight, the loading state, a successful sentiment result rendering in the
result card, and an API error rendering the error banner.

Use `npm run test:watch` for interactive watch mode while developing.

## Running the backend in Docker

The backend has a production-ready [Dockerfile](backend/Dockerfile) and
[.dockerignore](backend/.dockerignore). The image never bakes in your `.env`
file or secrets — configuration is supplied at container runtime via
environment variables, and it runs as a non-root user.

Build the image (from the `backend/` directory):

```bash
cd backend
docker build -t indian-equity-backend:latest .
```

Run it, passing the same variables you'd normally put in `.env`:

```bash
docker run -d \
  --name indian-equity-backend \
  -p 8000:8000 \
  -e HF_TOKEN=your_hf_token \
  -e HF_MODEL_ID=alexcruse07/indian-equity-sentiment-model \
  -e ALLOWED_ORIGINS=http://localhost:5173 \
  indian-equity-backend:latest
```

Or, load them from a local env file without ever copying it into the image:

```bash
docker run -d --name indian-equity-backend -p 8000:8000 \
  --env-file backend/.env \
  indian-equity-backend:latest
```

Verify it's healthy:

```bash
curl http://localhost:8000/api/v1/health
# {"status":"UP"}
```

Stop and remove it:

```bash
docker stop indian-equity-backend && docker rm indian-equity-backend
```
