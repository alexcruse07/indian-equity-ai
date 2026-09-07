# Indian Equity Sentiment Analyzer

> A production-style, full-stack web app that classifies the sentiment of
> Indian equity market text (news headlines, analyst comments, earnings
> commentary) into **Bullish / Bearish / Neutral** using a fine-tuned
> DistilBERT model, served by a FastAPI backend and a React/TypeScript
> frontend, deployed end-to-end on AWS.

## 🚀 Live demo

**➡️ https://d3ltrpuk8i675z.cloudfront.net/**

- Health: <https://d3ltrpuk8i675z.cloudfront.net/api/v1/health>
- API:    `POST https://d3ltrpuk8i675z.cloudfront.net/api/v1/sentiment`

Try one of the built-in example prompts, or paste any market-related
sentence and hit **Analyze**. First request after a cold start takes ~20–30 s
(model load); subsequent requests return in ~150–300 ms.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [The ML model](#the-ml-model)
4. [How the model is served](#how-the-model-is-served)
5. [Backend implementation](#backend-implementation)
6. [Frontend implementation](#frontend-implementation)
7. [AWS deployment & costs](#aws-deployment--costs)
8. [Running locally](#running-locally)
9. [Testing](#testing)
10. [Docker](#docker)
11. [Project structure](#project-structure)
12. [Environment variables](#environment-variables)
13. [Design decisions & tradeoffs](#design-decisions--tradeoffs)
14. [Production hardening / next steps](#production-hardening--next-steps)

---

## What it does

The user pastes any piece of Indian equity market text — a headline
("Reliance reported strong quarterly earnings"), an analyst comment, a
snippet of an earnings call transcript — and the app returns:

- **Sentiment**: `BULLISH`, `BEARISH`, or `NEUTRAL`
- **Confidence**: softmax probability of the winning class (0.0–1.0)
- **Model**: the model ID that produced the prediction

The UI is a clean single-page dashboard with example prompts, a loading
state, an error banner, and a color-coded result card.

## Architecture

Everything is served from a single origin (CloudFront), which routes by
path pattern to two independent origins — an S3 static site for the SPA
and an EC2-hosted FastAPI container for the API. This means **no CORS
plumbing and no mixed-content problems** — the frontend talks to the
backend via relative URLs (`/api/v1/*`) that never leave the CloudFront
domain from the browser's point of view.

```
                                         ┌──────────────────────────────┐
                                         │            AWS               │
 Browser ──HTTPS──► CloudFront ─────┬───►│  S3 bucket  (React SPA)      │
                    (single URL)    │    │                              │
                                    │    └──────────────────────────────┘
                                    │    ┌──────────────────────────────┐
                                    └───►│  EC2 t2.micro (Docker)       │
                                    /api │  ├─ FastAPI (Uvicorn)         │
                                         │  └─ transformers pipeline    │
                                         │     (DistilBERT, CPU-only)   │
                                         │                              │
                                         │  ← downloads model from     │
                                         │    huggingface.co on 1st req │
                                         └──────────────────────────────┘
```

### Request flow — `POST /api/v1/sentiment`

1. React calls `fetch("/api/v1/sentiment", …)` — relative URL, same origin.
2. CloudFront's `/api/*` behavior forwards the request to the EC2 origin
   over HTTP:8000.
3. FastAPI validates the request body with Pydantic (`text` present,
   non-empty, ≤ 1000 chars).
4. The endpoint calls `SentimentService.analyze_sentiment(text)` (injected
   via `Depends(get_sentiment_service)`).
5. The service first tries Hugging Face's **Inference Providers** API
   via `InferenceClient`.
6. That call currently returns 4xx for this model (see notes below), so
   the service **falls back to local inference** with a
   `transformers.pipeline("text-classification")` that runs the model
   in-process on the EC2 CPU.
7. Raw model output `[{"label": "BULLISH", "score": 0.58}]` is normalized
   into the domain response.
8. Response flows back through CloudFront to the browser and renders in
   the result card.

## The ML model

- **Model ID**: [`alexcruse07/indian-equity-sentiment-model`](https://huggingface.co/alexcruse07/indian-equity-sentiment-model)
- **Architecture**: [DistilBERT](https://huggingface.co/distilbert-base-uncased)
  fine-tuned for text classification
- **Task**: single-label classification (softmax over 3 classes)
- **Labels** (`id2label` from the model config):
  - `0` → `BEARISH`
  - `1` → `BULLISH`
  - `2` → `NEUTRAL`
- **Weights**: ~250 MB on disk, ~350 MB resident in RAM once loaded
- **Reported training accuracy**: ~70% (per the model card) — good enough
  for a demo but not a trading signal. The app is a UI/serving demo, not
  a market forecasting tool.

### Raw output shape

The model's raw output (from `transformers.pipeline("text-classification")`
with default `top_k=1`) is:

```python
[{"label": "BULLISH", "score": 0.5810990333557129}]
```

The service normalizes that into the app's domain response:

```json
{
  "sentiment": "BULLISH",
  "confidence": 0.5810990333557129,
  "model": "alexcruse07/indian-equity-sentiment-model"
}
```

## How the model is served

The service (`backend/app/services/sentiment_service.py`) uses a
**remote-first with local fallback** strategy:

### 1. Remote (Hugging Face Inference Providers)

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model=settings.hf_model_id,
    token=settings.hf_token,
    timeout=settings.hf_inference_timeout_seconds,
)
result = client.text_classification(text)
```

This talks to the modern `router.huggingface.co` provider router (NOT
the deprecated `api-inference.huggingface.co`). It's typed exception
handling maps:

| HTTP status | Behavior |
|---|---|
| 401 | `AuthenticationError` — token invalid, hard-fail |
| 400, 403, 404 | Fall back to local (model not deployed on any provider) |
| Timeout | `InferenceTimeoutError` — hard-fail |
| Other unexpected error | Log warning, fall back to local |

> **Reality check for this specific model:** at the time of writing,
> `alexcruse07/indian-equity-sentiment-model` has no
> `inferenceProviderMapping` on the Hub, so the remote call always
> returns a 400/404 and the service **always** falls back to local
> inference in production. The remote path is kept because the fallback
> pattern is the right production shape — if the model ever gets a
> provider deployment, the app immediately picks it up with no code
> change.

### 2. Local fallback (transformers pipeline)

```python
import transformers.pipelines as hf_pipelines

pipe = hf_pipelines.pipeline(
    "text-classification",
    model=settings.hf_model_id,
    token=settings.hf_token,
)
result = pipe(text)  # -> [{"label": "BULLISH", "score": 0.58}]
```

- **Lazy loaded** — only imported and instantiated on first use, so the
  ~250 MB torch/transformers cost isn't paid at startup if a remote
  provider is ever available.
- **Cached** — the pipeline is stored on the service instance and reused
  for every subsequent request in the process's lifetime.
- **CPU-only** — the Dockerfile installs the CPU-only PyTorch wheel via
  `--extra-index-url https://download.pytorch.org/whl/cpu`, keeping the
  image ~500 MB instead of the ~2.5 GB CUDA build. This is what makes
  the app fit on a t2.micro (1 GB RAM + 2 GB swap).

### Response normalization & validation

Every raw model response — remote or local, list or dict shape — flows
through `_normalize_client_result()` and `_to_domain_result()`, which:

- Reject empty / malformed shapes (`MalformedModelResponseError`)
- Reject non-numeric scores
- Reject labels not in the `Sentiment` enum (defensive against a model
  swap producing unexpected labels)
- Return a strongly-typed `SentimentResult` (`app/models/sentiment.py`)

## Backend implementation

Location: [`backend/`](backend/)

- **Framework**: FastAPI + Uvicorn
- **Validation**: Pydantic v2 models for both request and response
- **Config**: `pydantic-settings` reading environment variables (see
  [`app/config.py`](backend/app/config.py)); `@lru_cache` factory so the
  Settings object is a de-facto singleton
- **DI**: FastAPI `Depends(get_sentiment_service)` — the service can be
  swapped for a fake in tests by using `app.dependency_overrides`
- **Logging**: `logging.getLogger(__name__)` at INFO level; request
  bodies are never logged, only `text_length`
- **Exception mapping** (in [`app/api/sentiment.py`](backend/app/api/sentiment.py)):

  | Service exception | HTTP status |
  |---|---|
  | `InvalidInputError` | `400` |
  | `AuthenticationError` | `502` (masked as generic "inference failed") |
  | `InferenceTimeoutError` | `504` |
  | `MalformedModelResponseError` | `502` |
  | `InferenceError` | `502` |
  | any other unhandled | `500` (global handler in `main.py`) |

- **CORS**: `starlette.middleware.cors` — permissive in production
  because everything flows through CloudFront (same origin from the
  browser's perspective); tightenable via `ALLOWED_ORIGINS` env var

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/v1/health`    | Liveness check → `{"status": "UP"}` |
| POST | `/api/v1/sentiment` | Classify a piece of text |

Request:
```json
{ "text": "Reliance reported strong quarterly earnings." }
```
Response:
```json
{
  "sentiment": "BULLISH",
  "confidence": 0.5810990333557129,
  "model": "alexcruse07/indian-equity-sentiment-model"
}
```

## Frontend implementation

Location: [`frontend/`](frontend/)

- **Framework**: React 19 + TypeScript + Vite
- **Styling**: Custom CSS with a dark financial-dashboard theme (no
  Tailwind or component libraries — kept dependency-free on purpose)
- **State**: `useState` + a dedicated `useSentimentAnalysis` hook that
  encapsulates `{ result, error, isLoading, analyze, reset }`
- **API client**: [`src/services/api.ts`](frontend/src/services/api.ts)
  - Base URL read from `import.meta.env.VITE_API_BASE_URL` (empty string
    in production so calls are same-origin relative URLs)
  - Custom `ApiError` class distinguishes HTTP errors from network
    errors from malformed-JSON errors
  - No production URL is ever hardcoded
- **Components** (all in [`src/components/`](frontend/src/components/)):
  - `Header` — title + subtitle
  - `TextAnalyzerForm` — textarea + example prompts + submit button;
    owns only the input string, delegates analysis to a callback
  - `ExamplePrompts` — clickable seed sentences
  - `LoadingSpinner` — CSS-only spinner
  - `ResultCard` — sentiment badge + confidence bar + model name
  - `SentimentBadge` — color-coded pill (green / red / gray)
  - `ErrorBanner` — surfaces network / server error messages
- **Build output**: ~5.9 KB CSS + ~195 KB JS (61 KB gzipped)

## AWS deployment & costs

Everything lives in **`ap-south-1` (Mumbai)** and is well within
AWS free-tier limits for a demo workload.

| Service | Purpose | Free tier | Cost for this app |
|---|---|---|---|
| **Amazon S3** | Hosts the built React SPA (`dist/`) as private objects, served only through CloudFront via Origin Access Control | 5 GB storage, 20k GET, 2k PUT / month for 12 months | **Free** (app is ~211 KB total) |
| **Amazon CloudFront** | HTTPS-terminating CDN, single public URL, routes `/*` → S3 and `/api/*` → EC2; also caches static assets | 1 TB egress + 10M requests / month (always-free, not just 12 months) | **Free** |
| **Amazon EC2 (t2.micro)** | Runs the Docker container hosting FastAPI + the model | 750 hrs / month for 12 months | **Free** for first 12 months; ~$8.50/mo after |
| **Amazon EBS (30 GB gp3)** | Root volume for the EC2 instance | 30 GB gp3 / month for 12 months | **Free** for first 12 months |
| **Amazon VPC + default security group** | Networking + firewall | Always free | **Free** |
| **Data transfer OUT (EC2 → internet)** | Only used when CloudFront pulls from EC2 (rare given caching) | 100 GB / month (always-free) | **Free** |
| **CloudFront Origin Access Control** | Lets CloudFront read the private S3 bucket | Always free | **Free** |
| **EC2 key pair** | SSH access | Always free | **Free** |

**Total monthly cost while the app is running: $0.00** for the first
12 months, and **~$8.50/mo after** (only the EC2 t2.micro leaves the
free tier at the 12-month mark — everything else stays permanently free
for this workload).

> No paid services (Route 53 hosted zone, ACM certificate for a custom
> domain, RDS, Elastic IP on a stopped instance, NAT gateway, etc.) are
> used. If you attach an Elastic IP, note that AWS charges for it while
> the instance is stopped — leave the instance running or release the IP.

### Deployed resources (Mumbai / `ap-south-1`)

- EC2 instance: `i-0d118149859a7a6ca` (t2.micro, Amazon Linux 2023)
- Security group: `sg-00029cc28dbfed05d`
- S3 bucket:      `indian-equity-ai-frontend-590716140702`
- CloudFront:     `E2J3UOEXP5IWRE` → `d3ltrpuk8i675z.cloudfront.net`

## Running locally

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env       # then edit .env and add your HF_TOKEN
uvicorn app.main:app --reload
```

Backend runs at http://localhost:8000.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://localhost:5173 and by default calls
`http://localhost:8000` (via `VITE_API_BASE_URL` in `.env`).

## Testing

All tests mock every Hugging Face call — `pytest` and `npm test` are
fully offline and complete in seconds.

### Backend (pytest) — 31 tests

```bash
cd backend
source .venv/bin/activate
pytest
```

| File | Covers |
|---|---|
| `tests/test_health_api.py` | `GET /api/v1/health` |
| `tests/test_sentiment_api.py` (11 tests) | HTTP layer: success, empty/missing text, text exceeding max length, and every service-exception → HTTP-status mapping (400/502/504). Service is faked via a FastAPI dependency override. |
| `tests/test_sentiment_service.py` (19 tests) | Service layer: input validation, successful remote inference (list and dict shapes), 401 auth failure, timeout, 400/403/404/unexpected → local fallback, pipeline caching (loaded once), local pipeline failures, and malformed-response handling (missing score, empty list, unrecognized label, non-numeric score, empty local response). `InferenceClient` and `transformers.pipelines.pipeline` are both patched. |

### Frontend (Vitest + React Testing Library) — 6 tests

```bash
cd frontend
npm test
```

`src/App.test.tsx` mocks `services/api.ts` (no real HTTP) and covers:
- Renders header, subtitle, textarea, and Analyze button
- Typing into the textarea updates its value
- Analyze button is disabled with empty input
- Loading state renders "Analyzing…" and disables the button while a request is in flight
- Successful sentiment response renders in the result card (sentiment badge, confidence %, model name)
- API error renders in the error banner and re-enables the button

Watch mode: `npm run test:watch`.

## Docker

The backend has a production-ready [Dockerfile](backend/Dockerfile) and
[`.dockerignore`](backend/.dockerignore). Highlights:

- Base image: `python:3.13-slim`
- Multi-layer caching (deps installed before app copy) → code changes rebuild in seconds
- `.env` is never copied into the image — config comes from runtime env vars
- Runs as non-root `appuser`
- `HF_HOME` set to `/app/.cache/huggingface` (writable by `appuser`)
- Uses CPU-only PyTorch wheel → ~500 MB image instead of ~2.5 GB with CUDA
- Exposes port 8000, runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`

Local build & run:

```bash
cd backend
docker build -t indian-equity-backend:latest .

docker run -d --name indian-equity-backend -p 8000:8000 \
  --env-file .env \
  indian-equity-backend:latest

curl http://localhost:8000/api/v1/health
```

## Project structure

```text
indian-equity-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── health.py         GET /api/v1/health
│   │   │   └── sentiment.py      POST /api/v1/sentiment (Pydantic + DI)
│   │   ├── models/
│   │   │   └── sentiment.py      Sentiment enum + SentimentResult
│   │   ├── services/
│   │   │   └── sentiment_service.py   HF remote + local fallback logic
│   │   ├── config.py             pydantic-settings, env-driven config
│   │   └── main.py               FastAPI factory + CORS + exception handler
│   ├── tests/
│   │   ├── test_health_api.py
│   │   ├── test_sentiment_api.py
│   │   └── test_sentiment_service.py
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── .env.example
│   ├── requirements.txt          CPU-only torch + FastAPI stack
│   ├── requirements-dev.txt      pytest, httpx, etc.
│   └── test_model.py             Standalone HF/local model probe (dev tool)
├── frontend/
│   ├── src/
│   │   ├── components/           Header, TextAnalyzerForm, ResultCard, ...
│   │   ├── hooks/
│   │   │   └── useSentimentAnalysis.ts
│   │   ├── services/
│   │   │   └── api.ts            Typed API client, VITE_API_BASE_URL
│   │   ├── types/
│   │   │   └── sentiment.ts
│   │   ├── constants/
│   │   │   └── examplePrompts.ts
│   │   ├── App.tsx / App.css / index.css
│   │   └── App.test.tsx
│   ├── .env.example              VITE_API_BASE_URL=http://localhost:8000
│   ├── package.json
│   └── vite.config.ts            + vitest config
├── .gitignore
└── README.md
```

## Environment variables

### Backend

| Var | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | Yes | Hugging Face access token (read-only is sufficient) |
| `HF_MODEL_ID` | Yes | Defaults to `alexcruse07/indian-equity-sentiment-model` |
| `HF_INFERENCE_TIMEOUT_SECONDS` | No | Default `5.0` |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins. `*` for demo; specific origins in production |
| `APP_NAME` | No | Displayed in `/health` and logs |

### Frontend

| Var | Purpose |
|---|---|
| `VITE_API_BASE_URL` | Backend base URL. Empty string in production (same-origin via CloudFront), `http://localhost:8000` in local dev |

## Design decisions & tradeoffs

- **Same-origin routing through CloudFront** was chosen over separate
  frontend/backend URLs to avoid CORS setup, mixed-content restrictions
  (HTTPS frontend → HTTP backend), and to give the demo a single clean
  URL.
- **CPU-only PyTorch wheel** — the default `pip install torch` on Linux
  x86_64 pulls the ~2.5 GB CUDA build, which doesn't fit comfortably on
  a t2.micro. Explicitly using
  `--extra-index-url https://download.pytorch.org/whl/cpu` cuts image
  and RAM by ~5×.
- **2 GB swap on EC2** — DistilBERT + torch + Python is ~700 MB resident
  set size on a 1 GB RAM instance. Swap keeps first-load spikes from
  OOM-killing the container.
- **Lazy pipeline import** — `transformers` is imported inside the
  fallback method, not at module top level, so unit tests don't pay the
  cost. The lazy import subtly required patching
  `transformers.pipelines.pipeline` (not `transformers.pipeline`) in
  tests because `transformers` is a `_LazyModule`.
- **Remote-first with local fallback** even though remote never works
  today for this specific model — this is the right production shape;
  if the model is ever deployed to an Inference Provider, no code
  change is required.
- **No secrets in the image** — `.dockerignore` excludes `.env`; secrets
  come from `docker run --env-file` / EC2 user-data / AWS SSM in
  production.
- **Runs as non-root inside the container** even though it's on a
  single-tenant EC2 — small cost, correct posture.

## Production hardening / next steps

Things I deliberately didn't do for a portfolio demo, but would be
required for a real production deployment:

- **Custom domain + ACM certificate** on CloudFront (currently uses the
  default `cloudfront.net` domain with the default SSL cert).
- **Restrict EC2 origin ingress** to CloudFront's managed prefix list
  instead of `0.0.0.0/0`.
- **Move secrets to AWS SSM Parameter Store / Secrets Manager** and
  fetch at container start, instead of an `.env` file on the host.
- **CloudWatch logs agent** + a metrics dashboard (request rate, latency,
  429/5xx counts, memory).
- **Warmup call at FastAPI startup** so the first user doesn't wait
  ~20 s for the model to load.
- **Auto Scaling group + Application Load Balancer** if the workload
  actually needed to scale (this app doesn't).
- **CI/CD** — a GitHub Actions workflow that runs tests, builds the
  frontend, syncs to S3, invalidates CloudFront, and (on release)
  triggers a rolling redeploy of the EC2 container.
- **A better model** — the current fine-tune reports ~70% training
  accuracy and is clearly biased toward BULLISH in adversarial tests.
  A real product would need a stronger, better-evaluated model.
