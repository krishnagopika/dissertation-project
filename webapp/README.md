# Web front end

A single-page app over the deployed inference endpoint. Pick a MELD clip,
upload a file, or record yourself; get the emotion and sentiment the trained
pipeline predicts, plus the VAD and WER diagnostics the project uses to decide
whether an utterance is worth trusting.

The GPU work is **not** here. It runs on Modal (`src/modal/serve_emotion.py`),
scales to zero, and is reached through one authenticated proxy route. This app
is a stateless front end and holds no model.

Built with Next.js 15 (App Router), Tailwind v4, shadcn/ui, lucide icons and
Auth.js v5.

---

## Why Vercel rather than EC2

The question is only where the *proxy* lives; the model is on Modal either
way. On that framing Vercel wins on every axis that matters here:

| | Vercel Hobby | EC2 (t4g.small) |
|---|---|---|
| cost of the front end | $0 | ~$12/month, always on |
| TLS, CDN, custom domain | included | your problem |
| deploy | `git push` | AMI, systemd, nginx, certbot |
| function duration | 300s, enough for a cold start | unbounded |
| idle cost | $0 | full price |

EC2 earns its keep only if you need a **fixed IP** for allowlisting, request
bodies over **4.5 MB**, or **WebSockets**. None apply: audio is conditioned to
16 kHz mono in the browser, so 30 seconds is ~1.3 MB of base64.

The one thing you should *not* do is move the model onto EC2. A `g5.xlarge`
(A10G) is roughly **$730/month** running continuously, against Modal's **$0**
when nobody is calling it.

---

## Setup

```bash
cd webapp
npm install
cp .env.example .env.local     # then fill it in
npm run dev
```

### What goes in the environment

**Modal endpoints** — already live; `modal deploy src/modal/serve_emotion.py`
reprints them if you redeploy.

**Modal proxy auth** — dashboard → Settings → Proxy Auth Tokens. The endpoints
declare `requires_proxy_auth=True`, so without these every call 401s.

**Auth.js** — `npx auth secret` for `AUTH_SECRET`, then at least one identity
provider.

### Choosing an identity provider

All three are free; they differ in what sits in the auth path.

| | what you need | good for |
|---|---|---|
| **Google / GitHub** | client id + secret | fewest moving parts, no third party, no MAU ceiling |
| **WorkOS AuthKit** | Client ID + API Key | hosted sign-in page with social, password and magic link |
| **WorkOS SSO** | + a Connection ID | SAML, directory-backed IdPs |

`AUTH_WORKOS_CONNECTION` is the switch between the two WorkOS products, and
they are genuinely different endpoints: AuthKit lives under
`/user_management`, enterprise SSO under `/sso`. Auth.js bundles a provider
only for the second, so AuthKit is defined by hand in `auth.ts` — leave the
connection variable empty and Client ID + API Key is all AuthKit needs.

Redirect URI for whichever you pick:

```
https://<your-app>.vercel.app/api/auth/callback/<google|github|workos>
http://localhost:3000/api/auth/callback/<google|github|workos>
```

### Access control

`ALLOWED_EMAILS` gates sign-in; `ADMIN_EMAILS` gates the warm-container
controls. Both are comma-separated. **An empty `ALLOWED_EMAILS` denies
everyone** — a deployment that forgets to set it should lock its owner out,
not open a GPU to the internet.

## Deploy

```bash
npx vercel            # first run links the project
npx vercel --prod
```

Set every variable in Project Settings → Environment Variables. None are
`NEXT_PUBLIC_`, so none reach the browser — which is the point: with the Modal
key client-side, any visitor could pin a GPU at your expense.

---

## How it fits together

```
browser ──► /api/modal (Vercel, server-side)  ──► Modal endpoints (L4 GPU)
            session required (Auth.js)            scale to zero
            holds MODAL_KEY / MODAL_SECRET         ~$1.12/hr while live
            admin actions need ADMIN_EMAILS
```

`app/api/modal/route.ts` is the only path out. It sets `maxDuration = 300` so
a cold start does not hit the default timeout, and it re-checks the session
and the admin list server-side — hiding a button is not access control.

### The admin controls

`Keep warm` sets `min_containers=1`: one container stays up, every request is
warm, and billing runs continuously at about **$1.12/hour**. `Stop` returns it
to `min_containers=0` and idle costs nothing. The header shows a live spend
meter while pinned, because a warm container that someone forgot about is the
only way this deployment gets expensive.

Status is served by a **CPU-only** Modal function reading a heartbeat Dict, so
polling "is it warm?" never wakes a GPU to answer.

### Audio handling

Everything is decoded, downmixed to mono and resampled to 16 kHz **in the
browser** (`lib/audio.ts`) before upload. That keeps payloads inside Vercel's
4.5 MB body limit, guarantees the preview the user hears is the waveform the
model scored, and normalises across Chrome's webm/opus and Safari's mp4.

### Run history

Kept in `localStorage` (`lib/history.ts`), with a CSV export. Two latencies
per run, because they answer different questions: `server` is what the
endpoint spent on the audio, `wall` is what you waited — on a cold start those
differ by about a minute, and attributing that to the model would be wrong.

---

## What the app shows, and why

| panel | source | why it is there |
|---|---|---|
| emotion + sentiment | the trained fusion head | the actual prediction |
| full probability bars | both softmaxes | a flat distribution is a guess, whatever the top label says |
| transcript with word diff | Voxtral + Levenshtein alignment | WER is one number; the alignment says *which* words |
| WER, both policies | `src/evaluation/text_normalisation.py` | the gap between them is formatting rather than content |
| VAD timeline | Silero | a clip that is mostly silence is visibly that |
| quality gate | the `wer25` keep-list | "would this clip have been in the training set?" |
| per-stage timings | the endpoint | transcription dominates; it is the only autoregressive stage |
| run history | localStorage | what you tried, and how long each took |

Corpus clips come from the **dev and test splits only**, so nothing in the
picker was seen during fine-tuning.
