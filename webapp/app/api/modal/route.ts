/**
 * The only path between the browser and Modal.
 *
 * Everything the page needs goes through this one route, for two reasons:
 *
 *   Credentials. The Modal proxy-auth headers and the admin token live in
 *   server-side environment variables. If the browser called Modal directly
 *   it would have to carry them, and anyone with devtools could pin a GPU
 *   container to their own amusement at the project's expense.
 *
 *   Duration. A cold Modal container takes tens of seconds to load 9.3 GB of
 *   Voxtral onto an L4. Vercel Functions allow up to 300 s (Hobby and Pro
 *   alike, with Fluid compute), so the wait fits comfortably -- but only if
 *   maxDuration is raised from the default, which is what the export below
 *   does.
 */

import { NextRequest, NextResponse } from "next/server";

import { auth, isAdmin } from "@/auth";
import { listRuns, recordRun, runLogEnabled } from "@/lib/runs";

/** Cold start (~45 s) plus inference, with room to spare. Hobby caps at 300. */
export const maxDuration = 300;

/** No caching: every response depends on the audio in the request body. */
export const dynamic = "force-dynamic";

interface Env {
  classifyUrl: string;
  clipsUrl: string;
  clipUrl: string;
  adminUrl: string;
  key: string;
  secret: string;
}

/**
 * Read and validate the server environment.
 *
 * @returns The configured endpoints and credentials.
 * @throws If a required variable is missing, with a message naming it --
 *   a misconfigured deployment should say so rather than 401 mysteriously.
 */
function readEnv(): Env {
  const required = {
    classifyUrl: process.env.MODAL_CLASSIFY_URL,
    clipsUrl: process.env.MODAL_CLIPS_URL,
    clipUrl: process.env.MODAL_CLIP_URL,
    adminUrl: process.env.MODAL_ADMIN_URL,
    key: process.env.MODAL_KEY,
    secret: process.env.MODAL_SECRET,
  };

  for (const [name, value] of Object.entries(required)) {
    if (!value) {
      throw new Error(
        `Missing environment variable for "${name}". See webapp/.env.example.`,
      );
    }
  }

  return required as Record<keyof typeof required, string>;
}

/**
 * Call a Modal endpoint with proxy auth attached.
 *
 * @param url - The Modal endpoint.
 * @param env - Server configuration.
 * @param init - Fetch options; headers are merged, not replaced.
 * @returns The parsed JSON body.
 * @throws If Modal returns a non-2xx, with the status and body included so
 *   the failure is diagnosable from the browser console.
 */
async function callModal(
  url: string,
  env: Env,
  init: RequestInit = {},
): Promise<unknown> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "Modal-Key": env.key,
      "Modal-Secret": env.secret,
      ...(init.headers ?? {}),
    },
    cache: "no-store",
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`Modal returned ${response.status}: ${text.slice(0, 400)}`);
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Modal returned non-JSON: ${text.slice(0, 200)}`);
  }
}

/**
 * Proxy one action to Modal.
 *
 * Body: `{ action, ...payload }` where action is one of
 * `clips` (list the corpus), `clip` (fetch one clip's audio),
 * `classify` (analyse audio), `runs` (read the shared run log), or
 * `admin` (start/stop/status).
 *
 * @param request - The incoming Next.js request.
 * @returns Modal's JSON response, or `{ error }` with a 4xx/5xx status.
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  // Authorisation happens here, not only in middleware. Middleware keeps
  // anonymous traffic off the function; this keeps a signed-in but
  // non-admin user from reaching the start control by posting to the route
  // directly, which the UI alone could not prevent.
  const session = await auth();
  if (!session?.user?.email) {
    return NextResponse.json({ error: "Not signed in." }, { status: 401 });
  }

  let env: Env;
  try {
    env = readEnv();
  } catch (error) {
    return NextResponse.json(
      { error: (error as Error).message },
      { status: 500 },
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }

  const action = String(body.action ?? "");

  try {
    switch (action) {
      case "clips":
        return NextResponse.json(await callModal(env.clipsUrl, env));

      case "clip": {
        const key = String(body.key ?? "");
        if (!key) {
          return NextResponse.json({ error: "clip needs a key." }, { status: 400 });
        }
        const url = `${env.clipUrl}?key=${encodeURIComponent(key)}`;
        return NextResponse.json(await callModal(url, env));
      }

      case "classify": {
        const payload: Record<string, unknown> = {};
        if (body.key) payload.key = body.key;
        if (body.audio_b64) payload.audio_b64 = body.audio_b64;
        if (body.reference) payload.reference = body.reference;
        if (!payload.key && !payload.audio_b64) {
          return NextResponse.json(
            { error: "classify needs either a key or audio_b64." },
            { status: 400 },
          );
        }

        const started = Date.now();
        const analysis = await callModal(env.classifyUrl, env, {
          method: "POST",
          body: JSON.stringify(payload),
        });

        // Log the run, but never let logging break the analysis: recordRun
        // swallows its own failures and the await is guarded besides. The
        // email comes from the session, not the body, so a caller cannot
        // attribute a run to someone else.
        const result = analysis as Record<string, unknown>;
        if (!result.error) {
          await recordRun({
            email: session.user.email,
            analysis: result as never,
            wallMs: Date.now() - started,
            sourceKind: payload.key
              ? "corpus"
              : ((body.sourceKind as "upload" | "recording") ?? "upload"),
            source: String(payload.key ?? body.source ?? "audio"),
          });
        }

        return NextResponse.json(analysis);
      }

      case "runs":
        return NextResponse.json({
          enabled: runLogEnabled(),
          runs: await listRuns(Number(body.limit ?? 50)),
        });

      case "admin": {
        const adminAction = String(body.adminAction ?? "status");

        // `status` and `warm` are safe for any signed-in user: neither can
        // leave a GPU running, because the container expires on the scaledown
        // window regardless. Only `unpin`, which touches the autoscaler, needs
        // the admin list.
        if (adminAction === "unpin" && !isAdmin(session.user.email)) {
          return NextResponse.json(
            {
              error:
                "Only an admin can change the autoscaler. Ask the owner " +
                "to add you to ADMIN_EMAILS.",
            },
            { status: 403 },
          );
        }

        return NextResponse.json(
          await callModal(env.adminUrl, env, {
            method: "POST",
            body: JSON.stringify({
              action: adminAction,
              wait: Boolean(body.wait),
            }),
          }),
        );
      }

      default:
        return NextResponse.json(
          { error: `Unknown action "${action}".` },
          { status: 400 },
        );
    }
  } catch (error) {
    return NextResponse.json(
      { error: (error as Error).message },
      { status: 502 },
    );
  }
}
