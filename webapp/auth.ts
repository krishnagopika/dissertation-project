/**
 * Authentication: Auth.js v5 over OpenID Connect.
 *
 * Why this, and why it is free
 * ----------------------------
 * Auth.js (the library formerly called NextAuth) is MIT-licensed and free
 * with no user cap. It speaks generic OIDC, so the identity provider is a
 * configuration choice rather than a vendor lock-in. Three are wired up
 * below; the sign-in page shows whichever have credentials set.
 *
 *   Google / GitHub   Direct OAuth. Free forever, no user ceiling, no third
 *                     party sitting in the auth path, and about five minutes
 *                     to set up. The right default here.
 *
 *   WorkOS AuthKit    A hosted identity layer, free to a very high monthly
 *                     active user count. Worth it when you want a hosted
 *                     sign-in UI, email/password and magic links alongside
 *                     social, or — its real purpose — enterprise SSO: SAML,
 *                     SCIM directory sync, per-organisation connections.
 *                     For an allowlist of about five people none of that
 *                     applies, and it adds a vendor and a second dashboard
 *                     for no gain. Set AUTH_WORKOS_* if you want it anyway.
 *
 * A university IdP works too: add it as a generic OIDC provider —
 * `{ id, name, type: "oidc", issuer, clientId, clientSecret }` — and nothing
 * else in the app changes.
 *
 * Why an allowlist rather than "anyone with a Google account"
 * -----------------------------------------------------------
 * Every signed-in request can spend GPU seconds, and the start control can
 * pin a container that bills continuously. A public URL with open sign-up is
 * therefore a public wallet. `ALLOWED_EMAILS` gates entry and `ADMIN_EMAILS`
 * gates the warm-container controls, both as comma-separated lists.
 *
 * An empty `ALLOWED_EMAILS` denies everyone rather than admitting everyone.
 * That direction is deliberate: a deployment that forgets to configure the
 * list should lock its operator out, not open its GPU to the internet.
 */

import NextAuth from "next-auth";
import GitHub from "next-auth/providers/github";
import Google from "next-auth/providers/google";
import WorkOS from "next-auth/providers/workos";
import type { NextAuthConfig } from "next-auth";

/**
 * Parse a comma-separated env list into a lowercase set.
 *
 * @param raw - The environment variable value, possibly undefined.
 * @returns Lowercased, trimmed entries with blanks dropped.
 */
function emailSet(raw: string | undefined): Set<string> {
  return new Set(
    (raw ?? "")
      .split(",")
      .map((entry) => entry.trim().toLowerCase())
      .filter(Boolean),
  );
}

/**
 * Whether an email may use the app at all.
 *
 * @param email - The address from the OIDC profile.
 * @returns True when the address is on the allowlist.
 */
export function isAllowed(email: string | null | undefined): boolean {
  if (!email) return false;
  return emailSet(process.env.ALLOWED_EMAILS).has(email.toLowerCase());
}

/**
 * Whether an email may start and stop the GPU container.
 *
 * @param email - The address from the OIDC profile.
 * @returns True when the address is on the admin list.
 */
export function isAdmin(email: string | null | undefined): boolean {
  if (!email) return false;
  return emailSet(process.env.ADMIN_EMAILS).has(email.toLowerCase());
}


/**
 * WorkOS AuthKit as an Auth.js provider.
 *
 * Auth.js bundles a WorkOS provider, but it targets enterprise **SSO**
 * (`/sso/authorize`), which cannot run without a connection id. AuthKit --
 * the hosted sign-in page with social, password and magic-link, and the part
 * that is free to a high monthly-active-user count -- lives under
 * `/user_management` instead, so it needs its own definition.
 *
 * Two places where AuthKit departs from plain OAuth2 and this has to
 * compensate:
 *
 *   The token endpoint is `/authenticate`, and its response carries the
 *   `user` object inline rather than an `id_token` or a userinfo URL to go
 *   and fetch. So `userinfo.request` reads the user straight off the token
 *   set instead of making a second call.
 *
 *   That response omits `token_type`, which oauth4webapi treats as a
 *   protocol violation and rejects. `conform` puts it back.
 *
 * @returns A provider definition for AuthKit.
 */
function workosAuthKit(): NextAuthConfig["providers"][number] {
  const base = "https://api.workos.com/user_management";

  return {
    id: "workos",
    name: "WorkOS",
    type: "oauth",
    clientId: process.env.AUTH_WORKOS_ID,
    clientSecret: process.env.AUTH_WORKOS_SECRET,
    authorization: {
      url: `${base}/authorize`,
      params: { provider: "authkit", response_type: "code" },
    },
    token: {
      url: `${base}/authenticate`,
      async conform(response: Response) {
        const body = await response.clone().json();
        if (body.token_type) return response;
        return new Response(JSON.stringify({ ...body, token_type: "bearer" }), {
          status: response.status,
          headers: response.headers,
        });
      },
    },
    userinfo: {
      url: `${base}/authenticate`,
      async request({ tokens }: { tokens: Record<string, unknown> }) {
        return tokens.user;
      },
    },
    profile(profile: Record<string, unknown>) {
      return {
        id: String(profile.id),
        name:
          [profile.first_name, profile.last_name].filter(Boolean).join(" ") ||
          String(profile.email ?? ""),
        email: String(profile.email ?? ""),
        image: (profile.profile_picture_url as string | null) ?? null,
      };
    },
    checks: ["state"],
    style: { bg: "#6363f1", text: "#fff" },
  };
}

/** Providers, filtered to the ones actually configured. */
const providers: NextAuthConfig["providers"] = [];

if (process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET) {
  providers.push(Google);
}
if (process.env.AUTH_GITHUB_ID && process.env.AUTH_GITHUB_SECRET) {
  providers.push(GitHub);
}
if (process.env.AUTH_WORKOS_ID && process.env.AUTH_WORKOS_SECRET) {
  // WorkOS sells two different things behind one set of credentials, and
  // they use different endpoints. Which one you get is decided by whether a
  // connection id is configured.
  if (process.env.AUTH_WORKOS_CONNECTION) {
    // Enterprise SSO. Auth.js's bundled provider points at /sso/authorize,
    // which routes by connection rather than discovering one from the email
    // domain -- so the connection id is mandatory, not optional. Without it
    // WorkOS rejects the authorize request.
    providers.push(WorkOS({ connection: process.env.AUTH_WORKOS_CONNECTION }));
  } else {
    providers.push(workosAuthKit());
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers,
  pages: { signIn: "/signin", error: "/signin" },
  // JWT rather than a database session: this app stores nothing per user, so
  // a database would be infrastructure carrying no state.
  session: { strategy: "jwt" },
  callbacks: {
    /**
     * Refuse sign-in outright for addresses off the allowlist.
     *
     * Enforced here rather than only in the UI so a hand-crafted request to
     * the API cannot bypass it with a valid but unlisted Google account.
     */
    signIn({ profile, user }) {
      return isAllowed(profile?.email ?? user?.email);
    },
    /** Carry the admin flag on the token so the API need not re-read env. */
    jwt({ token }) {
      token.isAdmin = isAdmin(token.email);
      return token;
    },
    /** Expose the admin flag to the client, for showing or hiding controls. */
    session({ session, token }) {
      if (session.user) {
        session.user.isAdmin = Boolean(token.isAdmin);
      }
      return session;
    },
  },
});
