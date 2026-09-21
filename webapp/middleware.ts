/**
 * Require a session for everything except the sign-in flow and static assets.
 *
 * Belt and braces alongside the checks inside the API route: middleware keeps
 * unauthenticated traffic from reaching a function at all, which matters when
 * the function's job is to spend money on a GPU.
 */
export { auth as middleware } from "@/auth";

export const config = {
  matcher: [
    // Everything except Next internals, the auth endpoints, the sign-in page
    // and favicon. Negative lookahead rather than an allowlist so a new route
    // is protected by default.
    "/((?!api/auth|signin|_next/static|_next/image|favicon.ico).*)",
  ],
};
