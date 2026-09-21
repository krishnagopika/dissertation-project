/**
 * Widen the Auth.js session user with the admin flag set in auth.ts.
 *
 * Declared rather than cast at each use site so a component that reads
 * `session.user.isAdmin` is typechecked instead of trusting a cast.
 */
import type { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      isAdmin: boolean;
    } & DefaultSession["user"];
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    isAdmin?: boolean;
  }
}
