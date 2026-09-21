"use client";

/**
 * Email and password sign-in.
 *
 * Deliberately plain: one form, one error message. The error never
 * distinguishes "unknown email" from "wrong password", because a message
 * that did would turn the allowlist into an email-enumeration oracle.
 */

import { Loader2, LogIn } from "lucide-react";
import { signIn } from "next-auth/react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

/** Render the credentials form. */
export default function PasswordForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");

    // redirect:false so a failure re-renders this form with a message rather
    // than bouncing to Auth.js's own error page and losing what was typed.
    const result = await signIn("password", {
      email,
      password,
      redirect: false,
    });

    if (result?.error) {
      setError("That email and password combination was not accepted.");
      setBusy(false);
      return;
    }
    window.location.href = "/";
  };

  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="space-y-1.5">
        <Label htmlFor="email">Email</Label>
        <Input
          id="email"
          type="email"
          autoComplete="username"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="you@example.com"
          disabled={busy}
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor="password">Password</Label>
        <Input
          id="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          disabled={busy}
        />
      </div>

      {error && (
        <p className="border-bad/40 bg-bad/10 text-bad rounded-md border px-3 py-2 text-sm">
          {error}
        </p>
      )}

      <Button type="submit" className="w-full" size="lg" disabled={busy}>
        {busy ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <LogIn className="size-4" />
        )}
        Sign in
      </Button>
    </form>
  );
}
