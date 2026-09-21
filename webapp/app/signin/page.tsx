import { AudioLines, ShieldCheck } from "lucide-react";

import { signIn } from "@/auth";
import PasswordForm from "@/components/password-form";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Sign-in page.
 *
 * Renders a button per configured provider. Nothing is hard-coded to Google:
 * if only GitHub credentials are set, only GitHub appears. Sign-in itself is
 * still gated by `ALLOWED_EMAILS` in `auth.ts`, so a successful OAuth round
 * trip is necessary but not sufficient.
 *
 * @param props.searchParams - Auth.js redirects failures back here with an
 *   `error` parameter; `AccessDenied` is the allowlist rejection.
 */
export default async function SignInPage({
  searchParams,
}: {
  searchParams: Promise<{ error?: string }>;
}) {
  const { error } = await searchParams;

  const providers = [
    {
      id: "google",
      name: "Google",
      enabled: Boolean(
        process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET,
      ),
    },
    {
      id: "github",
      name: "GitHub",
      enabled: Boolean(
        process.env.AUTH_GITHUB_ID && process.env.AUTH_GITHUB_SECRET,
      ),
    },
    {
      // NOT "AuthKit". That is WorkOS's product name, and it means nothing to
      // the person looking at the button -- they do not care which vendor
      // hosts the page, they care what they are about to be asked for. The
      // label names the action; the helper text below says where it goes.
      id: "workos",
      name: process.env.AUTH_WORKOS_CONNECTION
        ? "your organisation"
        : "email or Google",
      enabled: Boolean(
        process.env.AUTH_WORKOS_ID && process.env.AUTH_WORKOS_SECRET,
      ),
    },
  ].filter((provider) => provider.enabled);

  const hasPassword = Boolean(
    process.env.AUTH_PASSWORD || process.env.AUTH_DEMO_PASSWORD,
  );

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="bg-primary/10 text-primary mb-2 flex size-10 items-center justify-center rounded-lg">
            <AudioLines className="size-5" />
          </div>
          <CardTitle className="text-xl">Speech Emotion Pipeline</CardTitle>
          <CardDescription>
            Emotion and sentiment from speech, with the ASR error and voice
            activity behind each prediction.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3">
          {error === "AccessDenied" && (
            <p className="border-bad/40 bg-bad/10 text-bad rounded-md border px-3 py-2 text-sm">
              That account is not on the allowlist. Ask the owner to add your
              address.
            </p>
          )}
          {error && error !== "AccessDenied" && (
            <p className="border-bad/40 bg-bad/10 text-bad rounded-md border px-3 py-2 text-sm">
              Sign-in failed ({error}). Check the provider configuration.
            </p>
          )}

          {hasPassword && <PasswordForm />}

          {hasPassword && providers.length > 0 && (
            <div className="flex items-center gap-3 py-1">
              <Separator className="flex-1" />
              <span className="text-muted-foreground text-xs">or</span>
              <Separator className="flex-1" />
            </div>
          )}

          {providers.length === 0 && !hasPassword ? (
            <p className="text-muted-foreground text-sm">
              No sign-in method is configured. Set{" "}
              <code className="font-mono text-xs">AUTH_PASSWORD</code> for
              email and password, or one of the OIDC credential pairs (
              <code className="font-mono text-xs">AUTH_WORKOS_*</code>,{" "}
              <code className="font-mono text-xs">AUTH_GOOGLE_*</code>,{" "}
              <code className="font-mono text-xs">AUTH_GITHUB_*</code>), and
              redeploy.
            </p>
          ) : (
            providers.map((provider) => (
              <form
                key={provider.id}
                action={async () => {
                  "use server";
                  await signIn(provider.id, { redirectTo: "/" });
                }}
              >
                <Button type="submit" className="w-full" size="lg">
                  Continue with {provider.name}
                </Button>
              </form>
            ))
          )}
        </CardContent>

        <CardFooter className="flex-col items-start gap-2">
          <p className="text-muted-foreground text-xs">
            Sign-in opens a secure hosted page, then returns you here.
          </p>
          <p className="text-muted-foreground flex items-start gap-2 text-xs">
            <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />
            Access is limited to an allowlist. Each analysis runs on a GPU that
            bills by the second, which is why this is not open sign-up.
          </p>
        </CardFooter>
      </Card>
    </main>
  );
}
