import { redirect } from "next/navigation";

import { auth } from "@/auth";
import SiteHeader from "@/components/site-header";
import Studio from "@/components/studio";

/**
 * The application page.
 *
 * A server component so the session is read before anything renders: the
 * admin flag decides whether the start/stop controls exist in the markup at
 * all, rather than being hidden client-side where they could be un-hidden.
 * The API route enforces the same rule again, since hiding a control is not
 * access control.
 */
export default async function Page() {
  const session = await auth();
  if (!session?.user?.email) {
    redirect("/signin");
  }

  return (
    <>
      <SiteHeader
        email={session.user.email}
        isAdmin={session.user.isAdmin}
      />

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
        <Studio isAdmin={session.user.isAdmin} />

        <footer className="text-muted-foreground mt-8 space-y-2 border-t pt-6 text-xs">
          <p className="max-w-[80ch]">
            <span className="text-foreground font-medium">
              What the numbers mean.
            </span>{" "}
            Weighted F1 on MELD&apos;s 7-class emotion task is 0.51 under ASR
            transcripts and 0.63 under gold. That 0.11 gap is the largest
            single effect in this project, and the reason the acoustic branch
            exists: text degrades under transcription error, audio does not.
          </p>
          <p className="max-w-[80ch]">
            <span className="text-foreground font-medium">
              The quality gate
            </span>{" "}
            is the project&apos;s own <code className="font-mono">wer25</code>{" "}
            filter — WER ≤ 0.25 and speech ratio ≥ 0.20 — the rule that
            produced the cleaned training set. A clip that fails it is outside
            the distribution the model was trained on.
          </p>
          <p className="max-w-[80ch]">
            MELD clips are reproduced here for research demonstration under the
            corpus&apos;s own terms; the audio is a 16 kHz mono downmix of
            short excerpts.
          </p>
        </footer>
      </main>
    </>
  );
}
