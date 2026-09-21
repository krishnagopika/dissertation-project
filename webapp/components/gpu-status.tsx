"use client";

/**
 * Container status, and an optional early warm-up.
 *
 * There is no "keep warm" pin here, deliberately. An earlier version set
 * `min_containers=1` behind a Start button with a Stop beside it, and that
 * was the wrong shape twice over: during an active session requests arrive
 * far more often than the scaledown window, so the pin bought nothing; and
 * the cheapest possible mistake — forgetting Stop — cost about $1.12/hour
 * indefinitely, roughly $80 over a long weekend.
 *
 * What replaced it is a 15-minute scaledown window, which does the same job
 * and cannot be forgotten, plus a "Warm up" that starts a container early
 * and then lets it expire on that same window. The worst case is one idle
 * tail, about $0.28.
 *
 * Status is served by a CPU-only Modal function reading a heartbeat, so
 * polling "is it warm?" never wakes a GPU to answer.
 */

import { Loader2, RefreshCw, Snowflake, Zap } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { admin } from "@/lib/api";
import type { AdminState } from "@/lib/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/** Poll cadence for status. Cheap, but not free, so not every second. */
const POLL_MS = 15_000;

/**
 * Format an elapsed duration compactly.
 *
 * @param seconds - Elapsed seconds.
 * @returns A short human-readable string.
 */
function duration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/**
 * Render the container status strip.
 *
 * @param props.onWarmChange - Told whenever warmth changes, so the page can
 *   set expectations before someone commits to a cold start.
 */
export default function GpuStatus({
  onWarmChange,
}: {
  onWarmChange?: (warm: boolean) => void;
}) {
  const [state, setState] = useState<AdminState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refresh = useCallback(async () => {
    try {
      const next = await admin("status");
      setState(next);
      onWarmChange?.(next.warm);
      setError("");
    } catch (caught) {
      setError((caught as Error).message);
    }
  }, [onWarmChange]);

  useEffect(() => {
    void refresh();
    pollRef.current = setInterval(() => void refresh(), POLL_MS);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [refresh]);

  const warmUp = async () => {
    setBusy(true);
    setError("");
    try {
      setState(await admin("warm"));
      // The warmup is spawned, not awaited, so the container appears a beat
      // later; nudge the poll rather than leaving the badge stale.
      setTimeout(() => void refresh(), 3000);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const coldSeconds = Math.round(state?.cold_start_seconds ?? 60);

  return (
    <Card className="py-0">
      <CardContent className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
        {!state ? (
          <span className="text-muted-foreground flex items-center gap-2 text-sm">
            <Loader2 className="size-4 animate-spin" />
            checking the GPU
          </span>
        ) : state.warm ? (
          <>
            <span className="text-ok flex items-center gap-2 text-sm font-medium">
              <Zap className="size-4" />
              GPU ready
            </span>
            <span className="text-muted-foreground text-xs">
              Analyses take about 3 seconds. The container shuts itself down{" "}
              {Math.round(state.scaledown_window_sec / 60)} minutes after the
              last one.
            </span>
          </>
        ) : (
          <>
            <span className="text-muted-foreground flex items-center gap-2 text-sm font-medium">
              <Snowflake className="size-4" />
              GPU asleep
            </span>
            <span className="text-muted-foreground text-xs">
              Nothing is running, so nothing is being billed. The first
              analysis waits about{" "}
              <span className="text-foreground font-medium">
                {coldSeconds} seconds
              </span>{" "}
              while the models load; after that each one takes ~3 seconds.
            </span>
          </>
        )}

        <div className="ml-auto flex items-center gap-2">
          {state && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge variant="secondary" className="tabular font-normal">
                  {state.warm
                    ? `$${state.container_usd_per_hour.toFixed(2)}/hr`
                    : "$0.00/hr"}
                </Badge>
              </TooltipTrigger>
              <TooltipContent className="max-w-[260px]">
                {state.warm
                  ? `Billed only while a container exists. If nobody runs
                     anything, it expires and this drops to zero — at most
                     $${state.idle_tail_usd?.toFixed(2)} of idle tail.`
                  : "Scaled to zero. Idle costs nothing."}
                {state.seconds_since_last_request !== null &&
                  ` Last request ${duration(state.seconds_since_last_request)} ago.`}
              </TooltipContent>
            </Tooltip>
          )}

          {state && !state.warm && (
            <Button size="sm" variant="outline" disabled={busy} onClick={() => void warmUp()}>
              {busy ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Zap className="size-3.5" />
              )}
              Warm up now
            </Button>
          )}

          <Button
            variant="ghost"
            size="icon"
            aria-label="Refresh status"
            onClick={() => void refresh()}
          >
            <RefreshCw className="size-3.5" />
          </Button>
        </div>

        {error && (
          <Alert variant="destructive" className="w-full">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
