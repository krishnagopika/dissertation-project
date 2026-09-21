"use client";

/**
 * Warm-container controls and cost meter.
 *
 * The endpoint scales to zero, which is what makes it nearly free — and also
 * what makes the first request of a session wait out a cold start. Start pins
 * one container (`min_containers=1`); Stop releases it.
 *
 * The running spend meter is not decoration. Pinning bills continuously
 * whether or not anyone uses the app, so the number that makes someone press
 * Stop has to be on screen.
 *
 * Status polling is free: it is served by a CPU-only Modal function reading a
 * heartbeat Dict, so asking "is it warm?" never wakes a GPU to answer.
 */

import {
  Activity,
  CircleDot,
  Loader2,
  Play,
  RefreshCw,
  Snowflake,
  Square,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { admin } from "@/lib/api";
import type { AdminState } from "@/lib/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/** Poll cadence for status. Cheap, but not free, so not every second. */
const POLL_MS = 10_000;

/**
 * Format an elapsed duration compactly.
 *
 * @param seconds - Elapsed seconds.
 * @returns A short human-readable string.
 */
function duration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/**
 * Render the container status strip.
 *
 * @param props.isAdmin - Whether to show the start/stop controls at all.
 * @param props.onWarmChange - Told whenever warmth changes, so the page can
 *   warn about a cold start before the user commits to one.
 */
export default function GpuStatus({
  isAdmin,
  onWarmChange,
}: {
  isAdmin: boolean;
  onWarmChange?: (warm: boolean) => void;
}) {
  const [state, setState] = useState<AdminState | null>(null);
  const [busy, setBusy] = useState<"start" | "stop" | null>(null);
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);
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

  // Animates the meter between polls; the server remains authoritative.
  useEffect(() => {
    const id = setInterval(() => setTick((value) => value + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const act = async (action: "start" | "stop") => {
    setBusy(action);
    setError("");
    try {
      const next = await admin(action);
      setState(next);
      onWarmChange?.(next.warm);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const liveSpend =
    state?.pinned && state.estimated_spend_usd !== undefined
      ? state.estimated_spend_usd +
        (tick % (POLL_MS / 1000)) * (state.container_usd_per_hour / 3600)
      : state?.estimated_spend_usd;

  const status = !state
    ? { icon: Loader2, label: "checking", tone: "text-muted-foreground", spin: true }
    : state.pinned
      ? { icon: CircleDot, label: "pinned — billing", tone: "text-bad", spin: false }
      : state.warm
        ? { icon: Zap, label: "warm", tone: "text-ok", spin: false }
        : { icon: Snowflake, label: "cold — scaled to zero", tone: "text-muted-foreground", spin: false };

  const StatusIcon = status.icon;

  return (
    <Card className="py-0">
      <CardContent className="flex flex-wrap items-center gap-x-4 gap-y-3 px-4 py-3">
        <div className="flex items-center gap-2">
          <StatusIcon
            className={`size-4 ${status.tone} ${status.spin ? "animate-spin" : ""}`}
          />
          <span className="text-sm font-medium">{status.label}</span>
        </div>

        {state?.load_seconds ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <Badge variant="secondary" className="tabular gap-1 font-normal">
                <Activity className="size-3" />
                {state.load_seconds}s load
              </Badge>
            </TooltipTrigger>
            <TooltipContent>
              How long the models last took to reach the GPU
            </TooltipContent>
          </Tooltip>
        ) : null}

        {state && !state.pinned && !state.warm && (
          <span className="text-muted-foreground text-xs">
            The next analysis pays a cold start of roughly{" "}
            {state.load_seconds ?? 60}s. Everything after is about a second.
          </span>
        )}

        {state?.pinned && (
          <span className="text-muted-foreground tabular text-xs">
            pinned {duration(state.pinned_seconds ?? 0)} ·{" "}
            <span className="text-bad font-medium">
              ${(liveSpend ?? 0).toFixed(4)}
            </span>{" "}
            spent · ${state.estimated_spend_per_day_usd?.toFixed(2)}/day if left on
          </span>
        )}

        <div className="ml-auto flex items-center gap-2">
          {state && (
            <span className="text-muted-foreground tabular hidden text-xs sm:inline">
              ${state.container_usd_per_hour.toFixed(2)}/hr live
            </span>
          )}

          <Button
            variant="ghost"
            size="icon"
            aria-label="Refresh status"
            onClick={() => void refresh()}
          >
            <RefreshCw className="size-3.5" />
          </Button>

          {isAdmin && (
            <>
              <Separator orientation="vertical" className="h-5" />
              <Button
                size="sm"
                disabled={busy !== null || state?.pinned}
                onClick={() => void act("start")}
              >
                {busy === "start" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Play className="size-3.5" />
                )}
                Keep warm
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={busy !== null || !state?.pinned}
                onClick={() => void act("stop")}
              >
                {busy === "stop" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Square className="size-3.5" />
                )}
                Stop
              </Button>
            </>
          )}
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
