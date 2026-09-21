"use client";

/**
 * Stage-by-stage progress while an analysis runs.
 *
 * The request is a single round trip, so the server cannot report progress
 * mid-flight. What this shows instead is the real pipeline in its real order,
 * advanced on the stage durations actually measured on the L4 — so the thing
 * lighting up is the thing running, even though the timing is a projection
 * rather than a report.
 *
 * That distinction is kept honest in two ways: nothing here is ever labelled
 * with a number, and the moment the response lands the panel is replaced by
 * the server's own measured per-stage timings. A progress indicator that
 * quoted milliseconds it had guessed would be lying; one that names the
 * current stage is just telling you where you are.
 */

import {
  AudioWaveform,
  Blocks,
  Check,
  FileAudio,
  Layers,
  Loader2,
  Mic,
  Rocket,
  Type,
} from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

/**
 * The pipeline, in execution order, with the durations measured on a warm
 * L4 for a ~9 s clip (see the endpoint's own `timings_ms`). Used only to
 * pace the indicator.
 */
const STAGES = [
  {
    id: "decode",
    icon: FileAudio,
    label: "Decoding audio",
    detail: "ffmpeg to 16 kHz mono",
    ms: 210,
  },
  {
    id: "vad",
    icon: AudioWaveform,
    label: "Detecting speech",
    detail: "Silero VAD segments the clip",
    ms: 280,
  },
  {
    id: "acoustic",
    icon: Layers,
    label: "Extracting acoustic features",
    detail: "Whisper encoder, masked-mean pooled to 1280-d",
    ms: 730,
  },
  {
    id: "transcribe",
    icon: Mic,
    label: "Transcribing",
    detail: "Voxtral-Mini, greedy decode — the slow one",
    ms: 1930,
  },
  {
    id: "text",
    icon: Type,
    label: "Encoding the transcript",
    detail: "fine-tuned XLM-RoBERTa, [CLS] to 768-d",
    ms: 40,
  },
  {
    id: "fuse",
    icon: Blocks,
    label: "Fusing and classifying",
    detail: "ContextThenFusion head, 7 emotions + 3 sentiments",
    ms: 100,
  },
] as const;

/** Measured cold start: container schedule, image pull, then model load. */
const COLD_START_MS = 60_000;

/**
 * Render the progress panel.
 *
 * @param props.warm - Whether a container is already up. When cold, a
 *   loading phase is shown first, because otherwise the pipeline stages
 *   appear frozen for a minute and look broken.
 */
export default function AnalysisProgress({ warm }: { warm: boolean }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const startedAt = Date.now();
    const id = setInterval(() => setElapsed(Date.now() - startedAt), 100);
    return () => clearInterval(id);
  }, []);

  const loadMs = warm ? 0 : COLD_START_MS;
  const inLoad = elapsed < loadMs;
  const sinceLoad = Math.max(0, elapsed - loadMs);

  // Which stage the clock has reached. The last stage stays active rather
  // than completing: the response, not the timer, decides when it is done.
  let cumulative = 0;
  let activeIndex = STAGES.length - 1;
  for (let i = 0; i < STAGES.length; i += 1) {
    cumulative += STAGES[i].ms;
    if (sinceLoad < cumulative) {
      activeIndex = i;
      break;
    }
  }

  const totalMs = STAGES.reduce((sum, stage) => sum + stage.ms, 0);
  const overall = inLoad
    ? (elapsed / loadMs) * 100
    : Math.min(99, (sinceLoad / totalMs) * 100);

  return (
    <Card>
      <CardContent className="space-y-4 py-1">
        {inLoad ? (
          <div className="flex items-start gap-3">
            <Rocket className="text-primary mt-0.5 size-4 shrink-0 animate-pulse" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">Starting the GPU container</p>
              <p className="text-muted-foreground text-xs">
                Scale-to-zero means nothing was running. Loading Voxtral-Mini
                and XLM-RoBERTa onto an L4 — about a minute, once. Every
                analysis after this one takes a few seconds.
              </p>
            </div>
            <span className="tabular text-muted-foreground shrink-0 text-xs">
              {(elapsed / 1000).toFixed(0)}s
            </span>
          </div>
        ) : (
          <ul className="space-y-2.5">
            {STAGES.map((stage, index) => {
              const done = index < activeIndex;
              const active = index === activeIndex;
              const Icon = stage.icon;

              return (
                <li
                  key={stage.id}
                  className={cn(
                    "flex items-start gap-3 transition-opacity duration-300",
                    !done && !active && "opacity-35",
                  )}
                >
                  <span className="mt-0.5 shrink-0">
                    {done ? (
                      <Check className="text-ok size-4" />
                    ) : active ? (
                      <Loader2 className="text-primary size-4 animate-spin" />
                    ) : (
                      <Icon className="text-muted-foreground size-4" />
                    )}
                  </span>

                  <div className="min-w-0 flex-1">
                    <p
                      className={cn(
                        "text-sm leading-tight",
                        active && "font-medium",
                      )}
                    >
                      {stage.label}
                    </p>
                    <p className="text-muted-foreground text-xs leading-tight">
                      {stage.detail}
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        <Progress value={overall} className="h-1" />
      </CardContent>
    </Card>
  );
}
