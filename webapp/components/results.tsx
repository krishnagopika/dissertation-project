"use client";

/**
 * The analysis panel: prediction, VAD, WER, quality gate, timings.
 *
 * Laid out so the two honest caveats are impossible to miss. The quality gate
 * sits beside the prediction rather than below the fold, because "this clip
 * would not have been in the training set" changes how much the label is
 * worth; and where ground truth exists, whether the prediction was right is
 * stated plainly rather than left for the reader to compare.
 */

import {
  AudioWaveform,
  Check,
  Clock,
  FileText,
  Gauge,
  Info,
  Timer,
  X,
} from "lucide-react";

import { alignWords } from "@/lib/diff";
import { cn } from "@/lib/utils";
import { EMOTIONS, SENTIMENTS } from "@/lib/types";
import type { Analysis, Emotion, Sentiment, WerRates } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/** Background fill per emotion, matching the chart tokens. */
const EMOTION_BG: Record<Emotion, string> = {
  neutral: "bg-chart-neutral",
  surprise: "bg-chart-surprise",
  fear: "bg-chart-fear",
  sadness: "bg-chart-sadness",
  joy: "bg-chart-joy",
  disgust: "bg-chart-disgust",
  anger: "bg-chart-anger",
};

const EMOTION_FG: Record<Emotion, string> = {
  neutral: "text-chart-neutral",
  surprise: "text-chart-surprise",
  fear: "text-chart-fear",
  sadness: "text-chart-sadness",
  joy: "text-chart-joy",
  disgust: "text-chart-disgust",
  anger: "text-chart-anger",
};

const SENTIMENT_BG: Record<Sentiment, string> = {
  negative: "bg-bad",
  neutral: "bg-chart-neutral",
  positive: "bg-ok",
};

const SENTIMENT_FG: Record<Sentiment, string> = {
  negative: "text-bad",
  neutral: "text-chart-neutral",
  positive: "text-ok",
};

/**
 * A stack of labelled probability bars.
 *
 * Every bar carries its name and its number, so colour is never the only
 * channel carrying the information.
 */
function Bars<T extends string>({
  probs,
  fills,
  order,
  top,
}: {
  probs: Record<T, number>;
  fills: Record<T, string>;
  order: readonly T[];
  top: T;
}) {
  return (
    <div className="space-y-1.5">
      {order.map((name) => {
        const value = probs[name] ?? 0;
        const isTop = name === top;
        return (
          <div key={name} className="grid grid-cols-[70px_1fr_46px] items-center gap-2">
            <span
              className={cn(
                "text-xs capitalize",
                isTop ? "text-foreground font-medium" : "text-muted-foreground",
              )}
            >
              {name}
            </span>
            <span className="bg-muted h-2 overflow-hidden rounded-full">
              <span
                className={cn(
                  "block h-full rounded-full transition-all duration-500",
                  fills[name],
                  !isTop && "opacity-40",
                )}
                style={{ width: `${Math.max(value * 100, 0.8)}%` }}
              />
            </span>
            <span
              className={cn(
                "tabular text-right text-xs",
                isTop ? "text-foreground font-medium" : "text-muted-foreground",
              )}
            >
              {(value * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Speech segments drawn against the full clip duration. */
function VadTimeline({
  duration,
  segments,
}: {
  duration: number;
  segments: { start: number; end: number }[];
}) {
  const WIDTH = 1000;
  const scale = (t: number) => (duration > 0 ? (t / duration) * WIDTH : 0);

  return (
    <svg
      viewBox={`0 0 ${WIDTH} 36`}
      preserveAspectRatio="none"
      className="h-9 w-full"
      role="img"
      aria-label={`${segments.length} speech segments across ${duration} seconds`}
    >
      <rect
        x={0}
        y={8}
        width={WIDTH}
        height={20}
        rx={4}
        className="fill-muted stroke-border"
      />
      {segments.map((segment, index) => (
        <rect
          key={index}
          x={scale(segment.start)}
          y={8}
          width={Math.max(scale(segment.end - segment.start), 2)}
          height={20}
          rx={3}
          className="fill-primary/75"
        />
      ))}
    </svg>
  );
}

/** Both WER policies side by side. */
function WerTable({
  policies,
}: {
  policies: { exact: WerRates | null; normalised: WerRates | null };
}) {
  const rows: [string, keyof WerRates][] = [
    ["WER", "wer"],
    ["MER", "mer"],
    ["WIL", "wil"],
    ["WIP", "wip"],
    ["CER", "cer"],
  ];

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="h-8">metric</TableHead>
          <TableHead className="h-8 text-right">normalised</TableHead>
          <TableHead className="h-8 text-right">exact</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {rows.map(([label, field]) => (
          <TableRow key={field}>
            <TableCell className="text-muted-foreground py-1.5">{label}</TableCell>
            <TableCell className="tabular py-1.5 text-right font-medium">
              {policies.normalised?.[field].toFixed(3) ?? "—"}
            </TableCell>
            <TableCell className="tabular text-muted-foreground py-1.5 text-right">
              {policies.exact?.[field].toFixed(3) ?? "—"}
            </TableCell>
          </TableRow>
        ))}
        <TableRow>
          <TableCell className="text-muted-foreground py-1.5">S/D/I/H</TableCell>
          {[policies.normalised, policies.exact].map((rates, index) => (
            <TableCell
              key={index}
              className="tabular text-muted-foreground py-1.5 text-right"
            >
              {rates
                ? `${rates.substitutions}/${rates.deletions}/${rates.insertions}/${rates.hits}`
                : "—"}
            </TableCell>
          ))}
        </TableRow>
      </TableBody>
    </Table>
  );
}

/** A headline metric tile. */
function Tile({
  label,
  children,
  sub,
}: {
  label: string;
  children: React.ReactNode;
  sub?: React.ReactNode;
}) {
  return (
    <Card className="gap-0 py-4">
      <CardContent className="px-4">
        <p className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          {label}
        </p>
        <div className="mt-1">{children}</div>
        {sub && <p className="text-muted-foreground mt-1 text-xs">{sub}</p>}
      </CardContent>
    </Card>
  );
}

/**
 * Render one analysis.
 *
 * @param props.result - The server's analysis payload.
 * @param props.wallMs - What the browser measured end to end. Shown next to
 *   the server's own total because on a cold start the two differ by a
 *   minute, and attributing that to the model would be wrong.
 */
export default function Results({
  result,
  wallMs,
}: {
  result: Analysis;
  wallMs: number;
}) {
  const { prediction, vad, wer, quality_gate: gate, timings_ms: timings } = result;
  const truth = result.ground_truth;
  const correct = result.correct;

  // Deliberately not "pass/fail". The gate is a statement about the INPUT,
  // not a verdict on the prediction: a clip outside the training
  // distribution still gets a label, it just deserves more scepticism. Naming
  // it "fails" reads as "the model failed", which is the wrong lesson --
  // flagging its own blind spot is the system working, not breaking.
  const gateVerdict =
    gate.passes === null
      ? { tone: "text-muted-foreground", label: "no reference" }
      : gate.passes
        ? { tone: "text-ok", label: "in distribution" }
        : { tone: "text-warn", label: "out of distribution" };

  const stages: [string, string][] = [
    ["ffmpeg decode", "decode_ms"],
    ["Silero VAD", "vad_ms"],
    ["Whisper encoder", "acoustic_ms"],
    ["Voxtral transcription", "transcribe_ms"],
    ["XLM-RoBERTa", "text_ms"],
    ["fusion head", "head_ms"],
  ];

  return (
    <div className="space-y-4">
      {/* Headline ------------------------------------------------------- */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Tile
          label="Emotion"
          sub={`${(prediction.emotion_confidence * 100).toFixed(1)}% confidence${
            truth ? ` · gold ${truth.emotion}` : ""
          }`}
        >
          <p
            className={cn(
              "text-2xl font-semibold capitalize",
              EMOTION_FG[prediction.emotion],
            )}
          >
            {prediction.emotion}
          </p>
        </Tile>

        <Tile
          label="Sentiment"
          sub={`${(prediction.sentiment_confidence * 100).toFixed(1)}% confidence${
            truth ? ` · gold ${truth.sentiment}` : ""
          }`}
        >
          <p
            className={cn(
              "text-2xl font-semibold capitalize",
              SENTIMENT_FG[prediction.sentiment],
            )}
          >
            {prediction.sentiment}
          </p>
        </Tile>

        <Tile
          label="Quality gate"
          sub={
            <>
              speech {(vad.speech_ratio * 100).toFixed(0)}%
              {wer?.policies.normalised
                ? ` · WER ${wer.policies.normalised.wer.toFixed(2)}`
                : " · no reference"}
            </>
          }
        >
          <p
            className={cn(
              "flex items-center gap-1.5 text-base font-semibold",
              gateVerdict.tone,
            )}
          >
            <Gauge className="size-4" />
            {gateVerdict.label}
          </p>
        </Tile>

        {correct ? (
          <Tile
            label="Against MELD gold"
            sub={`${truth?.speaker ? `${truth.speaker} · ` : ""}${truth?.split} split`}
          >
            <div className="flex flex-wrap gap-1.5">
              <Badge
                variant="outline"
                className={correct.emotion ? "text-ok" : "text-bad"}
              >
                {correct.emotion ? <Check className="size-3" /> : <X className="size-3" />}
                emotion
              </Badge>
              <Badge
                variant="outline"
                className={correct.sentiment ? "text-ok" : "text-bad"}
              >
                {correct.sentiment ? <Check className="size-3" /> : <X className="size-3" />}
                sentiment
              </Badge>
            </div>
          </Tile>
        ) : (
          <Tile
            label="Latency"
            sub={`${result.audio.duration_sec}s of audio · ${(wallMs / 1000).toFixed(1)}s end to end`}
          >
            <p className="tabular flex items-center gap-1.5 text-2xl font-semibold">
              <Timer className="text-muted-foreground size-5" />
              {((timings.total_ms ?? 0) / 1000).toFixed(2)}s
            </p>
          </Tile>
        )}
      </div>

      {gate.reasons.length > 0 && (
        <Card className="border-warn/40 bg-warn/5 py-3">
          <CardContent className="flex gap-2.5 px-4">
            <Info className="text-warn mt-0.5 size-4 shrink-0" />
            <div className="space-y-1 text-sm">
              <p>
                <span className="font-medium">
                  This clip sits outside the training distribution.
                </span>{" "}
                {gate.reasons.join("; ")}.
              </p>
              <p className="text-muted-foreground text-xs">
                The prediction above still stands — it is simply less
                trustworthy than its confidence suggests, and the system can
                tell you so. The{" "}
                <code className="font-mono">wer25</code> keep-list is the same
                rule that built the cleaned training set, so this is the model
                recognising its own blind spot rather than failing.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Distributions --------------------------------------------------- */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Emotion distribution</CardTitle>
            <CardDescription>
              All seven MELD classes. A flat distribution means the model is
              guessing, whatever the top label says.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Bars
              probs={prediction.emotion_probs}
              fills={EMOTION_BG}
              order={EMOTIONS}
              top={prediction.emotion}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Sentiment distribution</CardTitle>
            <CardDescription>
              The second head on the same fused representation, trained
              jointly.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Bars
              probs={prediction.sentiment_probs}
              fills={SENTIMENT_BG}
              order={SENTIMENTS}
              top={prediction.sentiment}
            />
          </CardContent>
        </Card>
      </div>

      {/* Transcript ------------------------------------------------------ */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <FileText className="size-4" />
            Transcript
          </CardTitle>
          <CardDescription>
            Voxtral-Mini, greedy decode. This is what the text branch actually
            read — not a gold transcript.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {wer ? (
            <>
              <p className="bg-muted/60 rounded-md border p-3 font-mono text-sm leading-7">
                {alignWords(wer.reference, result.transcript).map((token, index) => (
                  <Tooltip key={index}>
                    <TooltipTrigger asChild>
                      <span
                        className={cn(
                          "rounded px-0.5",
                          token.op === "sub" && "bg-warn/25",
                          token.op === "ins" && "bg-bad/20",
                          token.op === "del" &&
                            "text-muted-foreground line-through decoration-1",
                        )}
                      >
                        {token.text}{" "}
                      </span>
                    </TooltipTrigger>
                    {token.op !== "hit" && (
                      <TooltipContent>
                        {token.op === "sub"
                          ? `expected "${token.expected}"`
                          : token.op === "del"
                            ? "missing from the transcript"
                            : "not in the reference"}
                      </TooltipContent>
                    )}
                  </Tooltip>
                ))}
              </p>

              <div className="text-muted-foreground flex flex-wrap gap-3 text-xs">
                <span>matched</span>
                <span className="bg-warn/25 rounded px-1">substituted</span>
                <span className="bg-bad/20 rounded px-1">inserted</span>
                <span className="line-through">deleted</span>
              </div>

              <p className="text-muted-foreground text-xs">
                <span className="font-medium">Reference</span> (
                {wer.reference_source === "meld_gold" ? "MELD gold" : "yours"}):{" "}
                {wer.reference}
              </p>
            </>
          ) : (
            <>
              <p className="bg-muted/60 rounded-md border p-3 font-mono text-sm leading-7">
                {result.transcript || "(empty)"}
              </p>
              <p className="text-muted-foreground text-xs">
                No reference transcript, so WER cannot be computed. Type what
                was actually said in the reference box to score it.
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {/* Diagnostics ----------------------------------------------------- */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm">
              <AudioWaveform className="size-4" />
              Voice activity
            </CardTitle>
            <CardDescription>
              Silero VAD. Below {gate.vad_min_speech_ratio.toFixed(2)} the
              training filter would have dropped this clip.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <VadTimeline
              duration={result.audio.duration_sec}
              segments={vad.segments}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="outline"
                className={gate.speech_ratio_ok ? "text-ok" : "text-warn"}
              >
                {(vad.speech_ratio * 100).toFixed(1)}% speech
              </Badge>
              <span className="text-muted-foreground tabular text-xs">
                {vad.speech_seconds}s of {result.audio.duration_sec}s ·{" "}
                {vad.num_segments} segment{vad.num_segments === 1 ? "" : "s"}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Word error rate</CardTitle>
            <CardDescription>
              Both normalisation policies. The gap between them is formatting
              rather than content.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {wer ? (
              <>
                <WerTable policies={wer.policies} />
                <p className="text-muted-foreground mt-2 text-xs">
                  {wer.reference_words} reference words,{" "}
                  {wer.hypothesis_words} transcribed.
                </p>
              </>
            ) : (
              <p className="text-muted-foreground text-sm">
                No reference to score against. Pick a corpus clip, or supply
                the text yourself.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Timings --------------------------------------------------------- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Clock className="size-4" />
            Where the time went
          </CardTitle>
          <CardDescription>
            Per stage, on the GPU. Transcription dominates: it is the only
            stage that decodes autoregressively.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-1.5">
            {stages.map(([label, field]) => {
              const ms = timings[field] ?? 0;
              const share = timings.total_ms ? (ms / timings.total_ms) * 100 : 0;
              return (
                <div
                  key={field}
                  className="grid grid-cols-[150px_1fr_78px] items-center gap-2"
                >
                  <span className="text-muted-foreground truncate text-xs">
                    {label}
                  </span>
                  <span className="bg-muted h-2 overflow-hidden rounded-full">
                    <span
                      className="bg-primary/70 block h-full rounded-full"
                      style={{ width: `${Math.max(share, 0.6)}%` }}
                    />
                  </span>
                  <span className="tabular text-muted-foreground text-right text-xs">
                    {ms} ms · {share.toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>

          <div className="flex flex-wrap gap-4 border-t pt-3 text-xs">
            <span>
              <span className="text-muted-foreground">server total </span>
              <span className="tabular font-medium">{timings.total_ms ?? 0} ms</span>
            </span>
            <span>
              <span className="text-muted-foreground">end to end </span>
              <span className="tabular font-medium">{wallMs} ms</span>
            </span>
            <span className="text-muted-foreground">
              the difference is network, and container start when cold
            </span>
          </div>

          <p className="text-muted-foreground border-t pt-3 text-xs">
            {String(result.model.head)} · fusion {String(result.model.fusion)} ·
            context window K={String(result.model.context_window)} · dev
            weighted F1 {String(result.model.head_dev_emotion_wf1)}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
