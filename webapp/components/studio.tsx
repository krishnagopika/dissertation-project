"use client";

/**
 * The working surface: choose audio, analyse, read the result, see the runs.
 *
 * The corpus tab is first because it is the only input that can show WER and
 * a right/wrong verdict — the diagnostics are legible there before a visitor
 * tries their own audio and has nothing to score against.
 */

import { Library, Loader2, Play, Upload } from "lucide-react";
import { useEffect, useState } from "react";

import AnalysisProgress from "@/components/analysis-progress";
import AudioInput from "@/components/audio-input";
import ClipPicker from "@/components/clip-picker";
import GpuStatus from "@/components/gpu-status";
import HistoryPanel, {
  fromLocal,
  fromRunRow,
  type DisplayRun,
} from "@/components/history-panel";
import Results from "@/components/results";
import { analyseAudio, analyseClip, listRuns } from "@/lib/api";
import { toBase64 } from "@/lib/audio";
import { loadHistory, pushHistory, type HistoryEntry } from "@/lib/history";
import type { Analysis, Clip } from "@/lib/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

/** Above this, the wait was a container start rather than the model. */
const COLD_START_THRESHOLD_MS = 15_000;

/** Render the studio. */
export default function Studio() {
  // Upload/record first: it is the claim the project makes -- this works on
  // any speech, not only on the dataset it was tuned against. The corpus tab
  // is the evidence behind that claim and sits one click away.
  const [tab, setTab] = useState("upload");
  const [clip, setClip] = useState<Clip | null>(null);
  const [wav, setWav] = useState<Blob | null>(null);
  const [wavMeta, setWavMeta] = useState<{
    kind: "upload" | "recording";
    label: string;
  } | null>(null);
  const [reference, setReference] = useState("");
  const [result, setResult] = useState<Analysis | null>(null);
  const [wallMs, setWallMs] = useState(0);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [warm, setWarm] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [shared, setShared] = useState<DisplayRun[] | null>(null);

  // localStorage is unavailable during SSR, so the first paint must not read
  // it; hydrating from an effect keeps server and client markup identical.
  useEffect(() => setHistory(loadHistory()), []);

  // The shared log is authoritative when it is configured and reachable.
  // Local history stays as the fallback rather than being replaced, so a
  // database outage degrades to "your own runs" instead of "no runs".
  const refreshShared = async () => {
    try {
      const { enabled, runs } = await listRuns(50);
      setShared(enabled ? runs.map(fromRunRow) : null);
    } catch {
      setShared(null);
    }
  };

  useEffect(() => {
    void refreshShared();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canRun = tab === "corpus" ? clip !== null : wav !== null;

  const run = async () => {
    setBusy(true);
    setError("");
    setResult(null);

    const startedAt = performance.now();
    try {
      const analysis =
        tab === "corpus" && clip
          ? await analyseClip(clip.key)
          : await analyseAudio(
              await toBase64(wav!),
              reference,
              wavMeta?.kind ?? "upload",
              wavMeta?.label ?? "audio",
            );

      const elapsed = Math.round(performance.now() - startedAt);
      setResult(analysis);
      setWallMs(elapsed);

      const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setSelectedRun(id);
      setHistory(
        pushHistory({
          id,
          at: Date.now(),
          source: tab === "corpus" ? (clip?.key ?? "?") : (wavMeta?.label ?? "audio"),
          kind: tab === "corpus" ? "corpus" : (wavMeta?.kind ?? "upload"),
          emotion: analysis.prediction.emotion,
          sentiment: analysis.prediction.sentiment,
          emotionConfidence: analysis.prediction.emotion_confidence,
          transcript: analysis.transcript,
          durationSec: analysis.audio.duration_sec,
          speechRatio: analysis.vad.speech_ratio,
          wer: analysis.wer?.policies.normalised?.wer ?? null,
          gatePasses: analysis.quality_gate.passes,
          emotionCorrect: analysis.correct?.emotion ?? null,
          serverMs: analysis.timings_ms.total_ms ?? 0,
          wallMs: elapsed,
          coldStart: elapsed > COLD_START_THRESHOLD_MS,
          result: analysis,
        }),
      );
      void refreshShared();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <GpuStatus onWarmChange={setWarm} />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Analyse an utterance</CardTitle>
          <CardDescription>
            Voxtral-Mini transcribes the audio and, from the same forward pass,
            yields Whisper-encoder features. A fine-tuned XLM-RoBERTa reads the
            transcript; a trained fusion head combines both. No gold transcript
            is assumed anywhere — every ASR error propagates, which is the
            question this project asks.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="upload">
                <Upload className="size-3.5" />
                Upload or record
              </TabsTrigger>
              <TabsTrigger value="corpus">
                <Library className="size-3.5" />
                MELD corpus
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="mt-4 space-y-4">
              <p className="text-muted-foreground text-sm">
                Your own audio goes through exactly the same pipeline. Supply a
                reference transcript if you want WER; without one the
                prediction still works, and the WER panel says plainly that it
                has nothing to compare against.
              </p>
              <AudioInput
                disabled={busy}
                onAudio={(blob, kind, label) => {
                  setWav(blob);
                  setWavMeta({ kind, label });
                }}
              />
              <div className="space-y-1.5">
                <Label htmlFor="reference">
                  Reference transcript{" "}
                  <span className="text-muted-foreground font-normal">
                    — optional, enables WER
                  </span>
                </Label>
                <Input
                  id="reference"
                  value={reference}
                  onChange={(event) => setReference(event.target.value)}
                  placeholder="What was actually said"
                  disabled={busy}
                />
              </div>
            </TabsContent>
            <TabsContent value="corpus" className="mt-4 space-y-3">
              <p className="text-muted-foreground text-sm">
                Seventy clips, ten per emotion, from the dev and test splits
                only — never training data, or every number here would be
                optimistic. These carry MELD&apos;s gold transcript and gold
                label, so this is the only tab that can show WER and a
                right/wrong verdict. Use it to see what the diagnostics look
                like when there is something to check against.
              </p>
              <ClipPicker
                selected={clip?.key ?? null}
                onSelect={setClip}
                disabled={busy}
              />
            </TabsContent>

          </Tabs>

          <div className="flex flex-wrap items-center gap-3 border-t pt-4">
            <Button disabled={!canRun || busy} onClick={() => void run()}>
              {busy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              {busy ? "Analysing" : "Analyse"}
            </Button>

            {!busy && tab === "corpus" && clip && (
              <span className="text-muted-foreground text-sm">
                <span className="font-mono text-xs">{clip.key}</span> · gold{" "}
                {clip.emotion}/{clip.sentiment}
              </span>
            )}

            {!busy && tab === "upload" && wavMeta && (
              <span className="text-muted-foreground max-w-[30ch] truncate text-sm">
                {wavMeta.label}
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {busy && <AnalysisProgress warm={warm} />}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && <Results result={result} wallMs={wallMs} />}

      <HistoryPanel
        entries={shared ?? history.map(fromLocal)}
        shared={shared !== null}
        selectedId={selectedRun}
        onSelect={(entry) => {
          // Reopen from what was stored rather than re-running: the point of
          // the log is to look back at a result, not to pay for it twice.
          setResult(entry.result);
          setWallMs(entry.wallMs);
          setSelectedRun(entry.id);
          setError("");
        }}
      />
    </div>
  );
}
