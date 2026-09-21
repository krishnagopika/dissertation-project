"use client";

/**
 * Previous runs, with both latencies. Selecting a row reopens that analysis.
 *
 * Two latency columns because they answer different questions. `server` is
 * what the endpoint spent on the audio — the model's speed, comparable
 * between runs. `wall` is what you waited, which on a cold start includes the
 * minute of container start and model load. Showing only the first would hide
 * cold starts; only the second would blame the model for them.
 *
 * Scrolling note, because it was a bug: this uses ONE plain scroll container,
 * not a Radix ScrollArea wrapped around a shadcn Table. Table ships its own
 * `overflow-x-auto` wrapper, so nesting the two gave a vertical viewport
 * containing a horizontal viewport, and horizontal scroll got trapped between
 * them. One element owning both axes fixes it.
 */

import { Download, History } from "lucide-react";
import { useMemo } from "react";

import { toCsv, type HistoryEntry } from "@/lib/history";
import type { RunRow } from "@/lib/runs-shared";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

/**
 * Median of a numeric list.
 *
 * Median rather than mean because one cold start would drag a mean of five
 * warm requests somewhere that describes neither.
 *
 * @param values - The numbers.
 * @returns The median, rounded.
 */
function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return Math.round(
    sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
  );
}

const KIND_LABEL: Record<HistoryEntry["kind"], string> = {
  corpus: "clip",
  upload: "file",
  recording: "mic",
};

/** What a row needs to render, from either the database or localStorage. */
export interface DisplayRun {
  id: string;
  at: number;
  who: string | null;
  source: string;
  kind: HistoryEntry["kind"];
  emotion: string;
  sentiment: string;
  wer: number | null;
  gatePasses: boolean | null;
  emotionCorrect: boolean | null;
  serverMs: number;
  wallMs: number;
  coldStart: boolean;
  entry: HistoryEntry;
}

/**
 * Normalise a database row into the display shape.
 *
 * @param row - A row from the shared run log.
 * @returns The row as the table renders it.
 */
export function fromRunRow(row: RunRow): DisplayRun {
  return {
    id: row.id,
    at: new Date(row.created_at).getTime(),
    who: row.user_email,
    source: row.source,
    kind: row.source_kind,
    emotion: row.predicted_emotion,
    sentiment: row.predicted_sentiment,
    wer: row.wer_normalised,
    gatePasses: row.gate_passes,
    emotionCorrect: row.emotion_correct,
    serverMs: row.server_ms,
    wallMs: row.wall_ms,
    coldStart: row.cold_start,
    entry: {
      id: row.id,
      at: new Date(row.created_at).getTime(),
      source: row.source,
      kind: row.source_kind,
      emotion: row.predicted_emotion,
      sentiment: row.predicted_sentiment,
      emotionConfidence: row.emotion_confidence,
      transcript: row.transcript,
      durationSec: row.duration_sec,
      speechRatio: row.speech_ratio,
      wer: row.wer_normalised,
      gatePasses: row.gate_passes,
      emotionCorrect: row.emotion_correct,
      serverMs: row.server_ms,
      wallMs: row.wall_ms,
      coldStart: row.cold_start,
      result: row.result,
    },
  };
}

/**
 * Normalise a browser-local entry into the display shape.
 *
 * @param entry - A localStorage history entry.
 * @returns The entry as the table renders it. `who` is null: local history
 *   predates knowing who is signed in, and inventing an attribution would be
 *   worse than showing none.
 */
export function fromLocal(entry: HistoryEntry): DisplayRun {
  return { ...entry, who: null, entry };
}

/**
 * Render the run history.
 *
 * @param props.entries - Runs, newest first.
 * @param props.onSelect - Reopens a past run in the results panel.
 * @param props.selectedId - The run currently shown, highlighted in the list.
 */
export default function HistoryPanel({
  entries,
  onSelect,
  selectedId,
  shared,
}: {
  entries: DisplayRun[];
  onSelect: (entry: HistoryEntry) => void;
  selectedId: string | null;
  shared: boolean;
}) {
  const stats = useMemo(() => {
    if (entries.length === 0) return null;
    const warm = entries.filter((entry) => !entry.coldStart);
    const scored = entries.filter((entry) => entry.emotionCorrect !== null);
    return {
      runs: entries.length,
      medianServerMs: median(entries.map((entry) => entry.serverMs)),
      medianWarmWallMs: warm.length ? median(warm.map((e) => e.wallMs)) : null,
      coldStarts: entries.length - warm.length,
      accuracy: scored.length
        ? scored.filter((entry) => entry.emotionCorrect).length / scored.length
        : null,
      scored: scored.length,
    };
  }, [entries]);

  const download = () => {
    const blob = new Blob([toCsv(entries.map((e) => e.entry))], {
      type: "text/csv",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "emotion-runs.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <History className="size-4" />
          Run history
        </CardTitle>
        <CardDescription>
          {stats ? (
            <>
              {stats.runs} run{stats.runs === 1 ? "" : "s"} · median{" "}
              <span className="tabular">{stats.medianServerMs} ms</span> on the
              server
              {stats.medianWarmWallMs !== null && (
                <>
                  {" · "}
                  <span className="tabular">{stats.medianWarmWallMs} ms</span>{" "}
                  end to end when warm
                </>
              )}
              {stats.coldStarts > 0 && (
                <>
                  {" · "}
                  {stats.coldStarts} cold start
                  {stats.coldStarts === 1 ? "" : "s"}
                </>
              )}
              {stats.accuracy !== null && (
                <>
                  {" · "}
                  {(stats.accuracy * 100).toFixed(0)}% correct on {stats.scored}{" "}
                  labelled
                </>
              )}
              . Select a row to reopen it.
              {shared
                ? " Shared across everyone signed in."
                : " Stored in this browser only."}
            </>
          ) : (
            "Nothing yet. Analyse a clip and it will appear here."
          )}
        </CardDescription>
        {entries.length > 0 && (
          // Export, but no Clear. A one-click destructive control sitting
          // beside a log that is evidence for a write-up is a footgun, and
          // the list trims itself at MAX_ENTRIES anyway, so nothing needs
          // manual pruning. clearHistory() still exists for the console.
          <CardAction>
            <Button variant="ghost" size="sm" onClick={download}>
              <Download className="size-3.5" />
              Export CSV
            </Button>
          </CardAction>
        )}
      </CardHeader>

      {entries.length > 0 && (
        <CardContent>
          {/* One container, both axes. See the note at the top of the file. */}
          <div className="max-h-[300px] overflow-auto rounded-md border">
            <table className="w-full min-w-[880px] border-collapse text-sm">
              <thead className="bg-muted/80 supports-[backdrop-filter]:bg-muted/60 sticky top-0 z-10 backdrop-blur">
                <tr className="[&>th]:text-muted-foreground [&>th]:px-3 [&>th]:py-2 [&>th]:text-left [&>th]:text-[11px] [&>th]:font-medium [&>th]:tracking-wide [&>th]:uppercase">
                  <th className="w-[84px]">time</th>
                  <th className="w-[150px]">who</th>
                  <th className="w-[190px]">input</th>
                  <th className="w-[150px]">predicted</th>
                  <th className="w-[120px]">sentiment</th>
                  <th className="w-[76px] !text-right">WER</th>
                  <th className="w-[92px] !text-right">server</th>
                  <th className="w-[92px] !text-right">end&nbsp;to&nbsp;end</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => (
                  <tr
                    key={entry.id}
                    tabIndex={0}
                    role="button"
                    aria-pressed={selectedId === entry.id}
                    onClick={() => onSelect(entry.entry)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onSelect(entry.entry);
                      }
                    }}
                    className={cn(
                      "hover:bg-accent/60 focus-visible:ring-ring cursor-pointer border-t outline-none focus-visible:ring-2 focus-visible:ring-inset",
                      selectedId === entry.id && "bg-accent",
                      "[&>td]:px-3 [&>td]:py-2 [&>td]:align-middle",
                    )}
                  >
                    <td className="text-muted-foreground tabular text-xs whitespace-nowrap">
                      {new Date(entry.at).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </td>

                    <td className="text-muted-foreground text-xs">
                      <span className="block max-w-[140px] truncate">
                        {entry.who ?? "—"}
                      </span>
                    </td>

                    <td className="text-xs">
                      <span className="flex items-center gap-1.5">
                        <Badge
                          variant="secondary"
                          className="shrink-0 px-1.5 py-0 text-[10px] font-normal"
                        >
                          {KIND_LABEL[entry.kind]}
                        </Badge>
                        <span className="block max-w-[130px] truncate">
                          {entry.source}
                        </span>
                      </span>
                    </td>

                    <td>
                      <span className="flex items-center gap-1.5">
                        <span className="text-xs capitalize">
                          {entry.emotion}
                        </span>
                        {entry.emotionCorrect !== null && (
                          <Badge
                            variant="outline"
                            className={cn(
                              "shrink-0 px-1 py-0 text-[10px]",
                              entry.emotionCorrect ? "text-ok" : "text-bad",
                            )}
                          >
                            {entry.emotionCorrect ? "correct" : "wrong"}
                          </Badge>
                        )}
                      </span>
                    </td>

                    <td>
                      <span className="flex items-center gap-1.5">
                        <span className="text-xs capitalize">
                          {entry.sentiment}
                        </span>
                        {entry.gatePasses === false && (
                          <Badge
                            variant="outline"
                            className="text-warn shrink-0 px-1 py-0 text-[10px]"
                            title="Outside the training distribution — the label stands, but with less confidence than it claims"
                          >
                            OOD
                          </Badge>
                        )}
                      </span>
                    </td>

                    <td className="tabular text-right text-xs">
                      {entry.wer !== null ? entry.wer.toFixed(2) : "—"}
                    </td>

                    <td className="tabular text-right text-xs whitespace-nowrap">
                      {entry.serverMs} ms
                    </td>

                    <td className="tabular text-right text-xs whitespace-nowrap">
                      <span className={cn(entry.coldStart && "text-warn")}>
                        {(entry.wallMs / 1000).toFixed(1)}s
                      </span>
                      {entry.coldStart && (
                        <span className="text-muted-foreground ml-1 text-[10px]">
                          cold
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
