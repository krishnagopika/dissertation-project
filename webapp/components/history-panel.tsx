"use client";

/**
 * Previous runs, with both latencies.
 *
 * Two numbers per row because they answer different questions. `server` is
 * what the endpoint spent on the audio — the model's speed, comparable
 * between runs. `wall` is what you waited, which on a cold start includes the
 * minute of container start and model load. Showing only the first would hide
 * cold starts; only the second would blame the model for them.
 */

import { Download, History, Trash2 } from "lucide-react";
import { useMemo } from "react";

import { clearHistory, toCsv, type HistoryEntry } from "@/lib/history";
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
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

/**
 * Render the run history.
 *
 * @param props.entries - Runs, newest first.
 * @param props.onChange - Called with the new list after a clear.
 * @param props.onSelect - Replays a past run's result into the main panel.
 */
export default function HistoryPanel({
  entries,
  onChange,
  onSelect,
}: {
  entries: HistoryEntry[];
  onChange: (entries: HistoryEntry[]) => void;
  onSelect?: (entry: HistoryEntry) => void;
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
    const blob = new Blob([toCsv(entries)], { type: "text/csv" });
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
          {stats
            ? `${stats.runs} run${stats.runs === 1 ? "" : "s"} this browser · median ${stats.medianServerMs} ms on the server` +
              (stats.medianWarmWallMs !== null
                ? ` · ${stats.medianWarmWallMs} ms end to end when warm`
                : "") +
              (stats.coldStarts
                ? ` · ${stats.coldStarts} cold start${stats.coldStarts === 1 ? "" : "s"}`
                : "") +
              (stats.accuracy !== null
                ? ` · ${(stats.accuracy * 100).toFixed(0)}% correct on ${stats.scored} labelled`
                : "")
            : "Nothing yet. Analyse a clip and it will appear here."}
        </CardDescription>
        {entries.length > 0 && (
          <CardAction className="flex gap-1">
            <Button variant="ghost" size="sm" onClick={download}>
              <Download className="size-3.5" />
              CSV
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onChange(clearHistory())}
            >
              <Trash2 className="size-3.5" />
              Clear
            </Button>
          </CardAction>
        )}
      </CardHeader>

      {entries.length > 0 && (
        <CardContent>
          <ScrollArea className="h-[260px]">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="h-8">when</TableHead>
                  <TableHead className="h-8">input</TableHead>
                  <TableHead className="h-8">predicted</TableHead>
                  <TableHead className="h-8 text-right">WER</TableHead>
                  <TableHead className="h-8 text-right">server</TableHead>
                  <TableHead className="h-8 text-right">wall</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {entries.map((entry) => (
                  <TableRow
                    key={entry.id}
                    className={cn(onSelect && "cursor-pointer")}
                    onClick={() => onSelect?.(entry)}
                  >
                    <TableCell className="text-muted-foreground tabular py-1.5 text-xs whitespace-nowrap">
                      {new Date(entry.at).toLocaleTimeString()}
                    </TableCell>

                    <TableCell className="max-w-[180px] truncate py-1.5 text-xs">
                      <span className="text-muted-foreground mr-1">
                        {entry.kind === "corpus"
                          ? "clip"
                          : entry.kind === "upload"
                            ? "file"
                            : "mic"}
                      </span>
                      {entry.source}
                    </TableCell>

                    <TableCell className="py-1.5">
                      <span className="flex items-center gap-1.5">
                        <span className="text-xs capitalize">{entry.emotion}</span>
                        {entry.emotionCorrect !== null && (
                          <Badge
                            variant="outline"
                            className={cn(
                              "px-1 py-0 text-[10px]",
                              entry.emotionCorrect ? "text-ok" : "text-bad",
                            )}
                          >
                            {entry.emotionCorrect ? "✓" : "✗"}
                          </Badge>
                        )}
                        {entry.gatePasses === false && (
                          <Badge
                            variant="outline"
                            className="text-warn px-1 py-0 text-[10px]"
                            title="Outside the training distribution — the label stands, but with less confidence than it claims"
                          >
                            OOD
                          </Badge>
                        )}
                      </span>
                    </TableCell>

                    <TableCell className="tabular py-1.5 text-right text-xs">
                      {entry.wer !== null ? entry.wer.toFixed(2) : "—"}
                    </TableCell>

                    <TableCell className="tabular py-1.5 text-right text-xs">
                      {entry.serverMs} ms
                    </TableCell>

                    <TableCell className="tabular py-1.5 text-right text-xs">
                      <span className={cn(entry.coldStart && "text-warn")}>
                        {(entry.wallMs / 1000).toFixed(1)}s
                      </span>
                      {entry.coldStart && (
                        <span className="text-muted-foreground ml-1 text-[10px]">
                          cold
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </ScrollArea>
        </CardContent>
      )}
    </Card>
  );
}

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
