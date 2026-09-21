"use client";

/**
 * The MELD corpus picker.
 *
 * Drawn from dev and test only, class-balanced across the seven emotions —
 * see `src/modal/prepare_clips.py`. Each clip carries its gold label and gold
 * transcript, which is what lets the app show WER and a right/wrong verdict
 * rather than just a prediction.
 */

import { Loader2, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { fetchClipAudio, listClips } from "@/lib/api";
import { base64ToObjectUrl } from "@/lib/audio";
import { cn } from "@/lib/utils";
import { EMOTIONS } from "@/lib/types";
import type { Clip, Emotion } from "@/lib/types";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/** Tailwind text colour per emotion, matching the chart tokens. */
export const EMOTION_TEXT: Record<Emotion, string> = {
  neutral: "text-chart-neutral",
  surprise: "text-chart-surprise",
  fear: "text-chart-fear",
  sadness: "text-chart-sadness",
  joy: "text-chart-joy",
  disgust: "text-chart-disgust",
  anger: "text-chart-anger",
};

/**
 * Render the picker.
 *
 * @param props.selected - Currently selected clip key.
 * @param props.onSelect - Called with the clip the user picked.
 * @param props.disabled - Whether interaction is blocked.
 */
export default function ClipPicker({
  selected,
  onSelect,
  disabled,
}: {
  selected: string | null;
  onSelect: (clip: Clip) => void;
  disabled: boolean;
}) {
  const [clips, setClips] = useState<Clip[]>([]);
  const [filter, setFilter] = useState<Emotion | "all">("all");
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [fetching, setFetching] = useState<string | null>(null);

  useEffect(() => {
    listClips()
      .then(setClips)
      .catch((caught: Error) => setError(caught.message))
      .finally(() => setLoading(false));
  }, []);

  // Object URLs leak until revoked, and the picker makes one per preview.
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return clips.filter(
      (clip) =>
        (filter === "all" || clip.emotion === filter) &&
        (needle === "" ||
          clip.utterance.toLowerCase().includes(needle) ||
          clip.key.includes(needle) ||
          (clip.speaker ?? "").toLowerCase().includes(needle)),
    );
  }, [clips, filter, query]);

  const choose = async (clip: Clip) => {
    onSelect(clip);
    setFetching(clip.key);
    try {
      const base64 = await fetchClipAudio(clip.key);
      setAudioUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return base64ToObjectUrl(base64);
      });
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setFetching(null);
    }
  };

  if (loading) {
    // Content-shaped placeholders, not solid blocks. `bg-foreground/10`
    // rather than `bg-accent` because accent is a near-solid dark panel in
    // dark mode -- six of those read as black bars, which is what this
    // replaced. A tenth of the text colour stays a faint tint in both themes.
    const widths = ["w-4/5", "w-3/5", "w-11/12", "w-2/3", "w-3/4", "w-1/2"];
    return (
      <div className="space-y-3" aria-busy="true" aria-live="polite">
        <div className="flex gap-2">
          <div className="bg-foreground/10 h-9 w-[190px] animate-pulse rounded-md" />
          <div className="bg-foreground/10 h-9 flex-1 animate-pulse rounded-md" />
        </div>

        <div className="divide-y rounded-md border">
          {widths.map((width, index) => (
            <div
              key={index}
              className="flex animate-pulse items-center gap-3 px-3 py-2.5"
              style={{ animationDelay: `${index * 90}ms` }}
            >
              <div className="bg-foreground/10 h-5 w-[74px] shrink-0 rounded-full" />
              <div className={`bg-foreground/10 h-3.5 rounded ${width}`} />
              <div className="bg-foreground/10 ml-auto h-3 w-16 shrink-0 rounded" />
            </div>
          ))}
        </div>

        <p className="text-muted-foreground flex items-center gap-2 text-xs">
          <Loader2 className="size-3 animate-spin" />
          Loading the MELD corpus
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex flex-wrap gap-2">
        <Select
          value={filter}
          onValueChange={(value) => setFilter(value as Emotion | "all")}
          disabled={disabled}
        >
          <SelectTrigger className="w-[190px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All emotions ({clips.length})</SelectItem>
            {EMOTIONS.map((emotion) => (
              <SelectItem key={emotion} value={emotion}>
                <span className="capitalize">{emotion}</span> (
                {clips.filter((clip) => clip.emotion === emotion).length})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="relative min-w-[200px] flex-1">
          <Search className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search transcript, speaker or key"
            className="pl-8"
            disabled={disabled}
          />
        </div>
      </div>

      <ScrollArea className="h-[300px] rounded-md border">
        {visible.length === 0 ? (
          <p className="text-muted-foreground p-4 text-sm">Nothing matches.</p>
        ) : (
          <ul className="divide-y">
            {visible.map((clip) => (
              <li key={clip.key}>
                <button
                  type="button"
                  disabled={disabled}
                  aria-pressed={selected === clip.key}
                  onClick={() => void choose(clip)}
                  className={cn(
                    "hover:bg-accent/60 flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors disabled:opacity-50",
                    selected === clip.key && "bg-accent",
                  )}
                >
                  <Badge
                    variant="outline"
                    className={cn("w-[74px] justify-center", EMOTION_TEXT[clip.emotion])}
                  >
                    {clip.emotion}
                  </Badge>

                  <span className="min-w-0 flex-1 truncate text-sm">
                    {clip.speaker && (
                      <span className="font-medium">{clip.speaker}: </span>
                    )}
                    {clip.utterance}
                  </span>

                  {fetching === clip.key ? (
                    <Loader2 className="text-muted-foreground size-3.5 animate-spin" />
                  ) : (
                    <span className="text-muted-foreground tabular shrink-0 text-xs">
                      {clip.duration_sec}s · {clip.split}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </ScrollArea>

      {audioUrl && (
        <div className="space-y-1.5">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <audio controls src={audioUrl} className="w-full" />
          <p className="text-muted-foreground text-xs">
            This is the 16 kHz mono downmix the model receives, not the
            original video soundtrack.
          </p>
        </div>
      )}
    </div>
  );
}
