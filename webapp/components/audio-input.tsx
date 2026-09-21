"use client";

/**
 * Upload or record audio.
 *
 * Both paths converge on the same conditioning: decode, downmix to mono,
 * resample to 16 kHz, re-encode as wav (`lib/audio.ts`). So a phone recording
 * and a studio wav arrive at the model in the same shape, and the preview the
 * user hears is the waveform that was actually scored.
 */

import { Mic, Square, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { MAX_SECONDS, toWav16k } from "@/lib/audio";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";

/** Cosmetic ceiling on the record button; the server truncates at 30 s anyway. */
const RECORD_LIMIT_MS = MAX_SECONDS * 1000;

/**
 * Render the upload and record controls.
 *
 * @param props.onAudio - Called with the conditioned wav and how it arrived.
 * @param props.disabled - Whether interaction is blocked.
 */
export default function AudioInput({
  onAudio,
  disabled,
}: {
  onAudio: (wav: Blob, kind: "upload" | "recording", label: string) => void;
  disabled: boolean;
}) {
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [previewUrl]);

  const accept = async (
    raw: Blob,
    kind: "upload" | "recording",
    label: string,
  ) => {
    setError("");
    try {
      const wav = await toWav16k(raw);
      setPreviewUrl((previous) => {
        if (previous) URL.revokeObjectURL(previous);
        return URL.createObjectURL(wav);
      });
      onAudio(wav, kind, label);
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(false);
  };

  const startRecording = async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        // Release the microphone at once; a live track keeps the browser's
        // recording indicator on and reads as a bug.
        stream.getTracks().forEach((track) => track.stop());
        void accept(
          new Blob(chunksRef.current, { type: recorder.mimeType }),
          "recording",
          `microphone ${new Date().toLocaleTimeString()}`,
        );
      };

      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
      setElapsed(0);

      const startedAt = Date.now();
      timerRef.current = setInterval(() => {
        const ms = Date.now() - startedAt;
        setElapsed(ms);
        if (ms >= RECORD_LIMIT_MS) stopRecording();
      }, 100);
    } catch {
      setError(
        "Microphone access was refused or is unavailable. Recording needs " +
          "HTTPS and permission from the browser.",
      );
    }
  };

  return (
    <div className="space-y-3">
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex flex-wrap gap-2">
        <Input
          ref={fileRef}
          type="file"
          accept="audio/*,video/mp4,video/webm"
          className="hidden"
          disabled={disabled || recording}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void accept(file, "upload", file.name);
          }}
        />

        <Button
          variant="outline"
          disabled={disabled || recording}
          onClick={() => fileRef.current?.click()}
        >
          <Upload className="size-4" />
          Choose a file
        </Button>

        <Button
          variant={recording ? "destructive" : "outline"}
          disabled={disabled}
          onClick={() => (recording ? stopRecording() : void startRecording())}
        >
          {recording ? <Square className="size-4" /> : <Mic className="size-4" />}
          {recording
            ? `Stop (${(elapsed / 1000).toFixed(1)}s)`
            : "Record from microphone"}
        </Button>
      </div>

      {recording && (
        <Progress value={(elapsed / RECORD_LIMIT_MS) * 100} className="h-1.5" />
      )}

      <p className="text-muted-foreground text-xs">
        Anything the browser can decode. It is downmixed to 16 kHz mono before
        upload and truncated at {MAX_SECONDS} seconds, which is where
        Voxtral&apos;s encoder stops regardless.
      </p>

      {previewUrl && (
        // eslint-disable-next-line jsx-a11y/media-has-caption
        <audio controls src={previewUrl} className="w-full" />
      )}
    </div>
  );
}
