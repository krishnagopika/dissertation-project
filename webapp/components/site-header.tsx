"use client";

import { AudioLines, LogOut, Moon, Sun } from "lucide-react";
import { signOut } from "next-auth/react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/**
 * Application header: identity, theme, sign-out.
 *
 * @param props.email - The signed-in address, shown so it is obvious which
 *   account is spending the GPU budget.
 * @param props.isAdmin - Whether this account may start and stop the
 *   container; surfaced as a badge rather than hidden, because an admin who
 *   does not know they are one leaves the GPU pinned.
 */
export default function SiteHeader({
  email,
  isAdmin,
}: {
  email: string;
  isAdmin: boolean;
}) {
  const { resolvedTheme, setTheme } = useTheme();

  // next-themes cannot know the resolved theme until after hydration, so the
  // icon must not render until then or the markup mismatches.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <header className="bg-background/80 sticky top-0 z-40 border-b backdrop-blur-sm">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 sm:px-6">
        <div className="bg-primary/10 text-primary flex size-8 shrink-0 items-center justify-center rounded-md">
          <AudioLines className="size-4" />
        </div>

        <div className="min-w-0">
          <p className="truncate text-sm leading-tight font-semibold">
            Speech Emotion Pipeline
          </p>
          <p className="text-muted-foreground hidden truncate text-xs leading-tight sm:block">
            Voxtral + XLM-RoBERTa fusion, ASR condition
          </p>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {isAdmin && (
            <Badge variant="outline" className="hidden sm:inline-flex">
              admin
            </Badge>
          )}

          <span className="text-muted-foreground hidden max-w-[16ch] truncate text-xs md:inline">
            {email}
          </span>

          <Separator orientation="vertical" className="hidden h-5 sm:block" />

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Toggle theme"
                onClick={() =>
                  setTheme(resolvedTheme === "dark" ? "light" : "dark")
                }
              >
                {mounted && resolvedTheme === "dark" ? (
                  <Sun className="size-4" />
                ) : (
                  <Moon className="size-4" />
                )}
              </Button>
            </TooltipTrigger>
            <TooltipContent>Toggle theme</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Sign out"
                onClick={() => void signOut({ redirectTo: "/signin" })}
              >
                <LogOut className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Sign out</TooltipContent>
          </Tooltip>
        </div>
      </div>
    </header>
  );
}
