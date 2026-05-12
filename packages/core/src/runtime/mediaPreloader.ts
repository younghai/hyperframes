import { refreshRuntimeMediaCache, type RuntimeMediaClip } from "./media";

// Compositions with fewer than 6 timed clips rarely exceed browser memory
// limits during eager preload. The threshold avoids preload management
// overhead for typical compositions while catching the heavy-media case
// (e.g., 20 clips / 6GB reported in heygen-com/hyperframes#729).
const LAZY_THRESHOLD = 6;
const LOOKAHEAD_SECONDS = 10;
const LOOKAHEAD_MIN_CLIPS = 2;
// Cap on simultaneously promoted (buffered) clips. When the lookahead window
// contains more clips than this (e.g., many short clips), all window clips
// stay promoted — the cap is defense-in-depth, not a hard ceiling. The primary
// memory bound comes from window-based eviction in syncWindow().
const MAX_PROMOTED = 5;

export interface MediaPreloadManager {
  refresh(): void;
  sync(currentTimeSeconds: number): void;
  preloadAroundTime(timeSeconds: number): void;
  isLazy(): boolean;
}

export function createMediaPreloadManager(options?: {
  resolveStartSeconds?: (element: Element) => number;
  resolveDurationSeconds?: (element: HTMLVideoElement | HTMLAudioElement) => number | null;
  shouldIncludeElement?: (element: HTMLVideoElement | HTMLAudioElement) => boolean;
  onActivation?: (clipCount: number) => void;
}): MediaPreloadManager {
  let clips: RuntimeMediaClip[] = [];
  const promoted = new Set<HTMLMediaElement>();
  /** Insertion-order queue for LRU eviction (oldest first). */
  const promotionOrder: HTMLMediaElement[] = [];
  /** Stashed original src so we can restore after eviction. */
  const originalSrc = new Map<HTMLMediaElement, string>();
  let lazy = false;
  let activationEmitted = false;

  function refresh(): void {
    const cache = refreshRuntimeMediaCache(options);
    clips = cache.mediaClips;
    lazy = clips.length >= LAZY_THRESHOLD;
    if (lazy && !activationEmitted) {
      activationEmitted = true;
      options?.onActivation?.(clips.length);
    }
  }

  function evictClip(clip: RuntimeMediaClip): void {
    if (!promoted.has(clip.el)) return;
    // Stash original src before clearing
    if (!originalSrc.has(clip.el)) {
      originalSrc.set(clip.el, clip.el.src);
    }
    // Release buffered data: only way to free memory per MDN
    clip.el.removeAttribute("src");
    clip.el.load();
    clip.el.preload = "metadata";
    promoted.delete(clip.el);
    const idx = promotionOrder.indexOf(clip.el);
    if (idx !== -1) promotionOrder.splice(idx, 1);
  }

  function promoteClip(clip: RuntimeMediaClip): void {
    if (promoted.has(clip.el)) return;

    // Restore src if previously evicted
    const stashedSrc = originalSrc.get(clip.el);
    if (stashedSrc !== undefined && !clip.el.src) {
      clip.el.src = stashedSrc;
      originalSrc.delete(clip.el);
    }

    promoted.add(clip.el);
    promotionOrder.push(clip.el);

    if (clip.el.preload !== "auto") {
      clip.el.preload = "auto";
    }
    if (clip.el.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
      clip.el.load();
    }
  }

  function evictOutsideWindow(inWindow: Set<RuntimeMediaClip>): void {
    const windowEls = new Set<HTMLMediaElement>();
    for (const clip of inWindow) {
      windowEls.add(clip.el);
    }

    // Evict clips no longer in window, oldest first
    for (const clip of clips) {
      if (promoted.has(clip.el) && !windowEls.has(clip.el)) {
        evictClip(clip);
      }
    }

    // If still over budget after removing out-of-window clips,
    // evict the oldest promoted that isn't in the current window
    while (promotionOrder.length > MAX_PROMOTED) {
      const oldest = promotionOrder[0];
      if (windowEls.has(oldest)) break; // don't evict something currently needed
      const clip = clips.find((c) => c.el === oldest);
      if (clip) {
        evictClip(clip);
      } else {
        // Element no longer in clips list, just remove from tracking
        promoted.delete(oldest);
        promotionOrder.shift();
      }
    }
  }

  function getClipsInWindow(timeSeconds: number): Set<RuntimeMediaClip> {
    const windowEnd = timeSeconds + LOOKAHEAD_SECONDS;
    const inWindow = new Set<RuntimeMediaClip>();

    for (const clip of clips) {
      const active = timeSeconds >= clip.start && timeSeconds < clip.end;
      const inLookahead = clip.start >= timeSeconds && clip.start <= windowEnd;
      if (active || inLookahead) {
        inWindow.add(clip);
      }
    }

    if (inWindow.size < LOOKAHEAD_MIN_CLIPS) {
      const sorted = clips
        .filter((c) => c.start >= timeSeconds && !inWindow.has(c))
        .sort((a, b) => a.start - b.start);
      for (const clip of sorted) {
        inWindow.add(clip);
        if (inWindow.size >= LOOKAHEAD_MIN_CLIPS) break;
      }
    }

    return inWindow;
  }

  function syncWindow(timeSeconds: number): void {
    const window = getClipsInWindow(timeSeconds);
    evictOutsideWindow(window);
    for (const clip of clips) {
      if (window.has(clip)) {
        promoteClip(clip);
      }
    }
  }

  function sync(currentTimeSeconds: number): void {
    if (!lazy) return;
    syncWindow(currentTimeSeconds);
  }

  function preloadAroundTime(timeSeconds: number): void {
    if (!lazy) return;
    syncWindow(timeSeconds);
  }

  function isLazy(): boolean {
    return lazy;
  }

  return { refresh, sync, preloadAroundTime, isLazy };
}
