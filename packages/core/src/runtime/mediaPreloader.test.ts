import { describe, it, expect, beforeEach, vi } from "vitest";
import { createMediaPreloadManager } from "./mediaPreloader";

function mockMediaElement(attrs: {
  start: string;
  duration?: string;
  tag?: string;
}): HTMLMediaElement {
  const el = {
    tagName: (attrs.tag ?? "VIDEO").toUpperCase(),
    preload: "auto",
    readyState: 0,
    duration: Number.NaN,
    defaultPlaybackRate: 1,
    loop: false,
    src: `blob:mock-${attrs.start}`,
    dataset: {
      start: attrs.start,
      duration: attrs.duration,
    },
    hasAttribute: (name: string) => name === "data-start",
    getAttribute: (name: string) => {
      if (name === "data-start") return attrs.start;
      if (name === "data-duration") return attrs.duration ?? null;
      return null;
    },
    removeAttribute: (name: string) => {
      if (name === "src") {
        (el as Record<string, unknown>).src = "";
      }
    },
    closest: () => null,
    load: vi.fn(),
  } as unknown as HTMLMediaElement;
  return el;
}

function setupDOM(elements: HTMLMediaElement[]): void {
  const originalQuerySelector = document.querySelectorAll.bind(document);
  document.querySelectorAll = ((selector: string) => {
    if (selector === "video, audio") return elements as unknown as NodeListOf<Element>;
    return originalQuerySelector(selector);
  }) as typeof document.querySelectorAll;
}

describe("createMediaPreloadManager", () => {
  let elements: HTMLMediaElement[];

  beforeEach(() => {
    elements = [];
  });

  it("is not lazy when fewer than 6 media elements", () => {
    elements = [
      mockMediaElement({ start: "0", duration: "5" }),
      mockMediaElement({ start: "5", duration: "5" }),
    ];
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    expect(manager.isLazy()).toBe(false);
  });

  it("activates lazy mode at exactly LAZY_THRESHOLD (6 elements)", () => {
    elements = Array.from({ length: 6 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    expect(manager.isLazy()).toBe(true);
  });

  it("is not lazy with 5 elements (below threshold)", () => {
    elements = Array.from({ length: 5 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    expect(manager.isLazy()).toBe(false);
  });

  it("activates lazy mode with 8 media elements", () => {
    elements = Array.from({ length: 8 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    expect(manager.isLazy()).toBe(true);
  });

  it("sync promotes clips in the lookahead window", () => {
    elements = Array.from({ length: 8 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    manager.sync(0);

    expect(elements[0].preload).toBe("auto");
    expect(elements[1].preload).toBe("auto");
    expect(elements[7].preload).toBe("metadata");
  });

  it("preloadAroundTime promotes clips near seek target", () => {
    elements = Array.from({ length: 10 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    manager.preloadAroundTime(30);

    expect(elements[6].preload).toBe("auto");
    expect(elements[7].preload).toBe("auto");
    expect(elements[0].preload).toBe("metadata");
  });

  it("sync is a no-op when not lazy", () => {
    elements = [
      mockMediaElement({ start: "0", duration: "5" }),
      mockMediaElement({ start: "5", duration: "5" }),
    ];
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();
    manager.sync(0);

    expect(manager.isLazy()).toBe(false);
  });

  it("guarantees at least LOOKAHEAD_MIN_CLIPS are promoted", () => {
    elements = Array.from({ length: 8 }, (_, i) =>
      mockMediaElement({ start: String(i * 20), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    manager.sync(0);

    const promotedCount = elements.filter((el) => el.preload === "auto").length;
    expect(promotedCount).toBeGreaterThanOrEqual(2);
  });

  it("evicts clips when scrubbing away from them", () => {
    elements = Array.from({ length: 10 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    // Promote clips around t=0
    manager.sync(0);
    expect(elements[0].preload).toBe("auto");
    expect(elements[1].preload).toBe("auto");

    // Scrub to t=40 — clips 0,1 should be evicted
    manager.sync(40);
    expect(elements[0].preload).toBe("metadata");
    expect(elements[0].src).toBe("");
    expect(elements[8].preload).toBe("auto");
  });

  it("restores src when re-promoting a previously evicted clip", () => {
    elements = Array.from({ length: 10 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    const originalSrc0 = elements[0].src;

    // Promote at t=0, scrub away, scrub back
    manager.sync(0);
    manager.sync(40);
    expect(elements[0].src).toBe("");

    manager.sync(0);
    expect(elements[0].src).toBe(originalSrc0);
    expect(elements[0].preload).toBe("auto");
  });

  it("does not exceed MAX_PROMOTED (5) clips", () => {
    // 10 clips, each 5s long, spaced 5s apart
    elements = Array.from({ length: 10 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    // Sync at t=0 — window covers clips 0,1,2 (0-15s lookahead)
    manager.sync(0);
    const promotedAfterFirst = elements.filter((el) => el.preload === "auto").length;
    expect(promotedAfterFirst).toBeLessThanOrEqual(5);

    // Sync at different position — should evict old ones
    manager.sync(25);
    const totalPromoted = elements.filter((el) => el.preload === "auto").length;
    expect(totalPromoted).toBeLessThanOrEqual(5);
  });

  it("calls load() when evicting to release buffers", () => {
    elements = Array.from({ length: 10 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();

    for (const el of elements) {
      el.preload = "metadata";
    }

    manager.sync(0);
    const loadCallsBefore = (elements[0].load as ReturnType<typeof vi.fn>).mock.calls.length;

    // Scrub away — eviction should call load() to release buffers
    manager.sync(40);
    const loadCallsAfter = (elements[0].load as ReturnType<typeof vi.fn>).mock.calls.length;
    expect(loadCallsAfter).toBeGreaterThan(loadCallsBefore);
  });

  it("isLazy reports true with 6+ clips so caller can gate render-mode bypass", () => {
    elements = Array.from({ length: 6 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const manager = createMediaPreloadManager();
    manager.refresh();
    expect(manager.isLazy()).toBe(true);
  });

  it("calls onActivation when lazy mode activates", () => {
    elements = Array.from({ length: 8 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const onActivation = vi.fn();
    const manager = createMediaPreloadManager({ onActivation });
    manager.refresh();

    expect(onActivation).toHaveBeenCalledOnce();
    expect(onActivation).toHaveBeenCalledWith(8);
  });

  it("does not call onActivation below threshold", () => {
    elements = [
      mockMediaElement({ start: "0", duration: "5" }),
      mockMediaElement({ start: "5", duration: "5" }),
    ];
    setupDOM(elements);

    const onActivation = vi.fn();
    const manager = createMediaPreloadManager({ onActivation });
    manager.refresh();

    expect(onActivation).not.toHaveBeenCalled();
  });

  it("calls onActivation only once across multiple refreshes", () => {
    elements = Array.from({ length: 8 }, (_, i) =>
      mockMediaElement({ start: String(i * 5), duration: "5" }),
    );
    setupDOM(elements);

    const onActivation = vi.fn();
    const manager = createMediaPreloadManager({ onActivation });
    manager.refresh();
    manager.refresh();
    manager.refresh();

    expect(onActivation).toHaveBeenCalledOnce();
  });
});
