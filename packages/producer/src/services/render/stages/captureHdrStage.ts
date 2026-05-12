/**
 * captureHdrStage — Z-ordered HDR / shader-transition layered composite.
 *
 * The most complex capture path:
 *   - Spawns a dedicated `domSession` for transparent-background screenshots.
 *   - Spawns an `hdrEncoder` (`spawnStreamingEncoder` with
 *     `rawInputFormat: "rgb48le"`) accepting pre-composited HDR frames.
 *   - Opens raw HDR video frame files (`hdrVideoFrameSources`) and reads
 *     them per-frame for native-HDR video layers.
 *   - Decodes 16-bit HDR PNGs once and blits them as image layers.
 *   - Queries Chrome z-order at layout-change boundaries and groups
 *     elements into DOM / HDR video / HDR image layers.
 *   - Composites bottom-to-top in Node memory, writing rgb48le buffers
 *     to the encoder's stdin.
 *
 * Cleanup invariants the design doc explicitly flags as risky —
 * preserved verbatim from the in-process renderer:
 *   - `hdrEncoderClosed` / `domSessionClosed` flags gate defensive-close
 *     paths so they don't run twice when the success path already closed.
 *   - `hdrVideoFrameSources` is drained + cleared in the outer `finally`
 *     regardless of how the body exits.
 *   - The layered path unconditionally captures in screenshot mode
 *     because `captureAlphaPng` hangs under `--enable-begin-frame-control`.
 *     Previously the stage mutated `cfg.forceScreenshot = true` directly;
 *     the value is now derived into a local `hdrCfg` so the caller-owned
 *     `cfg` survives the stage unchanged. The sequencer is expected to
 *     pass `forceScreenshot: true` for the layered branch as a contract
 *     check.
 *
 * Known follow-up: same runtime import cycle pattern as the other
 * capture stages — the stage imports HDR helpers from
 * `renderOrchestrator.ts` (runtime), which imports the stage back.
 * Safe at runtime; a future PR will consolidate these helpers.
 */

import {
  existsSync,
  mkdirSync,
  openSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import {
  type BeforeCaptureHook,
  type CaptureOptions,
  type EngineConfig,
  type HdrTransfer,
  type StreamingEncoder,
  type TransitionFn,
  TRANSITIONS,
  applyDomLayerMask,
  blitRgba8OverRgb48le,
  captureAlphaPng,
  closeCaptureSession,
  createCaptureSession,
  crossfade,
  decodePng,
  decodePngToRgb48le,
  getEncoderPreset,
  initTransparentBackground,
  initializeSession,
  normalizeObjectFit,
  queryElementStacking,
  removeDomLayerMask,
  resampleRgb48leObjectFit,
  runFfmpeg,
  spawnStreamingEncoder,
} from "@hyperframes/engine";
import { fpsToFfmpegArg, fpsToNumber } from "@hyperframes/core";
import type { FileServerHandle } from "../../fileServer.js";
import type { ProducerLogger } from "../../../logger.js";
import { createHdrImageTransferCache } from "../../hdrImageTransferCache.js";
import {
  type HdrCompositeContext,
  type HdrDiagnostics,
  type HdrImageBuffer,
  type HdrPerfCollector,
  type HdrTransitionMeta,
  type HdrVideoFrameSource,
  type ProgressCallback,
  type RenderJob,
  type TransitionRange,
  addHdrTiming,
  blitHdrImageLayer,
  blitHdrVideoLayer,
  closeHdrVideoFrameSource,
  compositeHdrFrame,
  createHdrPerfCollector,
  resolveCompositeTransfer,
} from "../../renderOrchestrator.js";
import { updateJobStatus, type CompositionMetadata } from "../shared.js";

export interface CaptureHdrStageInput {
  job: RenderJob;
  cfg: EngineConfig;
  /**
   * Capture-mode flag threaded from `compileStage`. The HDR layered
   * branch requires `true` (see file header for the
   * `captureAlphaPng` / `--enable-begin-frame-control` constraint);
   * the stage throws if called with `false`. Stored locally as
   * `hdrCfg.forceScreenshot` so the caller-owned `cfg` is not mutated.
   */
  forceScreenshot: boolean;
  log: ProducerLogger;

  projectDir: string;
  compiledDir: string;
  framesDir: string;
  videoOnlyPath: string;

  width: number;
  height: number;
  totalFrames: number;

  composition: CompositionMetadata;
  hasHdrContent: boolean;
  effectiveHdr: { transfer: HdrTransfer } | undefined;
  nativeHdrVideoIds: Set<string>;
  nativeHdrImageIds: Set<string>;
  videoTransfers: Map<string, HdrTransfer>;
  imageTransfers: Map<string, HdrTransfer>;
  hdrImageSrcPaths: Map<string, string>;

  preset: ReturnType<typeof getEncoderPreset>;
  effectiveQuality: number;
  effectiveBitrate: string | undefined;

  fileServer: FileServerHandle;
  buildCaptureOptions: () => CaptureOptions;
  createRenderVideoFrameInjector: () => BeforeCaptureHook | null;

  /** Mutated in place (counters incremented). */
  hdrDiagnostics: HdrDiagnostics;

  abortSignal: AbortSignal | undefined;
  assertNotAborted: () => void;
  onProgress?: ProgressCallback;
}

export interface CaptureHdrStageResult {
  lastBrowserConsole: string[];
  hdrPerf: HdrPerfCollector | undefined;
  /** Wall-clock ms for the HDR capture phase. */
  captureDurationMs: number;
  /** ffmpeg-reported encode duration; overlapped with capture. */
  encodeMs: number;
}

export async function runCaptureHdrStage(
  input: CaptureHdrStageInput,
): Promise<CaptureHdrStageResult> {
  const {
    job,
    cfg,
    forceScreenshot,
    log,
    projectDir,
    compiledDir,
    framesDir,
    videoOnlyPath,
    width,
    height,
    totalFrames,
    composition,
    hasHdrContent,
    effectiveHdr,
    nativeHdrVideoIds,
    nativeHdrImageIds,
    videoTransfers,
    imageTransfers,
    hdrImageSrcPaths,
    preset,
    effectiveQuality,
    effectiveBitrate,
    fileServer,
    buildCaptureOptions,
    createRenderVideoFrameInjector,
    hdrDiagnostics,
    abortSignal,
    assertNotAborted,
    onProgress,
  } = input;

  if (!forceScreenshot) {
    throw new Error(
      "captureHdrStage requires forceScreenshot=true; the layered composite path uses captureAlphaPng which hangs under --enable-begin-frame-control.",
    );
  }

  const stageStart = Date.now();
  let lastBrowserConsole: string[] = [];
  let hdrPerf: HdrPerfCollector | undefined;
  let captureDurationMs = 0;
  let encodeMs = 0;
  // Recomputed here so the stage owns its own scope; matches the sequencer's
  // `nativeHdrIds = new Set([...nativeHdrVideoIds, ...nativeHdrImageIds])`
  // before the `if (useLayeredComposite)` branch.
  const nativeHdrIds = new Set([...nativeHdrVideoIds, ...nativeHdrImageIds]);

  log.info(
    hasHdrContent
      ? "[Render] HDR layered composite: z-ordered DOM + native HDR video/image layers"
      : "[Render] Shader transition composite: z-ordered SDR DOM layers",
  );
  hdrPerf = createHdrPerfCollector();

  // Layered compositing relies on captureAlphaPng (Page.captureScreenshot
  // with a transparent background) for DOM layers. That CDP call hangs
  // indefinitely when Chrome is launched with --enable-begin-frame-control
  // (the default on Linux/headless-shell), because the compositor is paused
  // and never produces a frame to capture. Use screenshot mode for the
  // entire layered path — same constraint as alpha output formats. We
  // derive a local `hdrCfg` instead of mutating the caller-owned `cfg`
  // so the value flowing through the rest of the pipeline is the one the
  // sequencer locked at compile time. (The HDR path is end-of-pipeline
  // today, but Phase 3 chunked rendering depends on stages not mutating
  // caller config.)
  const hdrCfg: EngineConfig = { ...cfg, forceScreenshot: true };

  // Use NATIVE HDR IDs (probed before SDR→HDR conversion) so only originally-HDR
  // videos are hidden + extracted natively. SDR videos stay in the DOM screenshot
  // (injected via the frame injector) and get sRGB→HLG conversion in the blit.
  // HDR images don't need an equivalent array — they're keyed off
  // `nativeHdrImageIds` directly (decoded once into `hdrImageBuffers` and blitted
  // by `blitHdrImageLayer`, with the DOM mask hiding them via `nativeHdrIds`).
  const hdrVideoIds = composition.videos
    .filter((v) => nativeHdrVideoIds.has(v.id))
    .map((v) => v.id);

  // Resolve HDR video source paths
  const hdrVideoSrcPaths = new Map<string, string>();
  for (const v of composition.videos) {
    if (!hdrVideoIds.includes(v.id)) continue;
    let srcPath = v.src;
    if (!srcPath.startsWith("/")) {
      const fromCompiled = join(compiledDir, srcPath);
      srcPath = existsSync(fromCompiled) ? fromCompiled : join(projectDir, srcPath);
    }
    hdrVideoSrcPaths.set(v.id, srcPath);
  }

  // Launch headless Chrome for DOM capture.
  // Pass the video frame injector so SDR videos are rendered correctly in Chrome.
  // HDR videos get injected too but are masked out via applyDomLayerMask
  // before each DOM screenshot — only the native FFmpeg-extracted HLG
  // frames are used for HDR pixels.
  if (!fileServer) throw new Error("fileServer must be initialized before HDR compositing");
  // Native HDR videos (e.g. HEVC) may be undecodable by Chrome on the
  // current platform — Linux headless-shell ships without HEVC support.
  // Their pixels come from out-of-band ffmpeg extraction, so the DOM
  // `<video>` element is only kept around for layout. Skip the per-page
  // readiness wait for these IDs; otherwise the render hangs 45s and
  // throws "video metadata not ready" even though we never asked the
  // browser to decode the video.
  const domSession = await createCaptureSession(
    fileServer.url,
    framesDir,
    buildCaptureOptions(),
    createRenderVideoFrameInjector(),
    hdrCfg,
  );
  // Track lifecycle of resources spawned during HDR rendering so the
  // outer finally block can defensively reclaim anything that wasn't
  // cleaned up via the success path. Both closeCaptureSession and
  // StreamingEncoder.close() are idempotent, but the flags let us avoid
  // redundant work and make the intent explicit.
  let hdrEncoder: StreamingEncoder | null = null;
  let hdrEncoderClosed = false;
  let domSessionClosed = false;
  // Open raw HDR frame files at this scope so cleanup can close descriptors
  // on both success and early failure paths.
  const hdrVideoFrameSources = new Map<string, HdrVideoFrameSource>();
  try {
    await initializeSession(domSession);
    assertNotAborted();
    lastBrowserConsole = domSession.browserConsoleBuffer;

    // Set transparent background once for this dedicated DOM session.
    // captureAlphaPng() per frame skips the per-frame CDP set/reset overhead.
    await initTransparentBackground(domSession.page);

    // ── Scene detection for shader transitions ──────────────────────────
    // Query the browser for transition metadata written by @hyperframes/shader-transitions
    // (window.__hf.transitions) and discover which elements belong to each scene.
    const transitionMeta: HdrTransitionMeta[] = await domSession.page.evaluate(() => {
      return window.__hf?.transitions ?? [];
    });

    // Contract: compositions using window.__hf.transitions must wrap each
    // scene's elements in a <div class="scene" id="sceneName"> where the id
    // matches the fromScene/toScene values declared in the transition metadata.
    const sceneElements: Record<string, string[]> = await domSession.page.evaluate(() => {
      const scenes = document.querySelectorAll(".scene");
      const map: Record<string, string[]> = {};
      for (const scene of scenes) {
        if (!scene.id) continue;
        const ids = new Set<string>([scene.id]);
        const els = scene.querySelectorAll("[id]");
        for (const el of els) {
          if (el.id) ids.add(el.id);
        }
        map[scene.id] = Array.from(ids);
      }
      return map;
    });

    const fpsDecimal = fpsToNumber(job.config.fps);
    const transitionRanges: TransitionRange[] = transitionMeta.map((t) => ({
      ...t,
      startFrame: Math.floor(t.time * fpsDecimal),
      endFrame: Math.ceil((t.time + t.duration) * fpsDecimal),
    }));

    if (transitionRanges.length > 0) {
      log.info("[Render] Detected shader transitions for layered compositing", {
        count: transitionRanges.length,
        transitions: transitionRanges.map((t) => ({
          shader: t.shader,
          from: t.fromScene,
          to: t.toScene,
          frames: `${t.startFrame}-${t.endFrame}`,
        })),
      });
    }

    // Spawn HDR streaming encoder accepting raw rgb48le composited frames.
    // Assigned to the let declared above so the outer finally can close it
    // if any of the work between here and hdrEncoder.close() throws.
    hdrEncoder = await spawnStreamingEncoder(
      videoOnlyPath,
      {
        fps: job.config.fps,
        width,
        height,
        codec: preset.codec,
        preset: preset.preset,
        quality: effectiveQuality,
        bitrate: effectiveBitrate,
        pixelFormat: preset.pixelFormat,
        hdr: preset.hdr,
        rawInputFormat: "rgb48le",
      },
      abortSignal,
      { ffmpegStreamingTimeout: 3_600_000 },
    );
    assertNotAborted();

    // ── Query element bounds for HDR extraction dimensions ────────────
    // Extract at each HDR video's display dimensions (not composition dimensions)
    // so the source stride matches the blit dimensions. Elements that aren't
    // visible at t=0 (e.g., data-start > 0) need to be queried at their own
    // start time so their layout dimensions are available.
    const hdrExtractionDims = new Map<string, { width: number; height: number }>();
    // CSS `object-fit` / `object-position` for HDR <img> elements. Captured
    // alongside `hdrExtractionDims` so the static-image decoder can resample
    // the rgb48le buffer into the element's layout box the same way the
    // browser would, instead of blitting the source PNG at native size.
    const hdrImageFitInfo = new Map<string, { fit: string; position: string }>();
    const hdrVideoStartTimes = new Map<string, number>();
    for (const v of composition.videos) {
      if (hdrVideoIds.includes(v.id)) {
        hdrVideoStartTimes.set(v.id, v.start);
      }
    }
    const hdrImageStartTimes = new Map<string, number>();
    for (const img of composition.images) {
      if (nativeHdrImageIds.has(img.id)) {
        hdrImageStartTimes.set(img.id, img.start);
      }
    }

    // Collect unique start times to minimize seek operations. Merge HDR
    // video AND image start times so an HDR image with `data-start > 0`
    // also gets a stacking-query pass at its appearance moment.
    const uniqueStartTimes = [
      ...new Set([...hdrVideoStartTimes.values(), ...hdrImageStartTimes.values()]),
    ].sort((a, b) => a - b);
    for (const seekTime of uniqueStartTimes) {
      await domSession.page.evaluate((t: number) => {
        if (window.__hf && typeof window.__hf.seek === "function") window.__hf.seek(t);
      }, seekTime);
      if (domSession.onBeforeCapture) {
        await domSession.onBeforeCapture(domSession.page, seekTime);
      }
      const stacking = await queryElementStacking(domSession.page, nativeHdrIds);
      for (const el of stacking) {
        // Use layout dimensions (offsetWidth/offsetHeight) for extraction — these
        // are unaffected by CSS transforms (GSAP scale/rotation). getBoundingClientRect
        // returns the transformed bounding box which can be wrong for extraction.
        if (
          el.isHdr &&
          el.layoutWidth > 0 &&
          el.layoutHeight > 0 &&
          !hdrExtractionDims.has(el.id)
        ) {
          hdrExtractionDims.set(el.id, { width: el.layoutWidth, height: el.layoutHeight });
        }
        // Record `object-fit` / `object-position` for HDR images so the
        // static-image decode pass can resample to layout dimensions with
        // the same semantics the browser would apply.
        if (el.isHdr && nativeHdrImageIds.has(el.id) && !hdrImageFitInfo.has(el.id)) {
          hdrImageFitInfo.set(el.id, {
            fit: el.objectFit,
            position: el.objectPosition,
          });
        }
      }
    }

    // Fallback probe for HDR images that weren't captured above.
    // When an image's `data-start` aligns with the exact visibility
    // boundary (or precedes a GSAP `from` tween that animates it in
    // later), Chrome reports 0 layout dimensions at that instant.
    // Re-probe slightly into the element's visible range so the
    // resample path gets real layout dims.
    for (const [imageId, startTime] of hdrImageStartTimes) {
      if (hdrExtractionDims.has(imageId)) continue;
      const img = composition.images.find((i) => i.id === imageId);
      if (!img) continue;
      const duration = img.end - img.start;
      const retryTime = startTime + Math.min(0.5, duration * 0.1);
      await domSession.page.evaluate((t: number) => {
        if (window.__hf && typeof window.__hf.seek === "function") window.__hf.seek(t);
      }, retryTime);
      if (domSession.onBeforeCapture) {
        await domSession.onBeforeCapture(domSession.page, retryTime);
      }
      const retryStacking = await queryElementStacking(domSession.page, nativeHdrIds);
      for (const el of retryStacking) {
        if (el.id === imageId && el.isHdr && el.layoutWidth > 0 && el.layoutHeight > 0) {
          hdrExtractionDims.set(el.id, { width: el.layoutWidth, height: el.layoutHeight });
          if (!hdrImageFitInfo.has(el.id)) {
            hdrImageFitInfo.set(el.id, { fit: el.objectFit, position: el.objectPosition });
          }
          break;
        }
      }
    }

    // ── Pre-extract all HDR video frames in a single FFmpeg pass ──────
    // Use raw rgb48le instead of PNG sequences so the hot loop can read a
    // fixed byte range per frame and skip PNG decode entirely.
    for (const [videoId, srcPath] of hdrVideoSrcPaths) {
      const video = composition.videos.find((v) => v.id === videoId);
      if (!video) continue;
      const frameDir = join(framesDir, `hdr_${videoId}`);
      mkdirSync(frameDir, { recursive: true });
      const duration = video.end - video.start;
      const dims = hdrExtractionDims.get(videoId) ?? { width, height };
      const rawPath = join(frameDir, "frames.rgb48le");
      const ffmpegArgs = [
        "-ss",
        String(video.mediaStart),
        "-i",
        srcPath,
        "-t",
        String(duration),
        "-r",
        // Pass the rational form to FFmpeg so NTSC stays exact end-to-end.
        fpsToFfmpegArg(job.config.fps),
        "-vf",
        `scale=${dims.width}:${dims.height}:force_original_aspect_ratio=increase,crop=${dims.width}:${dims.height}`,
        "-pix_fmt",
        "rgb48le",
        "-f",
        "rawvideo",
        "-y",
        rawPath,
      ];
      const result = await runFfmpeg(ffmpegArgs, { signal: abortSignal });
      if (!result.success) {
        hdrDiagnostics.videoExtractionFailures += 1;
        log.error("HDR frame pre-extraction failed; aborting render", {
          videoId,
          srcPath,
          stderr: result.stderr.slice(-400),
        });
        throw new Error(
          `HDR frame extraction failed for video "${videoId}". ` +
            `Aborting render to avoid shipping black HDR layers.`,
        );
      }
      const frameSize = dims.width * dims.height * 6;
      const frameCount = Math.floor(statSync(rawPath).size / frameSize);
      if (frameCount < 1) {
        hdrDiagnostics.videoExtractionFailures += 1;
        throw new Error(
          `HDR frame extraction produced no frames for video "${videoId}". ` +
            `Aborting render to avoid shipping black HDR layers.`,
        );
      }
      hdrVideoFrameSources.set(videoId, {
        dir: frameDir,
        rawPath,
        fd: openSync(rawPath, "r"),
        width: dims.width,
        height: dims.height,
        frameSize,
        frameCount,
        scratch: Buffer.allocUnsafe(frameSize),
      });
    }

    // ── Pre-decode all HDR image buffers once ────────────────────────
    // Static images decode exactly once, then the resulting rgb48le buffer
    // is blitted on every visible frame. Caching the decode here keeps the
    // per-frame cost to a memcpy + blit. Failures are logged and skipped so
    // a single broken file doesn't kill the render.
    //
    // We resample the decoded buffer to the element's *layout* dimensions
    // here (using CSS `object-fit` / `object-position` semantics), so the
    // affine blit downstream can treat the buffer as if the source was
    // sized to the element's box. Without this step, an `<img>` element
    // styled `object-fit: cover` would render its source PNG at native
    // pixel size inside the layout box — visually a small image floating
    // in the top-left corner of its container instead of filling it.
    const hdrImageBuffers = new Map<string, HdrImageBuffer>();
    for (const [imageId, srcPath] of hdrImageSrcPaths) {
      try {
        const decoded = decodePngToRgb48le(readFileSync(srcPath));
        const layout = hdrExtractionDims.get(imageId);
        const fitInfo = hdrImageFitInfo.get(imageId);
        if (layout && (layout.width !== decoded.width || layout.height !== decoded.height)) {
          const fit = normalizeObjectFit(fitInfo?.fit);
          const resampled = resampleRgb48leObjectFit(
            decoded.data,
            decoded.width,
            decoded.height,
            layout.width,
            layout.height,
            fit,
            fitInfo?.position,
          );
          hdrImageBuffers.set(imageId, {
            data: resampled,
            width: layout.width,
            height: layout.height,
          });
        } else {
          hdrImageBuffers.set(imageId, {
            data: Buffer.from(decoded.data),
            width: decoded.width,
            height: decoded.height,
          });
        }
      } catch (err) {
        hdrDiagnostics.imageDecodeFailures += 1;
        log.error("HDR image decode failed; aborting render", {
          imageId,
          srcPath,
          error: err instanceof Error ? err.message : String(err),
        });
        throw new Error(
          `HDR image decode failed for image "${imageId}". ` +
            `Aborting render to avoid shipping missing HDR image layers.`,
        );
      }
    }

    assertNotAborted();

    try {
      // The beforeCaptureHook injects SDR video frames into the DOM.
      // We call it manually since the HDR loop doesn't use captureFrame().
      const beforeCaptureHook = domSession.onBeforeCapture;

      // Track which HDR video raw frame sources have been cleaned up.
      // Once a video's last frame has been used (time > video.end), its
      // extraction directory is deleted to free disk space. This prevents
      // disk exhaustion on compositions with many HDR videos.
      const cleanedUpVideos = new Set<string>();
      // Build a map of video end times for quick lookup
      const hdrVideoEndTimes = new Map<string, number>();
      for (const v of composition.videos) {
        if (hdrVideoFrameSources.has(v.id)) {
          hdrVideoEndTimes.set(v.id, v.end);
        }
      }

      // ── HDR composite helper context ───────────────────────────────────
      // The actual layer-compositing logic lives at module scope in
      // `compositeHdrFrame`; we just pre-bind its long-lived dependencies
      // here so call sites stay short.
      const debugDumpEnabled = process.env.KEEP_TEMP === "1";
      const debugDumpDir = debugDumpEnabled ? join(framesDir, "debug-composite") : null;
      if (debugDumpDir && !existsSync(debugDumpDir)) {
        mkdirSync(debugDumpDir, { recursive: true });
      }
      const compositeTransfer = resolveCompositeTransfer(hasHdrContent, effectiveHdr);
      const hdrTargetTransfer = compositeTransfer === "srgb" ? undefined : compositeTransfer;
      // Per-job LRU cache for transfer-converted HDR image buffers. Static HDR
      // images that need PQ↔HLG conversion are converted exactly once per
      // (imageId, targetTransfer) and then reused for every subsequent frame
      // instead of paying a fresh `Buffer.from` + `convertTransfer` on every
      // composite. The cache is local to this render job so concurrent renders
      // do not share state.
      const hdrCacheMaxBytes = process.env.HDR_TRANSFER_CACHE_MAX_BYTES
        ? Number(process.env.HDR_TRANSFER_CACHE_MAX_BYTES)
        : undefined;
      const hdrImageTransferCache = createHdrImageTransferCache(
        hdrCacheMaxBytes !== undefined ? { maxBytes: hdrCacheMaxBytes } : {},
      );
      const hdrCompositeCtx: HdrCompositeContext = {
        log,
        domSession,
        beforeCaptureHook,
        width,
        height,
        fps: fpsToNumber(job.config.fps),
        compositeTransfer,
        nativeHdrImageIds,
        hdrImageBuffers,
        hdrImageTransferCache,
        hdrVideoFrameSources,
        hdrVideoStartTimes,
        imageTransfers,
        videoTransfers,
        debugDumpEnabled,
        debugDumpDir,
        hdrPerf,
      };

      // ── Pre-allocate transition buffers ─────────────────────────────────
      // Each buffer is width * height * 6 bytes (~37 MB at 1080p). Reused
      // across frames to avoid per-frame allocation in the hot loop.
      const bufSize = width * height * 6;
      const hasTransitions = transitionRanges.length > 0;
      const transBufferA = hasTransitions ? Buffer.alloc(bufSize) : null;
      const transBufferB = hasTransitions ? Buffer.alloc(bufSize) : null;
      const transOutput = hasTransitions ? Buffer.alloc(bufSize) : null;
      // Pre-allocate the normal-frame canvas too — reused via .fill(0) each iteration
      // to avoid ~37 MB allocation per frame in the hot loop.
      const normalCanvas = Buffer.alloc(bufSize);

      for (let i = 0; i < totalFrames; i++) {
        assertNotAborted();
        const time = (i * job.config.fps.den) / job.config.fps.num;
        if (hdrPerf) hdrPerf.frames += 1;

        // Seek timeline
        let timingStart = Date.now();
        await domSession.page.evaluate((t: number) => {
          if (window.__hf && typeof window.__hf.seek === "function") window.__hf.seek(t);
        }, time);
        addHdrTiming(hdrPerf, "frameSeekMs", timingStart);

        // Inject SDR video frames into the DOM
        if (beforeCaptureHook) {
          timingStart = Date.now();
          await beforeCaptureHook(domSession.page, time);
          addHdrTiming(hdrPerf, "frameInjectMs", timingStart);
        }

        // Query ALL timed elements for z-order analysis
        timingStart = Date.now();
        const stackingInfo = await queryElementStacking(domSession.page, nativeHdrIds);
        addHdrTiming(hdrPerf, "stackingQueryMs", timingStart);

        // Find active transition for this frame (if any)
        const activeTransition = transitionRanges.find((t) => i >= t.startFrame && i <= t.endFrame);

        // Per-frame debug snapshot (every 30 frames). The meta object
        // requires `Array.find` over `stackingInfo` plus a number-format
        // and conditional struct allocation — non-trivial work to do
        // every 30 frames in the encode hot loop. Gate the entire block
        // on the logger's level check so production runs (level=info)
        // pay nothing.
        //
        // Audit note (PR #383 review): this is the only per-frame log
        // site in the streaming HDR encode loop that constructs
        // non-trivial metadata. The `[diag]` log.info calls inside
        // compositeToBuffer (compositeToBuffer plan, hdr layer blit,
        // dom layer blit, compositeToBuffer end) are already gated by
        // `shouldLog = debugDumpEnabled && debugFrameIndex >= 0`, where
        // debugDumpEnabled is driven by KEEP_TEMP=1 — strictly stricter
        // than an isLevelEnabled check. The HDR blit error-path
        // log.debugs only fire on caught failures, not on the happy
        // path. Any new per-frame log site that builds meta should
        // follow the same `if (log.isLevelEnabled?.("level") ?? true)`
        // pattern (or stay behind `shouldLog`) so production stays
        // allocation-free in the hot loop.
        if (i % 30 === 0 && (log.isLevelEnabled?.("debug") ?? true)) {
          const hdrEl = stackingInfo.find((e) => e.isHdr);
          log.debug("[Render] HDR layer composite frame", {
            frame: i,
            time: time.toFixed(2),
            hdrElement: hdrEl
              ? { z: hdrEl.zIndex, visible: hdrEl.visible, width: hdrEl.width }
              : null,
            stackingCount: stackingInfo.length,
            activeTransition: activeTransition?.shader,
          });
        }

        if (activeTransition && transBufferA && transBufferB && transOutput) {
          if (hdrPerf) hdrPerf.transitionFrames += 1;
          const transitionTimingStart = Date.now();
          // ── Transition frame: dual-scene compositing ──────────────────
          const progress =
            activeTransition.endFrame === activeTransition.startFrame
              ? 1
              : (i - activeTransition.startFrame) /
                (activeTransition.endFrame - activeTransition.startFrame);

          // Resolve scene element IDs
          const sceneAIds = new Set(sceneElements[activeTransition.fromScene] ?? []);
          const sceneBIds = new Set(sceneElements[activeTransition.toScene] ?? []);

          // Zero-fill scene buffers (transition function writes every output pixel)
          timingStart = Date.now();
          transBufferA.fill(0);
          transBufferB.fill(0);
          addHdrTiming(hdrPerf, "canvasClearMs", timingStart);

          for (const [sceneBuf, sceneIds] of [
            [transBufferA, sceneAIds],
            [transBufferB, sceneBIds],
          ] as const) {
            // Re-check abort between scene A and scene B. Each scene
            // capture below performs a DOM seek, optional hook,
            // per-layer HDR blits, and a full-page screenshot — easily
            // hundreds of ms. Without this, an abort that arrives
            // during scene A's capture won't fire until the next outer
            // frame, after scene B has already been fully composited
            // and discarded.
            assertNotAborted();
            // Fresh state: seek + inject
            timingStart = Date.now();
            await domSession.page.evaluate((t: number) => {
              if (window.__hf && typeof window.__hf.seek === "function") window.__hf.seek(t);
            }, time);
            addHdrTiming(hdrPerf, "domLayerSeekMs", timingStart);
            if (beforeCaptureHook) {
              timingStart = Date.now();
              await beforeCaptureHook(domSession.page, time);
              addHdrTiming(hdrPerf, "domLayerInjectMs", timingStart);
            }

            // Blit all HDR videos/images for this scene
            for (const el of stackingInfo) {
              if (!el.isHdr || !sceneIds.has(el.id)) continue;
              if (nativeHdrImageIds.has(el.id)) {
                blitHdrImageLayer(
                  sceneBuf as Buffer,
                  el,
                  hdrImageBuffers,
                  hdrImageTransferCache,
                  width,
                  height,
                  log,
                  imageTransfers.get(el.id),
                  hdrTargetTransfer,
                  hdrPerf,
                );
              } else {
                blitHdrVideoLayer(
                  sceneBuf as Buffer,
                  el,
                  time,
                  fpsToNumber(job.config.fps),
                  hdrVideoFrameSources,
                  hdrVideoStartTimes,
                  width,
                  height,
                  log,
                  videoTransfers.get(el.id),
                  hdrTargetTransfer,
                  hdrPerf,
                );
              }
            }

            // Single DOM screenshot: mask the page so only this scene's DOM
            // elements paint. Same masking strategy as the per-layer DOM
            // branch — see applyDomLayerMask for details. Native HDR videos
            // and images are always inline-hidden so their fallback poster /
            // SDR thumbnail doesn't bleed into the DOM overlay (HDR pixels
            // are blitted separately by blitHdrVideoLayer / blitHdrImageLayer
            // above).
            const showIds = Array.from(sceneIds);
            const hideIds = stackingInfo
              .map((e) => e.id)
              .filter((id) => !sceneIds.has(id) || nativeHdrIds.has(id));
            if (hdrPerf) hdrPerf.domLayerCaptures += 1;
            timingStart = Date.now();
            await applyDomLayerMask(domSession.page, showIds, hideIds);
            addHdrTiming(hdrPerf, "domMaskApplyMs", timingStart);
            timingStart = Date.now();
            const domPng = await captureAlphaPng(domSession.page, width, height);
            addHdrTiming(hdrPerf, "domScreenshotMs", timingStart);
            timingStart = Date.now();
            await removeDomLayerMask(domSession.page, hideIds);
            addHdrTiming(hdrPerf, "domMaskRemoveMs", timingStart);

            try {
              timingStart = Date.now();
              const { data: domRgba } = decodePng(domPng);
              addHdrTiming(hdrPerf, "domPngDecodeMs", timingStart);
              timingStart = Date.now();
              blitRgba8OverRgb48le(domRgba, sceneBuf as Buffer, width, height, compositeTransfer);
              addHdrTiming(hdrPerf, "domBlitMs", timingStart);
            } catch (err) {
              log.warn("DOM layer decode/blit failed; skipping overlay for transition scene", {
                frameIndex: i,
                sceneIds: Array.from(sceneIds),
                error: err instanceof Error ? err.message : String(err),
              });
            }
          }

          // Apply shader transition blend directly in the active rgb48le
          // signal space. Linearizing HDR was attempted but destroys dark
          // PQ content — values below PQ ~5000 quantize to zero in 16-bit
          // linear, wiping out the bottom portion of dark video content.
          // SDR compositions use 16-bit-expanded sRGB, which matches the
          // shader design space.
          const transitionFn: TransitionFn = TRANSITIONS[activeTransition.shader] ?? crossfade;
          transitionFn(transBufferA, transBufferB, transOutput, width, height, progress);
          addHdrTiming(hdrPerf, "transitionCompositeMs", transitionTimingStart);

          timingStart = Date.now();
          hdrEncoder.writeFrame(transOutput);
          addHdrTiming(hdrPerf, "encoderWriteMs", timingStart);
        } else {
          if (hdrPerf) hdrPerf.normalFrames += 1;
          // ── Normal frame: full layer composite (no transition) ─────────
          timingStart = Date.now();
          normalCanvas.fill(0);
          addHdrTiming(hdrPerf, "canvasClearMs", timingStart);
          timingStart = Date.now();
          await compositeHdrFrame(hdrCompositeCtx, normalCanvas, time, stackingInfo, undefined, i);
          addHdrTiming(hdrPerf, "normalCompositeMs", timingStart);
          if (debugDumpEnabled && debugDumpDir && i % 30 === 0) {
            const previewPath = join(
              debugDumpDir,
              `frame_${String(i).padStart(4, "0")}_final_rgb48le.bin`,
            );
            writeFileSync(previewPath, normalCanvas);
          }
          timingStart = Date.now();
          hdrEncoder.writeFrame(normalCanvas);
          addHdrTiming(hdrPerf, "encoderWriteMs", timingStart);
        }

        // Clean up HDR raw frame sources for videos that have ended.
        // Frees disk space during long renders with many HDR videos.
        // Skip when KEEP_TEMP=1 so we can inspect intermediate state.
        if (process.env.KEEP_TEMP !== "1") {
          for (const [videoId, endTime] of hdrVideoEndTimes) {
            if (time > endTime && !cleanedUpVideos.has(videoId)) {
              // Also check no active transition references this video's scene
              const stillNeeded =
                activeTransition &&
                (sceneElements[activeTransition.fromScene]?.includes(videoId) ||
                  sceneElements[activeTransition.toScene]?.includes(videoId));
              if (!stillNeeded) {
                const frameSource = hdrVideoFrameSources.get(videoId);
                if (frameSource) {
                  closeHdrVideoFrameSource(frameSource, log);
                  try {
                    rmSync(frameSource.dir, { recursive: true, force: true });
                  } catch (err) {
                    log.warn("Failed to clean up HDR raw frame directory", {
                      videoId,
                      frameDir: frameSource.dir,
                      rawPath: frameSource.rawPath,
                      error: err instanceof Error ? err.message : String(err),
                    });
                  }
                  hdrVideoFrameSources.delete(videoId);
                }
                cleanedUpVideos.add(videoId);
              }
            }
          }
        }

        job.framesRendered = i + 1;
        if ((i + 1) % 10 === 0 || i + 1 === totalFrames) {
          const frameProgress = (i + 1) / totalFrames;
          updateJobStatus(
            job,
            "rendering",
            `Layered composite frame ${i + 1}/${job.totalFrames}`,
            Math.round(25 + frameProgress * 55),
            onProgress,
          );
        }
      }
    } finally {
      lastBrowserConsole = domSession.browserConsoleBuffer;
      await closeCaptureSession(domSession);
      domSessionClosed = true;
    }

    const hdrEncodeResult = await hdrEncoder.close();
    hdrEncoderClosed = true;
    assertNotAborted();
    if (!hdrEncodeResult.success) {
      throw new Error(`HDR encode failed: ${hdrEncodeResult.error}`);
    }

    captureDurationMs = Date.now() - stageStart;
    encodeMs = hdrEncodeResult.durationMs;
  } finally {
    // Defensive cleanup: if anything between domSession creation and the
    // success-path closes threw, the encoder ffmpeg subprocess and the
    // browser would otherwise be leaked. Both close() methods are
    // idempotent so it's safe to call them when the flags are already set,
    // but we skip the redundant work to keep logs clean.
    if (hdrEncoder && !hdrEncoderClosed) {
      try {
        await hdrEncoder.close();
      } catch (err) {
        log.warn("hdrEncoder defensive close failed", {
          err: err instanceof Error ? err.message : String(err),
        });
      }
    }
    if (!domSessionClosed) {
      await closeCaptureSession(domSession).catch((err) => {
        log.warn("closeCaptureSession defensive close failed", {
          err: err instanceof Error ? err.message : String(err),
        });
      });
    }
    // Close any raw frame files that survived in-loop cleanup (early
    // failures, KEEP_TEMP=1, videos still active when the render exits).
    // The on-disk frames themselves are torn down with workDir.
    for (const frameSource of hdrVideoFrameSources.values()) {
      closeHdrVideoFrameSource(frameSource, log);
    }
    hdrVideoFrameSources.clear();
  }

  return {
    lastBrowserConsole,
    hdrPerf,
    captureDurationMs,
    encodeMs,
  };
}
