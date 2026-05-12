/**
 * Shared types and pure helpers used by the staged render pipeline.
 *
 * Lives in its own module so the stage files in `./stages/` can import the
 * helpers they need without reaching back into `renderOrchestrator.ts` —
 * the orchestrator imports the stage functions, so a runtime cycle would
 * otherwise form (and grow as more stages are extracted).
 *
 * `renderOrchestrator.ts` re-exports everything declared here for
 * backwards compatibility with existing test files and external callers.
 */

import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { CANVAS_DIMENSIONS, type CanvasResolution } from "@hyperframes/core";
import type { AudioElement, ImageElement, VideoElement } from "@hyperframes/engine";
import type { CompiledComposition } from "../htmlCompiler.js";
import { defaultLogger, type ProducerLogger } from "../../logger.js";
import { isPathInside } from "../../utils/paths.js";
import type { ProgressCallback, RenderJob, RenderStatus } from "../renderOrchestrator.js";

export interface CompositionMetadata {
  duration: number;
  videos: VideoElement[];
  audios: AudioElement[];
  images: ImageElement[];
  width: number;
  height: number;
}

/**
 * Floating-point tolerance for reconciling browser-discovered media timing
 * against statically-parsed metadata. Used when the browser reports a
 * slightly different `end` / `mediaStart` / `volume` than the compiled
 * HTML and we want to ignore sub-millisecond float noise.
 */
export const BROWSER_MEDIA_EPSILON = 0.0001;

/**
 * Browser-discovered media inside inlined sub-compositions can still report
 * scene-local timing from the merged DOM (e.g. start=0, end=85.52) while the
 * compiled metadata is already offset into the parent host timeline
 * (e.g. start=4.417, end=89.937). Reproject browser end-time into the
 * compiled element's time origin before reconciling it back into the render
 * metadata.
 */
export function projectBrowserEndToCompositionTimeline(
  existingStart: number,
  browserStart: number,
  browserEnd: number,
): number {
  return browserEnd + (existingStart - browserStart);
}

/**
 * Translate the user-facing `--resolution` flag into a Chrome
 * `deviceScaleFactor`. The composition's intrinsic dimensions stay the
 * page-layout viewport; the screenshot lands at output dims via DPR.
 *
 * The scale must be a positive integer ≥ 1 — fractional DPRs introduce
 * visible aliasing and we'd rather fail loudly than produce a blurry
 * 4K render. Downsampling (output < composition) is rejected because
 * the user is unlikely to have intended it; if the use case appears
 * we can plumb a separate flag.
 *
 * Throws on:
 *   - HDR + outputResolution (HDR compositor processes raw pixel buffers
 *     at composition dimensions and would need parallel scaling).
 *   - Aspect-ratio mismatch (e.g. landscape composition → portrait-4k).
 *   - Non-integer scale ratio.
 *   - Downsampling (output dimensions smaller than composition).
 */
export function resolveDeviceScaleFactor(input: {
  compositionWidth: number;
  compositionHeight: number;
  outputResolution: CanvasResolution | undefined;
  hdrRequested: boolean;
  alphaRequested: boolean;
}): number {
  if (!input.outputResolution) return 1;
  if (input.hdrRequested) {
    throw new Error(
      "outputResolution cannot be combined with hdrMode='force-hdr'. " +
        "HDR rendering composites at composition dimensions and does not yet " +
        "support supersampling. Pick one or render in two passes.",
    );
  }
  if (input.alphaRequested) {
    throw new Error(
      "outputResolution cannot be combined with alpha output (--format webm|mov|png-sequence). " +
        "The alpha screenshot path does not yet apply deviceScaleFactor and would silently " +
        "produce composition-resolution frames. Render alpha at composition resolution and " +
        "upscale separately, or use --format mp4.",
    );
  }
  const target = CANVAS_DIMENSIONS[input.outputResolution];
  // Aspect-ratio compare via cross-multiplication so the equality is integer-
  // safe. Float division (`target.width / compositionWidth`) loses precision
  // for non-power-of-2 ratios (e.g. cinema 4K 4096×2160 = 1.8963…) and a
  // future preset could trip a false-mismatch on otherwise valid input.
  if (target.width * input.compositionHeight !== target.height * input.compositionWidth) {
    throw new Error(
      `outputResolution ${input.outputResolution} (${target.width}×${target.height}) ` +
        `does not match the aspect ratio of the composition ` +
        `(${input.compositionWidth}×${input.compositionHeight}). ` +
        `Pick a preset whose orientation matches.`,
    );
  }
  // Aspect ratios match → widthRatio === heightRatio. Compute once.
  const widthRatio = target.width / input.compositionWidth;
  if (widthRatio < 1) {
    throw new Error(
      `outputResolution ${input.outputResolution} (${target.width}×${target.height}) ` +
        `is smaller than the composition (${input.compositionWidth}×${input.compositionHeight}). ` +
        `Downsampling via --resolution is not supported.`,
    );
  }
  if (!Number.isInteger(widthRatio)) {
    throw new Error(
      `outputResolution ${input.outputResolution} requires a non-integer ` +
        `device scale factor (${widthRatio}×) to upsample from ` +
        `${input.compositionWidth}×${input.compositionHeight}. ` +
        `Pick a preset that's an integer multiple, or rescale the composition.`,
    );
  }
  return widthRatio;
}

/**
 * Write compiled HTML and sub-compositions to the work directory.
 *
 * Exported for integration tests. Not part of the stable public API —
 * callers outside this package should use `executeRenderJob` instead.
 */
export function writeCompiledArtifacts(
  compiled: CompiledComposition,
  workDir: string,
  includeSummary: boolean,
): void {
  const compileDir = join(workDir, "compiled");
  mkdirSync(compileDir, { recursive: true });

  writeFileSync(join(compileDir, "index.html"), compiled.html, "utf-8");

  for (const [srcPath, html] of compiled.subCompositions) {
    const outPath = join(compileDir, srcPath);
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(outPath, html, "utf-8");
  }

  // Copy external assets (files outside projectDir) into the compiled directory
  // so the file server can serve them. The safe-path check uses
  // `isPathInside()` rather than a hardcoded separator — on Windows,
  // `compileDir + "/"` never matches because paths use `\\`, which caused
  // every external asset to be wrongly rejected as "unsafe" (see GH #321).
  for (const [relativePath, absolutePath] of compiled.externalAssets) {
    const outPath = resolve(join(compileDir, relativePath));
    if (!isPathInside(outPath, compileDir)) {
      console.warn(`[Render] Skipping external asset with unsafe path: ${relativePath}`);
      continue;
    }
    mkdirSync(dirname(outPath), { recursive: true });
    copyFileSync(absolutePath, outPath);
  }

  if (includeSummary) {
    const summary = {
      width: compiled.width,
      height: compiled.height,
      staticDuration: compiled.staticDuration,
      videos: compiled.videos.map((v) => ({
        id: v.id,
        src: v.src,
        start: v.start,
        end: v.end,
        mediaStart: v.mediaStart,
      })),
      audios: compiled.audios.map((a) => ({
        id: a.id,
        src: a.src,
        start: a.start,
        end: a.end,
        mediaStart: a.mediaStart,
      })),
      subCompositions: Array.from(compiled.subCompositions.keys()),
      renderModeHints: compiled.renderModeHints,
      hasShaderTransitions: compiled.hasShaderTransitions,
    };
    writeFileSync(join(compileDir, "summary.json"), JSON.stringify(summary, null, 2), "utf-8");
  }
}

export interface RenderModeHintResult {
  /** Resolved capture-mode boolean after folding in the hint. */
  forceScreenshot: boolean;
  /** True iff the hint flipped a `false` input to `true` (warn log fired). */
  autoSelected: boolean;
}

/**
 * Fold the composition's `renderModeHints.recommendScreenshot` signal
 * into the caller's already-resolved `forceScreenshot` value. Pure: the
 * caller owns the assignment to its own config. When the hint is the
 * deciding factor (caller passed `false`, hint says recommend), fires
 * the auto-select warn log with the composition's reason codes.
 */
export function applyRenderModeHints(
  alreadyForced: boolean,
  compiled: CompiledComposition,
  log: ProducerLogger = defaultLogger,
): RenderModeHintResult {
  if (alreadyForced || !compiled.renderModeHints.recommendScreenshot) {
    return { forceScreenshot: alreadyForced, autoSelected: false };
  }
  log.warn("Auto-selected screenshot capture mode for render compatibility", {
    reasonCodes: compiled.renderModeHints.reasons.map((reason) => reason.code),
    reasons: compiled.renderModeHints.reasons.map((reason) => reason.message),
  });
  return { forceScreenshot: true, autoSelected: true };
}

/**
 * Mutate the `RenderJob` view of the pipeline's progress and fire the
 * caller's `onProgress` callback. Hoisted here (out of `renderOrchestrator.ts`)
 * so the stage modules can call it without forming a runtime cycle.
 *
 * `completedAt` is stamped on the terminal `"failed"` / `"complete"`
 * transitions so callers that poll the job state can tell when the
 * pipeline finished.
 */
export function updateJobStatus(
  job: RenderJob,
  status: RenderStatus,
  stage: string,
  progress: number,
  onProgress?: ProgressCallback,
): void {
  job.status = status;
  job.currentStage = stage;
  job.progress = progress;
  if (status === "failed" || status === "complete") job.completedAt = new Date();
  if (onProgress) onProgress(job, stage);
}
