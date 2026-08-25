/**
 * The shapes `src/api.py` actually returns. Kept beside the Python, not derived from it —
 * there is no codegen step and adding one for six types would cost more than it saves.
 *
 * If a field here disagrees with `src/api.py`, the Python is right: it is what the running
 * server sends. `Health.preprocess_sha256` and `Stage.params` both come from the checkpoint
 * and the config, never from anything typed on this side.
 */

export interface ClassEntry {
  index: number
  freq_hz: number
}

export interface Health {
  model_run: string
  preprocess_version: string
  preprocess_sha256: string
  device: string
  classes: ClassEntry[]
  model: string
  opencv: { package: string; version: string }
  /** True when the checkpoint was trained on `SYNTH-*` sessions. */
  synthetic: boolean
  /**
   * Null when this run's numbers may be quoted. A sentence of prose when they may not —
   * a synthetic run reaches ~100 % accuracy and that number measures nothing about a
   * plate. `docs/05` §3, §4: it gets a banner, on every view, not a console line.
   */
  do_not_report: string | null
}

export interface Candidate {
  class_index: number
  freq_hz: number
  /** Softmax probability, unrounded. Never round it before the reader sees it (§4). */
  p: number
}

export type StageKind = 'image' | 'bars'

export interface Stage {
  /** 1 to 13, in pipeline order (§8). */
  n: number
  key: string
  /** Catalan, from the server — the stage names are part of the didactic surface. */
  title: string
  kind: StageKind
  /** The exact OpenCV call, e.g. `cv2.Canny`. */
  call: string
  /** Its parameters, read from `config/preprocess.json`. Never retyped here (§8). */
  params: Record<string, unknown>
  /** Shape of the real pipeline array. */
  shape: number[]
  dtype?: string
  /** Set only when the PNG is a downscaled copy — stage 1. Show both to the reader. */
  display_shape?: number[] | null
  /** Real measured milliseconds for this step, or null when it was not timed (§8). */
  ms: number | null
  png_base64?: string
  bars?: Candidate[]
}

export interface Prediction {
  top3: Candidate[]
  /**
   * `homography` when the four ArUco markers were found — the geometry the model was
   * trained on. `center_crop` otherwise, which is a geometry it never saw. §4 gives that
   * case its own visible state.
   */
  roi_mode: 'homography' | 'center_crop'
  markers_found: number
  stages: Stage[]
  timings_ms: Record<string, number>
  elapsed_ms: number
}

/** `docs/05-webapp-spec.md` §9. A session in progress, held by the server. */
export interface DraftSummary {
  draft_id: string
  session_id: string
  status: 'new' | 'calibrated' | 'committing'
  /** The native path of the draft directory, for pasting into a file manager. */
  path: string
  created: string
  updated: string
  calibrated: boolean
  classes: number[]
  total: number
  /** Three categories, not two — see `_summary` in `src/dataset_api.py`. */
  accepted: number
  rejected: number
  pending: number
  duplicates: number
  /** Size of the draft directory on disk. */
  bytes: number
}

export interface CalibrationInfo {
  ok: boolean
  roi_mode: string
  n_markers: number
  marker_ids: number[]
  corners: number[][] | null
  px_per_mm: number | null
  flags: string[]
}

export interface FrameRecord {
  name: string
  original_filename: string
  freq_hz: number
  /** `pending` is stored-but-not-yet-gated: a real state, not a placeholder. */
  quality: 'ok' | 'blurred' | 'partial' | 'reject' | 'pending'
  reason: string | null
  metrics: Record<string, number>
  sha256: string
  preview: boolean
}

export interface Draft extends DraftSummary {
  session: Record<string, unknown>
  calibration: CalibrationInfo | null
  frames: Record<string, FrameRecord[]>
  existing_rows: number
}

export interface CommitReport {
  session_id: string
  total: number
  accepted: number
  rejected: number
  by_reason: Record<string, number>
  by_class: Record<string, { total: number; accepted: number }>
  roi_mode: string
  flags: string[]
}

export interface CommitStatus {
  status: 'idle' | 'running' | 'done' | 'failed' | 'interrupted'
  phase: 'stage' | 'ingest' | 'rename' | 'finished' | null
  done: number
  total: number
  report: CommitReport | null
  error: { type: string; message: string } | null
}

export interface SplitResult {
  seed: number
  moved: number
  sessions: Record<string, string[]>
  images: Record<string, number>
  rejected: number
}

export interface Matrix {
  sessions: string[]
  classes: number[]
  cells: Record<string, Record<string, { total: number; accepted: number }>>
  sessions_per_class: Record<string, number>
  min_sessions_per_class: number
}
