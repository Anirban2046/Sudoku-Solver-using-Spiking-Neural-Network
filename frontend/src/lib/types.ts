// Shapes of the NDJSON events streamed by POST /api/solve (see backend/snn/simulation.py).

export interface Connections {
  inhibitory: number;
  noise: number;
  stimulus: number;
  stimulus_active: number;
  recorder: number;
  total: number;
}

export interface MetaEvent {
  type: "meta";
  seed: number;
  segment: number;
  t_offset_ms: number;
  nest_version: string;
  threads: number;
  chunk_ms: number;
  noise_rate: number;
  stim_rate: number;
  pop_size: number;
  n_populations: number;
  n_neurons: number;
  neuron_model: string;
  neuron: { V_th: number; V_reset: number; E_L: number; tau_m: number; t_ref: number };
  connections: Connections;
  givens: number;
  build_seconds: number;
  wall_budget_s: number;
  /** how long a kick silences the winners of conflicting cells */
  kick_ms: number;
  /** windows without progress before the network is kicked */
  patience: number;
  /** cells hinted by earlier requests of this run */
  hints: number[];
  /** simulated time after which restarts add hints (null: never) */
  hint_after_ms: number | null;
}

export interface Kick {
  /** "kick": winners of the listed cells were briefly silenced;
   *  "reset": all membrane potentials were re-randomised;
   *  "hint": a reset that also stimulated the listed cells with their correct digit. */
  kind: "kick" | "reset" | "hint";
  cells: number[];
}

export interface StepEvent {
  type: "step";
  k: number;
  t_start_ms: number;
  t_end_ms: number;
  grid: number[];
  valid_rows: boolean[];
  valid_cols: boolean[];
  valid_boxes: boolean[];
  givens_overridden: number;
  performance: number;
  violations: number;
  cells_changed: number;
  total_spikes: number;
  rate_all_hz: number;
  rate_winners_hz: number;
  rate_losers_hz: number;
  kick: Kick | null;
  wall_seconds: number;
}

export interface DoneEvent {
  type: "done";
  solved: boolean;
  /** true when the request used up its time budget; the client continues with a new request */
  resume: boolean;
  solved_at_ms: number | null;
  t_end_ms: number;
  final_grid: number[];
  wall_seconds: number;
  cells: { counts: number[][]; margin: number[]; margin_mean: number };
  network: NetworkView;
  classical: { n_solutions: number; verdict: "match" | "valid_but_different" | "not_solved" };
  /** all hinted cells so far; passed on to a continuation request */
  hints: number[];
}

/** The spiking network in its final window (see backend analysis.network_view). */
export interface NetworkView {
  neurons_per_pop: number;
  /** every spike: neuron index (0..3644) and time in 0.1 ms from the window start */
  spikes: { neuron_b64: string; t_b64: string };
  /** V_m in centi-mV of the first neuron of each cell's winning and runner-up population */
  vm: { dt_ms: number; samples: number; winner_b64: string; rival_b64: string; rival_digit: number[] };
  rates: {
    window_ms: number;
    winners_hz: number;
    others_hz: number;
    hist_edges_hz: number[];
    hist_winners: number[];
    hist_others: number[];
  };
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type SolveEvent = MetaEvent | StepEvent | DoneEvent | ErrorEvent;

/** One request to the backend; a run is one or more segments. */
export interface Segment {
  meta: MetaEvent;
  done: DoneEvent | null;
}
