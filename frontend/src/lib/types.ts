// Shapes of the NDJSON events streamed by POST /api/solve (see backend).

export interface MetaEvent {
  type: "meta";
  seed: number;
  nest_version: string;
  threads: number;
  chunk_ms: number;
  max_sim_ms: number;
  noise_rate: number;
  stim_rate: number;
  pop_size: number;
  n_populations: number;
  n_neurons: number;
  raster_neurons_per_pop: number;
  neuron_model: string;
  neuron_params: Record<string, number | number[]>;
  weights: { inhibitory: number; stimulus: number; noise: number; delay_ms: number };
  connections: Connections;
  givens: number;
  build_seconds: number;
}

export interface Connections {
  inhibitory: number;
  noise: number;
  stimulus: number;
  stimulus_active: number;
  recorder: number;
  total: number;
}

export interface StepEvent {
  type: "step";
  k: number;
  t_start_ms: number;
  t_end_ms: number;
  grid: number[];
  tie: boolean[];
  silent: boolean[];
  valid_rows: boolean[];
  valid_cols: boolean[];
  valid_boxes: boolean[];
  givens_kept: boolean;
  performance: number;
  cells_changed: number;
  violations: number;
  total_spikes: number;
  pop_counts_b64: string;
  raster: { pop_b64: string; t_b64: string };
  wall_seconds: number;
}

export interface TraceWindow {
  k: number;
  t_start_ms: number;
  t_end_ms: number;
  grid: number[];
  performance: number;
  valid_rows: number;
  valid_cols: number;
  valid_boxes: number;
  invalid_rows: number[];
  invalid_cols: number[];
  invalid_boxes: number[];
  cells_changed: number;
  ties: number;
  silent: number;
  violations: number;
  givens_kept: boolean;
  total_spikes: number;
}

export interface PspInfo {
  weight: number;
  peak_mv: number;
  t_peak_ms: number;
  kernel_mv: number[];
}

export interface Analysis {
  outcome: {
    solved: boolean;
    solved_at_ms: number | null;
    sim_ms_total: number;
    wall_seconds_total: number;
    speed_ratio_sim_ms_per_wall_s: number;
    seed: number;
    givens_kept: boolean;
    best_performance: number;
    best_at_ms: number;
    best_grid: number[];
    final_invalid_rows: number[];
    final_invalid_cols: number[];
    final_invalid_boxes: number[];
    final_violations: number;
    max_sim_ms: number;
    build_seconds: number;
    threads: number;
    nest_version: string;
  };
  network: {
    n_populations: number;
    pop_size: number;
    n_neurons: number;
    connections: Connections;
    inhibitory_target_populations: { row: number; column: number; box_only: number; same_cell: number; total: number };
    fan_out_inhibitory: number;
    fan_in: { inhibitory: number; noise: number; stimulus: number; total: number };
    weights: MetaEvent["weights"];
  };
  neuron: {
    model: string;
    params: Record<string, number | number[]>;
    derived: {
      V_inf_mv: number;
      V_inf_above_threshold: boolean;
      tonic_isi_ms: number | null;
      tonic_rate_hz: number;
      psp_prefactor_ms: number;
      t_peak_ms: number;
      psp: { inhibitory: PspInfo; stimulus: PspInfo; noise: PspInfo };
      kernel_t_ms: number[];
      mean_current_noise: number;
      mean_current_stimulus: number;
      formulas: Record<string, string>;
    };
  };
  input: {
    givens: { row: number; col: number; digit: number; pop: number; rate_hz: number }[];
    n_givens: number;
    stim_rate_hz: number;
    noise_rate_hz: number;
    stim_weight_given: number;
    stim_weight_other: number;
    active_stimulus_connections: number;
  };
  readout: { chunk_ms: number; rule: string; trace: TraceWindow[] };
  firing: {
    t_end_ms: number[];
    rate_all_hz: number[];
    rate_winners_hz: number[];
    rate_losers_hz: number[];
    n_winner_neurons: number;
    n_loser_neurons: number;
    cell_counts_last: number[][];
    cell_counts_total: number[][];
    decision_margin: number[];
    decision_margin_mean: number;
    isi_cv_mean: number | null;
    isi_cv_median: number | null;
    isi_cv_neurons: number;
    isi_window_ms: number;
    neuron_rate_hist: { edges_hz: number[]; counts: number[] };
    total_spikes: number;
    mean_rate_hz: number;
    max_neuron_rate_hz: number;
  };
  convergence: {
    lock_in_ms: number[];
    lock_in_order: number[];
    changes_per_cell: number[];
    is_given: boolean[];
  };
  classical: {
    n_solutions: number;
    n_solutions_label: string;
    unique: boolean;
    solution: number[] | null;
    snn_equals_classical: boolean;
    verdict: "match" | "valid_but_different" | "not_solved" | "no_classical_solution";
  };
}

export interface DoneEvent {
  type: "done";
  solved: boolean;
  solved_at_ms: number | null;
  final_grid: number[];
  sim_ms_total: number;
  wall_seconds_total: number;
  analysis: Analysis;
}

export interface ErrorEvent {
  type: "error";
  message: string;
}

export type SolveEvent = MetaEvent | StepEvent | DoneEvent | ErrorEvent;

/** A window of raster data kept by the frontend (decoded once). */
export interface RasterWindow {
  k: number;
  tStart: number;
  pop: Uint16Array;
  t: Uint16Array; // 0.1 ms units relative to tStart
  grid: number[];
}
