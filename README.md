# Sudoku Solver using Spiking Neural Network

A web application that solves Sudoku puzzles with a **spiking neural network
(SNN)**. The network is simulated in the [NEST
Simulator](https://www.nest-simulator.org/), and its activity is shown live in
the browser.

**Live demo:** <https://sudoku-solver-using-spiking-neural-network.vercel.app/>

![The application after solving a puzzle: the solved board, the live statistics and the performance plot, followed by the network analysis down to the firing-rate histogram](docs/screenshot.png)

---

## Contents

1. [Overview](#1-overview)
2. [Features](#2-features)
3. [How the network solves Sudoku](#3-how-the-network-solves-sudoku)
4. [System architecture](#4-system-architecture)
5. [User interface](#5-user-interface)
6. [API reference](#6-api-reference)
7. [Results](#7-results)
8. [Technology stack](#8-technology-stack)
9. [Project structure](#9-project-structure)
10. [Running locally](#10-running-locally)
11. [Configuration](#11-configuration)
12. [Deployment](#12-deployment)
13. [Testing and quality](#13-testing-and-quality)
14. [Limitations](#14-limitations)
15. [Credits and licence](#15-credits-and-licence)

---

## 1. Overview

Sudoku can be written as a constraint-satisfaction problem, where every row,
column and 3×3 box must contain each digit exactly once. In this project, no
search algorithm is used in the solving loop. Instead, a network of 3,645 leaky
integrate-and-fire neurons encodes every possible digit in every cell, and the
rules of Sudoku are wired into the network as inhibitory synapses. Random
background spikes make the network explore until it settles into a state that
breaks no rule.

In the application, a user can enter or generate a puzzle and watch the network
solve it in real time. After that, the user can check how the network behaved:
which neurons won, how strongly they fired, and how their membrane potentials
changed over time.

| | |
|---|---|
| **Problem** | Solve any valid 9×9 Sudoku puzzle |
| **Method** | Stochastic spiking neural network, a winner-take-all constraint network |
| **Simulator** | NEST 3.10.0, model `iaf_psc_exp` |
| **Network size** | 729 populations, 3,645 neurons, 521,235 connections |
| **Interface** | Single-page web app with live streaming of the simulation |
| **Deployment** | Vercel (static frontend and Python serverless backend) |

---

## 2. Features

### Puzzle input

| Feature | Description |
|---|---|
| Editable 9×9 grid | Type digits 1–9. Backspace, Delete or 0 clears a cell, and the arrow keys move between cells |
| Live validation | Duplicate digits in a row, column or box are highlighted, with a message such as "5 appears twice in row 3" |
| **Randomly Initialize** | Generates a puzzle at once, with 36 given digits and exactly one solution |
| **Clear Grid** | Empties the board |
| **Start Solving** | Enabled only for a puzzle with at least 17 givens, no duplicates and a confirmed solution. Otherwise, a short note shows the reason |
| Editing after a run | When a cell is changed, editing continues from the board as it is shown |

### Live simulation

| Feature | Description |
|---|---|
| Live board | Updated after every 100 ms of simulated time. Given digits are white, the network's digits are light-blue and italic, and overridden givens are red. Invalid rows and columns get red markers, and invalid boxes get a red frame. Cells that changed in the latest window are highlighted |
| Live statistics | Simulated time, rule conflicts (cell pairs in a row, column or box that show the same digit), status, performance, spikes per window, network firing rate, and firing rate of the winning neurons |
| Performance plot | Fraction of valid rows, columns and boxes over simulated time. The time axis rescales as the run grows |
| Replay | After a run, a slider replays any 100 ms window on the board |

### Network analysis (shown after a run)

| Feature | Description |
|---|---|
| Summary metrics | Simulated time to solution, number of neurons and connections, total spikes, mean firing rate, winner vs. non-winner firing rate, wall-clock time |
| One cell, up close | When a cell is picked, a spike raster of all 45 of its neurons in the final 100 ms is shown. The membrane potential of a winning neuron is also shown against a runner-up neuron (recorded with a NEST multimeter) |
| Winner-take-all over time | Firing rate of the winning digit populations against all other neurons, over the whole run |
| Firing-rate distribution | Histogram of per-neuron firing rates, with winners and suppressed neurons shown separately |

### Robustness

| Feature | Description |
|---|---|
| Keeps searching | A convergence controller (§3.6) acts when the network stalls, and the run continues until the puzzle is solved |
| No overall time limit | Long runs are split into consecutive requests, and each request stays within the hosting time limit (§3.7) |
| Concurrent visitors | Several visitors can solve puzzles at the same time, because each simulation runs in its own process with its own network |
| Fault tolerance | Dropped connections, server errors and busy servers are retried automatically, and the run resumes from the last window |
| Input safety | The server checks every puzzle again and rejects any puzzle that it cannot confirm to be solvable |

### Presentation

* GitHub-style dark theme everywhere, including the charts and the print output
* Responsive layout, down to phone width
* Keyboard navigation, labelled controls and visible focus, with no automated
  accessibility violations found (axe-core)

---

## 3. How the network solves Sudoku

The network model is the NEST example `pynest/examples/sudoku`
(J. Gille, S. Furber, A. Rowley), which is based on an earlier SpiNNaker
implementation. The model is used without changes, except for one upstream
bug fix (§3.8).

In short, Sudoku is a **constraint-satisfaction problem**. Every possible
decision ("cell (r, c) holds digit d") is turned into a small group of spiking
neurons, and every rule is turned into inhibition between groups whose
decisions would clash. Each neuron is intrinsically active and also gets random
input, so the network keeps "trying out" assignments. An assignment that breaks
a rule makes itself weaker, while a complete solution keeps itself active. The
sections below describe the model (§3.1–§3.5), the controller that this project
adds (§3.6–§3.7), the theory behind the convergence (§3.9), the numerical
details (§3.10) and the background literature (§3.11).

### 3.1 Encoding

Each of the 81 cells has one **population** of 5 neurons for each digit 1–9:

```
81 cells × 9 digits = 729 populations
729 populations × 5 neurons = 3,645 neurons
population index = row × 81 + column × 9 + (digit − 1)
```

An active population means "this cell holds this digit". Formally, each
population stands for a binary variable $x_{r,c,d}$, which is 1 when cell $(r,
c)$ holds digit $d$. A valid solution satisfies four families of
constraints (an *exact cover* with 4 × 81 = 324 constraints):

$$\sum_{d=1}^{9} x_{r,c,d} = 1 \quad \text{(one digit per cell)}$$

$$\sum_{c=1}^{9} x_{r,c,d} = 1, \qquad \sum_{r=1}^{9} x_{r,c,d} = 1, \qquad \sum_{(r,c) \in b} x_{r,c,d} = 1 \quad \text{(each digit once per row, column and box)}$$

Each population has 5 neurons, not one, and this makes each decision robust.
The spike count of a population is an average over five noisy neurons, so no
single neuron can flip a decision alone.

### 3.2 Neuron model

All neurons are current-based leaky integrate-and-fire neurons with
exponentially decaying synaptic currents (NEST model `iaf_psc_exp`):

$$\frac{dV}{dt} = -\frac{V - E_L}{\tau_m} + \frac{I_{syn}(t) + I_e}{C_m}, \qquad \frac{dI_{syn}}{dt} = -\frac{I_{syn}}{\tau_{syn}}$$

When $V$ reaches $V_{th}$, the neuron emits a spike. It is then reset to
$V_{reset}$ and stays refractory for $t_{ref}$. Each incoming spike of weight
$w$ makes $I_{syn}$ jump by $w$. Excitatory and inhibitory currents decay
separately, but both decay with $\tau_{syn}$.

| Parameter | Symbol | Value |
|---|---|---|
| Membrane capacitance | $C_m$ | 0.25 |
| Bias current | $I_e$ | 0.5 |
| Membrane time constant | $\tau_m$ | 20 ms |
| Refractory period | $t_{ref}$ | 2 ms |
| Synaptic time constants (exc./inh.) | $\tau_{syn}$ | 5 ms |
| Resting potential | $E_L$ | −65 mV |
| Reset potential | $V_{reset}$ | −70 mV |
| Firing threshold | $V_{th}$ | −50 mV |
| Initial potential | $V_m(0)$ | uniform in [−65, −55] mV |

The parameters come from the original PyNN model (nF, nA), and NEST reads them
as pF and pA. However, only the ratios $I/C_m$ and $w/C_m$ enter the voltage
equation, so the dynamics in millivolts and milliseconds are the same.

**Intrinsic activity.** For a constant input current $I$, the membrane
potential moves towards

$$V_\infty(I) = E_L + \frac{\tau_m I}{C_m}$$

With the bias current alone, $V_\infty = -65 + 20 \cdot 0.5 / 0.25 = -25$ mV,
which is far above the threshold. Every neuron is therefore **intrinsically
active**: it would fire regularly even without any input. For a constant current above
threshold, the interspike interval and the firing rate are

$$T(I) = \tau_m \ln\frac{V_\infty(I) - V_{reset}}{V_\infty(I) - V_{th}}, \qquad f(I) = \frac{1}{t_{ref} + T(I)}$$

For the bias alone, this gives $T = 20 \ln(45/25) = 11.76$ ms and
$f = 72.7$ Hz. The refractory period limits the rate to $1/t_{ref} = 500$ Hz.

**Postsynaptic potentials.** When a single spike of weight $w$ arrives at
rest, it changes the membrane potential by

$$\Delta V(t) = \frac{w}{C_m} \cdot \frac{\tau_m \tau_{syn}}{\tau_m - \tau_{syn}} \left( e^{-t/\tau_m} - e^{-t/\tau_{syn}} \right)$$

This change is largest at $t_{peak} = \frac{\tau_m \tau_{syn}}{\tau_m - \tau_{syn}} \ln\frac{\tau_m}{\tau_{syn}} = 9.24$ ms:

| Input | Weight $w$ | Peak $\Delta V$ |
|---|---:|---:|
| Inhibition from one neuron | −0.2 | −2.52 mV |
| One stimulus spike (givens) | 1.3 | +16.38 mV |
| One noise spike | 1.6 | +20.16 mV |

**No reversal potentials.** The synapses inject currents instead of changing
conductances, so strong inhibition can drive $V$ far below $E_L$. In the
membrane-potential plot of the application, runner-up neurons drop below
−150 mV when many winning populations inhibit them at the same time.

### 3.3 Connectivity: the rules as inhibition

Every population inhibits the 28 populations that it conflicts with:

* the same digit in the other 8 cells of its **row**;
* the same digit in the other 8 cells of its **column**;
* the same digit in the 4 remaining cells of its **3×3 box**;
* the other 8 digits of its **own cell**.

Each neuron therefore inhibits 28 × 5 = 140 neurons.

Formally, the populations are the nodes of a **conflict graph**. Two nodes
$(r, c, d)$ and $(r', c', d')$ are connected when they are the same cell with
different digits, or the same digit in two cells that share a row, column or
box. Every node has exactly 28 neighbours (8 + 8 + 4 + 8).
Each connected pair of populations is wired all-to-all in both directions,
which gives 729 × 28 = 20,412 directed population pairs × 5 × 5 neurons =
510,300 inhibitory synapses.

This wiring makes two inhibitory motifs:

* **One winner per cell.** The 9 populations of a cell inhibit each other and
  form a *winner-take-all* circuit, which settles on one digit.
* **One occurrence per unit.** The populations of the same digit in a row,
  column or box also inhibit each other, so a digit that wins in one cell
  suppresses the same digit in all 20 peer cells.

| Connection type | Count | Weight |
|---|---:|---:|
| Inhibitory (neuron → neuron) | 510,300 | −0.2 |
| Background noise (Poisson generator → neuron) | 3,645 | 1.6 |
| Stimulus (Poisson generator → neuron) | 3,645 | 0, or 1.3 for given digits |
| Recording (neuron → spike recorder) | 3,645 | – |
| **Total** | **521,235** | |

All synapses are static and have a delay of 1 ms.

### 3.4 Input and noise

* **Noise:** every neuron gets an independent 350 Hz Poisson spike train, and
  this noise makes the network explore different digit assignments.
* **Givens:** each population has its own 200 Hz Poisson generator. Its
  weight is 0, except for the populations of the given digits (1.3), and this
  extra drive makes a given digit win its cell.

A Poisson spike train of rate $\nu$ that goes through synapses of weight $w$
produces a fluctuating current. The mean and the variance of this current
follow from Campbell's theorem:

$$\mu_I = w \tau_{syn} \nu, \qquad \sigma_I^2 = \frac{w^2 \nu \tau_{syn}}{2}$$

| Source | Rate $\nu$ | Mean current $\mu_I$ | Fluctuation $\sigma_I$ |
|---|---:|---:|---:|
| Background noise (all neurons) | 350 Hz | 2.80 | 1.50 |
| Stimulus (given digits only) | 200 Hz | 1.30 | 0.92 |

The noise fluctuations are of the same order as the margins that decide
between digits (§3.9), so neurons near the threshold fire at random moments.
This randomness works like a *temperature*: it lets the network leave
unfavourable states instead of freezing in the first state it reaches. A given
digit gets an extra mean current of +1.30, about 46 % more drive than the noise
gives. This pushes the population of the given digit to win.

### 3.5 Readout and performance

The simulation runs in windows of 100 ms. After each window, the spikes of
every population are counted, and each cell is read out as the digit whose
population fired the most (ties are broken at random):

$$\hat{d}_{r,c} = \arg\max_{d} N_{r,c,d}$$

Here, $N_{r,c,d}$ is the number of spikes fired in the window by the
population for digit $d$ in row $r$, column $c$. The board gets this score:

$$\text{performance} = \frac{R + C + B}{27}$$

Here, $R$, $C$ and $B$ are the numbers of valid rows, columns and boxes.

A puzzle is **solved** when all 27 units are valid and every given digit is
kept.

After the network has settled, the readout is clear. A winning population fires
at about 240 Hz (about 5 × 240 × 0.1 ≈ 120 spikes per window), while a
suppressed population fires at about 6 Hz (about 3 spikes). The
**decision margin** measures how clear a decision is:

$$m_{r,c} = \frac{N_{top} - N_{second}}{N_{top}}$$

This margin is shown in the "One cell, up close" view.

The number of rule violations is used as an **energy** of the read-out board:

$$E = \sum_{(i,j) \in P} [\hat{d}_i = \hat{d}_j]$$

Here, $P$ is the set of 810 unordered cell pairs that share a row, column or
box, and $[\cdot]$ is 1 if the condition is true and 0 otherwise. $E = 0$ holds
exactly for a valid grid, and the convergence controller (§3.6) watches this
value.

### 3.6 Convergence controller

The network is stochastic. It can stall with a few conflicts for a long time,
or it can settle on a valid grid that overrides a given digit (valid grids
are stable attractors). The original example simply stops after a fixed time.
This project adds a controller around the unchanged network, and the controller
escalates in two steps:

| Step | Trigger | Action |
|---|---|---|
| **Kick** | No improvement in conflicts for 5 windows (0.5 s) | The winning populations of the conflicting cells are silenced for 50 ms with a strong negative bias current, so that a rival digit can take over. If a kick does not help, the next kick silences every free cell that shares a row, column or box with a conflict |
| **Restart** | 4 kicks in a row without improvement | All membrane potentials are re-randomised. This restarts the search while the input of the givens is kept |

**Stall detection.** After window $k$, the controller computes
$b_k = E_k + G_k$, the number of conflicting cell pairs plus the number of
overridden givens. It counts the windows since $b$ last reached a new minimum,
and when this count reaches 5, the network is considered stuck.

**Kick.** For 50 ms, the bias current of the selected populations is set to
$I_e = -5$. Their total mean input becomes about $-5 + 2.8 = -2.2$, so
$V_\infty \approx -241$ mV and they fall silent. The inhibition that they
sent to their rivals decays with $\tau_{syn} = 5$ ms. This frees the other
digits of the same cells and the same digit in the peer cells, so a rival can
take over. After that, the bias goes back to normal, and the potential
recovers with $\tau_m = 20$ ms. A kick is the spiking version of a "flip" in
stochastic local search. If one kick does not help, the next kick silences the
whole neighbourhood of the conflict, so that this region is solved again
against the rest of the board.

**Restart.** All membrane potentials are drawn again, uniformly from
[−65, −55] mV (`SudokuNet.reset_V_m`). The synaptic currents die out within a
few $\tau_{syn}$, so the network starts a fresh transient from a new random
state, as in random-restart search. The input of the givens is kept.

Both steps only disturb the activity of the network, and every digit is still
decided by the spiking dynamics. Together with the continuation requests
(§3.7), the network keeps searching until it reaches a valid solution.

### 3.7 Running within serverless time limits

A serverless function can run for at most 300 s, so each request stops after a
wall-clock budget of 240 s and reports `resume: true`. The browser then sends a
follow-up request, which rebuilds the network with a new random
seed and continues the same timeline. For the user, this looks like one
continuous run.

### 3.8 Changes to the upstream code

* `SudokuNet.reset_V_m` drew from `uniform(-65, 55)`, which was corrected to
  `uniform(-65, -55)`.
* The upstream plotting code drew the frame of invalid boxes at `(3j, 3j)`, but
  the browser draws the frame at the correct position `(row 3i, column 3j)`.

### 3.9 Why the dynamics solve the puzzle

A mean-field argument shows why solutions are stable and why conflicting
states are not. In this argument, each input is replaced by its mean current
(§3.4), and each population is replaced by its firing rate $r_j$.

**Input to a neuron.** A neuron in population $i$ receives

$$I_i = I_e + \mu_{noise} + g_i \mu_{stim} + 5 w_{inh} \tau_{syn} \sum_{j \in \mathcal{N}(i)} r_j$$

Here, $g_i = 1$ for a given digit and 0 otherwise, $w_{inh} = -0.2$,
$\mathcal{N}(i)$ is the set of its 28 conflicting populations, and the factor 5
counts the neurons per population. The neuron fires when
$V_\infty(I_i) > V_{th}$, i.e. when

$$I_i > I_\theta = \frac{C_m (V_{th} - E_L)}{\tau_m} = 0.1875$$

**Effect of one winner.** A winning population fires at about 240 Hz, so it
adds $5 \cdot (-0.2) \cdot 0.005 \cdot 240 \approx -1.2$ to each population
that it inhibits. The effect of suppressed populations is negligible because
they fire very rarely. If $k$ neighbours of a population are winners:

| Winning neighbours $k$ | Input, free digit | | Input, given digit | |
|---:|---:|---|---:|---|
| 0 | +3.30 | fires | +4.60 | fires |
| 1 | +2.10 | fires | +3.40 | fires |
| 2 | +0.90 | fires | +2.20 | fires |
| 3 | −0.30 | **silent** | +1.00 | fires |
| 4 | −1.50 | **silent** | −0.20 | **silent** |

A free digit is silenced by **3 or more** winning neighbours, but a given digit
is silenced only by **4 or more**.

**Valid solutions are self-sustaining.** In a valid grid:

* every winning population has $k = 0$, because no other winner shares its cell
  and no peer cell has a winner with the same digit. Its input of 3.30 gives a
  mean-field rate of $f(3.3) \approx 282$ Hz (measured: 220–270 Hz);
* every losing population (digit $d$ in a cell that holds another digit) is
  inhibited by the winner of its own cell and by the cells in its row, column
  and box that hold $d$. These are 2 or 3 different cells, so $k$ is 3 or 4 and
  the population is silenced. In 30 solved grids, every
  losing population had exactly 3 or 4 winning neighbours.

The winners keep every loser silent, and the silent losers do not disturb the
winners. A valid solution is therefore a **self-consistent fixed point** (an
attractor) of the dynamics.

**Conflicting states are unstable.** If two cells in a unit show the same
digit, their winning populations inhibit each other ($k \geq 1$). Their drive
and firing rates go down, and the inhibition that they send to the rival
digits of their own cells becomes weaker. At the same time, some losing
populations have fewer than 3 winning neighbours, so they stay above
threshold. Such states are not self-consistent, and the noise (σ ≈ 1.5, comparable to the input margins in the table above)
soon pushes a rival to win. Each such move tends to reduce the number of
conflicts, so the energy $E$ of §3.5 slowly goes down towards 0.

**Why givens can be overridden.** A given digit is silenced only by 4 or more
winning neighbours. This is rare but possible: a complete valid grid
that disagrees with a given can surround the population of the given with 4
winners and still stay stable. These are the "valid grid that overrides a
given" attractors of §3.6, and the controller breaks them with a kick on the
overridden cell.

**Search as stochastic sampling.** As a whole, the network performs a noisy
descent on the conflict energy. States with fewer conflicts are more stable
and last longer. The noise still allows some moves to worse states, and in this
way the network can escape local minima. This is closely related to Hopfield
networks for optimisation and to the view of noisy spiking networks as sampling
from a Boltzmann-like distribution that concentrates on low-energy states
(§3.11). In this view, the solution with $E = 0$ is the most stable
state that the network can reach.

### 3.10 Numerical simulation details

| Aspect | Setting |
|---|---|
| Simulator | NEST 3.10.0 |
| Time step | 0.1 ms (NEST default resolution) |
| Integration | Exact integration of the linear subthreshold dynamics of `iaf_psc_exp` |
| Synaptic delay | 1 ms for all synapses |
| Noise generator | One `poisson_generator` connected to all neurons. NEST sends each target an independent spike train |
| Stimulus generators | One `poisson_generator` per population (729) |
| Spike recording | One `spike_recorder` per population, cleared after every 100 ms window |
| Membrane recording | One `multimeter` that samples $V_m$ of the first neuron of every population every 1 ms |
| Simulation loop | Repeated `Simulate(100 ms)` calls. The network state carries over, so the readout samples one continuous trajectory |
| Random seeds | One seed per request, used for NEST's random numbers and for the tie-breaking of the readout |
| Parallelism | OpenMP threads per simulation = available CPUs ÷ simultaneous runs |

A typical 100 ms window has about 11,700 spikes. Each spike goes to 140
inhibitory targets, so one window processes about 1.6 million synaptic events.

### 3.11 Background and references

* The network: NEST Simulator example `pynest/examples/sudoku`, by
  J. Gille, S. Furber and A. Rowley, which is based on the SpiNNaker
  [IntroLab Sudoku](https://github.com/SpiNNakerManchester/IntroLab/tree/master/sudoku)
  by S. Furber and A. Rowley.
* J. J. Hopfield and D. W. Tank (1985). "Neural" computation of decisions in
  optimization problems. *Biological Cybernetics*, 52, 141–152.
* L. Buesing, J. Bill, B. Nessler and W. Maass (2011). Neural dynamics as
  sampling: a model for stochastic computation in recurrent networks of
  spiking neurons. *PLoS Computational Biology*, 7(11), e1002211.
* Z. Jonke, S. Habenschuss and W. Maass (2016). Solving constraint
  satisfaction problems with networks of spiking neurons. *Frontiers in
  Neuroscience*, 10, 118.
* G. A. Fonseca Guerra and S. B. Furber (2017). Using stochastic spiking
  neural networks on SpiNNaker to solve constraint satisfaction problems.
  *Frontiers in Neuroscience*, 11, 714.
* S. Rotter and M. Diesmann (1999). Exact digital simulation of time-invariant
  linear systems with applications to neuronal modeling. *Biological
  Cybernetics*, 81, 381–402.

---

## 4. System architecture

```text
                 ┌──────────────────────────────┐
                 │ Browser                      │
                 │ React single-page app        │
                 └───┬───────────────┬────▲─────┘
              GET /  │  POST         │    │  NDJSON stream:
                     │  /api/solve   │    │  meta, a step every
                     │               │    │  100 ms, done
┌────────────────────┼───────────────┼────┼────────────────────┐
│ Vercel             │               │    │                    │
│                    ▼               ▼    │                    │
│  ┌───────────────────────┐   ┌──────────┴───────────────┐    │
│  │ Static frontend       │   │ FastAPI service          │    │
│  │ Vite build            │   │ main.py                  │    │
│  └───────────────────────┘   └─────────────┬────────────┘    │
│                                            ▼                 │
│                              ┌──────────────────────────┐    │
│                              │ Engine (engine.py):      │    │
│                              │ worker process per run   │    │
│                              └─────────────┬────────────┘    │
│                                            ▼                 │
│                              ┌──────────────────────────┐    │
│                              │ Simulation loop          │    │
│                              │ (simulation.py):         │    │
│                              │ readout and controller   │    │
│                              └─────────────┬────────────┘    │
│                                            ▼                 │
│                              ┌──────────────────────────┐    │
│                              │ NEST kernel              │    │
│                              │ 3,645 neurons            │    │
│                              └──────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

| Component | Responsibility |
|---|---|
| **Frontend** (`frontend/`) | Puzzle input and validation, puzzle generation, streaming client, live board and charts, network analysis views |
| **API** (`backend/main.py`) | Request validation, a limit on simultaneous simulations per instance, NDJSON streaming |
| **Engine** (`backend/snn/engine.py`) | Starts one worker process for each simulation and passes on its events. NEST keeps one global kernel per process, so separate processes let several visitors solve at the same time. A worker is stopped as soon as its client disconnects |
| **Worker** (`backend/snn/worker.py`) | Runs one simulation with its own NEST kernel and writes its events to the engine |
| **Simulation** (`backend/snn/simulation.py`) | Builds the network, runs the 100 ms windows, reads out the board, applies the convergence controller, and records spikes and membrane potentials |
| **Controller logic** (`backend/snn/control.py`) | Chooses which populations a kick silences (no NEST dependency) |
| **Analysis** (`backend/snn/analysis.py`) | Conflict counting, per-cell decision margins, final network view (spikes, membrane potentials, firing-rate statistics) |

**Request flow**

1. The browser posts the puzzle to `/api/solve`.
2. The server validates the puzzle, builds the network (a few seconds) and
   streams one `meta` event.
3. After that, it streams one `step` event for each 100 ms window. On Vercel,
   these events come about 0.5–0.6 s apart.
4. The run ends with a `done` event. If the time budget of the request ran
   out, this event carries `resume: true`, and the browser continues with a
   new request.

---

## 5. User interface

The page is a single screen, from top to bottom:

1. **Title and grid.** The 9×9 board, with three buttons below it:
   **Randomly Initialize**, **Clear Grid**, **Start Solving**.
2. **Live panel** (during and after a run). It shows time progressed, rule
   conflicts, status and current statistics on the left, and the performance
   plot on the right. A replay slider appears after the run ends.
3. **Inside the spiking network** (after a run):
   * summary tiles: solved-at time, neurons, connections, spikes fired,
     winning-neuron firing rate, wall time;
   * **One cell, up close:** a clickable mini-board (amber cells were the
     closest contests), the 45-neuron spike raster and the membrane-potential
     traces of the selected cell;
   * **Winner-take-all, across the board:** firing rates over time;
   * **How fast each neuron fires:** firing-rate histogram (log scale).

---

## 6. API reference

### `GET /api/health`

Runs a 10-neuron NEST smoke test, which is useful for warming up a cold instance.

```json
{ "nest": "3.10.0", "cpus": 2, "max_concurrent_runs": 3, "spikes": 60, "seconds": 0.56 }
```

When every simulation slot is in use, it answers at once with `"busy": true` and does not run the test.

### `POST /api/solve`

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `puzzle` | `int[9][9]` | yes | Digits 0–9, where 0 means empty |
| `seed` | `int` | no | Random seed for NEST and the readout (reproducible runs) |
| `t_offset_ms` | `int` | no | Continuation: simulated time that has already passed |
| `segment` | `int` | no | Continuation: index of this request within the run |

**Response:** `application/x-ndjson`, one JSON object per line.

| Event | When | Main fields |
|---|---|---|
| `meta` | once, after the network is built | seed, NEST version, neuron model and parameters, connection counts, build time |
| `step` | every 100 ms window | board, valid rows/columns/boxes, performance, conflicts, spikes, firing rates (all, winners, others), controller action |
| `done` | last line | `solved`, `resume`, `solved_at_ms`, final board, per-cell digit spike counts, final network view (spikes, membrane potentials, rate histogram) |
| `error` | on failure | `message` |

**Status codes**

| Code | Meaning |
|---|---|
| `200` | Stream started |
| `422` | Invalid puzzle: wrong shape, values out of range, duplicates, fewer than 17 givens, no solution, or solvability could not be confirmed |
| `429` | Every simulation slot on this instance is in use (`MAX_CONCURRENT_RUNS`), and the browser waits and retries for up to 3 minutes |

---

## 7. Results

These values were measured on the deployed application and in local
experiments. The network is stochastic, so the results of single runs vary.

### Typical behaviour

| Quantity | Observed value |
|---|---|
| Random 36-given puzzles | Usually solved within 0.4–10 s of simulated time |
| Simulation speed on Vercel | About 0.6 s of wall time per 100 ms window |
| Spikes per 100 ms window | About 11,700 |
| Mean firing rate (all neurons) | About 32 Hz |
| Winning neurons vs. the rest | About 220–270 Hz vs. 5–8 Hz |
| Network build time | A few seconds |

### Effect of the convergence controller

Ten puzzles were tested: six random 36-given puzzles and the NEST example
puzzles 1, 2, 5 and 6. Each puzzle got the same wall-clock budget of one
request:

| Configuration | Puzzles solved |
|---|---|
| Original network, no controller | 4 / 10 |
| With kicks and restarts | 6–7 / 10 |

The two hardest puzzles, NEST example 5 and example 6 (the so-called "world's
hardest Sudoku"), are still the difficult cases. With the controller,
they were still unsolved after about 67 s of simulated time (three
consecutive requests). Runs continue across requests, so the search on such
puzzles goes on, but it can take many minutes.

---

## 8. Technology stack

| Layer | Technology |
|---|---|
| Neural simulation | NEST Simulator 3.10.0, NumPy |
| Backend | Python 3.12, FastAPI, NDJSON streaming |
| Frontend | React 19, TypeScript, Vite 6, Chart.js 4, HTML canvas |
| Testing | pytest, Vitest, Testing Library |
| CI | GitHub Actions |
| Hosting | Vercel Services (static site and Python function) |
| Alternative hosting | Docker image for the backend |

---

## 9. Project structure

```
.
├── backend/
│   ├── main.py              FastAPI app: /api/health, /api/solve, validation, run limit
│   ├── snn/
│   │   ├── sudoku_net.py    NEST network definition (from the NEST example)
│   │   ├── simulation.py    Network build, 100 ms run loop, convergence controller
│   │   ├── control.py       Kick-target selection
│   │   ├── engine.py        One worker process per simulation, cleanup on disconnect
│   │   ├── worker.py        Runs one simulation in its own process
│   │   ├── analysis.py      Conflicts, decision margins, final network view
│   │   ├── helpers.py       Validation, classical solver, puzzle generator
│   │   └── encoding.py      Base64 packing of numeric arrays
│   ├── tests/               pytest suite
│   ├── requirements.txt     Runtime dependencies
│   ├── requirements-dev.txt Development and test dependencies
│   └── Dockerfile           Container image for alternative hosting
├── frontend/
│   ├── index.html
│   └── src/
│       ├── App.tsx          Page layout, run control, continuation and retries
│       ├── components/      Grid, controls, live panel, charts, raster, analysis
│       ├── lib/             Sudoku logic, API client, event types, decoding
│       ├── __tests__/       Vitest suite
│       └── styles.css       GitHub-style dark theme
├── docs/
│   ├── reference/sudoku_net.py  Unmodified upstream network, for comparison
│   └── screenshot.png
├── .github/workflows/ci.yml     Runs both test suites on every push
├── vercel.json                  Vercel services and routing
└── LICENSE                      GPL-2.0
```

---

## 10. Running locally

### Prerequisites

* Python 3.12 (NEST supports 3.9–3.13)
* Node.js 20 or newer
* Linux, macOS or WSL2. NEST has no native Windows build.

### Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn main:app --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The development server forwards `/api` requests
to the backend on port 8000.

---

## 11. Configuration

All settings are optional environment variables of the backend.

| Variable | Default | Description |
|---|---|---|
| `WALL_BUDGET_S` | `240` | Wall-clock seconds that one request may simulate before the browser continues with a new request (10–280) |
| `NOISE_RATE` | `350` | Background Poisson rate in Hz |
| `MAX_CONCURRENT_RUNS` | `3` | Number of simulations that may run at the same time on one server instance (1–16). Each one uses about 300 MB |
| `CORS_ORIGINS` | – | Comma-separated origins that may call a backend hosted separately |

Frontend build variable:

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE` | – | Base URL of the backend, only needed when the backend is hosted somewhere else |

---

## 12. Deployment

### Vercel (current deployment)

`vercel.json` defines one Vercel project with two services:

| Service | Root | Description |
|---|---|---|
| `web` | `frontend/` | Vite static build that installs only the production dependencies |
| `api` | `backend/` | FastAPI Python function, maximum duration 300 s |

Requests to `/api/*` go to the backend, and all other paths go to the frontend.
To deploy, import the GitHub repository in Vercel, set the **Framework Preset**
to **Services**, and keep the root directory at the repository root.
Every push to `main` deploys the project again.

### Docker (alternative backend host)

```bash
cd backend
docker build -t sudoku-snn-backend .
docker run -p 7860:7860 sudoku-snn-backend
```

Point the frontend to it with `VITE_API_BASE`, and allow the origin of the
frontend with `CORS_ORIGINS`.

---

## 13. Testing and quality

```bash
cd backend  && python -m pytest -q   # 34 tests
cd frontend && npm test              # 23 tests
```

| Area | Covered by tests |
|---|---|
| Sudoku logic | Conflict detection, solution counting, unique-puzzle generation, Start-button rules |
| Network | Connection counts (521,235), event order, payload sizes, recorded spikes and membrane potentials, solution validity |
| Controller | Kick and restart triggering, protection of given cells |
| API and concurrency | Validation errors, simultaneous solves, the concurrency limit, streaming, health check while busy, worker shutdown when a client disconnects |
| Frontend behaviour | Grid editing and clearing, continuation across requests, retries after errors, busy waiting |

GitHub Actions runs both suites and the frontend production build on every
push.

---

## 14. Limitations

* **Stochastic run time.** Easy puzzles usually finish within a few seconds
  of simulated time. There is no upper bound, and the hardest puzzles can take
  many minutes of wall time.
* **Shared CPU between simultaneous runs.** Up to `MAX_CONCURRENT_RUNS`
  (default 3) simulations run at the same time on one server instance, each one
  in its own process of about 300 MB. They share the CPUs of the
  instance, so simultaneous runs progress more slowly. Above this limit, more
  visitors wait for a short time or are served by another instance.
* **Hosting quotas.** The free Vercel plan includes 4 active-CPU hours per
  month. A solve uses about 1 CPU-second for each 100 ms window, so a typical
  solve uses about 5–100 CPU-seconds.

---

## 15. Credits and licence

The network (`backend/snn/sudoku_net.py`) and the validation helpers ported
into `backend/snn/helpers.py` come from the
[NEST Simulator](https://www.nest-simulator.org/) Sudoku example by
J. Gille, S. Furber and A. Rowley, © 2004 The NEST Initiative. The network was
first designed by Steve Furber and Andrew Rowley (University of Manchester)
for [SpiNNaker](https://github.com/SpiNNakerManchester/IntroLab/tree/master/sudoku).

This project is licensed under the GNU General Public License, version 2 or
(at your option) any later version, which is the licence of the NEST code that
the project includes. See [LICENSE](LICENSE).
