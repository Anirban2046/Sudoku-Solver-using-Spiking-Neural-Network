# Sudoku Solver using SNN

A web app that solves Sudoku with a **spiking neural network** simulated in
[NEST](https://www.nest-simulator.org/). You enter a puzzle (or generate one),
and the network runs on the server. Every 100 ms of simulated time is streamed
to the browser, so you watch the board, the performance curve and a spike
raster update live. A full analysis of the run appears when it finishes.

![Screenshot: the board after a run, with the performance curve and spike raster](docs/screenshot.png)

The network is the one from the NEST example
[`pynest/examples/sudoku`](https://github.com/nest/nest-simulator/tree/master/pynest/examples/sudoku)
by J. Gille, S. Furber and A. Rowley, which is based on the original SpiNNaker
implementation. It has 729 populations of 5 `iaf_psc_exp` neurons each (3,645
neurons) and 521,235 connections. The Sudoku rules are wired in as inhibition,
and the given digits are fed in as Poisson stimulation.

**Full specification, design decisions and the maths:** [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md)

## Run locally

You need Python 3.12 (3.9–3.13 work) and Node.js 20+. NEST has no native
Windows wheel, so on Windows use WSL2.

```bash
# backend (terminal 1)
cd backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn main:app --reload --port 8000

# frontend (terminal 2)
cd frontend
npm install
npm run dev            # http://localhost:5173, proxies /api to :8000
```

## Tests

```bash
cd backend && python -m pytest -q     # engine, analysis and API (about 30 s)
cd frontend && npm test               # sudoku logic and Start-button rules
```

GitHub Actions (`.github/workflows/ci.yml`) runs both suites on every push.

## Deploy (Vercel)

`vercel.json` defines one Vercel project with two services: the Vite frontend
(`web`) and the FastAPI backend (`api`, 300 s max duration, 2 GB). Deploy it
either way:

* **Dashboard:** import the GitHub repo into Vercel, keep the root directory
  at the repo root, and deploy.
* **CLI:** run `npm i -g vercel && vercel login && vercel` for a preview, then
  `vercel --prod`.

After deploying, open `/api/health`. It should report `"nest": "3.10.0"` and
`spikes > 0`.

| Env var | Default | Meaning |
|---|---|---|
| `MAX_SIM_MS` | 20000 | Simulated-time limit per run (keep ≤ 40000 on Hobby) |
| `NOISE_RATE` | 350 | Background Poisson rate (Hz) |
| `RASTER_NEURONS_PER_POP` | 1 | Neurons per population sent to the live raster (1–5) |
| `VITE_API_BASE` | *(empty)* | Set only when the backend runs elsewhere |
| `CORS_ORIGINS` | *(empty)* | Comma-separated origins allowed to call a separately hosted backend |

If NEST can't run as a Vercel function, deploy `backend/` with its
`Dockerfile` to another host and point the frontend at it with
`VITE_API_BASE`. The build guide's Appendix 2 lists the options.

## API

* `GET /api/health` runs a 10-neuron NEST smoke test. Use it to warm up a cold
  instance.
* `POST /api/solve` takes `{"puzzle": [[...9 ints...] x 9], "seed": 123}`
  (the seed is optional) and returns an NDJSON stream: one `meta` line, one
  `step` line per 100 ms window, then `done` (with the full `analysis`) or
  `error`. An invalid board returns `422`. If a run is already in progress, it
  returns `429` (the frontend retries 3 times).

## Layout

```
backend/   FastAPI app (main.py) and the snn package:
           sudoku_net.py (NEST network, verbatim + one upstream bug fix),
           simulation.py (build + run loop), engine.py (runs NEST on one
           dedicated thread), analysis.py, helpers.py, encoding.py
frontend/  Vite + React + TypeScript: grid, controls, live panel, report
docs/      build guide, reference copy of sudoku_net.py, screenshot
```

## Notes

* **The network is stochastic.** Most random 36-given puzzles converge within a
  few seconds of simulated time. Some don't converge within the limit. The app
  then shows the best state it reached, and the report says how close it got.
* **All NEST calls run on one dedicated thread** (`backend/snn/engine.py`).
  NEST segfaults if it is imported on one Python thread and driven from
  another, which a thread-pooled web server would otherwise do.

## Licence

GPL-2.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The network and
the ported helpers are © The NEST Initiative.
