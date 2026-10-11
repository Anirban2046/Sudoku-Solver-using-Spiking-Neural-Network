from fastapi import FastAPI

app = FastAPI()


@app.get("/api/health")
def health():
    import os
    import time

    import nest

    t = time.time()
    nest.ResetKernel()
    nest.verbosity = nest.VerbosityLevel.ERROR
    n = nest.Create("iaf_psc_exp", 10, {"I_e": 500.0})
    sr = nest.Create("spike_recorder")
    nest.Connect(n, sr)
    nest.Simulate(100.0)
    return {"nest": nest.__version__, "cpus": os.cpu_count(), "spikes": sr.n_events, "seconds": round(time.time() - t, 3)}
