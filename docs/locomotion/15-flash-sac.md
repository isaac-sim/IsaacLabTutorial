# 15. Explore Flash SAC after its integration lands

This exploratory lesson is gated on the merged Flash SAC integration. Verify its training/export interfaces, actor action bounds, observation groups, replay storage, temperature/critic updates, final-observation handling at timeouts, and resume semantics.

Run a bounded supported example first. Where Microduck is supported, compare from-scratch PPO and Flash SAC on matched physics, task/reward, resets, actor information, action semantics, transition budget, and fixed-command evaluation. PPO warm-start is a separate experiment.

Report wall time, actual environment transitions, optimizer updates, throughput excluding warmup, peak memory, task metrics, seeds, and failures. A single short run can teach the workflow; it cannot establish that one algorithm is generally faster or better.
