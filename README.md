# EBiM Challenge 2026 — Task 2

This branch contains Team **Sinica_CITI_Boy**'s Phase II work for deformable
thermal-pad handling. The strongest verified result in this repository is a
camera-conditioned, Code-as-Policy-inspired controller evaluated in Isaac Sim
5.1.0.

The final simulator controller does **not** use an LLM, Pi0.5 output, evaluator
state, object-pose topics, deformable-vertex topics, or other simulator ground
truth at run time. It uses robot state plus head/right-wrist RGB-D observations
to parameterize bounded RMPflow motions.

The complete method, comparisons, evidence, and limitations are documented in
[the Task 2 technical report](Technical_Report_Sinica_CITI_Boy_Task2.md).

## Current status

| Component | Status |
|---|---|
| Code-policy simulator pipeline | End-to-end operational |
| Randomized simulator validation | 2/2 retained runs above 0.5 IoU |
| ACT | Evaluated; rejected because of oscillatory actions |
| Pi0.5 | Evaluated; retained as an experimental fallback |
| Self-distillation | Investigated; teacher was not reliable enough |
| Phase II real-robot controller | Dataset/runtime safety work only; no actuation validation |

The two deadline validation runs used different seeds and different target PCB
slots:

| Seed | Target | IoU | Orientation | VLA decisions | Video |
|---:|:---:|---:|:---:|---:|:---|
| 1003 | B | 0.5150 | correct | 0 | [head camera](Technical_Report_Videos/code_policy_seed1003_target_B_head.mp4) |
| 1104 | D | 0.5514 | correct | 0 | [head camera](Technical_Report_Videos/code_policy_seed1104_target_D_head.mp4) |

Both videos are direct 1280x720 recordings of
`/isaac/head_camera/image_raw`. They contain only the Isaac Sim camera view,
not the desktop, terminal, or GUI controls.

These are deliberately limited results. They demonstrate task completion for
two randomized simulator layouts; they do not establish distribution-wide or
real-robot generalization.

## Final simulator policy

The complete control sequence is:

```text
base navigation
  -> base settle
  -> spine raise and settle
  -> left-arm safe hold + right-arm observation pose
  -> right-wrist RGB-D pad localization
  -> bounded pre-grasp correction and close
  -> continuous de-contact, peel, and transfer
  -> head RGB-D target retargeting
  -> supported wrist RGB-D alignment
  -> fast release
  -> backward/upward retreat
  -> evaluation
```

The nominal observation, latch, extraction, transport, and release landmarks
are robust statistics from 180 successful development trajectories across four
training-data collections. They are not a replay of one episode. At run time,
camera measurements adjust the pad cross-axis/forward alignment, target slot,
target position, and pad/target yaw within explicit safety bounds.

The load-bearing motion after closing is one continuous C1 curve. It starts
with a short vertical de-contact, then moves diagonally and forward to peel the
elastic pad clear of its symmetric base before lateral transport. This avoids
the internal stop-and-settle midpoints that previously caused pad creep and
premature drops. Near the target, the controller permits only a bounded
cross-axis correction, releases quickly above support, and retreats clear of
the pad before evaluation.

The implementation is primarily in:

- [`fixed_hybrid_transport.py`](task2_isaacsim/baselines/pi05/live/fixed_hybrid_transport.py)
- [`runner.py`](task2_isaacsim/baselines/pi05/live/runner.py)
- [`fixed_stage_base.py`](task2_isaacsim/baselines/pi05/live/fixed_stage_base.py)
- [`fixed_stage_spine.py`](task2_isaacsim/baselines/pi05/live/fixed_stage_spine.py)
- [`fixed_stage_observation.py`](task2_isaacsim/baselines/pi05/live/fixed_stage_observation.py)

## Requirements

- Linux with Docker Engine and NVIDIA Container Toolkit.
- An NVIDIA GPU supported by the official Isaac Sim 5.1.0 environment.
- The official simulator container running as `isaac-sim-5-1-0-workshop` and
  mounting this repository at `/workspace/EBiM_Challenge`.
- X11 access for the Isaac Sim GUI.
- The policy image `ebim-task2-pi05-submit:local`.
- The development artifacts under
  `/scratch1/2026_ebim/allen_task2_pi05/`: the retained 20k checkpoint,
  simulator dataset metadata, staging audit, pinned LeRobot source, cache, and
  evidence/output directories expected by `run_pi05.sh`.

No Internet access is required during policy execution. The current
development launcher still validates and mounts the retained Pi0.5 artifacts
even in `--code-policy` mode, although the final manipulation controller does
not consume Pi0.5 predictions.

Optional path overrides can be placed in
`task2_isaacsim/baselines/pi05/.env.pi05`; secrets and machine-specific paths
must not be committed.

## Run the randomized simulator evaluation

From the repository root, first confirm the retained model profiles:

```bash
cd task2_isaacsim/baselines/pi05
./run_pi05.sh models
```

Start the randomized room scene in Terminal 1:

```bash
cd task2_isaacsim/baselines/pi05
ROS_DOMAIN_ID=30 ./run_pi05.sh sim-up --gui --randomized
```

The randomization profile swaps the target among all four PCB slots, jitters
each PCB, and moves the pad together with its base by up to 1 cm in XY. Yaw
randomization is disabled in this deadline validation profile.

After the simulator reports that the room bridge is ready, run one policy in
Terminal 2:

```bash
cd task2_isaacsim/baselines/pi05

ROS_DOMAIN_ID=30 ./run_pi05.sh run \
  --code-policy \
  --seed 1003 \
  --run-label code-policy-seed1003 \
  --max-duration-s 300 \
  --base-stage-max-duration-s 180 \
  --spine-stage-max-duration-s 180 \
  --manipulation-stage-max-duration-s 240
```

Change `--seed` and `--run-label` for another independent trial. The launcher
resets the scene, checks the evaluation camera, executes all staging and
manipulation phases, retreats the arm, and calls the evaluator. A code-policy
run should report `policy_inference_decisions=0`.

Stop the development stack with:

```bash
./run_pi05.sh down
```

The concise launcher and model-contract reference is in
[`task2_isaacsim/baselines/pi05/README.md`](task2_isaacsim/baselines/pi05/README.md).

## Record only the Isaac Sim head camera

[`record_ros_image_video.py`](task2_isaacsim/scripts/record_ros_image_video.py)
records a ROS image stream directly to H.264. It avoids screen capture and
therefore excludes terminal and desktop content. From the repository root,
run it through the existing policy image:

```bash
docker run --rm --network host --entrypoint bash \
  -e ROS_DOMAIN_ID=30 \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e PYTHONPATH=/workspace/EBiM_Challenge \
  -v "$PWD:/workspace/EBiM_Challenge:ro" \
  -v "$PWD/Technical_Report_Videos:/videos" \
  ebim-task2-pi05-submit:local -lc \
  "source /opt/ros/jazzy/setup.bash && \
   python3 /workspace/EBiM_Challenge/task2_isaacsim/scripts/record_ros_image_video.py \
     --topic /isaac/head_camera/image_raw \
     --output /videos/code_policy_run_head.mp4 \
     --fps 10"
```

Press `Ctrl-C` after the policy finishes so FFmpeg writes the MP4 trailer.

## Experimental policy comparisons

The launcher retains three Pi0.5 profiles for controlled comparisons:

- `ours-20k`: two cameras, 8-D right-arm/right-gripper state and action,
  relative joint actions, normally evaluated at horizon 15.
- `submitted-30k`: saved 20-D whole-body transform with only its right 8-D
  command slice published in the safe default mode.
- `robot-dreams-20k --robot-dreams-native`: its original three-camera, 37-D
  state and absolute 20-D action contract, with base output forced to zero.

These interfaces load and publish valid actions, but none of the three Pi0.5
profiles demonstrated reliable end-to-end simulator completion. Our 20k model
could grasp the pad but did not transport it left consistently; it sometimes
opened or lost the gripper during transport. One earlier Formula 3 run did
complete the task, so Pi0.5 remains a feasible research path rather than the
final controller used for the two results above.

Failure examples are retained for method comparison:

- [ACT](Technical_Report_Videos/EBIM_Phase2_ACT.gif)
- [Pi0.5](Technical_Report_Videos/EBIM_Phase2_PI05.gif)
- [self-distillation](Technical_Report_Videos/EBIM_Phase2_Self_Distillation.gif)

## Verification

The final code review used the existing policy container to run the complete
Task 2 simulator test suite:

```bash
docker run --rm --network host --entrypoint bash \
  -e PYTHONPATH=/workspace/EBiM_Challenge \
  -v "$PWD:/workspace/EBiM_Challenge:ro" \
  ebim-task2-pi05-submit:local -lc \
  "source /opt/ros/jazzy/setup.bash && \
   /opt/lerobot/.venv/bin/python -m unittest discover \
     -s task2_isaacsim/tests -p 'test_*.py'"
```

Current result: **151/151 tests pass**. Python compilation, shell syntax, and
`git diff --check` also pass.

## Phase II real-robot boundary

[`task2_real/`](task2_real/README.md) contains the released real-dataset audit,
right-arm-only Pi0.5 adapter, site-calibration templates, and fail-closed shadow
runtime checks. It intentionally does not claim a deployable real-robot policy:

- no real command has been published by this branch;
- simulator base coordinates are not accepted as site calibration;
- spine units and hardware controller limits still require on-site validation;
- the deterministic simulator arm poses have not been collision-checked at the
  evaluation sites;
- offline loss reduction is not evidence of closed-loop task success.

For remote participation, on-site staff must first capture the actual ROS
interfaces and site calibration, then pass the zero-publication shadow gate
before any hardware actuation is enabled.

## Repository map

| Path | Purpose |
|---|---|
| [`task2_isaacsim/`](task2_isaacsim/README.md) | Isaac Sim Task 2 environment and controllers |
| [`task2_isaacsim/baselines/pi05/`](task2_isaacsim/baselines/pi05/README.md) | Unified simulator launcher, Pi0.5 experiments, and code policy |
| [`task2_real/`](task2_real/README.md) | Phase II real-data and runtime safety work |
| [`Technical_Report_Videos/`](Technical_Report_Videos/) | Retained success and comparison videos |
| [`Technical_Report_Sinica_CITI_Boy_Task2.md`](Technical_Report_Sinica_CITI_Boy_Task2.md) | Final Task 2 technical report |
| [`STATUS.md`](STATUS.md) | Wider benchmark repository component status |
| [`docs/developer_setup.md`](docs/developer_setup.md) | General developer setup |

The historical Phase I simulator-GT submission remains documented in
[`PHASE1_SUBMISSION_RUNBOOK.md`](task2_isaacsim/baselines/pi05/PHASE1_SUBMISSION_RUNBOOK.md)
and
[`PHASE1_POLICY_REPORT.md`](task2_isaacsim/baselines/pi05/PHASE1_POLICY_REPORT.md).
It is not the final Phase II simulator strategy described here.

## Scoring and limitations

Task 2 scoring is `Pick Success x Orientation Success x IoU`; wrong orientation
sets the score to zero. The local evaluator result is recorded only after the
release and retreat, so a high intermediate overlap cannot hide an early drop
or a pad displaced by the gripper during withdrawal.

Known limitations are:

- only two randomized seeds with two target slots are reported here;
- the deadline profile does not randomize target or pad yaw;
- deformable contact remains stochastic and an extreme edge grasp can still
  creep out during transport;
- camera segmentation uses simulator appearance assumptions that require
  recalibration or replacement for real imagery;
- the base route and arm/spine staging are simulator-specific;
- no Phase II hardware success is claimed.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
