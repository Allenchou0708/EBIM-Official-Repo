# EBiM Challenge 2026 Phase 2 — Technical Report (Task 2: Deformable Material Handling)

> Team: **Sinica_CITI_Boy**
> Track: Task 2 — Deformable Material Handling
> Submission type: **Repository Submission + Technical Report**
> Date: 2026-09-05

---

## 1. Executive Summary


We present a fully autonomous mobile manipulation system for EBiM Challenge 2026, Task 2: Deformable Material Handling, implemented and evaluated in the Isaac Sim environment.

The task requires the robot to autonomously approach the workspace, manipulate a deformable thermal pad, and accurately place it onto a designated region of a PCB. Our final simulator system combines closed-loop deterministic staging with a camera-conditioned, Code-as-Policy-inspired manipulation controller.

At the beginning of each episode, the mobile robot approaches the table from a predefined corner. Closed-loop deterministic controllers then raise and settle the spine, hold the left arm safely, and move the right arm to an observation pose with RMPflow. This provides a consistent starting state for the downstream manipulation controller.

After initialization, the controller uses head and right-wrist RGB-D observations to estimate the pad and target pose. Bounded visual corrections are converted into collision-aware RMPflow motions for grasping, transporting, aligning, releasing, and retreating. The final controller does not query an LLM or simulator ground truth at run time; the code structure was designed offline from the task specification and 180 successful development demonstrations.

During development, we explored several policy and training strategies, including ACT-based imitation learning, Pi0.5-based policy learning, self-distillation, and Code as Policy. We also investigated different action prediction horizons and model configurations to determine which approach was most suitable for the available simulation demonstrations.

Our experiments showed that the different approaches exhibited distinct failure modes. ACT frequently produced unstable and oscillatory actions during manipulation, while Pi0.5 provided smoother motion but could still release the thermal pad during transportation. We also explored self-distillation using task-state information as an additional teacher signal, but the teacher policy itself was not sufficiently reliable to provide a useful supervisory signal.

Among the approaches evaluated, the structured code policy achieved the most reliable overall behavior in our simulator experiments. Two deadline validation runs used different random seeds and different target PCB slots and both exceeded 0.5 IoU. These results demonstrate limited simulator generalization, not real-robot or distribution-wide success.

---

## 2. Related Works

### 2.1 Code as Policy

Integrating large foundation models into robotic systems has recently shifted the paradigm of robot learning from end-to-end policy training to hierarchical, language-guided orchestration. Specifically, the Code as Policy (CaP) framework (Liang et al., 2023) leverages the code-generation and reasoning capabilities of large language models (LLMs) to write executable robot control scripts directly from natural language task descriptions. Instead of predicting low-level joint commands or end-effector velocities through high-dimensional neural network policies, CaP employs the LLM as a high-level, few-shot planner. This planner reasons over semantic and spatial inputs to recursively invoke a predefined suite of API functions designed for perception, navigation, and manipulation.


### 2.2 Self Distillation

Knowledge distillation conventionalizes the transfer of capabilities from a larger, more powerful teacher model to a smaller student model. Recently, to eliminate the dependency on separate external teacher models, self-distillation has emerged as a prominent paradigm. For instance, On-Policy Self-Distillation (OPSD) instantiates both the teacher and student policies from a single model under different conditioning contexts. In OPSD, the teacher policy is granted access to privileged ground-truth reasoning traces, while the student policy only observes the query. By minimizing the per-token divergence over the student’s own on-policy rollouts, the model effectively teaches itself and achieves high sample efficiency.

In the field of robotics and reinforcement learning, a structurally parallel paradigm is widely adopted to bridge the sim-to-real gap, often drawing connections to classic online algorithms like DAgger. A privileged teacher policy, typically trained with full access to simulator states (e.g., precise joint positions or object coordinates), provides dense corrective supervision to a student policy constrained to realistic, raw sensory observations like point clouds.

---

## 3. Implementation Details & Results


### 3.1 Final Result: Code as Policy

Our best-performing simulator approach was a structured, Code-as-Policy-inspired controller. It is not an LLM call in the real-time control loop. Instead, task reasoning is compiled into a small set of auditable skills whose parameters are updated from current RGB-D observations:

1. Navigate the base to the calibrated table-relative pose and settle.
2. Raise and stabilize the spine, hold the left arm safely, and move the right arm to the observation pose with RMPflow.
3. Segment the blue pad in the right-wrist RGB-D image, transform its geometry into the world frame, and apply bounded forward, cross-axis, and yaw corrections before closing.
4. Keep the gripper closed while following one continuous C1 extraction-and-transfer curve. The curve includes a short vertical de-contact motion, a diagonal/forward peel that clears the symmetric base, and a smooth transfer without internal stop-and-settle waypoints.
5. Detect the red target from head-camera RGB-D, retarget the transport endpoint, and perform a supported late cross-axis/yaw correction from the wrist view.
6. Release quickly above the supported pad, retreat backward/upward, and return the arm to a clear observation pose before evaluation.

The nominal observation, latch, extraction, transport, and release landmarks were robust statistics from 180 successful development trajectories spanning four training-data collections. Runtime subscribes only to robot state and RGB-D camera topics; task-object poses, evaluator output, and simulator ground truth are not subscribed. Corrections are explicitly bounded to avoid turning a partial or occluded pad observation into a collision with the base or table.

For the deadline validation, Isaac Sim randomized the target among all four PCB slots, jittered every PCB, and moved the thermal pad together with its base by up to 1 cm in the image plane. The two retained runs used different target slots:

| Seed | Target slot | IoU | Orientation | Runtime VLA decisions | Video |
|---:|:---:|---:|:---:|---:|:---|
| 1003 | B | 0.5150 | correct | 0 | [head camera](Technical_Report_Videos/code_policy_seed1003_target_B_head.mp4) |
| 1104 | D | 0.5514 | correct | 0 | [head camera](Technical_Report_Videos/code_policy_seed1104_target_D_head.mp4) |

Both MP4 files are direct 1280x720 recordings of `/isaac/head_camera/image_raw`; they contain only the Isaac Sim camera view. The corresponding run evidence is stored under `live_technical-report-seed1003-targetB-align8_20260905_221618` and `live_technical-report-seed1104-targetD-head-align8_20260905_222239` in the experiment output directory.

This is deliberately a small validation set. Deformable contact remains stochastic, pad retention near an extreme edge grasp remains the dominant failure mode, and the controller has not been validated on Phase II hardware. Pi0.5 is retained as an experimental fallback because an earlier Formula 3 run completed the task, but neither final result above used Pi0.5 inference.



### 3.2 Using the ACT

We also trained an ACT policy from scratch using the available simulation demonstrations. The model contained approximately 83M trainable parameters, with an action prediction horizon of 5. The model was trained for 40000 steps, and the checkpoint with the lowest validation loss was selected for evaluation.

Despite decreasing training loss, the resulting policy exhibited unstable behavior during inference. The robot frequently produced oscillatory motions and repeatedly adjusted its end-effector position instead of committing to the grasping motion.

One possible explanation is that ACT does not explicitly provide a strong prior for the task-level manipulation behavior. Small differences between the predicted actions can accumulate during autoregressive action execution, resulting in unstable or oscillatory behavior around critical manipulation states.

Video: [ACT failure example](Technical_Report_Videos/EBIM_Phase2_ACT.gif)

### 3.3 Using the Pi0.5

We further evaluated Pi0.5 using the available simulation demonstrations. Our retained local checkpoint used two cameras, an 8-D right-arm/right-gripper action contract, and expert-only fine-tuning for 20,000 steps while keeping the VLM backbone frozen. It was evaluated with an action prediction horizon of 15. Separate whole-body and full-fine-tuning checkpoints were treated as interface ablations rather than equivalent models.

Compared with ACT, Pi0.5 produced noticeably smoother and more decisive motion. The longer action horizon reduced the oscillatory behavior observed with ACT, allowing the robot to maintain a more coherent motion trajectory during manipulation.

However, the policy still exhibited an important failure mode during the transportation stage. Although it could successfully grasp the thermal pad, the gripper sometimes opened or lost its grasp while moving the pad toward the target region. As a result, the robot was unable to complete the subsequent placement stage.

This indicates that although Pi0.5 provided better temporal consistency than ACT, the learned policy was still not sufficiently robust to maintain stable contact with the deformable object throughout the entire manipulation sequence.

Video: [Pi0.5 failure example](Technical_Report_Videos/EBIM_Phase2_PI05.gif)

### 3.4 Self Distillation

We also investigated self-distillation as a potential method for improving the Pi0.5 policy.

Our initial idea was to provide additional ground-truth state-change information in the language input of a teacher model, while removing this information from the student model. The teacher would therefore have access to additional task information and could potentially provide a stronger supervisory signal for the student.

However, preliminary experiments showed that the teacher policy itself was not sufficiently reliable. In particular, the teacher could still lose the thermal pad during the transportation stage. Since the teacher did not consistently produce successful behavior, its outputs could not provide a reliable target for distillation.

Therefore, we did not proceed with further self-distillation experiments and instead focused on the more reliable Code as Policy approach.


Video: [self-distillation failure example](Technical_Report_Videos/EBIM_Phase2_Self_Distillation.gif)

---



## 4. Simulator reproduction commands

The two reported simulator runs were produced with the repository launcher below. The Isaac Sim 5.1 container and evaluator must already be running on the same ROS domain. The policy container does not require Internet access at run time.

```bash
cd task2_isaacsim/baselines/pi05
export ROS_DOMAIN_ID=30

# Start the official room scene with board-slot and +/-1 cm object jitter.
./run_pi05.sh sim-up --gui --randomized

# In another terminal, run either reported seed.
./run_pi05.sh run \
  --code-policy \
  --seed 1003 \
  --run-label technical-report-seed1003-targetB-align8 \
  --max-duration-s 300 \
  --base-stage-max-duration-s 180 \
  --spine-stage-max-duration-s 180 \
  --manipulation-stage-max-duration-s 240

./run_pi05.sh run \
  --code-policy \
  --seed 1104 \
  --run-label technical-report-seed1104-targetD-head-align8 \
  --max-duration-s 300 \
  --base-stage-max-duration-s 180 \
  --spine-stage-max-duration-s 180 \
  --manipulation-stage-max-duration-s 240
```

The launcher executes `base navigation -> spine -> left safe hold/right observation pose -> camera-conditioned acquire -> continuous transport -> supported alignment -> release/retreat -> evaluation`. Although the common launcher retains Pi0.5 model profiles for ablations, `--code-policy` records `policy_inference_decisions=0` and does not use model output for the manipulation result.



---

## Contact

- Team: Sinica_CITI_Boy
- Email: 30033allen@gmail.com
- Repository: https://github.com/Allenchou0708/EBIM-Official-Repo
