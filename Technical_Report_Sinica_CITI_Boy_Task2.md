# EBiM Challenge 2026 Phase 2 — Technical Report (Task 2: Deformable Material Handling)

> Team: **Sinica_CITI_Boy**
> Track: Task 2 — Deformable Material Handling
> Submission type: **Repository Submission + Technical Report**
> Date: 2026-09-04

---

## 1. Executive Summary


We present a fully autonomous mobile manipulation system for EBiM Challenge 2026, Task 2: Deformable Material Handling, implemented and evaluated in the Isaac Sim environment.

The task requires the robot to autonomously approach the workspace, manipulate a deformable thermal pad, and accurately place it onto a designated region of a PCB. Our system combines a hard-coded motion controller for the initial robot positioning with a policy for the manipulation stage.

At the beginning of each episode, the mobile robot approaches the table from a predefined corner. A hard-coded ROS controller is then used to raise the robot's spine and position both arms at suitable initial configurations in front of the workspace. This provides a consistent starting state for the learned manipulation policy.

After initialization, the policy receives visual observations from the head camera and the left and right wrist cameras, together with a task description. Based on these observations, the policy continuously predicts robot actions for manipulating the thermal pad. The resulting behavior consists of three main stages: grasping the thermal pad, transporting it toward the target region, and placing and attaching it onto the PCB.

During development, we explored several policy and training strategies, including ACT-based imitation learning, Pi0.5-based policy learning, self-distillation, and Code as Policy. We also investigated different action prediction horizons and model configurations to determine which approach was most suitable for the available simulation demonstrations.

Our experiments showed that the different approaches exhibited distinct failure modes. ACT frequently produced unstable and oscillatory actions during manipulation, while Pi0.5 provided smoother motion but could still release the thermal pad during transportation. We also explored self-distillation using task-state information as an additional teacher signal, but the teacher policy itself was not sufficiently reliable to provide a useful supervisory signal.

Among the approaches evaluated, Code as Policy achieved the most reliable overall behavior in our experiments. Therefore, it was selected as our final approach for the competition submission. The final system uses visual observations to reason about the current manipulation state and invokes predefined robot control functions to generate fine-grained actions throughout the task.

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

Our best-performing approach was Code as Policy.

Instead of directly predicting low-level robot actions with a learned policy network, we use Codex to define a set of robot control functions that can be invoked during inference. The system receives visual observations from the head camera and the wrist cameras and uses the current visual state together with the task description to determine which control function should be executed. The selected functions then generate fine-grained robot motions for grasping, transporting, aligning, and placing the thermal pad.

The predefined functions provide a structured interface for controlling the robot, while the vision-language model is responsible for interpreting the current scene and selecting appropriate actions. This allows the system to combine high-level visual reasoning with precise low-level robot control.

video path : 



### 3.2 Using the ACT

We also trained an ACT policy from scratch using the available simulation demonstrations. The model contained approximately 83M trainable parameters, with an action prediction horizon of 5. The model was trained for 40000 steps, and the checkpoint with the lowest validation loss was selected for evaluation.

Although the decent training, the resulting policy exhibited unstable behavior during inference. The robot frequently produced oscillatory motions and repeatedly adjusted its end-effector position instead of committing to the grasping motion.

One possible explanation is that ACT does not explicitly provide a strong prior for the task-level manipulation behavior. Small differences between the predicted actions can accumulate during autoregressive action execution, resulting in unstable or oscillatory behavior around critical manipulation states.

video path : Technical_Report_Videos/EBIM_Phase2_ACT

### 3.3 Using the Pi0.5

We further evaluated Pi0.5 using the available simulation demonstrations. The model was fine-tuned on the simulation data with LoRA on the action expert and evaluated with an action prediction horizon of 15.

Compared with ACT, Pi0.5 produced noticeably smoother and more decisive motion. The longer action horizon reduced the oscillatory behavior observed with ACT, allowing the robot to maintain a more coherent motion trajectory during manipulation.

However, the policy still exhibited an important failure mode during the transportation stage. Although it could successfully grasp the thermal pad, the gripper sometimes opened or lost its grasp while moving the pad toward the target region. As a result, the robot was unable to complete the subsequent placement stage.

This indicates that although Pi0.5 provided better temporal consistency than ACT, the learned policy was still not sufficiently robust to maintain stable contact with the deformable object throughout the entire manipulation sequence.

video path : Technical_Report_Videos/EBIM_Phase2_PI05

### 3.4 self distillation

We also investigated self-distillation as a potential method for improving the Pi0.5 policy.

Our initial idea was to provide additional ground-truth state-change information in the language input of a teacher model, while removing this information from the student model. The teacher would therefore have access to additional task information and could potentially provide a stronger supervisory signal for the student.

However, preliminary experiments showed that the teacher policy itself was not sufficiently reliable. In particular, the teacher could still lose the thermal pad during the transportation stage. Since the teacher did not consistently produce successful behavior, its outputs could not provide a reliable target for distillation.

Therefore, we did not proceed with further self-distillation experiments and instead focused on the more reliable Code as Policy approach.


video path : Technical_Report_Videos/EBIM_Phase2_Self_Distillation

---



## 4. Build and run commands

Clone and check out the exact submitted revision:

git clone https://github.com/Allenchou0708/EBIM-Official-Repo.git
cd EBIM-Official-Repo
git checkout --detach <FINAL_PHASE2_COMMIT_SHA>

Build the policy image from the repository root:

docker build --pull
--file task2_real/Dockerfile
--tag academia-sinica-task2-phase2:<FINAL_PHASE2_COMMIT_SHA>
.

Optional container health check:

docker run --rm
--gpus all
academia-sinica-task2-phase2:<FINAL_PHASE2_COMMIT_SHA>
health

Launch the policy:

docker run --rm
--gpus all
--network host
--ipc host
--env ROS_DOMAIN_ID=<ORGANIZER_ASSIGNED_DOMAIN_ID>
--env RMW_IMPLEMENTATION=rmw_fastrtps_cpp
academia-sinica-task2-phase2:<FINAL_PHASE2_COMMIT_SHA>
run

The Docker image entrypoint is:

ENTRYPOINT ["/submission-entrypoint.sh"]

The run entrypoint command starts the complete policy in this order:

base navigation -> spine staging -> left-arm safe hold ->
right-arm observation pose -> right-arm/right-gripper PI0.5 inference.

The checkpoint, tokenizer, calibration schema, Python dependencies, and ROS 2
runtime are included in the image. No source-directory, dataset, checkpoint,
or cache bind mount is required. No Internet access is required at run time.
The only required external interface is the organizer's local ROS 2 network.



---

## Contact

- Team: Sinica_CITI_Boy
- Email: 30033allen@gmail.com
- Repository: https://github.com/Allenchou0708/EBIM-Official-Repo
