# SmolVLA + LIBERO Demo

Early-stage demonstration of [SmolVLA](https://huggingface.co/lerobot/smolvla_base) (Vision-Language-Action model) running on LIBERO robot arm simulation on a Mac (no NVIDIA GPU — uses Apple MPS).

---

## SmolVLA + EEG 脑电信号融合实验

![EEG评估结果](eval_output/eeg_eval_libero_spatial.png)

### 各EEG条件含义

图中的五个条件代表实验时人脑想象的不同动作类型，来自PhysioNet的运动想象（Motor Imagery）数据集：

- **no_eeg**：零EEG信号，相当于没有脑电输入。作为基准对照组，模型纯粹依靠摄像头画面、机械臂状态和语言指令来预测动作。

- **left_fist**：受试者在脑电采集时想象"握紧左拳"。这是一种经典的运动皮层激活信号，代表左侧肢体的运动意图。

- **right_fist**：受试者想象"握紧右拳"，激活右侧肢体对应的运动皮层区域。

- **both_fists**：受试者同时想象"双手握拳"。这个信号在语义上与机械臂的夹爪关闭动作最接近——都是"抓握"的意图。

- **both_feet**：受试者想象"双脚用力"，激活的是运动皮层中控制下肢的区域，与手臂抓取任务关联最弱。

**both_fists表现最好**（MAE=0.323，夹爪准确率54.3%）——"双手握拳"的脑电信号在语义层面与机械臂执行抓取任务时的夹爪闭合动作最为接近。**left_fist和both_feet表现最差**，夹爪准确率只有35%左右，说明这两类脑电信号与抓取任务的语义偏差最大，注入后反而干扰了模型的判断。

---

### 新模型（SmolVLA+EEG）总体介绍

#### 输入

模型同时接收四类输入：

| 输入 | 具体内容 |
|---|---|
| 摄像头画面 | 两路图像——俯视角摄像头 + 腕部摄像头，各128×128像素 |
| 机械臂状态 | 14维向量：末端执行器XYZ位置(3) + 姿态四元数(4) + 7个关节角度(7) |
| 语言指令 | 任务描述文本，例如"把黑碗放到盘子上"，编码为最多48个token |
| **EEG脑电信号** | 64通道 × 320个采样点（2秒，160Hz），来自佩戴脑电帽的人类操作者 |

#### 输出

与原SmolVLA相同：一次生成**50步连续动作序列**，每步动作为7维向量：

- Δx、Δy、Δz（末端执行器的平移量）
- Δroll、Δpitch、Δyaw（末端执行器的旋转量）
- 夹爪开合值

#### EEG在其中的作用

EEG信号扮演的是**"人类意图的隐式调节器"**，而非直接的控制信号。具体机制是：

1. **EEGNet编码器**（预训练，参数冻结）将64×320的原始脑电波形压缩成一个64维的嵌入向量，捕捉当前的运动意图类别

2. **投影层**（900个新参数，可训练）将64维嵌入线性映射到14维

3. **加法注入**：将这14维向量直接**加**到机械臂状态向量上，形成"增强状态"，再送入SmolVLA的动作专家网络

这种设计的含义是：EEG信号不是替代摄像头或语言，而是对机械臂状态表征做一个"偏置"——告诉模型"操作者此刻的意图偏向于抓握/向左/向前"，从而微调最终生成的动作序列。

训练时加入了**50%的随机丢弃（dropout）**，保证模型在没有脑电信号时也能正常工作，EEG只是锦上添花的辅助通道，而非必须依赖的输入。

> **根本局限**：目前EEG与机器人动作之间没有真实的配对数据——PhysioNet的脑电是人类坐着想象握拳时录制的，与机器人操作场景完全分离。若能采集"人戴脑电帽同时遥控机器人"的同步数据（如CMU的Reach-and-Grasp数据集），EEG的调节作用将会大幅提升。

---

## What this shows

| File | What it does |
|------|-------------|
| `quick_test.py` | Sanity check — verifies all imports and tensor shapes (~15s) |
| `demo_libero_env.py` | LIBERO env via lerobot's `LiberoEnv` wrapper, random policy, saves frames |
| `demo_smolvla_libero.py` | **Main demo** — LIBERO-configured SmolVLA (7-DOF, 2 cameras), random-init expert |
| `demo_smolvla_base_libero.py` | Loads pretrained `lerobot/smolvla_base` (SO100) adapted to LIBERO observations |
| `libero_smolvla_config.py` | LIBERO-specific `SmolVLAConfig` (14-dim state, 7-dim action, 2 cameras) |
| `utils.py` | Shared helpers: `obs_to_policy_batch`, `dummy_dataset_stats`, `save_episode_frames` |

## Environment setup

Python 3.12 with both `lerobot` (from source) and `libero` (via PYTHONPATH):

```bash
# lerobot installed from /Users/r/lerobot in the `lerobot` conda env
conda env list   # should show: lerobot, libero

# Run any script with:
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python <script.py>
```

## Quickstart

```bash
cd /Users/r/Projects/SmolVLA_cl

# 1. Verify setup (no model download needed, ~15s)
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python quick_test.py

# 2. LIBERO env sanity check with random actions
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_libero_env.py

# 3. Main demo: SmolVLA pipeline on LIBERO (randomly-init action expert)
#    Downloads SmolVLM2-500M backbone (~1GB) on first run
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_libero.py

# 4. Pretrained smolvla_base on LIBERO observations (cross-embodiment test)
#    Downloads lerobot/smolvla_base (~1GB) on first run
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_base_libero.py

# Run a different LIBERO suite/task
SUITE=libero_object TASK_ID=2 STEPS=30 PYTHONPATH=/Users/r/LIBERO \
    /opt/anaconda3/envs/lerobot/bin/python demo_smolvla_libero.py
```

## Performance on M5 (MPS)

SmolVLA uses **action chunking**: one VLM call generates 50 actions, which are executed sequentially.

| Phase | Latency (random weights) |
|-------|--------------------------|
| First inference (chunk of 50) | ~5–6 s |
| Steps 1–49 (pop from queue) | ~1 ms |
| Effective throughput | ~9 actions/s |

With the full VLM backbone loaded (SmolVLM2-500M), expect ~8–15s per chunk on MPS.

## LIBERO feature layout

```
observation.images.image    (1, 3, 256, 256)  ← agentview camera
observation.images.image2   (1, 3, 256, 256)  ← wrist camera
observation.state           (1, 14)           ← eef_pos(3) + eef_quat(4) + joint_pos(7)
observation.language.tokens (1, 48)           ← tokenized task description
action                      (7,)              ← delta_xyz(3) + delta_rpy(3) + gripper(1)
```

## Next steps: fine-tuning on LIBERO data

```bash
# Download LIBERO demonstration dataset (from HuggingFace)
# Then train SmolVLA with LIBERO config:
PYTHONPATH=/Users/r/LIBERO /opt/anaconda3/envs/lerobot/bin/python -m lerobot.scripts.train \
    --policy.type=smolvla \
    --dataset.repo_id=<user>/libero_spatial_lerobot \
    --policy.device=mps \
    --batch_size=4 \
    --steps=10000
```

## Architecture

```
LIBERO sim (robosuite/MuJoCo)
      │ OffScreenRenderEnv
      ▼
lerobot LiberoEnv (gymnasium wrapper)
      │ obs = {pixels: {image, image2}, robot_state: {eef, joints, gripper}}
      ▼
obs_to_policy_batch()
      │ normalize images [0,1], stack state, tokenize task
      ▼
SmolVLAPolicy.select_action(batch)
      │ SmolVLM2-500M VLM backbone (vision + language encoding)
      │ Flow-matching action expert (generates 50-action chunk)
      │ Pops one action from queue per step
      ▼
env.step(action)  ← 7-DOF delta control
```
