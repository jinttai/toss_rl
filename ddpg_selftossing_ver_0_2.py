# ----- Cell 2 : 공통 설정 -----
import numpy as np, torch, torch.nn as nn, gymnasium as gym, mujoco
import os
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])

def get_relative_rotation_quaternion_manual(q_initial, q_target):
    q_initial_inv = quaternion_conjugate(q_initial)
    q_relative_transform = quaternion_multiply(q_initial_inv, q_target)
    return q_relative_transform

def get_act_fn(name: str ):
    name = (name or "").lower()
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu":  nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        None:  None
    }.get(name, None)

def normalize_vector(v): # 3D 벡터 정규화 함수
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm

def jacobian_vel(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    return np.dot(jacp, data.qvel) # Returns 3D com velocity of the body

def wrap_angle(angle): # 각도를 [-pi, pi] 범위로 정규화
    return (angle + np.pi) % (2 * np.pi) - np.pi

CFG = {
    # -------- mujoco --------
    "model_path": "mujoco_src/spacerobot_twoarm_3dof.xml",
    "model_path_fixed": 'mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml',
    "release_timestep": 300,

    # -------- PD 제어 --------
    "kp": 50,
    "kd": 0,
    "max_vel": np.deg2rad(180.0),
    "target_xy_com_vel_components":  np.array([5, 5]), # 목표 COM 속도의 X, Y 성분

    # -------- RL 하이퍼파라미터 --------
    "gamma": 0.99,
    "tau":   0.001,
    "actor_lr":  1e-3,
    "critic_lr": 1e-3,
    "batch_size": 128, # 배치 사이즈 변경
    "buffer_size": 100_000,
    "episode_number": 30_000,
    "episode_length": 302,
    "start_random": 5_000,
    # 관절각도(6), 관절속도(6), COM위치(3), COM XY평면 속도 각도(1)
    "raw_observation_dimension": 16,
    "goal_dimension": 1,             # 목표 차원: COM XY평면 속도 각도 (스칼라)
    "her_replay_k": 4,


    # -------- 네트워크 파라미터 --------
    "actor_net": {
        "hidden": [400, 300],
        "hidden_activation": "tanh",
        "output_activation": "tanh"
    },
    "critic_net": {
        "hidden": [400, 300],
        "hidden_activation": "tanh",
        "output_activation": None
    },

    "save_dir": "rl_results/SelfTossing_AngleGoal",
    "actor_save_path": "actor_ddpg_her_angle_v0_2.pth",
    "critic_save_path": "critic_ddpg_her_angle_v0_2.pth",
    "results_save_path": "training_results_her_angle_v0_2.npz"
}

"""# 3. mujoco 환경 정의"""

class SpaceRobotEnv(gym.Env):
    def __init__(self, xml_path: str, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.model_fixed = mujoco.MjModel.from_xml_path(cfg['model_path_fixed'])
        self.data_fixed = mujoco.MjData(self.model_fixed)

        self.site_ee = self.model.site("end_effector").id
        self.q_start = 7 # 관절 qpos 시작 인덱스

        # 관찰 공간 정의 (raw_observation_dimension 사용)
        hi_raw = np.inf*np.ones(cfg["raw_observation_dimension"], np.float32)
        self.observation_space = gym.spaces.Box(-hi_raw, hi_raw, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1, 1, (6,), dtype=np.float32) # 6 DoF 팔

        self.horizon = self.cfg["episode_length"]
        self.step_cnt = 0
        self.similarity = 0.0

        self.dt = self.model.opt.timestep
        self.release_timestep = self.cfg['release_timestep']
        self.goal_timestamp_steps = self.release_timestep + 1

        # 목표 COM 속도 XY 성분으로부터 목표 각도 계산
        target_xy_components = cfg['target_xy_com_vel_components']
        self.original_target_com_angle = np.arctan2(target_xy_components[1], target_xy_components[0])
        self.current_episode_goal_angle = np.array([self.original_target_com_angle]) # 목표를 1D 배열로 저장

    def _raw_obs(self):
        qpos_joints = self.data.qpos[self.q_start:].copy() # 6개 관절 각도
        qvel_joints = self.data.qvel[6:].copy() # 6개 관절 속도
        
        com_pos_3d, com_vel_3d = self._calculate_com() # 전체 시스템의 3D COM 위치 및 속도
        
        # COM XY 평면 속도 각도 계산
        com_vel_xy_angle = np.arctan2(com_vel_3d[1], com_vel_3d[0])
        
        # 관찰: 관절각도(6), 관절속도(6), COM위치(3), COM XY평면 속도 각도(1) = 16
        obs = np.concatenate([
            qpos_joints,
            qvel_joints,
            com_pos_3d,
            np.array([com_vel_xy_angle])
        ]).astype(np.float32)
        
        if len(obs) != self.cfg["raw_observation_dimension"]:
            print(f"Error: Observation dimension mismatch. Expected {self.cfg['raw_observation_dimension']}, Got {len(obs)}")
        return obs

    def _pid_velocity_control(self, target_qvel, current_qvel, qacc, Kp, Kd):
        error_vel = target_qvel - current_qvel
        qacc_command = Kp * error_vel
        return qacc_command

    def _apply_pd(self, des_vel):
        self.data_fixed.qpos = self.data.qpos[7:13].copy()
        self.data_fixed.qvel = self.data.qvel[6:12].copy()
        self.data_fixed.qpos[0:3] = -self.data.qpos[9:6:-1].copy()
        self.data_fixed.qvel[0:3] = -self.data.qvel[8:5:-1].copy()
        mujoco.mj_forward(self.model_fixed, self.data_fixed)
        desnseM_nominal = np.zeros((self.model_fixed.nv, self.model_fixed.nv))
        mujoco.mj_fullM(self.model_fixed, desnseM_nominal, self.data_fixed.qM)
        C = self.data_fixed.qfrc_bias.copy()
        des_vel_nominal = des_vel.copy()
        des_vel_nominal[0:3] = -des_vel[2::-1].copy()
        target_acc = self._pid_velocity_control(des_vel_nominal, self.data_fixed.qvel, self.data_fixed.qacc, self.cfg['kp'], self.cfg['kd'])
        torque_nominal = np.dot(desnseM_nominal, target_acc) + C
        torque = torque_nominal.copy()
        torque[0:3] = -torque_nominal[2::-1].copy()
        self.data.ctrl = torque

    def step(self, action): # action is 6D for arm joints
        des_vel_joints = np.clip(action, -1, 1) * self.cfg["max_vel"] # desired joint velocities
        self._apply_pd(des_vel_joints)

        if self.step_cnt >= self.release_timestep:
            if hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0:
                 self.data.eq_active[0] = 0 # Deactivate weld / equality constraint
            else:
                 # print("Warning: eq_active not available or empty.")
                 pass

        mujoco.mj_step(self.model, self.data)
        self.step_cnt += 1
        
        raw_obs_next = self._raw_obs()
        _, current_com_vel_3d = self._calculate_com() # For info
        
        term = False # Termination condition (e.g., task success, robot failure)
        trunc = self.step_cnt >= self.horizon
        
        reward = 0.0 
        
        info = {
            'com_vel_3d': current_com_vel_3d.copy(), # Full 3D COM velocity
            'step_count_in_episode': self.step_cnt
        }
        return raw_obs_next, reward, term, trunc, info



    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_cnt = 0
        mujoco.mj_resetData(self.model, self.data)
        rng = self.np_random

        # Initialize qpos with some randomness for the 6 arm joints
        initial_joint_qpos = rng.uniform(-0.5, 0.5, 6) # For 6DoF arm
        self._initialize_qpos(initial_joint_qpos)

        mujoco.mj_forward(self.model, self.data)

        self.current_episode_goal_angle = np.array([self.original_target_com_angle]) # Reset to original goal angle
        
        return self._raw_obs(), {"current_goal": self.current_episode_goal_angle.copy()}

    def _calculate_com(self): # Calculates 3D COM position and velocity
        com_pos = np.zeros(3)
        com_vel = np.zeros(3)
        total_mass = 0.0
        for i in range(1, self.model.nbody): # Iterate up to nbody
            body_name_c = self.model.body(i).name
            if not body_name_c: continue # Skip if no name

            body_id_mj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name_c)
            if body_id_mj == -1 : continue

            body_mass = self.model.body_mass[body_id_mj]
            if body_mass <= 1e-6: continue # Skip massless bodies

            # data.xipos gives CoM position of the body in global frame
            # jacobian_vel calculates CoM velocity of the body in global frame
            current_body_com_pos = self.data.xipos[body_id_mj]
            current_body_com_vel = jacobian_vel(self.model, self.data, body_id_mj)

            com_pos += body_mass * current_body_com_pos
            com_vel += body_mass * current_body_com_vel
            total_mass += body_mass
            
        if total_mass > 1e-6:
            com_pos /= total_mass
            com_vel /= total_mass
        else:
            com_pos = np.zeros(3) # Default if total mass is negligible
            com_vel = np.zeros(3)
        return com_pos, com_vel

    def compute_reward(self, com_vel_at_eval_step_3d, desired_angle_goal_scalar, eval_step_count):
        reward = 0.0
        self.similarity = 0.0 # Reset similarity

        if eval_step_count == self.goal_timestamp_steps:
            # Calculate achieved XY angle from 3D COM velocity
            achieved_com_vel_xy_norm = np.linalg.norm(com_vel_at_eval_step_3d[:2])
            
            if achieved_com_vel_xy_norm < 1e-3: # If no significant movement in XY plane
                reward = -0.5 
                self.similarity = -1.0 # Indicate no/negligible movement
            else:
                achieved_angle = np.arctan2(com_vel_at_eval_step_3d[1], com_vel_at_eval_step_3d[0])
                
                angle_diff = wrap_angle(desired_angle_goal_scalar - achieved_angle)
                self.similarity = np.cos(angle_diff) # Similarity: 1 for perfect alignment, -1 for opposite
                
                if self.similarity > 0.95: # Corresponds to ~18 degrees error
                    reward = 5.0 + self.similarity # Max reward around 2.0
                elif self.similarity < 0:
                    reward = -1 + self.similarity
                else:
                    reward = -0.1 + self.similarity # Penalize larger deviations
        return reward

    def _initialize_qpos(self, qpos_arm_joints): # qpos_arm_joints is 6D for the arm
        weld_quat = np.array([1, 0, 0, 0])
        weld_pos = np.array([1.0, 1.0, 1.0])
        # find transpose from based fixed model (not weld, base fixed)
        model_fixed = mujoco.MjModel.from_xml_path(self.cfg['model_path_fixed'])
        data_fixed = mujoco.MjData(model_fixed)

        data_fixed.qpos = qpos_arm_joints
        data_fixed.qvel = np.zeros(6)
        data_fixed.qacc = np.array([0, 0, 0, 0, 0, 0])
        mujoco.mj_forward(model_fixed, data_fixed)

        site_id = model_fixed.site("end_effector").id
        body_id = model_fixed.body("arm1_ee").id
        site_xquat = data_fixed.xquat[body_id]
        site_xpos = data_fixed.site_xpos[site_id]
        site_xmat = data_fixed.site_xmat[site_id]

        # initialize weld model
        self.data.qpos[7:13] = qpos_arm_joints
        quat_relative = get_relative_rotation_quaternion_manual(site_xquat, weld_quat)
        self.data.qpos[3:7] = quaternion_multiply(quat_relative, self.data.qpos[3:7])
        self.data.qpos[0:3] = weld_pos - self._rotate_vector_by_quaternion(site_xpos, quat_relative)
        self.data.qvel = np.zeros(12)



    def _rotate_vector_by_quaternion(self, vector, quat_rotation_wxyz): 
        quat_xyzw = quat_rotation_wxyz[[1,2,3,0]]
        r = R.from_quat(quat_xyzw)
        return r.apply(vector)

"""# 4. ReplayBuffer"""

class ReplayBuffer:
    def __init__(self, size, raw_obs_dim, act_dim, goal_dim):
        self.size = size
        self.ptr  = 0
        self.full = False
        self.raw_obs  = np.zeros((size, raw_obs_dim), np.float32)
        self.act  = np.zeros((size, act_dim), np.float32)
        self.rew  = np.zeros((size, 1), np.float32)
        self.raw_nobs = np.zeros((size, raw_obs_dim), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.goal = np.zeros((size, goal_dim), np.float32) # goal_dim is now 1

    def add(self, ro, a, r, rno, d, g): # g is now a 1D goal (angle)
        self.raw_obs[self.ptr]  = ro
        self.act[self.ptr]  = a
        self.rew[self.ptr]  = r
        self.raw_nobs[self.ptr] = rno
        self.done[self.ptr] = d
        self.goal[self.ptr] = g # Ensure g is stored correctly as [angle]
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size_arg):
        max_i = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_i, size=batch_size_arg)
        return (torch.as_tensor(self.raw_obs[idx]).to(device),
                torch.as_tensor(self.act[idx]).to(device),
                torch.as_tensor(self.rew[idx]).to(device),
                torch.as_tensor(self.raw_nobs[idx]).to(device),
                torch.as_tensor(self.done[idx]).to(device),
                torch.as_tensor(self.goal[idx]).to(device))

"""# 5. Network, DDPG agent"""

def build_mlp(in_dim, out_dim, cfg_net):
    hidden = cfg_net["hidden"]
    ActH   = get_act_fn(cfg_net["hidden_activation"])
    ActOut = get_act_fn(cfg_net["output_activation"])

    layers, dim = [], in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), ActH()]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    if ActOut is not None:
        layers.append(ActOut())
    return nn.Sequential(*layers)

class DDPGAgent:
    def __init__(self, raw_obs_dim, act_dim, goal_dim, act_lim, cfg=CFG):
        self.gamma, self.tau = cfg["gamma"], cfg["tau"]
        self.act_lim = act_lim

        actor_input_dim = raw_obs_dim + goal_dim # goal_dim is now 1
        critic_input_dim = raw_obs_dim + act_dim + goal_dim # goal_dim is now 1

        self.actor  = build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device)
        self.critic = build_mlp(critic_input_dim, 1,    cfg["critic_net"]).to(device)
        self.targ_actor  = build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device)
        self.targ_critic = build_mlp(critic_input_dim,1,cfg["critic_net"]).to(device)

        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.targ_critic.load_state_dict(self.critic.state_dict())

        self.a_optim = torch.optim.Adam(self.actor.parameters(),  lr=cfg["actor_lr"])
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

        self.noise_std = 0.2

        self.save_dir = cfg['save_dir']
        self.actor_save_path = os.path.join(self.save_dir, cfg['actor_save_path'])
        self.critic_save_path = os.path.join(self.save_dir, cfg['critic_save_path'])
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def act(self, raw_obs, goal_angle_array, noise=True): # goal_angle_array is [angle]
        raw_obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(goal_angle_array, dtype=torch.float32, device=device).unsqueeze(0)
        obs_goal_cat = torch.cat([raw_obs_t, goal_t], dim=1)

        a = self.actor(obs_goal_cat).squeeze(0).cpu().numpy()
        if noise:
            a += np.random.normal(0, self.noise_std * self.act_lim, a.shape)
        return np.clip(a, -self.act_lim, self.act_lim)

    def update(self, replay_buffer, batch_size_arg):
        if not replay_buffer.full and replay_buffer.ptr < batch_size_arg:
            return

        ro, a, r, rno, d, g_angle_batch = replay_buffer.sample(batch_size_arg) # g is now angle batch

        o_g = torch.cat([ro, g_angle_batch], dim=1)
        no_g = torch.cat([rno, g_angle_batch], dim=1)

        with torch.no_grad():
            next_a = self.targ_actor(no_g)
            q_tar  = self.targ_critic(torch.cat([rno, next_a, g_angle_batch], dim=1))
            y = r + self.gamma * (1 - d) * q_tar

        q = self.critic(torch.cat([ro, a, g_angle_batch], dim=1))
        c_loss = nn.functional.mse_loss(q, y)
        self.c_optim.zero_grad(); c_loss.backward(); self.c_optim.step()

        a_loss = -self.critic(torch.cat([ro, self.actor(o_g), g_angle_batch], dim=1)).mean()
        self.a_optim.zero_grad(); a_loss.backward(); self.a_optim.step()

        for net, tnet in ((self.actor, self.targ_actor), (self.critic, self.targ_critic)):
            for p, tp in zip(net.parameters(), tnet.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save_models(self):
        torch.save(self.actor.state_dict(), self.actor_save_path)
        torch.save(self.critic.state_dict(), self.critic_save_path)
        print(f"모델 저장 완료: {self.actor_save_path} 및 {self.critic_save_path}")

    def load_models(self):
        if os.path.exists(self.actor_save_path) and os.path.exists(self.critic_save_path):
            self.actor.load_state_dict(torch.load(self.actor_save_path, map_location=device))
            self.critic.load_state_dict(torch.load(self.critic_save_path, map_location=device))
            self.targ_actor.load_state_dict(self.actor.state_dict())
            self.targ_critic.load_state_dict(self.critic.state_dict())
            print("저장된 모델 로드 완료.")
        else:
            print("저장된 모델을 찾을 수 없습니다. 처음부터 학습합니다.")

"""# 6. 학습루프

"""
env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
raw_obs_dim = env.observation_space.shape[0] # Should be 16
act_dim = env.action_space.shape[0]
goal_dim = CFG["goal_dimension"] # Should be 1

agent = DDPGAgent(raw_obs_dim, act_dim, goal_dim, env.action_space.high[0], cfg=CFG)

buf = ReplayBuffer(CFG["buffer_size"], raw_obs_dim, act_dim, goal_dim)

total_trained_steps = 0
ret_window = deque(maxlen=100)
avg_returns_log = []
steps_at_avg_log = []

SUCCESS_THRESHOLD = 1.8 # Adjusted for new reward scale (max reward ~2.0 if similarity > 0.95)
success_window = deque(maxlen=100)
success_rate_log = []

print(f"--- 1단계: 초기 무작위 탐험 ({CFG['start_random']} 스텝) ---")
raw_obs, reset_info = env.reset()
current_episode_target_angle_array = reset_info["current_goal"] # [angle]

for step_count in range(CFG['start_random']):
    action = env.action_space.sample() # Random action for exploration
    next_raw_obs, _, term, trunc, info_step = env.step(action)

    # For random exploration, reward is based on the original target angle
    reward_for_transition = env.compute_reward(
        info_step['com_vel_3d'], # Pass 3D COM velocity
        current_episode_target_angle_array[0], # Pass scalar target angle
        info_step['step_count_in_episode']
    )
    done = term or trunc
    buf.add(raw_obs, action, reward_for_transition, next_raw_obs, float(done), current_episode_target_angle_array)
    raw_obs = next_raw_obs
    total_trained_steps += 1

    if done:
        raw_obs, reset_info = env.reset()
        current_episode_target_angle_array = reset_info["current_goal"]
        env.step_cnt = 0

    if (step_count + 1) % 1000 == 0:
        print(f"무작위 탐험 진행: {step_count + 1}/{CFG['start_random']} 스텝")

print(f"\n--- 2단계: 에이전트 학습 ({CFG['episode_number']} 에피소드) ---")
try:
    for episode_num in range(CFG['episode_number']):
        raw_obs, reset_info = env.reset()
        current_original_target_angle_array = reset_info["current_goal"] # [angle]
        episode_transitions_buffer = []
        episode_original_reward_sum = 0.0
        episode_len = 0
        current_similarity_at_goal_time = 0.0

        for s_idx in range(CFG["episode_length"]):
            episode_len += 1
            action = agent.act(raw_obs, current_original_target_angle_array, noise=True)
            next_raw_obs, _, term, trunc, info_step = env.step(action)

            episode_transitions_buffer.append((raw_obs, action, next_raw_obs, term, trunc, info_step))
            raw_obs = next_raw_obs
            total_trained_steps += 1
            if term or trunc:
                break
        
        achieved_com_vel_3d_at_goal_ts = None
        for _, _, _, _, _, info_s_t_plus_1_from_ep in episode_transitions_buffer:
            if info_s_t_plus_1_from_ep['step_count_in_episode'] == env.goal_timestamp_steps:
                achieved_com_vel_3d_at_goal_ts = info_s_t_plus_1_from_ep['com_vel_3d']
                break
        
        for ro_t, a_t, rno_t, term_t, trunc_t, info_s_t_plus_1 in episode_transitions_buffer:
            done_flag = float(term_t or trunc_t)
            
            reward_original = env.compute_reward(
                info_s_t_plus_1['com_vel_3d'],
                current_original_target_angle_array[0], # scalar target angle
                info_s_t_plus_1['step_count_in_episode']
            )
            buf.add(ro_t, a_t, reward_original, rno_t, done_flag, current_original_target_angle_array)

            if info_s_t_plus_1['step_count_in_episode'] == env.goal_timestamp_steps:
                episode_original_reward_sum += reward_original
                current_similarity_at_goal_time = env.similarity


        if buf.ptr > CFG["batch_size"] or (buf.full and CFG["batch_size"] <= buf.size) :
            agent.update(buf, CFG["batch_size"])

        ret_window.append(episode_original_reward_sum)
        if episode_original_reward_sum >= SUCCESS_THRESHOLD:
            success_window.append(1.0)
        else:
            success_window.append(0.0)

        if episode_num % 20 == 0: # Log every 20 episodes
            avg_ret = np.mean(ret_window) if len(ret_window) > 0 else 0.0
            current_success_rate = np.mean(success_window) if len(success_window) > 0 else 0.0
            avg_returns_log.append(avg_ret)
            success_rate_log.append(current_success_rate)
            steps_at_avg_log.append(total_trained_steps)
            print(f"에피소드 {episode_num:>5} | 길이 {episode_len:>4} | "
                  f"보상(원본) {episode_original_reward_sum:>7.2f} | 유사도 {current_similarity_at_goal_time:>6.3f} | "
                  f"최근100 평균보상 {avg_ret:>7.2f} | "
                  f"최근100 성공률 {current_success_rate:.2f} | "
                  f"총스텝 {total_trained_steps}")

        if episode_num > 0 and episode_num % 500 == 0:
            agent.save_models()
            intermediate_results_path = os.path.join(CFG["save_dir"], CFG["results_save_path"])
            np.savez(intermediate_results_path,
                     avg_returns=np.array(avg_returns_log),
                     steps_at_avg=np.array(steps_at_avg_log),
                     success_rates=np.array(success_rate_log))
            print(f"중간 결과 저장 완료: {intermediate_results_path} at episode {episode_num}")
finally:
    print("\n--- 학습 루프 종료 또는 중단. 최종 모델 및 결과 저장 시도 ---")
    if 'agent' in locals():
        agent.save_models()
    if len(avg_returns_log) > 0: # Ensure logs are not empty before saving
        final_results_path = os.path.join(CFG["save_dir"], CFG["results_save_path"])
        os.makedirs(CFG["save_dir"], exist_ok=True)
        np.savez(final_results_path,
                avg_returns=np.array(avg_returns_log),
                steps_at_avg=np.array(steps_at_avg_log),
                success_rates=np.array(success_rate_log))
        print(f"최종 결과 저장 완료: {final_results_path}")
    else:
        print("저장할 학습 결과 데이터가 없습니다.")
    print("---- 학습 과정 완전 종료 ----")

"""# Plot Reward"""
# 저장 디렉토리 생성 (없으면)
os.makedirs(CFG["save_dir"], exist_ok=True)
results_file_path = os.path.join(CFG["save_dir"], CFG["results_save_path"])

avg_returns_log_plot = np.array([])
steps_at_avg_log_plot = np.array([])
success_rate_log_plot = np.array([])

try:
    data = np.load(results_file_path)
    avg_returns_log_plot = data['avg_returns']
    steps_at_avg_log_plot = data['steps_at_avg']
    if 'success_rates' in data:
        success_rate_log_plot = data['success_rates']
    else:
        print(f"경고: '{results_file_path}' 파일에 'success_rates' 키가 없습니다.")
        if len(avg_returns_log_plot) > 0:
             success_rate_log_plot = np.clip(avg_returns_log_plot / (SUCCESS_THRESHOLD + 0.2) , 0, 1) # Crude fallback
        else:
             success_rate_log_plot = np.array([])

except FileNotFoundError:
    print(f"오류: '{results_file_path}' 에서 결과 파일을 찾을 수 없습니다. 빈 플롯이 생성될 수 있습니다.")
except Exception as e:
    print(f"결과 파일 로드 중 오류 발생: {e}. 빈 플롯이 생성될 수 있습니다.")

# 이동 평균 계산 함수
def moving_average(data, window_size):
    if len(data) < window_size or window_size <= 0:
        return data
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

window_size = 10
if len(steps_at_avg_log_plot) <= window_size :
    window_size = max(1, len(steps_at_avg_log_plot) // 5 if len(steps_at_avg_log_plot) > 0 else 1)

avg_returns_smooth = avg_returns_log_plot
steps_for_smooth_rewards = steps_at_avg_log_plot
success_rate_smooth = success_rate_log_plot
steps_for_smooth_success = steps_at_avg_log_plot

if len(avg_returns_log_plot) >= window_size and window_size > 0:
    avg_returns_smooth = moving_average(avg_returns_log_plot, window_size)
    if len(avg_returns_smooth) > 0:
        steps_for_smooth_rewards = steps_at_avg_log_plot[window_size-1 : window_size-1 + len(avg_returns_smooth)]

if len(success_rate_log_plot) >= window_size and window_size > 0:
    success_rate_smooth = moving_average(success_rate_log_plot, window_size)
    if len(success_rate_smooth) > 0:
         steps_for_smooth_success = steps_at_avg_log_plot[window_size-1 : window_size-1 + len(success_rate_smooth)]

fig, ax1 = plt.subplots(figsize=(12, 7))
color_reward = 'tab:blue'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Average Reward', color=color_reward)

if len(steps_at_avg_log_plot) > 0 and len(avg_returns_log_plot) > 0:
    ax1.plot(steps_at_avg_log_plot, avg_returns_log_plot, color=color_reward, alpha=0.25, label='Reward Average (Raw)')
if len(steps_for_smooth_rewards) == len(avg_returns_smooth) and len(avg_returns_smooth) > 0:
    ax1.plot(steps_for_smooth_rewards, avg_returns_smooth, color=color_reward, linestyle='-', label=f'Reward Average (Smoothed, w={window_size})')

ax1.tick_params(axis='y', labelcolor=color_reward)
ax1.grid(True, linestyle=':')
ax2 = ax1.twinx()
color_success = 'tab:green'
ax2.set_ylabel('Success Rate', color=color_success)

if len(steps_at_avg_log_plot) > 0 and len(success_rate_log_plot) > 0:
    ax2.plot(steps_at_avg_log_plot, success_rate_log_plot, color=color_success, alpha=0.25, label='Success rate (Raw)')
if len(steps_for_smooth_success) == len(success_rate_smooth) and len(success_rate_smooth) > 0:
    ax2.plot(steps_for_smooth_success, success_rate_smooth, color=color_success, linestyle='--', label=f'Success rate (Smoothed, w={window_size})')

ax2.tick_params(axis='y', labelcolor=color_success)
ax2.set_ylim(0, 1.05)
plt.title('Reward and Success rate', pad=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
if lines1 or lines2:
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
fig.tight_layout()
plot_save_path = os.path.join(CFG["save_dir"], "training_performance_plot_angle.png")
plt.savefig(plot_save_path)
print(f"그래프 저장 완료: {plot_save_path}")
plt.show()