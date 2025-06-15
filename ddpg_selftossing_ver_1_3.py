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
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "leakyrelu": nn.LeakyReLU, None: None}.get(name, None)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm >= 1e-6 else np.zeros_like(v)

def jacobian_vel(model, data, body_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    return jacp @ data.qvel

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

CFG = {
    # -------- mujoco --------
    "model_path": "mujoco_src/spacerobot_twoarm_3dof.xml",
    "model_path_fixed": 'mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml', # 원래대로 복원

    # -------- PD 제어 --------
    "kp": 50,
    "kd": 0,
    "max_vel": np.deg2rad(10.0),
    "target_xy_com_vel_components":  np.array([1, 1]),

    # -------- RL 하이퍼파라미터 --------
    "gamma": 0.99,
    "tau":   0.005,
    "actor_lr":  1e-4,
    "critic_lr": 1e-4,
    "batch_size": 256,
    "buffer_size": 500_000,
    "episode_number": 30_000,
    "episode_length": 302,
    "start_random": 5_000,
    "raw_observation_dimension": 13,
    "goal_dimension": 1,
    "her_replay_k": 4,
    "velocity_reward_weight": 0.1,
    "angle_release_threshold_deg": 1.0, # 물체 놓기 결정 각도 임계값 (기존 유지)
    "success_angle_threshold_deg": 5.0, # 최종 성공 판단 각도 임계값 (새로 추가)
    "max_torque": 5.0, # 최대 토크 (기존 유지)
    "velocity_threshold": 1, # 릴리즈 성공 판단을 위한 속도 임계값 (기존 유지)

    # -------- 노이즈 파라미터 --------
    "initial_noise_std": 0.3,
    "final_noise_std": 0.05,
    "noise_decay_ratio": 0.5,

    # -------- 네트워크 파라미터 --------
    "actor_net": {"hidden": [400, 300], "hidden_activation": "tanh", "output_activation": "tanh"},
    "critic_net": {"hidden": [400, 300], "hidden_activation": "tanh", "output_activation": None},

    # 저장 경로도 원래대로 또는 새 버전에 맞게 조정 (여기서는 이전 dense reward 버전으로 유지)
    "save_dir": "rl_results/SelfTossing_DenseReward_HER_Torque_Control", # 경로명 변경하여 구분
    "actor_save_path": "actor_ddpg_her_dense_originit_v1_3.pth",
    "critic_save_path": "critic_ddpg_her_dense_originit_v1_3.pth",
    "results_save_path": "training_results_her_dense_originit_v1_3.npz"
}

# ---

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
        self.q_start = 7

        hi_raw = np.inf*np.ones(cfg["raw_observation_dimension"], np.float32)
        self.observation_space = gym.spaces.Box(-hi_raw, hi_raw, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)

        self.horizon = self.cfg["episode_length"]
        self.step_cnt = 0
        
        self.dt = self.model.opt.timestep

        target_xy_components = cfg['target_xy_com_vel_components']
        self.original_target_com_angle = np.arctan2(target_xy_components[1], target_xy_components[0])
        self.current_episode_goal_angle = np.array([self.original_target_com_angle])

    def _raw_obs(self):
        qpos_joints = self.data.qpos[self.q_start:].copy()
        qvel_joints = self.data.qvel[6:].copy()
        com_pos_3d, com_vel_3d = self._calculate_com()
        com_vel_xy_angle = np.arctan2(com_vel_3d[1], com_vel_3d[0])
        
        obs = np.concatenate([
            qpos_joints, qvel_joints, np.array([com_vel_xy_angle])
        ]).astype(np.float32)
        
        if len(obs) != self.cfg["raw_observation_dimension"]:
            print(f"Error: Obs dim mismatch. Expected {self.cfg['raw_observation_dimension']}, Got {len(obs)}")
        return obs

    def _pid_velocity_control(self, target_qvel, current_qvel, Kp, Kd):
        error_vel = target_qvel - current_qvel
        return Kp * error_vel

    def _apply_pd(self, des_vel):
        self.data.ctrl = self.cfg['kp'] * (des_vel - self.data.qvel[6:]) - self.cfg['kd'] * self.data.qacc[6:]

    def step(self, action):
        torque_input = np.clip(action, -1, 1) * self.cfg["max_torque"]
        self.data.ctrl = torque_input

        current_com_pos_3d, current_com_vel_3d = self._calculate_com()
        current_com_vel_xy_angle = np.arctan2(current_com_vel_3d[1], current_com_vel_3d[0])
        target_angle = self.current_episode_goal_angle[0]
        angle_diff_rad = wrap_angle(target_angle - current_com_vel_xy_angle)
        angle_diff_deg = np.rad2deg(np.abs(angle_diff_rad))
        velocity = np.linalg.norm(current_com_vel_3d[:2])

        released_this_step = False
        weld_active = False
        if hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0:
            weld_active = (self.data.eq_active[0] == 1)

        term = False 
        if weld_active:
            if (angle_diff_deg <= self.cfg["angle_release_threshold_deg"] and velocity > self.cfg["velocity_threshold"]) or self.step_cnt >= self.horizon -1 :
                if hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0:
                    self.data.eq_active[0] = 0 
                    released_this_step = True
                    term = True

        mujoco.mj_step(self.model, self.data)
        self.step_cnt += 1
        
        raw_obs_next = self._raw_obs()
        reward = self.compute_reward(current_com_vel_3d, target_angle)

        if (angle_diff_deg <= self.cfg["angle_release_threshold_deg"] and velocity > self.cfg["velocity_threshold"]):
            reward += 500 # 보상 추가: 릴리즈 성공 시 5점 추가
        trunc = self.step_cnt >= self.horizon
        
        info = {
            'com_vel_3d': current_com_vel_3d.copy(),
            'step_count_in_episode': self.step_cnt,
            'achieved_goal_angle': np.array([raw_obs_next[-1]]),
            'angle_diff_deg': angle_diff_deg,
            'released_this_step': released_this_step,
            'original_reward_components': {
                'angle_cos': np.cos(angle_diff_rad),
                'vel_mag_sq': np.linalg.norm(current_com_vel_3d[:2])**2
            }
        }
        return raw_obs_next, reward, term, trunc, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_cnt = 0
        mujoco.mj_resetData(self.model, self.data)
        if hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0:
            self.data.eq_active[0] = 1

        rng = self.np_random if seed is None else np.random.RandomState(seed)
        initial_joint_qpos = np.zeros(6) #rng.uniform(-0.5, 0.5, 6)
        self._initialize_qpos(initial_joint_qpos) # 원래 함수로 복원
        mujoco.mj_forward(self.model, self.data)
        
        self.current_episode_goal_angle = np.array([self.original_target_com_angle])
        raw_obs = self._raw_obs()
        return raw_obs, {"current_goal": self.current_episode_goal_angle.copy(),
                         "achieved_goal_angle": np.array([raw_obs[-1]])}

    def _calculate_com(self):
        com_pos = np.zeros(3)
        com_vel = np.zeros(3)
        total_mass = 0.0
        for i in range(1, self.model.nbody - 1):
            if not self.model.body(i).name: continue
            body_id_mj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.body(i).name)
            if body_id_mj == -1 or self.model.body_mass[body_id_mj] <= 1e-6: continue
            
            body_mass = self.model.body_mass[body_id_mj]
            com_pos += body_mass * self.data.xipos[body_id_mj]
            com_vel += body_mass * jacobian_vel(self.model, self.data, body_id_mj)
            total_mass += body_mass
            
        if total_mass > 1e-6:
            com_pos /= total_mass
            com_vel /= total_mass
        return com_pos, com_vel

    def compute_reward(self, com_vel_3d_at_this_step, desired_angle_goal_scalar):
        current_com_vel_xy_angle = np.arctan2(com_vel_3d_at_this_step[1], com_vel_3d_at_this_step[0])
        angle_diff_rad = wrap_angle(desired_angle_goal_scalar - current_com_vel_xy_angle)
        velocity_reward = (np.linalg.norm(com_vel_3d_at_this_step[:2]) - self.cfg["velocity_threshold"]) ** 2 * self.cfg["velocity_reward_weight"]
        reward =  - abs(angle_diff_rad) ** 2 + velocity_reward

        return reward

    # --- _initialize_qpos를 원래 코드로 복원 ---
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
        # Scipy's Rotation expects quaternion as [x, y, z, w]
        quat_xyzw = quat_rotation_wxyz[[1,2,3,0]] 
        return R.from_quat(quat_xyzw).apply(vector)

# --- (ReplayBuffer, build_mlp, DDPGAgent are mostly the same)

"""# 4. ReplayBuffer"""
class ReplayBuffer:
    def __init__(self, size, raw_obs_dim, act_dim, goal_dim):
        self.size, self.ptr, self.full = size, 0, False
        self.raw_obs  = np.zeros((size, raw_obs_dim), np.float32)
        self.act  = np.zeros((size, act_dim), np.float32)
        self.rew  = np.zeros((size, 1), np.float32)
        self.raw_nobs = np.zeros((size, raw_obs_dim), np.float32)
        self.done = np.zeros((size, 1), np.float32)
        self.goal = np.zeros((size, goal_dim), np.float32)

    def add(self, ro, a, r, rno, d, g):
        self.raw_obs[self.ptr], self.act[self.ptr], self.rew[self.ptr] = ro, a, r
        self.raw_nobs[self.ptr], self.done[self.ptr], self.goal[self.ptr] = rno, d, g
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size_arg):
        idx = np.random.randint(0, self.size if self.full else self.ptr, size=batch_size_arg)
        return (torch.as_tensor(self.raw_obs[idx]).to(device),
                torch.as_tensor(self.act[idx]).to(device),
                torch.as_tensor(self.rew[idx]).to(device),
                torch.as_tensor(self.raw_nobs[idx]).to(device),
                torch.as_tensor(self.done[idx]).to(device),
                torch.as_tensor(self.goal[idx]).to(device))

"""# 5. Network, DDPG agent"""
def build_mlp(in_dim, out_dim, cfg_net): # Same as before
    hidden = cfg_net["hidden"]
    ActH   = get_act_fn(cfg_net["hidden_activation"])
    ActOut = get_act_fn(cfg_net["output_activation"])
    layers, dim = [], in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), ActH()]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    if ActOut is not None: layers.append(ActOut())
    return nn.Sequential(*layers)
    
class DDPGAgent: # Same as before
    def __init__(self, raw_obs_dim, act_dim, goal_dim, act_lim, cfg=CFG):
        self.cfg = cfg
        self.gamma, self.tau = cfg["gamma"], cfg["tau"]
        self.act_lim = act_lim
        actor_input_dim = raw_obs_dim + goal_dim
        critic_input_dim = raw_obs_dim + act_dim + goal_dim
        self.actor  = build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device)
        self.critic = build_mlp(critic_input_dim, 1,    cfg["critic_net"]).to(device)
        self.targ_actor  = build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device)
        self.targ_critic = build_mlp(critic_input_dim,1,cfg["critic_net"]).to(device)
        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.targ_critic.load_state_dict(self.critic.state_dict())
        self.a_optim = torch.optim.Adam(self.actor.parameters(),  lr=cfg["actor_lr"])
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.initial_noise_std = cfg["initial_noise_std"]
        self.final_noise_std = cfg["final_noise_std"]
        self.noise_decay_steps = cfg["episode_number"] * cfg["episode_length"] * cfg["noise_decay_ratio"]
        self.current_noise_std = self.initial_noise_std
        self.save_dir = cfg['save_dir']
        self.actor_save_path = os.path.join(self.save_dir, cfg['actor_save_path'])
        self.critic_save_path = os.path.join(self.save_dir, cfg['critic_save_path'])
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def act(self, raw_obs, goal_angle_array, current_total_steps_for_noise_calc, add_noise=True):
        raw_obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(goal_angle_array, dtype=torch.float32, device=device).unsqueeze(0)
        obs_goal_cat = torch.cat([raw_obs_t, goal_t], dim=1)
        a = self.actor(obs_goal_cat).squeeze(0).cpu().numpy()
        if add_noise:
            effective_steps = max(0, current_total_steps_for_noise_calc - self.cfg["start_random"])
            if self.noise_decay_steps > 0 :
                fraction = min(1.0, effective_steps / float(self.noise_decay_steps))
                self.current_noise_std = self.initial_noise_std - fraction * (self.initial_noise_std - self.final_noise_std)
            else: self.current_noise_std = self.final_noise_std
            a += np.random.normal(0, self.current_noise_std * self.act_lim, a.shape)
        return np.clip(a, -self.act_lim, self.act_lim)

    def update(self, replay_buffer, batch_size_arg):
        if not replay_buffer.full and replay_buffer.ptr < batch_size_arg: return 0.0, 0.0 
        ro, a, r, rno, d, g_angle_batch = replay_buffer.sample(batch_size_arg)
        o_g, no_g = torch.cat([ro, g_angle_batch], dim=1), torch.cat([rno, g_angle_batch], dim=1)
        with torch.no_grad():
            next_a = self.targ_actor(no_g)
            q_tar  = self.targ_critic(torch.cat([rno, next_a, g_angle_batch], dim=1))
            y = r + self.gamma * (1 - d) * q_tar
        q = self.critic(torch.cat([ro, a, g_angle_batch], dim=1))
        c_loss = nn.functional.mse_loss(q, y)
        self.c_optim.zero_grad(); c_loss.backward(); self.c_optim.step()
        for p in self.critic.parameters(): p.requires_grad = False
        a_loss = -self.critic(torch.cat([ro, self.actor(o_g), g_angle_batch], dim=1)).mean()
        self.a_optim.zero_grad(); a_loss.backward(); self.a_optim.step()
        for p in self.critic.parameters(): p.requires_grad = True
        for net, tnet in ((self.actor, self.targ_actor), (self.critic, self.targ_critic)):
            for p, tp in zip(net.parameters(), tnet.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return c_loss.item(), a_loss.item()

    def save_models(self): # Same
        torch.save(self.actor.state_dict(), self.actor_save_path)
        torch.save(self.critic.state_dict(), self.critic_save_path)
        print(f"모델 저장 완료: {self.actor_save_path} & {self.critic_save_path}")

    def load_models(self): # Same
        if os.path.exists(self.actor_save_path) and os.path.exists(self.critic_save_path):
            self.actor.load_state_dict(torch.load(self.actor_save_path, map_location=device))
            self.critic.load_state_dict(torch.load(self.critic_save_path, map_location=device))
            self.targ_actor.load_state_dict(self.actor.state_dict())
            self.targ_critic.load_state_dict(self.critic.state_dict())
            print("저장된 모델 로드 완료.")
        else: print("저장된 모델 없음. 처음부터 학습.")
# ... (DDPGAgent 클래스 정의 후) ...
"""# 6. 학습루프"""
env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
raw_obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
goal_dim = CFG["goal_dimension"]

agent = DDPGAgent(raw_obs_dim, act_dim, goal_dim, env.action_space.high[0], cfg=CFG)
# agent.load_models()

buf = ReplayBuffer(CFG["buffer_size"], raw_obs_dim, act_dim, goal_dim)

total_trained_steps = 0
ret_window, success_window = deque(maxlen=100), deque(maxlen=100)
avg_returns_log, steps_at_avg_log, success_rate_log = [], [], []
actor_loss_log, critic_loss_log = deque(maxlen=100), deque(maxlen=100)
final_angle_diff_log = deque(maxlen=100)


plot_log_dir = os.path.join(CFG["save_dir"], "episode_plots")
os.makedirs(plot_log_dir, exist_ok=True)

print(f"--- 1단계: 초기 무작위 탐험 ({CFG['start_random']} 스텝) ---")
# ... (초기 무작위 탐험 부분은 이전과 동일) ...
raw_obs, reset_info = env.reset()
current_episode_target_angle_array = reset_info["current_goal"]

for step_count in range(CFG['start_random']):
    action = env.action_space.sample()
    next_raw_obs, reward, term, trunc, info_step = env.step(action)
    done = term or trunc
    buf.add(raw_obs, action, reward, next_raw_obs, float(done), current_episode_target_angle_array)
    raw_obs = next_raw_obs
    total_trained_steps += 1
    if done:
        raw_obs, reset_info = env.reset()
        current_episode_target_angle_array = reset_info["current_goal"]
    if (step_count + 1) % 1000 == 0:
        print(f"무작위 탐험: {step_count + 1}/{CFG['start_random']} 스텝. 버퍼: {buf.ptr if not buf.full else buf.size}")


print(f"\n--- 2단계: 에이전트 학습 ({CFG['episode_number']} 에피소드) ---")
try:
    for episode_num in range(CFG['episode_number']):
        raw_obs, reset_info = env.reset()
        current_original_target_angle_array = reset_info["current_goal"]
        
        episode_transitions_buffer = []
        episode_reward_sum = 0.0
        episode_len = 0
        
        log_qpos_joints, log_qvel_joints, log_des_qvel_joints, log_com_vel_xy = [], [], [], []
        last_angle_diff_deg_in_ep = 180.0

        for s_idx in range(CFG["episode_length"]):
            episode_len += 1
            action = agent.act(raw_obs, current_original_target_angle_array, total_trained_steps, add_noise=True)
            
            log_qpos_joints.append(raw_obs[:6].copy()) 
            log_qvel_joints.append(raw_obs[6:12].copy())
            current_des_vel_joints = np.clip(action, -1, 1) * CFG["max_vel"]
            log_des_qvel_joints.append(current_des_vel_joints.copy())

            next_raw_obs, reward, term, trunc, info_step = env.step(action)
            
            log_com_vel_xy.append(info_step['com_vel_3d'][:2].copy())

            episode_reward_sum += reward
            last_angle_diff_deg_in_ep = info_step['angle_diff_deg']
            
            # episode_transitions_buffer에는 (obs, action, original_reward, next_obs, term, trunc, info) 저장
            episode_transitions_buffer.append((raw_obs, action, reward, next_raw_obs, term, trunc, info_step))
            raw_obs = next_raw_obs
            total_trained_steps += 1

            if buf.ptr > CFG["batch_size"] or (buf.full and CFG["batch_size"] <= buf.size) :
                c_loss_val, a_loss_val = agent.update(buf, CFG["batch_size"])
                critic_loss_log.append(c_loss_val); actor_loss_log.append(a_loss_val)
            
            if term or trunc:
                final_vel = np.linalg.norm(info_step['com_vel_3d'][:2])
                break 
        
        # 1. 원본 목표에 대한 트랜지션 저장
        #    episode_transitions_buffer에 이미 원래 보상(sparse bonus 포함 가능)이 들어있음
        for ro_t, a_t, r_t, rno_t, term_t, trunc_t, info_s_t_plus_1 in episode_transitions_buffer:
            buf.add(ro_t, a_t, r_t, rno_t, float(term_t or trunc_t), current_original_target_angle_array)
        
        # --- HER 로직 다시 적용 ---
        # episode_transitions_buffer의 각 튜플은 (obs, action, original_reward, next_obs, term, trunc, info_step)
        # info_step은 'com_vel_3d', 'achieved_goal_angle' 등을 포함
        for t_idx, (ro_t, a_t, _, rno_t, term_t, trunc_t, info_s_t_plus_1) in enumerate(episode_transitions_buffer):
            # _ 는 original_reward 자리인데, HER에서는 재계산하므로 사용 안 함
            for _ in range(CFG["her_replay_k"]): # CFG["her_replay_k"]가 0이면 이 루프는 실행되지 않음
                if episode_len == 0: continue # 에피소드 길이가 0이면 건너뛰기 (이론상 발생 안 함)
                future_offset = np.random.randint(t_idx, episode_len) # 현재 t_idx부터 에피소드 끝 사이에서 랜덤 선택
                
                # future_offset에 해당하는 트랜지션의 info_step 가져오기
                # episode_transitions_buffer[future_offset]는 (obs, act, rew, nobs, term, trunc, info) 형태
                # 이 info에서 'achieved_goal_angle'을 새로운 목표로 사용
                her_new_desired_goal = episode_transitions_buffer[future_offset][6]['achieved_goal_angle']
                
                # HER 보상 재계산: env.compute_reward는 dense 부분만 계산
                # info_s_t_plus_1은 현재 t_idx 트랜지션의 결과 정보
                reward_her = env.compute_reward(info_s_t_plus_1['com_vel_3d'], her_new_desired_goal[0])
                
                buf.add(ro_t, a_t, reward_her, rno_t, float(term_t or trunc_t), her_new_desired_goal)
        # --- ----------------- ---
        
        is_success_episode = (last_angle_diff_deg_in_ep <= CFG["success_angle_threshold_deg"])
        success_window.append(1.0 if is_success_episode else 0.0)

        ret_window.append(episode_reward_sum)
        final_angle_diff_log.append(last_angle_diff_deg_in_ep)

        if episode_num % 20 == 0:
            # ... (로그 출력 부분은 이전과 동일) ...
            avg_total_ret_val = np.mean(ret_window) if ret_window else 0.0 
            avg_a_loss_val = np.mean(actor_loss_log) if actor_loss_log else 0.0
            avg_c_loss_val = np.mean(critic_loss_log) if critic_loss_log else 0.0
            avg_final_angle_val = np.mean(final_angle_diff_log) if final_angle_diff_log else 180.0
            current_success_rate_val = np.mean(success_window) if success_window else 0.0
            avg_returns_log.append(avg_total_ret_val)
            success_rate_log.append(current_success_rate_val)
            steps_at_avg_log.append(total_trained_steps)
            print(f"E {episode_num:>5} | L {episode_len:>3} | EpR {episode_reward_sum:>7.2f} | FinalAng {last_angle_diff_deg_in_ep:>6.1f}° | "
                  f"AvgEpR {avg_total_ret_val:>7.2f} | AvgFA {avg_final_angle_val:>6.1f}° | Suc% {current_success_rate_val*100:>3.0f} | "
                  f"Noise {agent.current_noise_std:.3f} | ALoss {avg_a_loss_val:.2e} | Final_vel {final_vel} | Steps {total_trained_steps}")


        if episode_num > 0 and episode_num % 1000 == 0 and episode_len > 0:
            # ... (에피소드 데이터 그래프 그리는 부분은 이전과 동일) ...
            print(f"에피소드 {episode_num}: 데이터 로깅 그래프 생성 중...")
            log_qpos_joints_arr = np.array(log_qpos_joints)
            log_qvel_joints_arr = np.array(log_qvel_joints)
            log_des_qvel_joints_arr = np.array(log_des_qvel_joints)
            log_com_vel_xy_arr = np.array(log_com_vel_xy)
            time_steps = np.arange(episode_len)
            fig, axs = plt.subplots(4, 1, figsize=(14, 15), sharex=True)
            for i in range(log_qpos_joints_arr.shape[1]): axs[0].plot(time_steps, log_qpos_joints_arr[:, i], label=f'qpos[{i}]')
            axs[0].set_ylabel('Joint Pos (rad)'); axs[0].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small'); axs[0].grid(True)
            axs[0].set_title(f'Ep {episode_num} Data')
            for i in range(log_qvel_joints_arr.shape[1]): axs[1].plot(time_steps, log_qvel_joints_arr[:, i], label=f'qvel[{i}] (act)')
            axs[1].set_ylabel('Actual Joint Vel (rad/s)'); axs[1].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small'); axs[1].grid(True)
            for i in range(log_des_qvel_joints_arr.shape[1]): axs[2].plot(time_steps, log_des_qvel_joints_arr[:, i], linestyle='--', label=f'qvel[{i}] (des)')
            axs[2].set_ylabel('Desired Joint Vel (rad/s)'); axs[2].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small'); axs[2].grid(True)
            axs[3].plot(time_steps, log_com_vel_xy_arr[:, 0], label='CoM Vel X'); axs[3].plot(time_steps, log_com_vel_xy_arr[:, 1], label='CoM Vel Y')
            axs[3].set_ylabel('CoM XY Vel (m/s)'); axs[3].set_xlabel('Time Steps'); axs[3].legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small'); axs[3].grid(True)
            plt.tight_layout(rect=[0,0,0.85,0.97]); plot_filename = os.path.join(plot_log_dir, f"ep_{episode_num}_final_angle_succ_HER.png") # 파일명에 HER 명시
            plt.savefig(plot_filename); plt.close(fig); print(f"그래프 저장: {plot_filename}")
        
        if episode_num > 0 and episode_num % 500 == 0:
            # ... (모델 및 결과 저장 부분은 이전과 동일) ...
            agent.save_models()
            np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
                     avg_returns=np.array(avg_returns_log), 
                     steps_at_avg=np.array(steps_at_avg_log), 
                     success_rates=np.array(success_rate_log))
            print(f"중간 결과 저장 완료 at E {episode_num}")

finally:
    # ... (finally 블록은 이전과 동일) ...
    print("\n--- 학습 루프 종료. 최종 저장 ---")
    if 'agent' in locals(): agent.save_models()
    if avg_returns_log:
        os.makedirs(CFG["save_dir"], exist_ok=True)
        np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
                 avg_returns=np.array(avg_returns_log), 
                 steps_at_avg=np.array(steps_at_avg_log), 
                 success_rates=np.array(success_rate_log))
        print(f"최종 결과 저장 완료.")
    else: print("저장할 결과 없음.")
    print("---- 학습 완전 종료 ----")

# --- Plotting (메인 학습 그래프, 이전과 동일) ---
# ... (메인 학습 그래프 플로팅 코드는 이전과 동일) ...
os.makedirs(CFG["save_dir"], exist_ok=True)
results_file_path = os.path.join(CFG["save_dir"], CFG["results_save_path"])
avg_returns_log_plot, steps_at_avg_log_plot, success_rate_log_plot = np.array([]), np.array([]), np.array([])
try:
    data = np.load(results_file_path)
    avg_returns_log_plot = data['avg_returns'] 
    steps_at_avg_log_plot = data['steps_at_avg']
    if 'success_rates' in data: 
        success_rate_log_plot = data['success_rates']
    else:
        print(f"경고: '{results_file_path}'에 'success_rates' 키 없음.")
except FileNotFoundError: print(f"오류: '{results_file_path}' 파일 로드 실패.")
except Exception as e: print(f"결과 파일 로드 중 오류: {e}.")

def moving_average(data, window_size): # 이전과 동일
    if not data.any() or len(data) < window_size or window_size <= 0: return data
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

plot_window_size = 20 
if len(steps_at_avg_log_plot) <= plot_window_size : # 이전과 동일
    plot_window_size = max(1, len(steps_at_avg_log_plot) // 5 if len(steps_at_avg_log_plot) > 0 else 1)

avg_returns_smooth = moving_average(avg_returns_log_plot, plot_window_size) # 이전과 동일
steps_for_smooth_rewards = steps_at_avg_log_plot[plot_window_size-1 : plot_window_size-1 + len(avg_returns_smooth)] if len(avg_returns_smooth) > 0 else np.array([])
success_rate_smooth = moving_average(success_rate_log_plot, plot_window_size) # 이전과 동일
steps_for_smooth_success = steps_at_avg_log_plot[plot_window_size-1 : plot_window_size-1 + len(success_rate_smooth)] if len(success_rate_smooth) > 0 else np.array([])

fig, ax1 = plt.subplots(figsize=(12, 7)) # 이전과 동일
color_reward = 'tab:blue'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel(f'Avg Episode Total Reward (Smoothed w={plot_window_size})', color=color_reward) 
if steps_at_avg_log_plot.any() and avg_returns_log_plot.any():
    ax1.plot(steps_at_avg_log_plot, avg_returns_log_plot, color=color_reward, alpha=0.25, label='Avg Episode Total Reward (Raw)')
if steps_for_smooth_rewards.any() and avg_returns_smooth.any():
    ax1.plot(steps_for_smooth_rewards, avg_returns_smooth, color=color_reward, linestyle='-', label=f'Avg Ep Total Reward (Smoothed, w={plot_window_size})')
ax1.tick_params(axis='y', labelcolor=color_reward); ax1.grid(True, linestyle=':')
ax2 = ax1.twinx()
color_success = 'tab:green'
ax2.set_ylabel(f'Success Rate (Final Angle <= {CFG["success_angle_threshold_deg"]} deg, Smoothed w={plot_window_size})', color=color_success)
if steps_at_avg_log_plot.any() and success_rate_log_plot.any():
    ax2.plot(steps_at_avg_log_plot, success_rate_log_plot, color=color_success, alpha=0.25, label='Success rate (Raw)')
if steps_for_smooth_success.any() and success_rate_smooth.any():
    ax2.plot(steps_for_smooth_success, success_rate_smooth, color=color_success, linestyle='--', label=f'Success rate (Smoothed, w={plot_window_size})')
ax2.tick_params(axis='y', labelcolor=color_success); ax2.set_ylim(0, 1.05)
plt.title(f'DDPG+HER: Dense Reward (Success by Final Angle $\leq$ {CFG["success_angle_threshold_deg"]}$^\circ$)', pad=20)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
if labels1 or labels2: ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
fig.tight_layout()
plt.savefig(os.path.join(CFG["save_dir"], "training_performance_plot_final_angle_succ_HER.png")) # 파일명에 HER 명시
print(f"메인 학습 그래프 저장 완료.")
plt.show()