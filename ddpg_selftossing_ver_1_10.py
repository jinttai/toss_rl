# ==============================================================================
# Cell 1 : 공통 설정, 유틸리티 함수 및 설정(CFG)
# ==============================================================================
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco
import os
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time

# PyTorch 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 유틸리티 함수 ---

def quaternion_conjugate(q):
    """ 쿼터니언의 켤레(conjugate)를 계산합니다. """
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    """ 두 쿼터니언을 곱합니다. """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def get_relative_rotation_quaternion_manual(q_initial, q_target):
    """ 두 쿼터니언 사이의 상대 회전을 나타내는 쿼터니언을 계산합니다. """
    q_initial_inv = quaternion_conjugate(q_initial)
    q_relative_transform = quaternion_multiply(q_initial_inv, q_target)
    return q_relative_transform

def get_act_fn(name: str):
    """ 문자열 이름에 해당하는 활성화 함수 객체를 반환합니다. """
    name = (name or "").lower()
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "leakyrelu": nn.LeakyReLU, None: None}.get(name, None)

def normalize_vector(v):
    """ 벡터를 정규화합니다. """
    norm = np.linalg.norm(v)
    return v / norm if norm >= 1e-6 else np.zeros_like(v)

def jacobian_vel(model, data, body_id):
    """ 특정 바디의 자코비안을 이용해 속도를 계산합니다. """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    return jacp @ data.qvel

def wrap_angle(angle):
    """ 각도를 -pi와 +pi 사이로 래핑합니다. """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def huber(x, delta=0.5):
    """일차 구간‧이차 구간이 만나는 지점이 delta인 Huber loss"""
    abs_x = np.abs(x)
    quad = 0.5 * (x ** 2)
    lin  = delta * (abs_x - 0.5 * delta)
    return np.where(abs_x <= delta, quad, lin)

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


# --- Configuration (CFG) ---
CFG = {
    # -------- mujoco --------
    "model_path": "mujoco_src/spacerobot_twoarm_3dof.xml",
    "model_path_fixed": 'mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml',

    # -------- PD 제어 --------
    "kp": 10.0 * np.array([0.3, 0.5, 0.8, 0.1, 0.05, 0.01]),#np.array([0.8, 0.5, 0.3, 0.1, 0.05, 0.01]), # PD 제어 비례 상수
    "kd": 0,
    "max_vel": 0.5, # rad/s
    ### NOTE ###: 아래 값은 이제 기본값으로만 사용되며, 실제 목표는 매 에피소드마다 무작위로 설정됩니다.
    "target_xy_com_vel_components":  np.array([1, 1]), 
    "target_velocity_magnitude": 0.2,
    "nsubstep": 10,  # MuJoCo 시뮬레이션을 위한 서브스텝 수

    # -------- RL 하이퍼파라미터 --------
    "gamma": 0.99,
    "tau":   0.005,
    "actor_lr":  3e-4,
    "critic_lr": 3e-4,
    "batch_size": 256,
    "buffer_size": 500_000,
    "episode_number": 20_000,
    "episode_length": 501,
    "start_random": 20_000,
    "raw_observation_dimension": 16,
    "goal_dimension": 2,
    "her_replay_k": 4,
    "velocity_reward_weight": 0.1,
    "angle_release_threshold_deg": 5.0,
    "success_angle_threshold_deg": 5.0,
    "max_torque": 5.0,
    "velocity_threshold": 0.5,
    "velocity_reward_weight": 50,

    # -------- TD3 및 보상 함수 파라미터 --------
    "policy_delay": 2,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
    "action_rate_penalty_weight": 0.001,
    "update_step": 2,  # 에이전트 업데이트 빈도

    # -------- 탐험 노이즈 파라미터 --------
    "noise_sigma": 0.1,
    "noise_decay_rate": 0.9999,
    "noise_min_sigma": 0.01,

    # -------- 네트워크 파라미터 --------
    "actor_net": {"hidden": [400, 300], "hidden_activation": "tanh", "output_activation": "tanh"},
    "critic_net": {"hidden": [400, 300], "hidden_activation": "tanh", "output_activation": None},

    # -------- 저장 경로 --------
    "save_dir": "rl_results/SelfTossing_V1_10_TD3"+ time.strftime("_%Y%m%d_%H%M%S", time.localtime()),
    "actor_save_path": "actor_td3_random_10.pth",
    "critic_save_path": "critic_td3_random_10.pth",
    "results_save_path": "training_results_td3_random.npz",
    "normalizer_save_path": "obs_normalizer_stats_random_9.npz",

    # -------- Normalizer Gamma/Beta --------
    "normalizer_gamma": 1.0,
    "normalizer_beta": 0.0
}

# ==============================================================================
# Cell 2 : 관측 정규화 클래스
# ==============================================================================
class Normalizer:
    """ Welford's algorithm 기반의 실행 평균/분산 계산 및 정규화 클래스 """
    def __init__(self, num_inputs, clip_range=5.0, gamma=1.0, beta=0.0):
        self.n = 0
        self.mean = np.zeros(num_inputs, dtype=np.float64)
        self.M2 = np.zeros(num_inputs, dtype=np.float64)
        self.std = np.ones(num_inputs, dtype=np.float64)
        self.clip_range = clip_range
        self.gamma = gamma
        self.beta = beta

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.n > 1:
            self.std = np.sqrt(self.M2 / (self.n - 1))

    def normalize(self, inputs):
        obs_mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        obs_std = torch.as_tensor(self.std, dtype=torch.float32, device=device)
        normalized_inputs = (inputs - obs_mean) / (obs_std + 1e-8)
        clipped_inputs = torch.clamp(normalized_inputs, -self.clip_range, self.clip_range)
        scaled_shifted_inputs = self.gamma * clipped_inputs + self.beta
        return scaled_shifted_inputs

    def save_stats(self, path):
        np.savez(path, n=self.n, mean=self.mean, M2=self.M2)
        print(f"Normalizer 통계량 저장 완료: {path}")

    def load_stats(self, path):
        if os.path.exists(path):
            data = np.load(path)
            self.n = data['n']
            self.mean = data['mean']
            self.M2 = data['M2']
            if self.n > 1:
                self.std = np.sqrt(self.M2 / (self.n - 1))
            print("저장된 Normalizer 통계량 로드 완료.")
        else:
            print("저장된 Normalizer 통계량 없음. 새로 시작.")

# ==============================================================================
# Cell 3 : ### MODIFIED ### 목표를 무작위로 설정하는 MuJoCo 환경
# ==============================================================================
class SpaceRobotEnv(gym.Env):
    """ 우주 로봇팔 제어를 위한 커스텀 Gym 환경 """
    def __init__(self, xml_path: str, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.model_fixed = mujoco.MjModel.from_xml_path(cfg['model_path_fixed'])
        self.data_fixed = mujoco.MjData(self.model_fixed)

        self.q_start = 0

        hi_raw = np.inf*np.ones(cfg["raw_observation_dimension"], np.float32)
        self.observation_space = gym.spaces.Box(-hi_raw, hi_raw, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        
        self.last_action = np.zeros(self.action_space.shape[0])

        self.horizon = self.cfg["episode_length"]
        self.step_cnt = 0
        
        self.dt = self.model.opt.timestep

        # 기본 목표 각도 설정 (이제 reset에서 무작위로 덮어쓰여짐)
        target_xy_components = cfg['target_xy_com_vel_components']
        self.original_target_com_angle = np.arctan2(target_xy_components[1], target_xy_components[0])
        self.current_episode_goal_angle = np.array([self.original_target_com_angle])
        self.current_episode_goal = np.array([self.original_target_com_angle, self.cfg["target_velocity_magnitude"]])


    def _raw_obs(self):
        qpos_joints = self.data.qpos[7:13].copy()
        qvel_joints = self.data.qvel[6:12].copy()
        _, com_vel_3d = self._calculate_com()
        com_vel_xy_angle = np.arctan2(com_vel_3d[1], com_vel_3d[0])
        obs = np.concatenate([np.array([self.step_cnt]), qpos_joints, qvel_joints, com_vel_3d[:2], np.array([com_vel_xy_angle])]).astype(np.float32)
        return obs
    
    def _apply_pd_control(self, des_velocity):
        current_joint_velocity = self.data.qvel[6:12].copy()
        qacc = self.data.qacc[6:12].copy()
        self.data.ctrl = self.cfg["kp"] * (des_velocity - current_joint_velocity) - self.cfg["kd"] * qacc 

    def step(self, action):
        des_velocity = np.clip(action, -1, 1) * self.cfg["max_vel"]
        self._apply_pd_control(des_velocity)
        _, current_com_vel_3d = self._calculate_com()
        term = False 
        for i in range(self.cfg['nsubstep']) : mujoco.mj_step(self.model, self.data)
        self.step_cnt += 1
        raw_obs_next = self._raw_obs()
        reward = self.compute_reward(self.step_cnt, current_com_vel_3d, self.current_episode_goal, action, self.last_action)
        self.last_action = action.copy() # 다음 스텝을 위해 마지막 행동 업데이트
        trunc = self.step_cnt >= self.horizon
        angle_diff_deg = np.rad2deg(np.abs(wrap_angle(self.current_episode_goal[0] - np.arctan2(current_com_vel_3d[1], current_com_vel_3d[0]))))
        info = {
            'com_vel_3d': current_com_vel_3d.copy(),
            'achieved_goal_angle': np.array([raw_obs_next[-1]]),
            'angle_diff_deg': angle_diff_deg,
        }
        return raw_obs_next, reward, term, trunc, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_cnt = 0
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        qpos_init= rng.uniform(-np.pi/18, np.pi/18, 6)
        self._initialize_qpos(qpos_init)
        mujoco.mj_forward(self.model, self.data)
        
        self.last_action = np.zeros(self.action_space.shape[0])
        random_target_angle = np.pi / 4 + rng.uniform(-np.pi/18, np.pi/18)
        self.current_episode_goal = np.array([
            random_target_angle,
            self.cfg["target_velocity_magnitude"]
        ])
        self.current_episode_goal_angle = np.array([random_target_angle])        
        raw_obs = self._raw_obs()

        return raw_obs, {"current_goal": self.current_episode_goal.copy()}

    def compute_reward(self, time_step, com_vel_3d, desired_goal, action, last_action):
        # 환경 보상
        desired_angle, desired_speed = desired_goal
        cur_angle = np.arctan2(com_vel_3d[1], com_vel_3d[0])
        cur_speed = np.linalg.norm(com_vel_3d[:2])
        angle_err = wrap_angle(desired_angle - cur_angle)
        speed_err = (desired_speed + 0.1 - cur_speed) * 30
        w_t = (time_step / self.horizon)**2
        reward_angle = -w_t * (angle_err**2 + np.log10(1e-6 + np.abs(angle_err))) * 0.1
        reward_velocity = -w_t * (speed_err**2 + np.log10(1e-6 + np.abs(speed_err))) *0.1

        # 행동 변화율에 대한 패널티
        action_rate_penalty = -self.cfg["action_rate_penalty_weight"] * np.mean((action - last_action)**2)
        reward = reward_angle + action_rate_penalty + reward_velocity
        if (time_step >= self.horizon - 1) and (np.abs(angle_err) <= self.cfg["angle_release_threshold_deg"] * np.pi / 180) and (np.abs(cur_speed) >= self.cfg["target_velocity_magnitude"]):
            reward += 100
        return reward

    def _initialize_qpos(self, qpos_arm_joints):
        """ 로봇팔의 초기 자세를 설정하고, 그에 맞게 베이스의 위치와 방향을 조정합니다. """
        weld_quat, weld_pos = np.array([1, 0, 0, 0]), np.array([1.0, 1.0, 1.0])
        
        self.data_fixed.qpos[:] = qpos_arm_joints
        self.data_fixed.qvel[:] = np.zeros_like(self.data_fixed.qvel)
        mujoco.mj_forward(self.model_fixed, self.data_fixed)

        site_id = mujoco.mj_name2id(self.model_fixed, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        body_id = self.model_fixed.body("arm1_ee").id
        site_xquat = self.data_fixed.xquat[body_id]
        site_xpos = self.data_fixed.site_xpos[site_id]

        self.data.qpos[7:13] = qpos_arm_joints
        quat_relative = get_relative_rotation_quaternion_manual(site_xquat, weld_quat)
        self.data.qpos[3:7] = quaternion_multiply(quat_relative, self.data.qpos[3:7])
        self.data.qpos[0:3] = weld_pos - self._rotate_vector_by_quaternion(site_xpos, quat_relative)
        self.data.qvel[:] = np.zeros_like(self.data.qvel)

    def _calculate_com(self):
        com_pos, com_vel, total_mass = np.zeros(3), np.zeros(3), 0.0
        for i in range(1, self.model.nbody-1):
            body_mass = self.model.body_mass[i]
            if body_mass <= 1e-6: continue
            com_pos += body_mass * self.data.xipos[i]
            com_vel += body_mass * jacobian_vel(self.model, self.data, i)
            total_mass += body_mass
        if total_mass > 1e-6:
            com_pos /= total_mass
            com_vel /= total_mass
        return com_pos, com_vel
    
    def _rotate_vector_by_quaternion(self, vector, quat_rotation_wxyz): 
        # Scipy's Rotation expects quaternion as [x, y, z, w]
        quat_xyzw = quat_rotation_wxyz[[1,2,3,0]] 
        return R.from_quat(quat_xyzw).apply(vector)

# ==============================================================================
# Cell 4 : 리플레이 버퍼
# ==============================================================================
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

# ==============================================================================
# Cell 5 : TD3 에이전트
# ==============================================================================
def build_mlp(in_dim, out_dim, cfg_net):
    hidden = cfg_net["hidden"]
    ActH = get_act_fn(cfg_net["hidden_activation"])
    ActOut = get_act_fn(cfg_net["output_activation"])
    layers, dim = [], in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), ActH()]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    if ActOut is not None: layers.append(ActOut())
    return nn.Sequential(*layers)

class TD3Agent:
    """ TD3 에이전트 (Twin-Delayed DDPG) """
    def __init__(self, raw_obs_dim, act_dim, goal_dim, act_lim, obs_normalizer, cfg=CFG):
        self.cfg = cfg
        self.act_dim = act_dim
        self.gamma, self.tau, self.act_lim = cfg["gamma"], cfg["tau"], act_lim
        self.obs_normalizer = obs_normalizer
        self.policy_delay = cfg["policy_delay"]
        self.target_noise_std = cfg["target_noise_std"]
        self.target_noise_clip = cfg["target_noise_clip"]
        self.noise_sigma = cfg["noise_sigma"]
        self.update_counter = 0

        actor_input_dim = raw_obs_dim + goal_dim
        critic_input_dim = raw_obs_dim + act_dim + goal_dim

        self.actor = torch.jit.script(build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device))
        self.targ_actor = torch.jit.script(build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device))
        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])

        self.critic1 = torch.jit.script(build_mlp(critic_input_dim, 1, cfg["critic_net"]).to(device))
        self.critic2 = torch.jit.script(build_mlp(critic_input_dim, 1, cfg["critic_net"]).to(device))
        self.targ_critic1 = torch.jit.script(build_mlp(critic_input_dim, 1, cfg["critic_net"]).to(device))
        self.targ_critic2 = torch.jit.script(build_mlp(critic_input_dim, 1, cfg["critic_net"]).to(device))
        self.targ_critic1.load_state_dict(self.critic1.state_dict())
        self.targ_critic2.load_state_dict(self.critic2.state_dict())
        self.c_optim = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=cfg["critic_lr"])

        self.save_dir = cfg['save_dir']
        self.actor_save_path = os.path.join(self.save_dir, cfg['actor_save_path'])
        self.critic_save_path = os.path.join(self.save_dir, cfg['critic_save_path'])
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def act(self, raw_obs, goal_angle_array, add_noise=True):
        raw_obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(goal_angle_array, dtype=torch.float32, device=device).unsqueeze(0)
        normalized_raw_obs_t = self.obs_normalizer.normalize(raw_obs_t)
        obs_goal_cat = torch.cat([normalized_raw_obs_t, goal_t], dim=1)
        a = self.actor(obs_goal_cat).squeeze(0).cpu().numpy()
        if add_noise:
            noise = np.random.normal(0, self.noise_sigma, size=self.act_dim)
            a += noise
        return np.clip(a, -self.act_lim, self.act_lim)
    
    def update(self, replay_buffer, batch_size_arg):
        if not replay_buffer.full and replay_buffer.ptr < batch_size_arg: return 0.0, 0.0, 0.0
        self.update_counter += 1
        ro, a, r, rno, d, g_batch = replay_buffer.sample(batch_size_arg)
        normalized_ro = self.obs_normalizer.normalize(ro)
        normalized_rno = self.obs_normalizer.normalize(rno)
        no_g = torch.cat([normalized_rno, g_batch], dim=1)
        
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise_std).clamp(-self.target_noise_clip, self.target_noise_clip)
            next_a = (self.targ_actor(no_g) + noise).clamp(-self.act_lim, self.act_lim)
            q1_tar = self.targ_critic1(torch.cat([normalized_rno, next_a, g_batch], dim=1))
            q2_tar = self.targ_critic2(torch.cat([normalized_rno, next_a, g_batch], dim=1))
            q_tar = torch.min(q1_tar, q2_tar)
            y = r + self.gamma * (1 - d) * q_tar

        q1 = self.critic1(torch.cat([normalized_ro, a, g_batch], dim=1))
        q2 = self.critic2(torch.cat([normalized_ro, a, g_batch], dim=1))
        c1_loss = nn.functional.mse_loss(q1, y)
        c2_loss = nn.functional.mse_loss(q2, y)
        c_loss = c1_loss + c2_loss
        self.c_optim.zero_grad()
        c_loss.backward()
        self.c_optim.step()
        
        a_loss = torch.tensor(0.0)
        if self.update_counter % self.policy_delay == 0:
            o_g = torch.cat([normalized_ro, g_batch], dim=1)
            a_loss = -self.critic1(torch.cat([normalized_ro, self.actor(o_g), g_batch], dim=1)).mean()
            self.a_optim.zero_grad()
            a_loss.backward()
            self.a_optim.step()
            for net, tnet in [(self.actor, self.targ_actor), (self.critic1, self.targ_critic1), (self.critic2, self.targ_critic2)]:
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return c_loss.item() / 2, a_loss.item()

    def save_models(self):
        torch.save(self.actor.state_dict(), self.actor_save_path)
        critic_state = {'critic1': self.critic1.state_dict(), 'critic2': self.critic2.state_dict()}
        torch.save(critic_state, self.critic_save_path)
        print(f"모델 저장 완료.")

    def load_models(self):
        if os.path.exists(self.actor_save_path) and os.path.exists(self.critic_save_path):
            self.actor.load_state_dict(torch.load(self.actor_save_path, map_location=device))
            critic_state = torch.load(self.critic_save_path, map_location=device)
            self.critic1.load_state_dict(critic_state['critic1'])
            self.critic2.load_state_dict(critic_state['critic2'])
            self.targ_actor.load_state_dict(self.actor.state_dict())
            self.targ_critic1.load_state_dict(self.critic1.state_dict())
            self.targ_critic2.load_state_dict(self.critic2.state_dict())
            print("저장된 모델 로드 완료.")
        else:
            print("저장된 모델 없음. 처음부터 학습.")

# ==============================================================================
# Cell 6 : ### MODIFIED ### 학습 루프
# ==============================================================================
def main():
    print(f"사용 디바이스: {device}")
    env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
    raw_obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    goal_dim = CFG["goal_dimension"]
    obs_normalizer = Normalizer(raw_obs_dim, gamma=CFG["normalizer_gamma"], beta=CFG["normalizer_beta"])
    agent = TD3Agent(raw_obs_dim, act_dim, goal_dim, env.action_space.high[0], obs_normalizer, cfg=CFG)
    buf = ReplayBuffer(CFG["buffer_size"], raw_obs_dim, act_dim, goal_dim)
    total_trained_steps = 0
    ret_window, success_window = deque(maxlen=100), deque(maxlen=100)
    avg_returns_log, steps_at_avg_log, success_rate_log = [], [], []
    final_angle_diff_log = deque(maxlen=100)
    plot_log_dir = os.path.join(CFG["save_dir"], "episode_plots")
    os.makedirs(plot_log_dir, exist_ok=True)

    print(f"\n--- 1단계: 초기 무작위 탐험 ({CFG['start_random']} 스텝) ---")
    raw_obs, reset_info = env.reset()
    current_episode_goal = reset_info["current_goal"]
    for step_count in range(CFG['start_random']):
        action = env.action_space.sample()
        next_raw_obs, reward, term, trunc, info_step = env.step(action)
        done = term or trunc
        obs_normalizer.update(raw_obs)
        buf.add(raw_obs, action, reward, next_raw_obs, float(done), current_episode_goal)
        raw_obs = next_raw_obs
        total_trained_steps += 1
        if done:
            raw_obs, reset_info = env.reset()
            current_episode_goal = reset_info["current_goal"]
        if (step_count + 1) % 1000 == 0:
            print(f"무작위 탐험: {step_count + 1}/{CFG['start_random']} 스텝. 버퍼: {buf.ptr if not buf.full else buf.size}")

    print(f"\n--- 2단계: 에이전트 학습 ({CFG['episode_number']} 에피소드) ---")
    for episode_num in range(CFG['episode_number']):
        raw_obs, reset_info = env.reset()
        current_original_target_angle_array = reset_info["current_goal"]
        agent.noise_sigma = max(agent.noise_sigma * CFG["noise_decay_rate"], CFG["noise_min_sigma"])
        episode_transitions_buffer, episode_reward_sum, episode_len = [], 0.0, 0
        last_angle_diff_deg_in_ep, final_vel = 180.0, 0.0
        com_vel_history, joint_vel_history = [], []

        for s_idx in range(CFG["episode_length"]):
            action = agent.act(raw_obs, current_original_target_angle_array, add_noise=True)
            next_raw_obs, reward, term, trunc, info_step = env.step(action)
            
            episode_reward_sum += reward
            last_angle_diff_deg_in_ep = info_step['angle_diff_deg']
            episode_transitions_buffer.append((raw_obs, action, reward, next_raw_obs, term, trunc, info_step))
            com_vel_history.append(info_step['com_vel_3d'][:2])
            joint_vel_history.append(env.data.qvel.copy())
            obs_normalizer.update(raw_obs)
            raw_obs = next_raw_obs
            total_trained_steps += 1
            episode_len += 1
            if (buf.ptr > CFG["batch_size"] or (buf.full and CFG["batch_size"] <= buf.size)) and s_idx % CFG['update_step'] == 0:
                agent.update(buf, CFG["batch_size"])
            if term or trunc:
                final_vel = np.linalg.norm(info_step['com_vel_3d'][:2])
                break
        
        # 실제 경험을 버퍼에 추가
        for ro_t, a_t, r_t, rno_t, term_t, trunc_t, _ in episode_transitions_buffer:
            buf.add(ro_t, a_t, r_t, rno_t, float(term_t or trunc_t), current_original_target_angle_array)
        
        
        is_success_episode = (last_angle_diff_deg_in_ep <= CFG["success_angle_threshold_deg"] and final_vel >= CFG["target_velocity_magnitude"])
        success_window.append(1.0 if is_success_episode else 0.0)
        ret_window.append(episode_reward_sum)
        final_angle_diff_log.append(last_angle_diff_deg_in_ep)

        if episode_num % 10 == 0:
            avg_ret = np.mean(ret_window) if ret_window else 0.0
            avg_angle = np.mean(final_angle_diff_log) if final_angle_diff_log else 180.0
            suc_rate = np.mean(success_window) if success_window else 0.0
            avg_returns_log.append(avg_ret)
            steps_at_avg_log.append(total_trained_steps)
            success_rate_log.append(suc_rate)
            print(f"E {episode_num:>5} | L {episode_len:>3} | EpR {episode_reward_sum:>7.2f} | "
                  f"FinalAng {last_angle_diff_deg_in_ep:>6.1f}° | FinalVel {final_vel:>5.2f} | AvgEpR {avg_ret:>7.2f} | "
                  f"AvgFA {avg_angle:>6.1f}° | Suc% {suc_rate*100:>3.0f} | "
                  f"Noise {agent.noise_sigma:.3f} | Steps {total_trained_steps}")

        if episode_num > 0 and episode_num % 500 == 0:
            agent.save_models()
            obs_normalizer.save_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"]))
            np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
                     avg_returns=np.array(avg_returns_log), steps_at_avg=np.array(steps_at_avg_log), success_rates=np.array(success_rate_log))
            print(f"중간 결과 저장 완료 at E {episode_num}")
            plot_episode_trajectories(com_vel_history, joint_vel_history, episode_num, plot_log_dir, current_original_target_angle_array)

    print("\n--- 학습 루프 종료. 최종 저장 ---")
    agent.save_models()
    obs_normalizer.save_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"]))
    np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
             avg_returns=np.array(avg_returns_log), steps_at_avg=np.array(steps_at_avg_log), success_rates=np.array(success_rate_log))
    print("\n---- 학습 완전 종료 ----")
    plot_results()
    
# ==============================================================================
# Cell 7 : 결과 시각화
# ==============================================================================
def plot_episode_trajectories(com_vel_history, joint_vel_history, episode_num, save_path, target_goal):
    if not com_vel_history or not joint_vel_history:
        print(f"E {episode_num}: 궤적 데이터가 비어있어 그래프를 생성할 수 없습니다.")
        return
    com_vels = np.array(com_vel_history)
    target_angle, target_mag = target_goal
    target_vx = target_mag * np.cos(target_angle)
    target_vy = target_mag * np.sin(target_angle)
    plt.figure(figsize=(8, 8))
    plt.plot(com_vels[:, 0], com_vels[:, 1], 'b-', label='CoM Velocity Trajectory', alpha=0.7)
    plt.plot(com_vels[0, 0], com_vels[0, 1], 'go', markersize=10, label='Start')
    plt.plot(com_vels[-1, 0], com_vels[-1, 1], 'ro', markersize=10, label='End')
    plt.quiver(0, 0, target_vx, target_vy, angles='xy', scale_units='xy', scale=1, color='k', label=f'Target Velocity (Mag: {target_mag:.1f})')
    max_val = max(np.abs(com_vels).max(), target_mag) * 1.1 if com_vels.size > 0 else target_mag * 1.1
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'CoM Velocity Trajectory - Episode {episode_num}')
    plt.xlabel('Velocity X (m/s)')
    plt.ylabel('Velocity Y (m/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"episode_{episode_num}_com_velocity_trajectory.png"))
    plt.close()
    joint_vels = np.rad2deg(np.array(joint_vel_history))
    timesteps = np.arange(len(joint_vels))
    plt.figure(figsize=(12, 7))
    for i in range(joint_vels.shape[1]):
        plt.plot(timesteps, joint_vels[:, i], label=f'Joint {i+1}')
    plt.title(f'Joint Velocities - Episode {episode_num}')
    plt.xlabel('Time Step')
    plt.ylabel('Joint Velocity (deg/s)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"episode_{episode_num}_joint_velocities.png"))
    plt.close()
    print(f"E {episode_num}: 궤적 그래프 저장 완료.")

def plot_results():
    print("\n--- 결과 시각화 시작 ---")
    results_file_path = os.path.join(CFG["save_dir"], CFG["results_save_path"])
    try:
        data = np.load(results_file_path)
        avg_returns_log_plot = data['avg_returns']
        steps_at_avg_log_plot = data['steps_at_avg']
        success_rate_log_plot = data.get('success_rates', np.array([]))
    except (FileNotFoundError, KeyError) as e:
        print(f"오류: '{results_file_path}' 파일 로드 실패 ({e}). 시각화를 건너뜁니다.")
        return

    def moving_average(data, window_size):
        if data is None or not data.any() or len(data) < window_size: return data
        return np.convolve(data, np.ones(window_size), 'valid') / window_size

    plot_window_size = 10
    avg_returns_smooth = moving_average(avg_returns_log_plot, plot_window_size)
    steps_for_smooth_rewards = steps_at_avg_log_plot[plot_window_size - 1:] if len(steps_at_avg_log_plot) >= plot_window_size else steps_at_avg_log_plot
    success_rate_smooth = moving_average(success_rate_log_plot, plot_window_size)
    steps_for_smooth_success = steps_at_avg_log_plot[plot_window_size - 1:] if len(steps_at_avg_log_plot) >= plot_window_size else steps_at_avg_log_plot

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color_reward = 'tab:blue'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel(f'Avg Episode Total Reward (Smoothed w={plot_window_size})', color=color_reward)
    ax1.plot(steps_at_avg_log_plot, avg_returns_log_plot, color=color_reward, alpha=0.25)
    if avg_returns_smooth is not None:
        ax1.plot(steps_for_smooth_rewards, avg_returns_smooth, color=color_reward, linestyle='-', label=f'Avg Ep Reward (Smoothed)')
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.grid(True, linestyle=':')

    ax2 = ax1.twinx()
    color_success = 'tab:green'
    ax2.set_ylabel(f'Success Rate (Smoothed w={plot_window_size})', color=color_success)
    if success_rate_log_plot is not None and success_rate_log_plot.any():
        ax2.plot(steps_at_avg_log_plot, success_rate_log_plot, color=color_success, alpha=0.25)
        if success_rate_smooth is not None:
            ax2.plot(steps_for_smooth_success, success_rate_smooth, color=color_success, linestyle='--', label=f'Success rate (Smoothed)')
    ax2.tick_params(axis='y', labelcolor=color_success)
    ax2.set_ylim(0, 1.05)

    plt.title(f'TD3+HER (Random Goal, Tuned PD): Training Performance', pad=20)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plot_path = os.path.join(CFG["save_dir"], "training_performance_plot.png")
    plt.savefig(plot_path)
    print(f"메인 학습 그래프 저장 완료: {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()