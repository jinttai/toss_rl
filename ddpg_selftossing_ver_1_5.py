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

# --- Configuration (CFG) ---
CFG = {
    # -------- mujoco --------
    "model_path": "mujoco_src/spacerobot_twoarm_3dof.xml",
    "model_path_fixed": 'mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml',

    # -------- PD 제어 --------
    "kp": 10,
    "kd": 0.1, # 요청에 따라 변경하지 않음
    "max_vel": np.deg2rad(10.0),
    "target_xy_com_vel_components":  np.array([1, 1]),
    "target_velocity_magnitude": 1.0,

    # -------- RL 하이퍼파라미터 --------
    "gamma": 0.99,
    "tau":   0.005,
    "actor_lr":  1e-3,
    "critic_lr": 1e-3,
    "batch_size": 256,
    "buffer_size": 500_000,
    "episode_number": 30_000,
    "episode_length": 302,
    "start_random": 50_000,
    "raw_observation_dimension": 16,
    "goal_dimension": 2,
    "her_replay_k": 4,
    "velocity_reward_weight": 0.1,
    "angle_release_threshold_deg": 5.0,
    "success_angle_threshold_deg": 5.0,
    "max_torque": 5.0,
    "velocity_threshold": 0.5,

    # -------- Ornstein-Uhlenbeck 노이즈 파라미터 --------
    "noise_sigma": 0.2, # 노이즈의 크기
    "noise_theta": 0.15, # 노이즈가 평균으로 회귀하는 속도
    "noise_decay_rate": 0.9995, # 노이즈 감소율 (추가)
    "noise_min_sigma": 0.01, # 노이즈 최소값 (추가)


    # -------- 네트워크 파라미터 --------
    "actor_net": {"hidden": [400, 400], "hidden_activation": "tanh", "output_activation": "tanh"},
    "critic_net": {"hidden": [400, 400], "hidden_activation": "tanh", "output_activation": None},

    # -------- 저장 경로 --------
    "save_dir": "rl_results/SelfTossing_DenseReward_HER_Velocity_ver3_OU_Noise_Normalized",
    "actor_save_path": "actor_ddpg_her_ou_norm.pth",
    "critic_save_path": "critic_ddpg_her_ou_norm.pth",
    "results_save_path": "training_results_her_ou_norm.npz",
    "normalizer_save_path": "obs_normalizer_stats.npz", # Normalizer 상태 저장 경로

    # -------- Normalizer Gamma/Beta (추가) --------
    "normalizer_gamma": 1.0, # Normalizer 출력에 곱할 값
    "normalizer_beta": 0.0 # Normalizer 출력에 더할 값
}

# ==============================================================================
# Cell 2 : 에이전트 노이즈 및 관측 정규화 클래스
# ==============================================================================
class OrnsteinUhlenbeckActionNoise:
    """ 시간적으로 상관관계가 있는 노이즈 생성을 위한 Ornstein-Uhlenbeck 프로세스 클래스 """
    def __init__(self, mu, sigma, theta, dt=1e-2, x_initial=None, size=None):
        self.theta = theta
        self.mu = mu * np.ones(size)
        self.sigma = sigma # CFG로부터 동적으로 업데이트될 수 있도록 저장
        self.dt = dt
        self.x_initial = x_initial if x_initial is not None else np.zeros_like(self.mu)
        self.x_prev = self.x_initial.copy()
        self.size = size

    def __call__(self):
        noise = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +
                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size))
        self.x_prev = noise
        return noise

    def reset(self):
        self.x_prev = self.x_initial.copy()

class Normalizer:
    """ Welford's algorithm 기반의 실행 평균/분산 계산 및 정규화 클래스 """
    def __init__(self, num_inputs, clip_range=5.0, gamma=1.0, beta=0.0): # gamma, beta 추가
        self.n = 0
        self.mean = np.zeros(num_inputs, dtype=np.float64)
        self.M2 = np.zeros(num_inputs, dtype=np.float64)
        self.std = np.ones(num_inputs, dtype=np.float64)
        self.clip_range = clip_range
        self.gamma = gamma # Gamma 변수 저장
        self.beta = beta   # Beta 변수 저장

    def update(self, x):
        """ 새로운 데이터(x)로 평균과 분산을 업데이트합니다. """
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.n > 1:
            self.std = np.sqrt(self.M2 / (self.n - 1))

    def normalize(self, inputs):
        """ 입력 데이터를 정규화 (평균 0, 표준편차 1)하고 클리핑합니다.
            그리고 gamma와 beta를 적용합니다.
        """
        obs_mean = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        obs_std = torch.as_tensor(self.std, dtype=torch.float32, device=device)
        
        # 정규화: (X - mu) / sigma
        normalized_inputs = (inputs - obs_mean) / (obs_std + 1e-8)
        
        # 클리핑
        clipped_inputs = torch.clamp(normalized_inputs, -self.clip_range, self.clip_range)
        
        # Gamma와 Beta 적용: gamma * X_norm + beta
        scaled_shifted_inputs = self.gamma * clipped_inputs + self.beta # Gamma, Beta 적용
        
        return scaled_shifted_inputs

    def save_stats(self, path):
        """ 계산된 통계량(평균, 분산 등)을 파일에 저장합니다. """
        np.savez(path, n=self.n, mean=self.mean, M2=self.M2)
        print(f"Normalizer 통계량 저장 완료: {path}")

    def load_stats(self, path):
        """ 파일에서 통계량을 불러옵니다. """
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
# Cell 3 : MuJoCo 환경 정의
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
        """ 환경의 원시(raw) 관측치를 반환합니다. """
        qpos_joints = self.data.qpos[self.q_start:].copy()
        qvel_joints = self.data.qvel[6:].copy()
        _, com_vel_3d = self._calculate_com()
        com_vel_xy_angle = np.arctan2(com_vel_3d[1], com_vel_3d[0])
        
        obs = np.concatenate([np.array([self.step_cnt]), qpos_joints, qvel_joints, com_vel_3d[:2], np.array([com_vel_xy_angle])]).astype(np.float32)
        
        return obs
    
    def _apply_pd_control(self, des_velocity):
        current_joint_velocity = self.data.qvel[6:]
        qacc = self.data.qacc[6:]
        self.data.ctrl = self.cfg["kp"] * (des_velocity - current_joint_velocity) - self.cfg["kd"] * qacc 

    def step(self, action):
        """ 환경을 한 스텝 진행시킵니다. """
        des_velocity = np.clip(action, -1, 1) * self.cfg["max_vel"]
        self._apply_pd_control(des_velocity)

        _, current_com_vel_3d = self._calculate_com()
        current_com_vel_xy_angle = np.arctan2(current_com_vel_3d[1], current_com_vel_3d[0])
        target_angle = self.current_episode_goal_angle[0]
        angle_diff_rad = wrap_angle(target_angle - current_com_vel_xy_angle)
        angle_diff_deg = np.rad2deg(np.abs(angle_diff_rad))
        velocity = np.linalg.norm(current_com_vel_3d[:2])

        released_this_step = False
        weld_active = (hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0 and self.data.eq_active[0] == 1)

        term = False 
        if weld_active:
            if self.step_cnt >= self.horizon - 1:
                if hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0:
                    self.data.eq_active[0] = 0 
                    released_this_step = True
                    term = True

        mujoco.mj_step(self.model, self.data)
        self.step_cnt += 1
        
        raw_obs_next = self._raw_obs()
        reward = self.compute_reward(self.step_cnt, current_com_vel_3d, self.current_episode_goal)

        trunc = self.step_cnt >= self.horizon

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
        if hasattr(self.data, 'eq_active') and len(self.data.eq_active) > 0:
            self.data.eq_active[0] = 1
        rng = np.random.default_rng(seed)
        initial_joint_qpos = rng.uniform(-np.pi, np.pi, 6)
        self._initialize_qpos(initial_joint_qpos)
        mujoco.mj_forward(self.model, self.data)
        
        self.current_episode_goal = np.array([
            self.original_target_com_angle,
            self.cfg["target_velocity_magnitude"]
        ])
        raw_obs = self._raw_obs()

        return raw_obs, {"current_goal": self.current_episode_goal.copy()}

    def compute_reward(self, time_step, com_vel_3d, desired_goal):
        desired_angle, desired_speed = desired_goal        # rad, m/s
        cur_angle = np.arctan2(com_vel_3d[1], com_vel_3d[0])
        cur_speed = np.linalg.norm(com_vel_3d[:2])

        angle_err = wrap_angle(desired_angle - cur_angle)  # [-π, π]
        speed_err = desired_speed - cur_speed
        delta_angle = np.deg2rad(self.cfg["angle_release_threshold_deg"])
        vel_weight = self.cfg["velocity_reward_weight"]
        delta_vel = self.cfg["velocity_threshold"]
        episode_horizon = self.horizon

        # shaping 부분 (음수값)
        reward = -huber(angle_err, delta_angle) -vel_weight * huber(speed_err, delta_vel)

        # 종단 보너스/패널티
        if time_step >= episode_horizon - 1:
            reward += -abs(angle_err) * 50

        return reward

    def _calculate_com(self):
        com_pos, com_vel, total_mass = np.zeros(3), np.zeros(3), 0.0
        for i in range(1, self.model.nbody - 1):
            body_id_mj = i
            body_mass = self.model.body_mass[body_id_mj]
            if body_mass <= 1e-6: continue
            
            com_pos += body_mass * self.data.xipos[body_id_mj]
            com_vel += body_mass * jacobian_vel(self.model, self.data, body_id_mj)
            total_mass += body_mass
            
        if total_mass > 1e-6:
            com_pos /= total_mass
            com_vel /= total_mass
        return com_pos, com_vel


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

    def _rotate_vector_by_quaternion(self, vector, quat_rotation_wxyz):
        """ 쿼터니언을 이용해 벡터를 회전시킵니다. """
        quat_xyzw = quat_rotation_wxyz[[1,2,3,0]] 
        return R.from_quat(quat_xyzw).apply(vector)

# ==============================================================================
# Cell 4 : 리플레이 버퍼
# ==============================================================================
class ReplayBuffer:
    """ DDPG 학습을 위한 경험 리플레이 버퍼 (GPU 최적화 버전) """
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
        # 이미 GPU에 있으므로 변환 없이 바로 반환 (데이터는 numpy 배열에서 Tensor로 변환)
        return (torch.as_tensor(self.raw_obs[idx]).to(device),
                torch.as_tensor(self.act[idx]).to(device),
                torch.as_tensor(self.rew[idx]).to(device),
                torch.as_tensor(self.raw_nobs[idx]).to(device),
                torch.as_tensor(self.done[idx]).to(device),
                torch.as_tensor(self.goal[idx]).to(device))

# ==============================================================================
# Cell 5 : DDPG 에이전트 및 신경망
# ==============================================================================
def build_mlp(in_dim, out_dim, cfg_net):
    """ MLP(Multi-Layer Perceptron) 신경망을 생성합니다. """
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

class DDPGAgent:
    """ DDPG 에이전트 (Actor-Critic 구조) """
    def __init__(self, raw_obs_dim, act_dim, goal_dim, act_lim, obs_normalizer, cfg=CFG): # normalizer 인자 추가
        self.cfg = cfg
        self.gamma, self.tau, self.act_lim = cfg["gamma"], cfg["tau"], act_lim
        self.obs_normalizer = obs_normalizer # normalizer 저장
        
        actor_input_dim = raw_obs_dim + goal_dim
        critic_input_dim = raw_obs_dim + act_dim + goal_dim
        
        self.actor = torch.jit.script(build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device))
        self.critic = torch.jit.script(build_mlp(critic_input_dim, 1, cfg["critic_net"]).to(device))
        
        self.targ_actor = torch.jit.script(build_mlp(actor_input_dim, act_dim, cfg["actor_net"]).to(device))
        self.targ_critic = torch.jit.script(build_mlp(critic_input_dim, 1, cfg["critic_net"]).to(device))
        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.targ_critic.load_state_dict(self.critic.state_dict())
        
        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        
        self.noise = OrnsteinUhlenbeckActionNoise(mu=0., sigma=cfg["noise_sigma"], theta=cfg["noise_theta"], size=act_dim)
        
        self.save_dir = cfg['save_dir']
        self.actor_save_path = os.path.join(self.save_dir, cfg['actor_save_path'])
        self.critic_save_path = os.path.join(self.save_dir, cfg['critic_save_path'])
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def act(self, raw_obs, goal_angle_array, add_noise=True):
        """ 주어진 관측에 대해 행동을 결정합니다. """
        raw_obs_t = torch.as_tensor(raw_obs, dtype=torch.float32, device=device).unsqueeze(0)
        goal_t = torch.as_tensor(goal_angle_array, dtype=torch.float32, device=device).unsqueeze(0)
        
        # 정규화된 관측치를 Actor의 입력으로 사용
        normalized_raw_obs_t = self.obs_normalizer.normalize(raw_obs_t) # <--- 정규화 적용
        obs_goal_cat = torch.cat([normalized_raw_obs_t, goal_t], dim=1)
        a = self.actor(obs_goal_cat).squeeze(0).cpu().numpy()
        
        if add_noise:
            a += self.noise()
        
        return np.clip(a, -self.act_lim, self.act_lim)
    
    def reset_noise(self):
        """ 행동 노이즈를 리셋합니다. """
        self.noise.reset()

    def update(self, replay_buffer, batch_size_arg):
        """ 리플레이 버퍼의 샘플을 사용하여 신경망을 업데이트합니다. """
        if not replay_buffer.full and replay_buffer.ptr < batch_size_arg: return 0.0, 0.0
        
        ro, a, r, rno, d, g_batch = replay_buffer.sample(batch_size_arg)
        
        # 정규화된 관측치를 Actor와 Critic의 입력으로 사용
        normalized_ro = self.obs_normalizer.normalize(ro)   # <--- 정규화 적용
        normalized_rno = self.obs_normalizer.normalize(rno) # <--- 정규화 적용
        
        # Actor 입력 (정규화된 관측치 + 목표)
        o_g = torch.cat([normalized_ro, g_batch], dim=1)
        # Target Critic 입력 (정규화된 다음 관측치 + 목표)
        no_g = torch.cat([normalized_rno, g_batch], dim=1)
        
        with torch.no_grad():
            next_a = self.targ_actor(no_g)
            q_tar = self.targ_critic(torch.cat([normalized_rno, next_a, g_batch], dim=1)) # <--- 정규화된 다음 관측치 사용
            y = r + self.gamma * (1 - d) * q_tar
            
        q = self.critic(torch.cat([normalized_ro, a, g_batch], dim=1)) # <--- 정규화된 현재 관측치 사용
        c_loss = nn.functional.mse_loss(q, y)
        self.c_optim.zero_grad(); c_loss.backward(); self.c_optim.step()
        
        for p in self.critic.parameters(): p.requires_grad = False
        a_loss = -self.critic(torch.cat([normalized_ro, self.actor(o_g), g_batch], dim=1)).mean() # <--- 정규화된 현재 관측치 사용
        self.a_optim.zero_grad(); a_loss.backward(); self.a_optim.step()
        for p in self.critic.parameters(): p.requires_grad = True
        
        for net, tnet in ((self.actor, self.targ_actor), (self.critic, self.targ_critic)):
            for p, tp in zip(net.parameters(), tnet.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                
        return c_loss.item(), a_loss.item()

    def save_models(self):
        """ 학습된 모델 통계량을 저장합니다. """
        torch.save(self.actor.state_dict(), self.actor_save_path)
        torch.save(self.critic.state_dict(), self.critic_save_path)
        print(f"모델 저장 완료.")

    def load_models(self):
        if os.path.exists(self.actor_save_path) and os.path.exists(self.critic_save_path):
            self.actor.load_state_dict(torch.load(self.actor_save_path, map_location=device))
            self.critic.load_state_dict(torch.load(self.critic_save_path, map_location=device))
            self.targ_actor.load_state_dict(self.actor.state_dict())
            self.targ_critic.load_state_dict(self.critic.state_dict())
            print("저장된 모델 로드 완료.")
        else:
            print("저장된 모델 없음. 처음부터 학습.")
# ==============================================================================
# Cell 6 : 학습 루프
# ==============================================================================
def main():
    # --- 시간 측정 시작 및 설정 ---
    overall_start_time = time.time()
    print(f"사용 디바이스: {device}")
    
    # 타이머 딕셔너리 초기화
    timing_stats = {
        "agent_act": 0.0,
        "env_step": 0.0,
        "standard_buffer_add": 0.0,
        "her_replay_add": 0.0,
        "agent_update": 0.0,
        "logging_saving": 0.0
    }

    env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
    raw_obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    goal_dim = CFG["goal_dimension"]

    # --- Normalizer 인스턴스 생성 (Gamma, Beta 포함) ---
    obs_normalizer = Normalizer(raw_obs_dim, 
                                gamma=CFG["normalizer_gamma"], 
                                beta=CFG["normalizer_beta"])
    # obs_normalizer.load_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"])) # 이전 통계 로드 (선택 사항)

    agent = DDPGAgent(raw_obs_dim, act_dim, goal_dim, env.action_space.high[0], obs_normalizer, cfg=CFG)
    # agent.load_models() # 이전 학습을 이어가려면 이 줄의 주석을 해제하세요.

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

        obs_normalizer.update(raw_obs) # <--- 무작위 탐험 중 관측치 업데이트
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
        agent.reset_noise()
        # --- 노이즈 Decay 적용 ---
        agent.noise.sigma = max(agent.noise.sigma * CFG["noise_decay_rate"], CFG["noise_min_sigma"]) # <--- 노이즈 decay
        current_original_target_angle_array = reset_info["current_goal"]
        
        episode_transitions_buffer, episode_reward_sum, episode_len = [], 0.0, 0
        last_angle_diff_deg_in_ep, final_vel = 180.0, 0.0

        for s_idx in range(CFG["episode_length"]):
            episode_len += 1
            action = agent.act(raw_obs, current_original_target_angle_array, add_noise=True)
            next_raw_obs, reward, term, trunc, info_step = env.step(action)
            episode_reward_sum += reward
            last_angle_diff_deg_in_ep = info_step['angle_diff_deg']
            episode_transitions_buffer.append((raw_obs, action, reward, next_raw_obs, term, trunc, info_step))
            
            obs_normalizer.update(raw_obs) # <--- 학습 중 관측치 업데이트
            raw_obs = next_raw_obs
            total_trained_steps += 1

            if buf.ptr > CFG["batch_size"] or (buf.full and CFG["batch_size"] <= buf.size):
                agent.update(buf, CFG["batch_size"])
            
            if term or trunc:
                final_vel = np.linalg.norm(info_step['com_vel_3d'][:2])
                break
        
        for ro_t, a_t, r_t, rno_t, term_t, trunc_t, info_s_t_plus_1 in episode_transitions_buffer:
            buf.add(ro_t, a_t, r_t, rno_t, float(term_t or trunc_t), current_original_target_angle_array)
        
        # HER (Hindsight Experience Replay) 적용
        for t_idx, (ro_t, a_t, _, rno_t, term_t, trunc_t, info_s_t_plus_1) in enumerate(episode_transitions_buffer):
            future_info = episode_transitions_buffer[-1][-1] # 에피소드 마지막 스텝의 info 사용
            her_new_desired_goal = np.array([future_info['achieved_goal_angle'][0], np.linalg.norm(future_info['com_vel_3d'][:2])])
            current_com_vel = info_s_t_plus_1['com_vel_3d'] # 현재 transition의 다음 상태에서의 속도
            reward_her = env.compute_reward(t_idx, current_com_vel, her_new_desired_goal)
            buf.add(ro_t, a_t, reward_her, rno_t, float(term_t or trunc_t), her_new_desired_goal)
        
        is_success_episode = (last_angle_diff_deg_in_ep <= CFG["success_angle_threshold_deg"]) and (final_vel >= CFG["target_velocity_magnitude"])
        success_window.append(1.0 if is_success_episode else 0.0)
        ret_window.append(episode_reward_sum)
        final_angle_diff_log.append(last_angle_diff_deg_in_ep)

        if episode_num % 100 == 0:
            avg_total_ret_val = np.mean(ret_window) if ret_window else 0.0
            avg_final_angle_val = np.mean(final_angle_diff_log) if final_angle_diff_log else 180.0
            current_success_rate_val = np.mean(success_window) if success_window else 0.0
            
            avg_returns_log.append(avg_total_ret_val)
            success_rate_log.append(current_success_rate_val)
            steps_at_avg_log.append(total_trained_steps)
            
            print(f"E {episode_num:>5} | L {episode_len:>3} | EpR {episode_reward_sum:>7.2f} | "
                  f"FinalAng {last_angle_diff_deg_in_ep:>6.1f}° | FinalVel {final_vel:>5.2f} | AvgEpR {avg_total_ret_val:>7.2f} | "
                  f"AvgFA {avg_final_angle_val:>6.1f}° | Suc% {current_success_rate_val*100:>3.0f} | "
                  f"NoiseSigma {agent.noise.sigma:.3f} | Steps {total_trained_steps}")

        if episode_num > 0 and episode_num % 500 == 0:
            agent.save_models()
            obs_normalizer.save_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"])) # <--- Normalizer 상태 저장
            np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
                     avg_returns=np.array(avg_returns_log), 
                     steps_at_avg=np.array(steps_at_avg_log), 
                     success_rates=np.array(success_rate_log))
            print(f"중간 결과 저장 완료 at E {episode_num}")


    print("\n--- 학습 루프 종료. 최종 저장 ---")
    if 'agent' in locals(): agent.save_models()
    if 'obs_normalizer' in locals(): obs_normalizer.save_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"])) # <--- 최종 Normalizer 상태 저장
    if avg_returns_log:
        np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
                 avg_returns=np.array(avg_returns_log), 
                 steps_at_avg=np.array(steps_at_avg_log), 
                 success_rates=np.array(success_rate_log))
        print(f"최종 결과 저장 완료.")
    else:
        print("저장할 결과 없음.")

    print("\n---- 학습 완전 종료 ----")
    
    # 학습 종료 후 결과 시각화
    plot_results()
# ==============================================================================
# Cell 7 : 결과 시각화
# ==============================================================================
def plot_results():
    """ 학습 결과를 불러와 그래프로 시각화합니다. """
    print("\n--- 결과 시각화 시작 ---")
    results_file_path = os.path.join(CFG["save_dir"], CFG["results_save_path"])
    
    try:
        data = np.load(results_file_path)
        avg_returns_log_plot = data['avg_returns']
        steps_at_avg_log_plot = data['steps_at_avg']
        success_rate_log_plot = data.get('success_rates', np.array([]))
    except FileNotFoundError:
        print(f"오류: '{results_file_path}' 파일 로드 실패. 시각화를 건너뜁니다.")
        return
    except Exception as e:
        print(f"결과 파일 로드 중 오류: {e}. 시각화를 건너뜁니다.")
        return

    def moving_average(data, window_size):
        if not data.any() or len(data) < window_size: return data
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
    ax1.plot(steps_for_smooth_rewards, avg_returns_smooth, color=color_reward, linestyle='-', label=f'Avg Ep Reward (Smoothed)')
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.grid(True, linestyle=':')

    ax2 = ax1.twinx()
    color_success = 'tab:green'
    ax2.set_ylabel(f'Success Rate (Smoothed w={plot_window_size})', color=color_success)
    if success_rate_log_plot.any():
        ax2.plot(steps_at_avg_log_plot, success_rate_log_plot, color=color_success, alpha=0.25)
        ax2.plot(steps_for_smooth_success, success_rate_smooth, color=color_success, linestyle='--', label=f'Success rate (Smoothed)')
    ax2.tick_params(axis='y', labelcolor=color_success)
    ax2.set_ylim(0, 1.05)

    plt.title(f'DDPG+HER (OU-Noise, Normalized Obs): Training Performance', pad=20)
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    
    plot_path = os.path.join(CFG["save_dir"], "training_performance_plot.png")
    plt.savefig(plot_path)
    print(f"메인 학습 그래프 저장 완료: {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()