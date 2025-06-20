"""
Self-Tossing Space Robot RL
TD3 → SAC 전체 통합 버전 (Python ≥ 3.9, PyTorch ≥ 2.0)

원본 TD3 코드 구조(셀 주석) 유지 + SACAgent, 하이퍼파라미터·학습 루프 교체
기존 유틸리티·환경·리플레이·시각화 함수는 그대로 포함
"""

# ==============================================================================
# Cell 0 : 공통 import, 설정(CFG)
# ==============================================================================
import os, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco
from collections import deque
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_act_fn(name: str):
    name = (name or "").lower()
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU,
            "leakyrelu": nn.LeakyReLU, None: None}.get(name, None)

CFG = {
    # 모델
    "model_path":         "mujoco_src/spacerobot_twoarm_3dof.xml",
    "model_path_fixed":   "mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml",

    # PD 제어
    "kp":    10.0 * np.array([0.3, 0.5, 0.8, 0.1, 0.05, 0.01]),
    "kd":    0.0,
    "max_vel": 0.5,

    # 목표 속도(에피소드마다 각도만 랜덤)
    "target_xy_com_vel_components": np.array([1, 1]),
    "target_velocity_magnitude":    0.2,
    "nsubstep": 10,

    # SAC
    "gamma": 0.99,
    "tau":   0.005,
    "actor_lr":  3e-4,
    "critic_lr": 3e-4,
    "alpha_lr":  3e-4,
    "auto_alpha": True,
    "target_entropy": None,

    # 버퍼·배치
    "batch_size":     256,
    "buffer_size":    500_000,
    "episode_number": 30_000,
    "episode_length": 501,
    "start_random":   20_000,

    # 관측/목표 차원
    "raw_observation_dimension": 16,
    "goal_dimension": 2,

    # 성공 판정
    "angle_release_threshold_deg": 5.0,
    "success_angle_threshold_deg": 5.0,
    "velocity_threshold": 0.5,

    # 네트워크 구조
    "actor_net":  {"hidden": [400, 300], "hidden_activation": "tanh"},
    "critic_net": {"hidden": [400, 300], "hidden_activation": "tanh"},

    # 저장
    "save_dir": "rl_results/SelfTossing_V1_10_SAC" +
                time.strftime("_%Y%m%d_%H%M%S", time.localtime()),
    "actor_save_path":      "actor_sac.pth",
    "critic_save_path":     "critic_sac.pth",
    "results_save_path":    "training_results_sac.npz",
    "normalizer_save_path": "obs_normalizer_stats_sac.npz",

    # Normalizer
    "normalizer_gamma": 1.0,
    "normalizer_beta":  0.0,
}

# ==============================================================================
# Cell 1 : 유틸리티 함수
# ==============================================================================
def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def get_relative_rotation_quaternion_manual(q_initial, q_target):
    return quaternion_multiply(quaternion_conjugate(q_initial), q_target)

def normalize_vector(v):
    n = np.linalg.norm(v)
    return v / n if n >= 1e-6 else np.zeros_like(v)

def jacobian_vel(model, data, body_id):
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    return jacp @ data.qvel

def wrap_angle(ang):  # [-π, π]
    return (ang + np.pi) % (2*np.pi) - np.pi

def huber(x, delta=0.5):
    ax = np.abs(x); quad = 0.5 * x**2; lin = delta*(ax-0.5*delta)
    return np.where(ax <= delta, quad, lin)

# ==============================================================================
# Cell 2 : 관측 정규화 클래스
# ==============================================================================
class Normalizer:
    def __init__(self, dim, clip_range=5.0, gamma=1.0, beta=0.0):
        self.n = 0
        self.mean = np.zeros(dim, np.float64)
        self.M2   = np.zeros(dim, np.float64)
        self.std  = np.ones(dim, np.float64)
        self.clip_range, self.gamma, self.beta = clip_range, gamma, beta

    def update(self, x):
        x = np.asarray(x, np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.n > 1:
            self.std = np.sqrt(self.M2 / (self.n - 1))

    def normalize(self, t):
        m = torch.as_tensor(self.mean, dtype=torch.float32, device=device)
        s = torch.as_tensor(self.std,  dtype=torch.float32, device=device)
        z = (t - m) / (s + 1e-8)
        z = torch.clamp(z, -self.clip_range, self.clip_range)
        return self.gamma * z + self.beta

    def save_stats(self, path):
        np.savez(path, n=self.n, mean=self.mean, M2=self.M2)
        print(f"Normalizer 저장: {path}")

    def load_stats(self, path):
        if os.path.exists(path):
            d = np.load(path)
            self.n, self.mean, self.M2 = d['n'], d['mean'], d['M2']
            if self.n > 1: self.std = np.sqrt(self.M2 / (self.n - 1))
            print("Normalizer 로드 완료.")

# ==============================================================================
# Cell 3 : SpaceRobotEnv
# ==============================================================================
class SpaceRobotEnv(gym.Env):
    def __init__(self, xml_path: str, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self.model_fixed = mujoco.MjModel.from_xml_path(cfg['model_path_fixed'])
        self.data_fixed  = mujoco.MjData(self.model_fixed)

        hi = np.inf * np.ones(cfg["raw_observation_dimension"], np.float32)
        self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
        self.action_space      = gym.spaces.Box(-1, 1, (6,), np.float32)

        self.horizon = cfg["episode_length"]; self.step_cnt = 0
        self.dt = self.model.opt.timestep
        self.last_action = np.zeros(self.action_space.shape[0])

        tc = cfg['target_xy_com_vel_components']
        self.original_angle = np.arctan2(tc[1], tc[0])
        self.current_goal = np.array([self.original_angle, cfg["target_velocity_magnitude"]])

    # ---------- 내부 메서드 ----------
    def _raw_obs(self):
        qpos = self.data.qpos[7:13].copy(); qvel = self.data.qvel[6:12].copy()
        _, com_vel = self._calculate_com()
        ang = np.arctan2(com_vel[1], com_vel[0])
        return np.concatenate(([self.step_cnt], qpos, qvel, com_vel[:2], [ang])).astype(np.float32)

    def _apply_pd(self, des_vel):
        cur_v = self.data.qvel[6:12].copy(); qacc = self.data.qacc[6:12].copy()
        self.data.ctrl = self.cfg["kp"]*(des_vel-cur_v) - self.cfg["kd"]*qacc

    def _initialize_qpos(self, q):
        weld_q, weld_p = np.array([1,0,0,0]), np.array([1.0,1.0,1.0])
        self.data_fixed.qpos[:] = q; self.data_fixed.qvel[:] = 0
        mujoco.mj_forward(self.model_fixed, self.data_fixed)

        site_id = mujoco.mj_name2id(self.model_fixed, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        body_id = self.model_fixed.body("arm1_ee").id
        ee_quat = self.data_fixed.xquat[body_id]; ee_pos = self.data_fixed.site_xpos[site_id]

        self.data.qpos[7:13] = q
        rel_q = get_relative_rotation_quaternion_manual(ee_quat, weld_q)
        self.data.qpos[3:7] = quaternion_multiply(rel_q, self.data.qpos[3:7])
        self.data.qpos[0:3] = weld_p - self._rot_vec(ee_pos, rel_q)
        self.data.qvel[:] = 0

    def _rot_vec(self, v, q_wxyz):
        q_xyzw = q_wxyz[[1,2,3,0]]
        return R.from_quat(q_xyzw).apply(v)

    def _calculate_com(self):
        pos = vel = np.zeros(3); m = 0
        for i in range(1, self.model.nbody-1):
            bm = self.model.body_mass[i]
            if bm < 1e-6: continue
            pos += bm * self.data.xipos[i]
            vel += bm * jacobian_vel(self.model, self.data, i)
            m += bm
        if m > 1e-6:
            pos /= m; vel /= m
        return pos, vel

    # ---------- Gym API ----------
    def step(self, action):
        des_vel = np.clip(action, -1, 1) * self.cfg["max_vel"]
        self._apply_pd(des_vel)
        _, com_vel = self._calculate_com()
        for _ in range(self.cfg['nsubstep']):
            mujoco.mj_step(self.model, self.data)
        self.step_cnt += 1
        obs_next = self._raw_obs()

        rew = self._reward(self.step_cnt, com_vel, self.current_goal,
                           action, self.last_action)
        self.last_action = action.copy()
        trunc = self.step_cnt >= self.horizon
        ang_diff = np.rad2deg(abs(wrap_angle(self.current_goal[0] -
                                             np.arctan2(com_vel[1], com_vel[0]))))
        info = dict(com_vel_3d=com_vel.copy(),
                    achieved_goal_angle=np.array([obs_next[-1]]),
                    angle_diff_deg=ang_diff)
        return obs_next, rew, False, trunc, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.step_cnt = 0
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        q0 = rng.uniform(-np.pi/18, np.pi/18, 6)
        self._initialize_qpos(q0); mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(self.action_space.shape[0])
        rand_ang = np.pi/4 + rng.uniform(-np.pi/18, np.pi/18)
        self.current_goal = np.array([rand_ang, self.cfg["target_velocity_magnitude"]])
        return self._raw_obs(), {"current_goal": self.current_goal.copy()}

    def _reward(self, t, com_vel, goal, act, last_act):
        d_ang, d_spd = goal
        cur_ang = np.arctan2(com_vel[1], com_vel[0])
        cur_spd = np.linalg.norm(com_vel[:2])
        a_err = wrap_angle(d_ang - cur_ang)
        s_err = (d_spd + 0.1 - cur_spd)*30
        w = (t / self.horizon)**2
        r_ang = -w*(a_err**2 + np.log10(1e-6+abs(a_err)))*0.1
        r_spd = -w*(s_err**2 + np.log10(1e-6+abs(s_err)))*0.1
        r_rate = -0.001 * np.mean((act-last_act)**2)
        r = r_ang + r_spd + r_rate
        if (t >= self.horizon-1 and abs(a_err) <=
            self.cfg["angle_release_threshold_deg"]*np.pi/180 and
            abs(cur_spd) >= self.cfg["target_velocity_magnitude"]):
            r += 100
        return r

# ==============================================================================
# Cell 4 : ReplayBuffer
# ==============================================================================
class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim, goal_dim):
        self.size, self.ptr, self.full = size, 0, False
        self.ro   = np.zeros((size, obs_dim),  np.float32)
        self.act  = np.zeros((size, act_dim),  np.float32)
        self.rew  = np.zeros((size, 1),        np.float32)
        self.rno  = np.zeros((size, obs_dim),  np.float32)
        self.done = np.zeros((size, 1),        np.float32)
        self.goal = np.zeros((size, goal_dim), np.float32)
    def add(self, ro, a, r, rno, d, g):
        self.ro[self.ptr], self.act[self.ptr], self.rew[self.ptr] = ro, a, r
        self.rno[self.ptr], self.done[self.ptr], self.goal[self.ptr] = rno, d, g
        self.ptr = (self.ptr+1) % self.size; self.full |= self.ptr == 0
    def sample(self, bs):
        idx = np.random.randint(0, self.size if self.full else self.ptr, size=bs)
        to_t = lambda x: torch.as_tensor(x[idx]).to(device)
        return to_t(self.ro), to_t(self.act), to_t(self.rew),\
               to_t(self.rno), to_t(self.done), to_t(self.goal)

# ==============================================================================
# Cell 5 : SACAgent
# ==============================================================================
def build_mlp(in_dim, out_dim, cfg_net):
    hidden = cfg_net["hidden"]; Act = get_act_fn(cfg_net["hidden_activation"])
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d,h), Act()]; d = h
    layers.append(nn.Linear(d,out_dim))
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, in_dim, act_dim, cfg_net):
        super().__init__()
        self.net = build_mlp(in_dim, 2*act_dim, cfg_net)
        self.log_std_min, self.log_std_max = -20, 2
        self.act_dim = act_dim
    def forward(self, x):
        mu_logstd = self.net(x)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + eps*std
        a = torch.tanh(pre)
        logp = (-0.5*((pre-mu)/std).pow(2) - log_std - 0.5*np.log(2*np.pi)).sum(-1, keepdim=True)
        logp -= (2*(np.log(2) - pre - nn.functional.softplus(-2*pre))).sum(-1, keepdim=True)
        return a, logp

class SACAgent:
    def __init__(self, obs_dim, act_dim, goal_dim, act_lim, norm, cfg=CFG):
        self.cfg, self.act_dim, self.act_lim, self.norm = cfg, act_dim, act_lim, norm
        actor_in  = obs_dim + goal_dim
        critic_in = obs_dim + act_dim + goal_dim
        self.actor = GaussianPolicy(actor_in, act_dim, cfg["actor_net"]).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        def _critic(): return build_mlp(critic_in, 1, cfg["critic_net"]).to(device)
        self.q1, self.q2, self.q1_t, self.q2_t = _critic(), _critic(), _critic(), _critic()
        self.q1_t.load_state_dict(self.q1.state_dict()); self.q2_t.load_state_dict(self.q2.state_dict())
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),
                                           lr=cfg["critic_lr"])
        if cfg["auto_alpha"]:
            self.log_alpha = torch.tensor(0.0, device=device, requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["alpha_lr"])
            self.target_entropy = cfg["target_entropy"] or -act_dim
        else:
            self.log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=False)

    @property
    def alpha(self): return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, ro, goal, deterministic=False):
        r = torch.as_tensor(ro,    dtype=torch.float32, device=device).unsqueeze(0)
        g = torch.as_tensor(goal,  dtype=torch.float32, device=device).unsqueeze(0)
        z = self.norm.normalize(r); inp = torch.cat([z,g],1)
        if deterministic:
            mu = self.actor.net(inp)[:,:self.act_dim]
            a = torch.tanh(mu)
        else:
            a,_ = self.actor(inp)
        return a.squeeze(0).cpu().numpy()

    def update(self, buf, bs):
        if not buf.full and buf.ptr < bs: return 0,0,0
        ro,a,r,rno,d,g = buf.sample(bs)
        nro = self.norm.normalize(ro); nrno = self.norm.normalize(rno)

        # 1) Critic
        with torch.no_grad():
            nxt_in = torch.cat([nrno, g],1)
            na, nlogp = self.actor(nxt_in)
            q1_t = self.q1_t(torch.cat([nrno, na, g],1))
            q2_t = self.q2_t(torch.cat([nrno, na, g],1))
            y = r + self.cfg["gamma"]*(1-d)*(torch.min(q1_t,q2_t) - self.alpha*nlogp)
        q1 = self.q1(torch.cat([nro,a,g],1))
        q2 = self.q2(torch.cat([nro,a,g],1))
        c_loss = nn.functional.mse_loss(q1,y) + nn.functional.mse_loss(q2,y)
        self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

        # 2) Actor
        in_cat = torch.cat([nro,g],1)
        pi, logp = self.actor(in_cat)
        q1_pi = self.q1(torch.cat([nro,pi,g],1))
        q2_pi = self.q2(torch.cat([nro,pi,g],1))
        a_loss = (self.alpha*logp - torch.min(q1_pi,q2_pi)).mean()
        self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

        # 3) Alpha
        if self.cfg["auto_alpha"]:
            al_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(); al_loss.backward(); self.alpha_opt.step()
        else: al_loss = torch.tensor(0.0)

        # 4) Target soft update
        for net, tnet in ((self.q1,self.q1_t),(self.q2,self.q2_t)):
            for p,tp in zip(net.parameters(), tnet.parameters()):
                tp.data.mul_(1-self.cfg["tau"]).add_(self.cfg["tau"]*p.data)
        return c_loss.item()/2, a_loss.item(), al_loss.item()

    # 저장/로드
    def save(self, d):
        torch.save(self.actor.state_dict(), os.path.join(d, CFG["actor_save_path"]))
        torch.save({'q1':self.q1.state_dict(),'q2':self.q2.state_dict()},
                   os.path.join(d, CFG["critic_save_path"]))
        if self.cfg["auto_alpha"]:
            torch.save({'log_alpha':self.log_alpha.detach()}, os.path.join(d,"alpha.pth"))
    def load(self, d):
        ap, cp = os.path.join(d, CFG["actor_save_path"]), os.path.join(d, CFG["critic_save_path"])
        if os.path.exists(ap) and os.path.exists(cp):
            self.actor.load_state_dict(torch.load(ap, map_location=device))
            c = torch.load(cp, map_location=device)
            self.q1.load_state_dict(c['q1']); self.q2.load_state_dict(c['q2'])
            self.q1_t.load_state_dict(self.q1.state_dict()); self.q2_t.load_state_dict(self.q2.state_dict())
            if self.cfg["auto_alpha"]:
                apath = os.path.join(d,"alpha.pth")
                if os.path.exists(apath): self.log_alpha.data = torch.load(apath)['log_alpha']

# ==============================================================================
# Cell 6 : 학습 루프
# ==============================================================================
def main():
    print(f"device: {device}")
    env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    goal_dim = CFG["goal_dimension"]

    os.makedirs(CFG["save_dir"], exist_ok=True)
    norm = Normalizer(obs_dim, gamma=CFG["normalizer_gamma"], beta=CFG["normalizer_beta"])
    agent = SACAgent(obs_dim, act_dim, goal_dim, env.action_space.high[0], norm)
    buf = ReplayBuffer(CFG["buffer_size"], obs_dim, act_dim, goal_dim)

    steps, ret_w, suc_w, ang_w = 0, deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    avg_r_log, step_log, suc_log = [], [], []

    plot_dir = os.path.join(CFG["save_dir"], "episode_plots"); os.makedirs(plot_dir, exist_ok=True)

    # 초기 랜덤
    obs, info = env.reset(); cur_goal = info["current_goal"]
    for s in range(CFG["start_random"]):
        act = env.action_space.sample()
        nobs, r, _, trunc, inf = env.step(act)
        done = trunc
        norm.update(obs); buf.add(obs, act, r, nobs, float(done), cur_goal)
        obs = nobs; steps += 1
        if done: obs, info = env.reset(); cur_goal = info["current_goal"]
        if (s+1) % 1000 == 0:
            print(f"Random {s+1}/{CFG['start_random']}")

    # 학습
    for ep in range(CFG["episode_number"]):
        obs, info = env.reset(); cur_goal = info["current_goal"]
        ep_ret = 0.0; ep_len = 0; last_ang = 180.0; final_vel = 0.0
        com_hist, joint_hist = [], []

        for t in range(CFG["episode_length"]):
            act = agent.act(obs, cur_goal)
            nobs, r, _, trunc, inf = env.step(act)
            done = trunc
            norm.update(obs); buf.add(obs, act, r, nobs, float(done), cur_goal)
            c_l, a_l, al_l = agent.update(buf, CFG["batch_size"])
            ep_ret += r; ep_len += 1
            last_ang = inf['angle_diff_deg']
            final_vel = np.linalg.norm(inf['com_vel_3d'][:2])
            com_hist.append(inf['com_vel_3d'][:2])
            joint_hist.append(env.data.qvel.copy())
            obs = nobs; steps += 1
            if done: break

        success = (last_ang <= CFG["success_angle_threshold_deg"] and
                   final_vel >= CFG["target_velocity_magnitude"])
        ret_w.append(ep_ret); suc_w.append(float(success)); ang_w.append(last_ang)

        if ep % 10 == 0:
            avg_r = np.mean(ret_w) if ret_w else 0
            suc_r = np.mean(suc_w) if suc_w else 0
            avg_ang = np.mean(ang_w) if ang_w else 180
            avg_r_log.append(avg_r); step_log.append(steps); suc_log.append(suc_r)
            print(f"E{ep:5d}|L{ep_len:3d}|R{ep_ret:8.1f}|Ang{last_ang:5.1f}|Vel{final_vel:.2f}"
                  f"|AvgR{avg_r:8.1f}|Suc{100*suc_r:3.0f}%")

        if ep and ep % 500 == 0:
            agent.save(CFG["save_dir"])
            norm.save_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"]))
            np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
                     avg_returns=np.array(avg_r_log),
                     steps_at_avg=np.array(step_log),
                     success_rates=np.array(suc_log))
            plot_episode_trajectories(com_hist, joint_hist, ep, plot_dir, cur_goal)

    agent.save(CFG["save_dir"])
    norm.save_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"]))
    np.savez(os.path.join(CFG["save_dir"], CFG["results_save_path"]),
             avg_returns=np.array(avg_r_log), steps_at_avg=np.array(step_log),
             success_rates=np.array(suc_log))
    plot_results()

# ==============================================================================
# Cell 7 : 결과 시각화
# ==============================================================================
def plot_episode_trajectories(com_hist, joint_hist, ep, d, goal):
    if not com_hist or not joint_hist: return
    cvs = np.asarray(com_hist)
    ang, mag = goal
    tvx, tvy = mag*np.cos(ang), mag*np.sin(ang)
    plt.figure(figsize=(8,8))
    plt.plot(cvs[:,0], cvs[:,1], 'b-', alpha=0.7, label='CoM vel')
    plt.plot(cvs[0,0], cvs[0,1], 'go', label='start'); plt.plot(cvs[-1,0], cvs[-1,1], 'ro', label='end')
    plt.quiver(0,0,tvx,tvy, scale_units='xy', angles='xy', scale=1, color='k', label='target')
    m = max(np.abs(cvs).max(), mag)*1.1
    plt.xlim(-m,m); plt.ylim(-m,m); plt.gca().set_aspect('equal')
    plt.title(f'CoM Velocity Trajectory - Episode {ep}')
    plt.xlabel('Vx'); plt.ylabel('Vy'); plt.grid(True); plt.legend()
    plt.savefig(os.path.join(d, f"episode_{ep}_com_vel.png")); plt.close()

    jv = np.rad2deg(np.asarray(joint_hist))
    plt.figure(figsize=(12,7))
    ts = np.arange(len(jv))
    for i in range(jv.shape[1]): plt.plot(ts, jv[:,i], label=f'Joint {i+1}')
    plt.title(f'Joint Velocities - Episode {ep}')
    plt.xlabel('step'); plt.ylabel('deg/s'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(d, f"episode_{ep}_joint_vel.png")); plt.close()

def plot_results():
    pf = os.path.join(CFG["save_dir"], CFG["results_save_path"])
    try:
        d = np.load(pf)
        r, s, suc = d['avg_returns'], d['steps_at_avg'], d['success_rates']
    except Exception as e:
        print(e); return
    def ma(x,w): return np.convolve(x, np.ones(w),'valid')/w if len(x)>=w else x
    w = 10; rs, sucs = ma(r,w), ma(suc,w)
    plt.figure(figsize=(12,7))
    plt.plot(s,r,alpha=0.25); plt.plot(s[w-1:],rs,label='AvgR smoothed')
    plt.ylabel('Avg Ep Reward'); plt.xlabel('steps'); plt.grid(True); plt.legend(loc='upper left')
    ax2 = plt.gca().twinx()
    ax2.plot(s,suc,alpha=0.25,color='g')
    ax2.plot(s[w-1:],sucs,'--',color='g',label='Success smoothed'); ax2.set_ylim(0,1.05)
    ax2.set_ylabel('Success rate'); plt.title('Training performance')
    plt.legend(); plt.tight_layout()
    p = os.path.join(CFG["save_dir"], "training_performance.png"); plt.savefig(p); plt.show()
    print(f"Saved plot: {p}")

# ==============================================================================
# main
# ==============================================================================
if __name__ == "__main__":
    main()
