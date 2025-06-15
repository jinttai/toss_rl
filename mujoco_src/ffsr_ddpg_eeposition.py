import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import mujoco # Ensure MuJoCo is correctly installed and configured
from collections import deque
import matplotlib.pyplot as plt
from typing import Optional # <--- MODIFIED: ADD THIS IMPORT

# 2. Configuration and Helper Functions

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_act_fn(name: Optional[str]): # <--- MODIFIED: Changed type hint
    name = (name or "").lower()
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu":  nn.ELU,
        "leakyrelu": nn.LeakyReLU,
        None:  None # This None corresponds to a string "none" effectively, or an actual None value for name
    }.get(name, None) # If name is not in dict keys, or if name is None and processed to "" then not found

CFG = {
    # -------- mujoco --------
    # IMPORTANT: Update this path to your local XML file location
    "model_path": "spacerobot_cjt_simple.xml", # e.g., "./models/spacerobot_cjt_simple.xml"
    "episode_length": 1000,

    # -------- PD 제어 --------
    "kp": np.array([100.593455, 45.00005263, 12.5932949, 3.09995538,
                    0.69789491, 0.003071458]),
    "kd": np.array([1.00654459e-01, 1.07497233e-01, 1.91707960e-01,
                    5.01610700e-03, 8.47781584e-04, 1.47819615e-05]),
    "max_vel": np.deg2rad(90.0),
    # "target" is now initialized randomly in env.reset, so not needed here globally.

    # -------- RL 하이퍼파라미터 --------
    "gamma": 0.99,
    "tau":   0.001,
    "actor_lr":  1e-3,
    "critic_lr": 1e-3,
    "batch_size": 128,
    "buffer_size": 50_000,
    "episode_number": 5_000,
    "start_random": 5_000, # Steps for initial random data collection
    "observation_dimention": 21, # 6 (qpos) + 6 (qvel) + 3 (ee_pos) + 3 (ee_vel) + 3 (target_pos)

    # -------- 네트워크 파라미터 --------
    "actor_net": {
        "hidden": [400, 300],
        "hidden_activation": "tanh",
        "output_activation": "tanh"
    },
    "critic_net": {
        "hidden": [400, 300],
        "hidden_activation": "tanh",
        "output_activation": None # This will be passed as name=None to get_act_fn
    }
}

# 3. MuJoCo Environment Definition
class SpaceRobotEnv(gym.Env):
    metadata = {"render_modes": []} # Add render_modes if you plan to use rendering

    def __init__(self, xml_path: str, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except Exception as e:
            print(f"Error loading MuJoCo model from {xml_path}: {e}")
            print("Please ensure the model_path in CFG is correct and the XML file is valid.")
            raise
        self.data  = mujoco.MjData(self.model)

        self.site_ee = self.model.site("end_effector").id
        self.body_ee_stick = self.model.body("ee_stick").id
        self.q_start = 7  # freejoint(7) 이후 6개가 팔 관절 (qpos: 0-6 base, 7-12 arm; qvel: 0-5 base, 6-11 arm)

        # Initialize np_random. It's good practice to do this in __init__ or ensure reset() is called.
        # gym.Env's np_random property will handle lazy initialization if not explicitly set.
        # self.seed() # Call this if you need to seed early, or rely on reset()
        
        # Target position will be initialized in reset()
        self.target_pos = np.zeros(3) # Placeholder

        hi = np.inf * np.ones(cfg["observation_dimention"], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-hi, hi, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

        self.sparse_reward_achieved_in_episode = False # Renamed for clarity
        self.horizon = self.cfg["episode_length"]
        self.step_cnt = 0

    def _obs(self):
        qpos_arm = self.data.qpos[self.q_start:].copy()
        qvel_arm = self.data.qvel[6:].copy() # Arm velocities start at index 6 in qvel
        # qacc_arm = self.data.qacc[6:].copy() # qacc might not be directly needed in obs for DDPG, depends on design
        
        ee_pos = self.data.site_xpos[self.site_ee].copy()
        # For ee_vel, cvel is often used. Ensure 'ee_stick' body has the correct index for its velocity.
        # cvel has shape (nbody, 6), where 3:6 are angular velocities, 0:3 are linear.
        # Typically, we need linear velocity of the end-effector.
        ee_vel_linear = self.data.cvel[self.body_ee_stick, 3:6].copy() # Linear velocity of ee_stick body

        target_pos_obs = self.target_pos.copy()
        
        # Concatenate: qpos_arm (6) + qvel_arm (6) + ee_pos (3) + ee_vel_linear (3) + target_pos_obs (3) = 21
        return np.concatenate([qpos_arm, qvel_arm, ee_pos, ee_vel_linear, target_pos_obs]).astype(np.float32)

    def _apply_pd(self, des_vel_normalized_action):
        # Action is normalized [-1, 1], scale it to desired velocity limits
        des_vel = np.clip(des_vel_normalized_action, -1, 1) * self.cfg["max_vel"]
        
        qvel_arm = self.data.qvel[6:] # Arm velocities
        # PD controller usually uses position error, not acceleration (qacc).
        # If this PD formulation is specific: Kp*(vel_desired - vel_current) - Kd*acc_current
        # However, a more standard velocity PD controller might be: Kp*(vel_desired - vel_current)
        # Or a position PD controller outputting torques: Kp*(pos_desired - pos_current) - Kd*(vel_desired - vel_current)
        # The original code uses qacc for damping, which is non-standard for typical PD velocity control, but we'll keep it as is.
        # Ensure qacc is up-to-date if used:
        mujoco.mj_inverse(self.model, self.data) # Computes accelerations if needed (though mj_step also does this)
        qacc_arm = self.data.qacc[6:].copy() # Arm accelerations
        
        torque = self.cfg["kp"] * (des_vel - qvel_arm) - self.cfg["kd"] * qacc_arm
        self.data.ctrl[:] = torque # Assuming ctrl maps directly to the 6 arm actuators

    def step(self, action):
        self._apply_pd(action) # action is the desired velocity profile (normalized)
        mujoco.mj_step(self.model, self.data)
        self.step_cnt += 1

        obs = self._obs()
        ee_pos = obs[12:15] # ee_pos is at index 12, 13, 14 in the concatenated obs
        joint_vel = obs[6:12]
        

        # -------- reward function ----------
        Kd_dist = 0.1 # Coefficient for distance penalty
        R_bonus = 1.0   # Bonus reward
        dist_threshold_bonus = 0.15 # Distance threshold for bonus

        Kd_vel = 0.005
        vel_term = np.sum(np.square(joint_vel)) * Kd_vel

        dist = np.linalg.norm(ee_pos - self.target_pos)
        
        reward = -dist * Kd_dist - vel_term # Base penalty for distance

        if dist < dist_threshold_bonus:
            reward += R_bonus

        terminated = dist < 0.05 # Goal reached
        truncated = self.step_cnt >= self.horizon
        
        info = {"dist": dist, "target_pos": self.target_pos.copy()}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed) # Essential for proper seeding of self.np_random
        self.step_cnt = 0
        self.sparse_reward_achieved_in_episode = False # Reset bonus flag

        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize initial joint positions and velocities for the arm
        # self.np_random comes from super().reset(seed=seed)
        qpos_arm_init = self.np_random.uniform(-0.5, 0.5, size=6)
        qvel_arm_init = self.np_random.uniform(-0.1, 0.1, size=6)
        
        self.data.qpos[self.q_start:] = qpos_arm_init
        self.data.qvel[6:] = qvel_arm_init # Arm velocities start at index 6 in qvel

        # Randomize target position
        self.target_pos = np.array([
            self.np_random.uniform(-0.8, 0.8),
            self.np_random.uniform(-0.8, 0.8),
            self.np_random.uniform(0.4, 0.8)
        ])
        
        mujoco.mj_forward(self.model, self.data) # Update simulation state (e.g., site positions) after setting qpos/qvel

        return self._obs(), {"target_pos": self.target_pos.copy()}

    # def seed(self, seed=None): # Deprecated in Gymnasium, use reset(seed=...)
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return [seed]

# 4. ReplayBuffer
class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.size = int(size) # Ensure size is integer
        self.ptr  = 0
        self.current_size = 0 # Tracks current number of elements
        self.obs  = np.zeros((self.size, obs_dim), np.float32)
        self.act  = np.zeros((self.size, act_dim), np.float32)
        self.rew  = np.zeros((self.size, 1), np.float32)
        self.nobs = np.zeros((self.size, obs_dim), np.float32)
        self.done = np.zeros((self.size, 1), np.float32) # Stores terminated, not truncated

    def add(self, o, a, r, no, d):
        self.obs[self.ptr]  = o
        self.act[self.ptr]  = a
        self.rew[self.ptr]  = r
        self.nobs[self.ptr] = no
        self.done[self.ptr] = d # d should be 'terminated' (boolean for goal achievement)
        self.ptr = (self.ptr + 1) % self.size
        self.current_size = min(self.current_size + 1, self.size)

    def sample(self, batch_size):
        if self.current_size == 0:
            return None # Or raise an error if buffer is empty
        idx = np.random.randint(0, self.current_size, size=batch_size)
        return (torch.as_tensor(self.obs[idx], device=device),
                torch.as_tensor(self.act[idx], device=device),
                torch.as_tensor(self.rew[idx], device=device),
                torch.as_tensor(self.nobs[idx], device=device),
                torch.as_tensor(self.done[idx], device=device))

# 5. Network and DDPG Agent
def build_mlp(in_dim, out_dim, cfg_net):
    hidden_dims = cfg_net["hidden"]
    ActH   = get_act_fn(cfg_net["hidden_activation"])
    ActOut = get_act_fn(cfg_net["output_activation"])

    layers = []
    current_dim = in_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        if ActH: layers.append(ActH())
        current_dim = h_dim
    
    layers.append(nn.Linear(current_dim, out_dim))
    if ActOut:
        layers.append(ActOut())
    return nn.Sequential(*layers)

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, act_lim_high, cfg=CFG): # act_lim_high is usually 1.0 for normalized actions
        self.gamma, self.tau = cfg["gamma"], cfg["tau"]
        self.act_dim = act_dim

        self.actor  = build_mlp(obs_dim, act_dim, cfg["actor_net"]).to(device)
        self.critic = build_mlp(obs_dim + act_dim, 1, cfg["critic_net"]).to(device)
        self.targ_actor  = build_mlp(obs_dim, act_dim, cfg["actor_net"]).to(device)
        self.targ_critic = build_mlp(obs_dim + act_dim, 1, cfg["critic_net"]).to(device)
        
        self.targ_actor.load_state_dict(self.actor.state_dict())
        self.targ_critic.load_state_dict(self.critic.state_dict())
        # Freeze target networks
        for p in self.targ_actor.parameters(): p.requires_grad = False
        for p in self.targ_critic.parameters(): p.requires_grad = False

        self.a_optim = torch.optim.Adam(self.actor.parameters(),  lr=cfg["actor_lr"])
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

        self.noise_std = 0.2 # Exploration noise standard deviation
        self.act_lim_high = act_lim_high # Max absolute value of action components

    @torch.no_grad()
    def act(self, obs, add_noise=True):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dimension
        a = self.actor(obs_t).squeeze(0).cpu().numpy() # Remove batch dimension
        if add_noise:
            noise = np.random.normal(0, self.noise_std * self.act_lim_high, size=self.act_dim)
            a += noise
        return np.clip(a, -self.act_lim_high, self.act_lim_high)

    def update(self, buf, batch_size):
        sample = buf.sample(batch_size)
        if sample is None: return # Buffer not ready or empty
        o, a, r, no, d = sample

        with torch.no_grad():
            next_a = self.targ_actor(no)
            q_target_next = self.targ_critic(torch.cat([no, next_a], dim=1))
            y = r + self.gamma * (1 - d) * q_target_next # d is 0 if not done, 1 if done

        # Critic loss
        q_current = self.critic(torch.cat([o, a], dim=1))
        c_loss = nn.functional.mse_loss(q_current, y)
        self.c_optim.zero_grad()
        c_loss.backward()
        self.c_optim.step()

        # Actor loss
        # Freeze critic parameters temporarily for actor update calculation
        for p in self.critic.parameters(): p.requires_grad = False
        
        a_pred = self.actor(o)
        q_val_for_actor = self.critic(torch.cat([o, a_pred], dim=1))
        a_loss = -q_val_for_actor.mean()
        self.a_optim.zero_grad()
        a_loss.backward()
        self.a_optim.step()

        # Unfreeze critic parameters
        for p in self.critic.parameters(): p.requires_grad = True
        
        # Soft update target networks
        with torch.no_grad():
            for net, tnet in ((self.actor, self.targ_actor), (self.critic, self.targ_critic)):
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - self.tau)
                    tp.data.add_(self.tau * p.data)

# 6. Main Training Loop
def main():
    print(f"Initializing environment with model: {CFG['model_path']}")
    env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_lim_high = env.action_space.high[0] # Assuming symmetric action space

    print(f"Observation dimension: {obs_dim}, Action dimension: {act_dim}, Action limit: {act_lim_high}")

    agent = DDPGAgent(obs_dim, act_dim, act_lim_high, cfg=CFG)
    buf = ReplayBuffer(CFG["buffer_size"], obs_dim, act_dim)

    obs, info = env.reset()
    ret_window = deque(maxlen=100)
    avg_returns_history = [] # Store average returns for plotting
    # steps_at_avg_history = [] # If you need to log steps corresponding to avg returns

    total_steps = 0

    # --- Phase 1: Collecting initial random experiences ---
    print(f"--- Phase 1: Collecting initial random experiences for {CFG['start_random']} steps ---")
    for step in range(int(CFG['start_random'])):
        action = env.action_space.sample() # Random action
        nobs, r, term, trunc, info = env.step(action)
        buf.add(obs, action, r, nobs, float(term)) # Store terminated as float
        obs = nobs
        total_steps += 1
        if term or trunc:
            obs, info = env.reset()
        if (step + 1) % 1000 == 0:
            print(f"Initial collection: Step {step+1}/{CFG['start_random']}")

    # --- Phase 2: Agent learning ---
    print(f"\n--- Phase 2: Agent learning for {CFG['episode_number']} episodes ---")
    obs, info = env.reset() # Reset env again before starting agent learning
    current_episode_return = 0.0
    current_episode_length = 0

    for episode_num in range(int(CFG["episode_number"])):
        for step_in_episode in range(int(CFG['episode_length'])):
            action = agent.act(obs, add_noise=True)
            nobs, r, term, trunc, info = env.step(action)
            
            buf.add(obs, action, r, nobs, float(term))
            obs = nobs
            current_episode_return += r
            current_episode_length += 1
            total_steps += 1

            # Update agent
            if buf.current_size >= CFG["batch_size"]: # Ensure buffer has enough samples
                 agent.update(buf, CFG["batch_size"])

            if term or trunc:
                ret_window.append(current_episode_return)
                avg_return_latest = np.mean(ret_window) if ret_window else current_episode_return
                avg_returns_history.append(avg_return_latest)
                
                print(f"Ep {episode_num+1:>5} | Len {current_episode_length:>4} | Ret {current_episode_return:>7.2f} "
                      f"| AvgRet100 {avg_return_latest:>7.2f} | Dist {info.get('dist', -1):.3f} "
                      f"| Target {info.get('target_pos', [0,0,0])}")
                
                obs, info = env.reset()
                current_episode_return = 0.0
                current_episode_length = 0
                break # End current episode

    # --- Plot Learning Curve ---
    plt.figure(figsize=(12, 6))
    plt.plot(avg_returns_history)
    plt.title("Learning Curve (Average Return over last 100 episodes)")
    plt.xlabel("Episode (after initial collection, updates every 100 actual episodes)")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve_ddpg.png") # Save the plot
    plt.show()

    # --- Save Trained Model ---
    model_save_path = "actor_ddpg.pth"
    torch.save(agent.actor.state_dict(), model_save_path)
    print(f"Actor model saved to {model_save_path}")

    # --- Evaluation ---
    print("\n--- Evaluating trained agent ---")
    eval_env = SpaceRobotEnv(CFG["model_path"], cfg=CFG) # Fresh env for eval
    distance_history = []
    eval_episodes = 3

    for ep in range(eval_episodes):
        o, info = eval_env.reset(seed=ep) # Use different seeds for eval episodes
        terminated, truncated = False, False
        eval_episode_return = 0
        eval_episode_steps = 0
        while not (terminated or truncated):
            a = agent.act(o, add_noise=False) # Evaluate with deterministic policy
            o, r, terminated, truncated, info = eval_env.step(a)
            distance_history.append(info["dist"])
            eval_episode_return += r
            eval_episode_steps +=1
            if eval_episode_steps >= CFG["episode_length"]: # Safety break if episode is too long
                truncated = True 

        print(f"Eval episode {ep+1}: Return {eval_episode_return:.2f}, Steps {eval_episode_steps}, Final Dist {info['dist']:.3f}")
    
    eval_env.close() # Good practice to close env if it has resources like renderers

    # --- Plot Distance History from Evaluation ---
    if distance_history:
        plt.figure(figsize=(12, 6))
        plt.plot(distance_history)
        plt.title("Distance to Target During Evaluation")
        plt.xlabel("Time Step (across all evaluation episodes)")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("evaluation_distance_history.png")
        plt.show()
    
    env.close()


if __name__ == "__main__":
    main()