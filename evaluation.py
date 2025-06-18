# evaluate_policy.py
import os
import argparse
import numpy as np
import torch

# 학습 스크립트에서 정의했던 클래스와 설정 가져오기
from ddpg_selftossing_ver_1_10 import SpaceRobotEnv, Normalizer, TD3Agent, CFG

def evaluate(num_episodes: int = 100,
             render: bool = False,
             save_traj: bool = False,
             traj_dir: str = "eval_trajectories") -> None:
    """
    Trained actor를 로드해 num_episodes 만큼 환경을 실행하고
    평균 리턴, 성공률 등을 출력한다.

    Args
    ----
    num_episodes: 에피소드 수
    render      : MuJoCo viewer로 렌더링할지 여부
    save_traj   : 각 에피소드의 CoM·관절 속도 궤적을 파일로 저장할지 여부
    traj_dir    : 궤적 저장 디렉터리
    """

    env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
    raw_obs_dim = env.observation_space.shape[0]
    act_dim     = env.action_space.shape[0]
    goal_dim    = CFG["goal_dimension"]

    # 정규화 통계 불러오기
    obs_norm = Normalizer(raw_obs_dim,
                          gamma=CFG["normalizer_gamma"],
                          beta=CFG["normalizer_beta"])
    obs_norm.load_stats(os.path.join(CFG["save_dir"],
                                     CFG["normalizer_save_path"]))

    # 에이전트(가중치 포함) 불러오기
    agent = TD3Agent(raw_obs_dim, act_dim, goal_dim,
                     env.action_space.high[0], obs_norm, cfg=CFG)
    agent.load_models()
    agent.noise_sigma = 0.0  # 완전 결정론적 행동

    # 메트릭 누적 변수
    ep_returns, successes, final_angles = [], [], []

    # 궤적 저장용 디렉터리
    if save_traj:
        os.makedirs(traj_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs, info_reset = env.reset()
        goal = info_reset["current_goal"]

        ep_ret, traj_com, traj_joint = 0.0, [], []

        for t in range(CFG["episode_length"]):
            act = agent.act(obs, goal, add_noise=False)
            nxt_obs, reward, term, trunc, info = env.step(act)

            ep_ret += reward
            obs = nxt_obs

            if render:
                env.render()

            if save_traj:
                traj_com.append(info["com_vel_3d"][:2])
                traj_joint.append(env.data.qvel.copy())

            if term or trunc:
                break

        # 성공 여부 평가
        success = (info['angle_diff_deg'] <= CFG["success_angle_threshold_deg"]
                   and np.linalg.norm(info['com_vel_3d'][:2])
                   >= CFG["target_velocity_magnitude"])

        ep_returns.append(ep_ret)
        successes.append(float(success))
        final_angles.append(info['angle_diff_deg'])

        if save_traj:
            np.savez_compressed(
                os.path.join(traj_dir, f"episode_{ep:04d}.npz"),
                com_vel=np.asarray(traj_com),
                joint_vel=np.asarray(traj_joint),
                goal=goal,
                success=success)

        print(f"EP {ep:03d} | Ret {ep_ret:8.2f} | "
              f"Final ∠ {info['angle_diff_deg']:6.2f}° | "
              f"Success {success}")

    # ---------------- 통계 출력 ---------------- #
    print("\n=== Evaluation Summary ===")
    print(f"Episodes           : {num_episodes}")
    print(f"Avg Return         : {np.mean(ep_returns):.2f}")
    print(f"Success Rate       : {np.mean(successes)*100:.1f}%")
    print(f"Avg Final Angle Err: {np.mean(final_angles):.2f}°")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100,
                        help="평가 에피소드 수")
    parser.add_argument("--render", action="store_true",
                        help="MuJoCo 렌더링 활성화")
    parser.add_argument("--save_traj", action="store_true",
                        help="궤적 npz 파일 저장 여부")
    parser.add_argument("--traj_dir", type=str, default="eval_trajectories",
                        help="궤적 저장 경로")
    args = parser.parse_args()

    evaluate(num_episodes=args.episodes,
             render=args.render,
             save_traj=args.save_traj,
             traj_dir=args.traj_dir)
