# evaluate_policy.py
# -----------------------------------------------------------
# TD3(Self-tossing) 학습 모델 평가 스크립트
#   • 평균 리턴 / 성공률 / 최종 각도 오차 출력
#   • 옵션:
#       --render        : MuJoCo 패시브 뷰어 렌더
#       --save_traj     : 각 에피소드 궤적(npz) 저장
#       --save_plot     : 시간축 그래프(PNG) 저장
# -----------------------------------------------------------
import os, argparse, numpy as np, torch, matplotlib.pyplot as plt

# 학습 스크립트(딥러닝·환경 정의)에서 클래스/설정 가져오기
from ddpg_selftossing_ver_1_11 import SpaceRobotEnv, Normalizer, TD3Agent, CFG

def plot_timeseries(mags, angs, ep_idx, out_dir):
    """‖v‖·angle-error 시계열 그래프 두 장 저장"""
    t = np.arange(len(mags))

    # 속도 크기
    plt.figure(figsize=(6,3))
    plt.plot(t, mags)
    plt.xlabel("timestep"); plt.ylabel("‖v‖ (m/s)")
    plt.title(f"EP{ep_idx:03d}  speed magnitude")
    plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ep{ep_idx:03d}_mag.png"))
    plt.close()

    # 각도 오차
    plt.figure(figsize=(6,3))
    plt.plot(t, angs)
    plt.xlabel("timestep"); plt.ylabel("angle error (deg)")
    plt.title(f"EP{ep_idx:03d}  angle error")
    plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"ep{ep_idx:03d}_angle.png"))
    plt.close()

def evaluate(num_episodes: int = 100,
             render: bool = False,
             save_traj: bool = False,
             save_plot: bool = False,
             out_dir: str = "eval_output") -> None:
    """
    저장된 에이전트(Actor) 로드 후 num_episodes 회 평가.

    Parameters
    ----------
    num_episodes : 평가 에피소드 수
    render       : MuJoCo viewer 렌더링 여부
    save_traj    : 각 에피소드의 CoM·joint 속도 npz 저장
    save_plot    : 속도 크기·각도 오차 시계열 PNG 저장
    out_dir      : 출력(궤적/그래프) 디렉터리
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 환경 & 에이전트 준비 ----------
    env = SpaceRobotEnv(CFG["model_path"], cfg=CFG)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    goal_dim = CFG["goal_dimension"]

    # Normalizer
    norm = Normalizer(obs_dim,
                      gamma=CFG["normalizer_gamma"],
                      beta=CFG["normalizer_beta"])
    norm.load_stats(os.path.join(CFG["save_dir"], CFG["normalizer_save_path"]))

    # TD3 에이전트
    agent = TD3Agent(obs_dim, act_dim, goal_dim,
                     env.action_space.high[0], norm, cfg=CFG)
    agent.load_models()
    agent.noise_sigma = 0.0   # 평가: 결정론적

    # ---------- 메트릭 누적 ----------
    returns, successes, final_angles = [], [], []

    for ep in range(num_episodes):
        obs, reset_info = env.reset()
        goal = reset_info["current_goal"]

        ep_ret = 0.0
        traj_com, traj_joint = [], []
        mag_hist, ang_hist = [], []     # NEW

        for t in range(CFG["episode_length"]):
            action = agent.act(obs, goal, add_noise=False)
            obs, reward, term, trunc, info = env.step(action)

            # 기록
            v_xy = info["com_vel_3d"][:2]
            speed_mag = np.linalg.norm(v_xy)
            angle_err = info["angle_diff_deg"]

            if save_traj:
                traj_com.append(v_xy)
                traj_joint.append(env.data.qvel.copy())
            if save_plot:
                mag_hist.append(speed_mag)
                ang_hist.append(angle_err)

            ep_ret += reward
            if render: env.render()
            if term or trunc: break

        # 성공 판정
        success = (info['angle_diff_deg'] <= CFG["success_angle_threshold_deg"] and
                   np.linalg.norm(info['com_vel_3d'][:2]) >= CFG["target_velocity_magnitude"])

        returns.append(ep_ret)
        successes.append(float(success))
        final_angles.append(info['angle_diff_deg'])

        # ----- 파일 저장 -----
        ep_tag = f"ep{ep:04d}"
        if save_traj:
            np.savez_compressed(os.path.join(out_dir, f"{ep_tag}.npz"),
                                com_vel=np.asarray(traj_com),
                                joint_vel=np.asarray(traj_joint),
                                goal=goal,
                                success=success)
        if save_plot:
            plot_timeseries(mag_hist, ang_hist, ep, out_dir)

        print(f"EP {ep:03d} | Ret {ep_ret:8.2f} | "
              f"Final ∠ {info['angle_diff_deg']:6.2f}° | "
              f"Success {success}")

    # ---------- 요약 ----------
    print("\n=== Evaluation Summary ===")
    print(f"Episodes           : {num_episodes}")
    print(f"Average Return     : {np.mean(returns):.2f}")
    print(f"Success Rate       : {np.mean(successes)*100:.1f}%")
    print(f"Avg Final AngleErr : {np.mean(final_angles):.2f}°")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int,  default=100, help="평가 에피소드 수")
    parser.add_argument("--render",     action="store_true",    help="MuJoCo 렌더링")
    parser.add_argument("--save_traj",  action="store_true",    help="궤적 npz 저장")
    parser.add_argument("--save_plot",  action="store_true",    help="시계열 그래프 저장")
    parser.add_argument("--out_dir",    type=str, default="eval_output", help="출력 폴더")
    args = parser.parse_args()

    evaluate(num_episodes=args.episodes,
             render=args.render,
             save_traj=args.save_traj,
             save_plot=args.save_plot,
             out_dir=args.out_dir)
