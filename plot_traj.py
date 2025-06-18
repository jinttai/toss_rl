# plot_saved_episodes.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_episode(npz_path: str,
                 save_dir: str,
                 target_vel_mag: float = 0.2) -> None:
    """
    단일 에피소드(npz)의 궤적을 읽어 두 개의 PNG 저장.
    """
    data = np.load(npz_path)
    com_vel = data["com_vel"]          # (T, 2)
    joint_vel = np.rad2deg(data["joint_vel"])  # (T, 6) → deg/s
    goal_angle, goal_speed = data["goal"]      # [angle, magnitude]
    success = bool(data["success"])

    ep_id = int(os.path.splitext(os.path.basename(npz_path))[0].split("_")[-1])

    # ---------- CoM velocity 궤적 ---------- #
    tgt_vx = goal_speed * np.cos(goal_angle)
    tgt_vy = goal_speed * np.sin(goal_angle)

    plt.figure(figsize=(6, 6))
    plt.plot(com_vel[:, 0], com_vel[:, 1], label="CoM velocity", alpha=0.7)
    plt.scatter(com_vel[0, 0], com_vel[0, 1], c="g", s=60, label="start")
    plt.scatter(com_vel[-1, 0], com_vel[-1, 1], c="r", s=60, label="end")
    plt.quiver(0, 0, tgt_vx, tgt_vy,
               angles="xy", scale_units="xy", scale=1, color="k",
               label=f"target (mag={goal_speed:.2f})")
    lim = max(np.abs(com_vel).max(), goal_speed) * 1.1
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle=":")
    plt.title(f"Episode {ep_id}  success={success}")
    plt.xlabel("vx (m/s)"); plt.ylabel("vy (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ep_{ep_id:04d}_com.png"))
    plt.close()

    # ---------- Joint velocity ---------- #
    t = np.arange(len(joint_vel))
    plt.figure(figsize=(8, 5))
    for j in range(joint_vel.shape[1]):
        plt.plot(t, joint_vel[:, j], label=f"joint{j+1}")
    plt.grid(True, linestyle=":")
    plt.xlabel("timestep"); plt.ylabel("joint vel (deg/s)")
    plt.title(f"Episode {ep_id}")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ep_{ep_id:04d}_joint.png"))
    plt.close()


def main(traj_dir: str = "eval_trajectories",
         out_dir: str = "episode_plots") -> None:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(traj_dir, "episode_*.npz")))
    if not files:
        print(f"No *.npz found in {traj_dir}")
        return
    for f in files:
        plot_episode(f, out_dir)
    print(f"Saved {len(files) * 2} PNGs to {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traj_dir", default="eval_trajectories",
                   help="npz trajectory directory")
    p.add_argument("--out_dir",  default="episode_plots",
                   help="PNG output directory")
    args = p.parse_args()
    main(args.traj_dir, args.out_dir)
