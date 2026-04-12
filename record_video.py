"""
record_video.py - Record a cinematic MP4 video of the trained PPO agent
Features: Compact HUD overlay, high-resolution, cinematic quality
"""

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
import os
import cv2
import imageio
import numpy as np

from config import ENV_CONFIG

# Highway-env Discrete Meta Actions translation
ACTION_NAMES = ["LANE LEFT", "IDLE", "LANE RIGHT", "FASTER", "SLOWER"]
ACTION_COLORS = {
    "LANE LEFT": (200, 100, 255),   # Purple/Magenta
    "IDLE": (100, 200, 255),        # Cyan
    "LANE RIGHT": (255, 150, 50),   # Orange
    "FASTER": (50, 255, 100),       # Green
    "SLOWER": (50, 180, 255),       # Light Blue
}


def draw_compact_hud(frame, speed, action_name, episode_reward, step_count, episode_num):
    """Draw a compact, semi-transparent HUD that doesn't obscure the driving."""
    height, width = frame.shape[:2]
    overlay = frame.copy()

    # ─── TOP BAR (thin strip across the top) ───
    bar_h = 50
    cv2.rectangle(overlay, (0, 0), (width, bar_h), (10, 12, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = cv2.FONT_HERSHEY_DUPLEX

    # Title on left
    cv2.putText(frame, "PPO AI DRIVER", (15, 35), font_bold, 1.0, (100, 255, 130), 2)
    cv2.putText(frame, "[AUTONOMOUS]", (270, 35), font, 0.6, (100, 200, 150), 1)

    # Live stats on right side of top bar
    stats_right = width - 20
    stat_color = (200, 220, 255)

    ep_text = f"EP: {episode_num:03d}"
    step_text = f"STEP: {step_count:04d}"
    cv2.putText(frame, step_text, (stats_right - 180, 35), font, 0.65, stat_color, 1)
    cv2.putText(frame, ep_text, (stats_right - 350, 35), font, 0.65, stat_color, 1)

    # ─── BOTTOM PANEL (compact info strip) ───
    panel_h = 90
    panel_y = height - panel_h
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (0, panel_y), (width, height), (10, 12, 20), -1)
    cv2.addWeighted(bottom_overlay, 0.7, frame, 0.3, 0, frame)

    # Green accent line at top of bottom panel
    cv2.line(frame, (0, panel_y), (width, panel_y), (50, 200, 100), 2)

    # --- Bottom panel content: 3 sections ---
    section_w = width // 3
    label_y = panel_y + 30
    value_y = panel_y + 65
    label_color = (150, 160, 180)

    # Section 1: SPEED
    sec1_x = 30
    cv2.putText(frame, "VELOCITY", (sec1_x, label_y), font, 0.6, label_color, 1)
    speed_color = (50, 255, 100) if speed > 20 else (100, 255, 200) if speed > 10 else (200, 200, 50)
    cv2.putText(frame, f"{speed:.1f} m/s", (sec1_x, value_y), font_bold, 1.2, speed_color, 2)

    # Speed bar
    bar_x = sec1_x + 200
    bar_w = 180
    bar_h2 = 16
    bar_y2 = value_y - 14
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h2), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h2), (80, 80, 100), 1)
    fill = int(min(speed / 35.0, 1.0) * (bar_w - 4))
    cv2.rectangle(frame, (bar_x + 2, bar_y2 + 2), (bar_x + 2 + fill, bar_y2 + bar_h2 - 2), speed_color, -1)

    # Section 2: ACTION
    sec2_x = section_w + 30
    cv2.putText(frame, "ACTION", (sec2_x, label_y), font, 0.6, label_color, 1)
    action_color = ACTION_COLORS.get(action_name, (100, 200, 255))

    # Action pill/badge
    badge_overlay = frame.copy()
    badge_x1 = sec2_x
    badge_x2 = sec2_x + 220
    badge_y1 = value_y - 22
    badge_y2 = value_y + 8
    cv2.rectangle(badge_overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), action_color, -1)
    cv2.addWeighted(badge_overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), action_color, 1)
    cv2.putText(frame, f"{action_name}", (sec2_x + 10, value_y), font_bold, 0.9, action_color, 2)

    # Section 3: REWARD
    sec3_x = 2 * section_w + 30
    cv2.putText(frame, "REWARD", (sec3_x, label_y), font, 0.6, label_color, 1)

    if episode_reward > 30:
        reward_color = (50, 255, 100)    # Green
    elif episode_reward > 15:
        reward_color = (100, 255, 200)   # Cyan
    elif episode_reward > 5:
        reward_color = (255, 200, 50)    # Yellow
    else:
        reward_color = (100, 150, 255)   # Light blue

    cv2.putText(frame, f"{episode_reward:+.1f}", (sec3_x, value_y), font_bold, 1.2, reward_color, 2)

    # Reward bar
    rbar_x = sec3_x + 180
    rbar_w = 200
    cv2.rectangle(frame, (rbar_x, bar_y2), (rbar_x + rbar_w, bar_y2 + bar_h2), (30, 30, 40), -1)
    cv2.rectangle(frame, (rbar_x, bar_y2), (rbar_x + rbar_w, bar_y2 + bar_h2), (80, 80, 100), 1)
    rfill = int(min(max(episode_reward, 0) / 50.0, 1.0) * (rbar_w - 4))
    cv2.rectangle(frame, (rbar_x + 2, bar_y2 + 2), (rbar_x + 2 + rfill, bar_y2 + bar_h2 - 2), reward_color, -1)

    # ─── NEURAL NETWORK STATUS (bottom-right tiny badge) ───
    status_text = "NEURAL NET ACTIVE"
    text_x = width - 280
    text_y = panel_y + 25
    cv2.circle(frame, (text_x - 12, text_y - 5), 5, (50, 255, 100), -1)  # green dot
    cv2.putText(frame, status_text, (text_x, text_y), font, 0.5, (100, 255, 150), 1)

    return frame


def create_title_card(width, height, text_lines, duration_frames=45):
    """Create cinematic title card frames."""
    frames = []
    for i in range(duration_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Subtle gradient background
        for y in range(height):
            intensity = int(15 + 10 * (y / height))
            frame[y, :] = [intensity, intensity + 3, intensity + 8]

        # Fade in effect
        alpha = min(1.0, i / 20.0)

        font_title = cv2.FONT_HERSHEY_DUPLEX
        font_sub = cv2.FONT_HERSHEY_SIMPLEX

        for idx, (text, font_ref, scale, color, thickness) in enumerate(text_lines):
            text_size = cv2.getTextSize(text, font_ref, scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2 - 60 + idx * 60

            faded_color = tuple(int(c * alpha) for c in color)
            cv2.putText(frame, text, (text_x, text_y), font_ref, scale, faded_color, thickness)

        # Accent line
        line_alpha = min(1.0, i / 30.0)
        line_w = int(400 * line_alpha)
        line_x = (width - line_w) // 2
        cv2.line(frame, (line_x, height // 2 + 40), (line_x + line_w, height // 2 + 40),
                 (int(50 * alpha), int(200 * alpha), int(100 * alpha)), 2)

        frames.append(frame)
    return frames


def record():
    print("=" * 60)
    print("RECORDING CINEMATIC PPO AGENT VIDEO")
    print("=" * 60)

    model_path = "./models/ppo_highway_final"
    video_folder = "./results/videos"
    os.makedirs(video_folder, exist_ok=True)

    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find model at {model_path}. Make sure to train first!")
        return

    # Create the environment with cinematic high-res render settings
    base_env = gym.make(ENV_CONFIG["env_id"], render_mode="rgb_array")

    # Upgrade visuals to cinematic quality
    base_env.unwrapped.configure({
        "screen_width": 1920,        # Full HD width
        "screen_height": 600,        # Cinematic ultra-wide
        "scaling": 10,               # High detail cars/assets
        "show_trajectories": True,   # Show AI's predicted paths
    })

    fps = 15  # Always 15fps regardless of env metadata (highway-fast-v0 reports 2 fps which is wrong)

    # Recording parameters
    target_episodes = 50   # Record a full evaluation suite
    min_frames_per_episode = 2

    # Create intro title card
    font_title = cv2.FONT_HERSHEY_DUPLEX
    font_sub = cv2.FONT_HERSHEY_SIMPLEX
    intro_lines = [
        ("PPO AUTONOMOUS DRIVER", font_title, 1.8, (100, 255, 130), 3),
        ("Highway-fast-v0  |  Reinforcement Learning", font_sub, 0.9, (150, 180, 200), 1),
        ("Shreedhar K B  |  23BCS126", font_sub, 0.8, (120, 160, 180), 1),
    ]
    intro_frames = create_title_card(1920, 600, intro_lines, duration_frames=60)

    gameplay_frames = []
    total_episodes = 0
    total_steps = 0
    global_reward = 0

    print(f"\nRecording {target_episodes} full episodes @ {fps} fps...")
    print("This provides a complete visual evaluation audit.\n")

    while total_episodes < target_episodes:
        obs, info = base_env.reset()
        done = False
        truncated = False

        episode_reward = 0
        episode_steps = 0
        episode_frames = []

        total_episodes += 1
        print(f"Recording Episode {total_episodes}/{target_episodes}...", end=" ", flush=True)

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = base_env.step(action)
            episode_reward += reward
            global_reward += reward
            episode_steps += 1
            total_steps += 1

            frame = np.ascontiguousarray(base_env.render())

            speed = info.get("speed", 0)
            action_name = ACTION_NAMES[int(action)] if int(action) < len(ACTION_NAMES) else str(action)

            # Draw compact HUD
            frame = draw_compact_hud(frame, speed, action_name, episode_reward, episode_steps, total_episodes)

            episode_frames.append(frame)
            gameplay_frames.append(frame)

        if len(episode_frames) >= min_frames_per_episode:
            print(f"✓ Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        else:
            gameplay_frames = gameplay_frames[:-len(episode_frames)]
            total_episodes -= 1
            print("(skipped - too short)")

    base_env.close()

    # Create outro title card
    outro_lines = [
        ("TRAINING COMPLETE", font_title, 1.8, (100, 255, 130), 3),
        (f"Total Reward: {global_reward:.1f}  |  Episodes: 50 (Evaluation)", font_sub, 0.9, (150, 180, 200), 1),
        ("PPO with GAE  |  Actor-Critic Architecture", font_sub, 0.8, (120, 160, 180), 1),
    ]
    outro_frames = create_title_card(1920, 600, outro_lines, duration_frames=45)

    # Combine: intro + gameplay + outro
    all_frames = intro_frames + gameplay_frames + outro_frames

    print("\n" + "=" * 60)
    print("ENCODING VIDEO...")
    print("=" * 60)

    video_path = f"{video_folder}/highway_ppo_annotated.mp4"
    # Save as high-quality MP4 video
    imageio.mimsave(video_path, all_frames, fps=fps, codec='libx264', pixelformat='yuv420p')

    video_duration = len(all_frames) / fps

    print(f"\n✓ RECORDING COMPLETE!")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Global Reward: {global_reward:.2f}")
    print(f"  Video Duration: {video_duration:.1f} seconds")
    print(f"  Frames: {len(all_frames)} (intro:{len(intro_frames)} + gameplay:{len(gameplay_frames)} + outro:{len(outro_frames)})")
    print(f"  FPS: {fps}")
    print(f"  ✓ Video saved to: {video_path}")


if __name__ == "__main__":
    record()
