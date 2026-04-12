"""
record_video.py - Save an annotated MP4 video of the trained PPO agent
IMPROVED: Better graphics, longer duration, enhanced HUD
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

def draw_progress_bar(frame, x, y, width, height, value, max_value, color):
    """Draw progress bar with glow effect."""
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 40), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 120), 2)
    
    # Fill bar
    if max_value > 0:
        fill_width = int((value / max_value) * (width - 4))
        cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), color, -1)
        
        # Glow effect - draw slightly larger bar behind with low opacity
        glow_overlay = frame.copy()
        cv2.rectangle(glow_overlay, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), 
                     (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80)), -1)
        cv2.addWeighted(glow_overlay, 0.3, frame, 0.7, 0, frame)

def draw_game_hud(frame, speed, action_name, episode_reward, step_count, total_episodes):
    """Draw game-quality HUD overlay."""
    height, width = frame.shape[:2]
    
    # === LEFT PANEL: Main Stats ===
    panel_x, panel_y = 25, 20
    panel_w, panel_h = 700, 520
    
    overlay = frame.copy()
    
    # Main panel with gradient effect (using multiple rectangles)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                  (10, 12, 20), -1)
    
    # Glowing border effect (multiple layers)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                  (0, 255, 100), 3)
    cv2.rectangle(overlay, (panel_x + 2, panel_y + 2), (panel_x + panel_w - 2, panel_y + panel_h - 2), 
                  (50, 200, 80), 1)
    
    # Blend for better look
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Font setup
    font_title = cv2.FONT_HERSHEY_DUPLEX
    font_large = cv2.FONT_HERSHEY_DUPLEX
    font_medium = cv2.FONT_HERSHEY_SIMPLEX
    
    # === TITLE SECTION ===
    title_x = panel_x + 30
    title_y = panel_y + 50
    
    # Main title with glow
    cv2.putText(frame, "PPO AI DRIVER", (title_x, title_y), font_title, 2.2, (100, 255, 100), 5)
    cv2.putText(frame, "PPO AI DRIVER", (title_x, title_y), font_title, 2.2, (150, 255, 150), 2)
    
    # Subtitle
    cv2.putText(frame, "[AUTONOMOUS MODE]", (title_x + 10, title_y + 45), font_medium, 0.95, 
                (100, 200, 150), 2)
    
    # Decorative line
    cv2.line(frame, (panel_x + 20, title_y + 65), (panel_x + panel_w - 20, title_y + 65), 
             (100, 255, 100), 2)
    
    # === SPEED GAUGE ===
    speed_section_y = title_y + 100
    
    cv2.putText(frame, "VELOCITY", (title_x, speed_section_y), font_medium, 0.9, 
                (180, 180, 200), 1)
    
    # Speed value display
    speed_color = (50, 255, 100) if speed > 20 else (100, 255, 200) if speed > 10 else (200, 200, 50)
    cv2.putText(frame, f"{speed:.1f}", (title_x, speed_section_y + 50), font_large, 
                2.0, speed_color, 4)
    cv2.putText(frame, "m/s", (title_x + 160, speed_section_y + 50), font_medium, 
                1.0, (180, 180, 200), 2)
    
    # Speed bar
    draw_progress_bar(frame, title_x, speed_section_y + 70, 300, 25, speed, 35, speed_color)
    
    # === ACTION INDICATOR ===
    action_section_y = speed_section_y + 130
    
    cv2.putText(frame, "CURRENT ACTION", (title_x, action_section_y), font_medium, 0.9, 
                (180, 180, 200), 1)
    
    action_color = ACTION_COLORS.get(action_name, (100, 200, 255))
    
    # Action box with glow
    action_box_overlay = frame.copy()
    cv2.rectangle(action_box_overlay, (title_x, action_section_y + 15), 
                  (title_x + 250, action_section_y + 70), action_color, -1)
    cv2.addWeighted(action_box_overlay, 0.2, frame, 0.8, 0, frame)
    
    # Action text
    cv2.putText(frame, f"● {action_name}", (title_x + 15, action_section_y + 55), 
                font_medium, 1.3, action_color, 3)
    
    # === REWARD SCORE ===
    reward_section_y = action_section_y + 100
    
    cv2.putText(frame, "EPISODE REWARD", (title_x, reward_section_y), font_medium, 0.9, 
                (180, 180, 200), 1)
    
    # Reward value with dynamic color
    if episode_reward > 50:
        reward_color = (50, 255, 100)  # Green for excellent
    elif episode_reward > 30:
        reward_color = (100, 255, 200)  # Cyan for good
    elif episode_reward > 10:
        reward_color = (255, 200, 50)  # Yellow for okay
    else:
        reward_color = (255, 100, 100)  # Red for poor
    
    cv2.putText(frame, f"{episode_reward:+.2f}", (title_x, reward_section_y + 50), 
                font_large, 2.0, reward_color, 4)
    
    # Reward bar
    draw_progress_bar(frame, title_x, reward_section_y + 70, 300, 25, max(0, episode_reward), 100, reward_color)
    
    # === RIGHT SIDE: Performance Stats ===
    right_panel_x = panel_x + panel_w + 20
    right_panel_y = panel_y
    right_panel_w = 320
    
    # Right panel background
    right_overlay = frame.copy()
    cv2.rectangle(right_overlay, (right_panel_x, right_panel_y), 
                  (right_panel_x + right_panel_w, right_panel_y + 300), (10, 20, 15), -1)
    cv2.rectangle(right_overlay, (right_panel_x, right_panel_y), 
                  (right_panel_x + right_panel_w, right_panel_y + 300), (100, 150, 255), 2)
    cv2.addWeighted(right_overlay, 0.8, frame, 0.2, 0, frame)
    
    # Title
    right_stat_y = right_panel_y + 40
    cv2.putText(frame, "DIAGNOSTICS", (right_panel_x + 20, right_stat_y), font_medium, 1.0, 
                (100, 200, 255), 2)
    
    cv2.line(frame, (right_panel_x + 15, right_stat_y + 15), 
             (right_panel_x + right_panel_w - 15, right_stat_y + 15), (100, 200, 255), 1)
    
    # Stats
    stat_y = right_stat_y + 50
    stat_color = (180, 220, 255)
    
    cv2.putText(frame, f"Episode: {total_episodes:03d}", (right_panel_x + 20, stat_y), 
                font_medium, 0.8, stat_color, 1)
    stat_y += 40
    cv2.putText(frame, f"Step: {step_count:05d}", (right_panel_x + 20, stat_y), 
                font_medium, 0.8, stat_color, 1)
    stat_y += 40
    cv2.putText(frame, f"FPS: 15", (right_panel_x + 20, stat_y), 
                font_medium, 0.8, stat_color, 1)
    stat_y += 40
    cv2.putText(frame, f"Status: ACTIVE", (right_panel_x + 20, stat_y), 
                font_medium, 0.8, (100, 255, 100), 1)
    
    # === BOTTOM STATUS BAR ===
    status_bar_y = panel_y + panel_h + 20
    
    # Status bar background
    cv2.rectangle(frame, (panel_x, status_bar_y), (panel_x + panel_w + right_panel_w + 20, status_bar_y + 40), 
                  (15, 15, 25), -1)
    cv2.rectangle(frame, (panel_x, status_bar_y), (panel_x + panel_w + right_panel_w + 20, status_bar_y + 40), 
                  (100, 100, 120), 1)
    
    # Status text
    cv2.putText(frame, "● NEURAL NETWORK ACTIVE | REAL-TIME INFERENCE | OPTIMAL PERFORMANCE", 
                (panel_x + 20, status_bar_y + 27), font_medium, 0.7, (100, 255, 150), 1)
    
    return frame

def record():
    print("=" * 60)
    print("RECORDING ENHANCED ANNOTATED AGENT VIDEO")
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
    
    # Upgrade standard visuals to cinematic/realistic quality
    base_env.unwrapped.configure({
        "screen_width": 1920,          # Full HD width
        "screen_height": 600,          # Cinematic ultra-wide height
        "scaling": 10,                 # Up-scales the cars/assets making them more detailed
        "show_trajectories": True,     # Renders the AI's future predicted paths
    })
    
    # Extract original FPS from environment metadata (usually 15 for highway-v0)
    fps = base_env.metadata.get("render_fps", 15)
    
    # Recording parameters - aim for ~90 seconds of video (900 frames at 15 fps)
    target_frames = 900
    min_frames_per_episode = 2  # Don't record episodes that end too quickly
    
    frames = []
    total_episodes = 0
    total_steps = 0
    global_reward = 0
    
    print(f"\nRecording approximately 60 seconds of video (target: {target_frames} frames @ {fps} fps)...")
    print("This may span multiple episodes...\n")
    
    # Record multiple episodes until we reach target frame count
    while len(frames) < target_frames:
        obs, info = base_env.reset()
        done = False
        truncated = False
        
        episode_reward = 0
        episode_steps = 0
        episode_frames = []
        
        total_episodes += 1
        print(f"Recording Episode {total_episodes}...", end=" ", flush=True)
        
        # Record one episode
        while not (done or truncated) and len(frames) < target_frames:
            # Predict the best action
            action, _states = model.predict(obs, deterministic=True)
            
            # Take the action
            obs, reward, done, truncated, info = base_env.step(action)
            episode_reward += reward
            global_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Grab current rendering and enforce C-contiguous memory for OpenCV
            frame = np.ascontiguousarray(base_env.render())
            
            # Extract vehicle specifics
            speed = info.get("speed", 0)
            action_name = ACTION_NAMES[int(action)] if int(action) < len(ACTION_NAMES) else str(action)
            
            # Draw the game-quality HUD
            frame = draw_game_hud(frame, speed, action_name, episode_reward, episode_steps, total_episodes)
            
            episode_frames.append(frame)
            frames.append(frame)
        
        # Only keep frames from episodes with meaningful length
        if len(episode_frames) >= min_frames_per_episode:
            print(f"✓ Reward: {episode_reward:.2f}, Steps: {episode_steps}")
        else:
            # Remove short episode frames
            frames = frames[:-len(episode_frames)]
            total_episodes -= 1
            print("(skipped - too short)")
    
    base_env.close()
    
    # Trim to exact target if we went over slightly
    frames = frames[:target_frames]
    
    print("\n" + "=" * 60)
    print("ENCODING VIDEO...")
    print("=" * 60)
    
    video_path = f"{video_folder}/highway_ppo_annotated.mp4"
    imageio.mimsave(video_path, frames, fps=fps, codec='libx264', pixelformat='yuv420p')
    
    video_duration = len(frames) / fps
    
    print(f"\n✓ RECORDING COMPLETE!")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Global Reward: {global_reward:.2f}")
    print(f"  Video Duration: {video_duration:.1f} seconds")
    print(f"  Frames Recorded: {len(frames)}")
    print(f"  FPS: {fps}")
    print(f"  ✓ Video saved to: {video_path}")
    print(f"\n  💡 Tip: Play at 2x speed (0:{video_duration/2:.0f}) for ~{video_duration/2:.0f} seconds of content!")

if __name__ == "__main__":
    record()
