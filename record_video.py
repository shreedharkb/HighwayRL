"""
record_video.py - Save an annotated MP4 video of the trained PPO agent
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

def record():
    print("=" * 50)
    print("RECORDING ANNOTATED AGENT VIDEO")
    print("=" * 50)
    
    model_path = "./models/ppo_highway_final"
    video_folder = "./results/videos"
    os.makedirs(video_folder, exist_ok=True)
    
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"Error: Could not find model at {model_path}. Make sure to train first!")
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
    
    obs, info = base_env.reset()
    done = False
    truncated = False
    
    score = 0
    steps = 0
    frames = []
    
    print("\nRecording 1 full episode with HUD overlay... Please wait.")
    
    # Extract original FPS from environment metadata (usually 15 for highway-v0)
    fps = base_env.metadata.get("render_fps", 15)
    
    # Loop manually instead of using RecordVideo wrapper to let us annotate it
    while not (done or truncated):
        # Predict the best action
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the action
        obs, reward, done, truncated, info = base_env.step(action)
        score += reward
        steps += 1
        
        # Grab current rendering and enforce C-contiguous memory for OpenCV
        frame = np.ascontiguousarray(base_env.render())
        
        # Extract vehicle specifics
        speed = info.get("speed", 0)
        action_name = ACTION_NAMES[int(action)] if int(action) < len(ACTION_NAMES) else str(action)
        
        # Draw the HUD Backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (520, 310), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Use OpenCV to inject text labels permanently into the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "PPO AI DRIVER", (45, 75), font, 1.4, (100, 255, 100), 3)
        cv2.putText(frame, "(Controlled Green Car)", (45, 115), font, 0.9, (100, 255, 100), 2)
        
        cv2.putText(frame, f"Speed:  {speed:.1f} m/s", (45, 180), font, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Action: {action_name}", (45, 235), font, 1.2, (50, 200, 255), 2)
        cv2.putText(frame, f"Reward: {score:.2f}", (45, 290), font, 1.2, (255, 150, 50), 2)
        
        frames.append(frame)
        
    base_env.close()
    
    print("\nEncoding High-Resolution MP4...")
    video_path = f"{video_folder}/highway_ppo_annotated.mp4"
    imageio.mimsave(video_path, frames, fps=fps)
    
    print(f"\nRecording Complete!")
    print(f"  Final Score: {score:.2f}")
    print(f"  Total Steps: {steps}")
    print(f"  Annotated video saved to: {video_path}")

if __name__ == "__main__":
    record()
