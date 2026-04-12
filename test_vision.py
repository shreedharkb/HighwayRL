import gymnasium as gym
import highway_env
import cv2
import numpy as np

env = gym.make("highway-v0", render_mode="rgb_array")
env.unwrapped.configure({
    "screen_width": 1920,
    "screen_height": 600,
    "scaling": 10,
    "show_trajectories": True
})
obs, info = env.reset()
frame = env.render()
env.close()

# Mask green
green_lower = np.array([0, 100, 0])
green_upper = np.array([100, 255, 100])
mask_green = cv2.inRange(frame, green_lower, green_upper)
y_green, x_green = np.where(mask_green > 0)

if len(x_green) > 0:
    print(f"Green car centroid: X={int(np.mean(x_green))}, Y={int(np.mean(y_green))}")
else:
    print("Green car not found.")

# Mask blue
blue_lower = np.array([0, 50, 100])
blue_upper = np.array([100, 150, 255])
mask_blue = cv2.inRange(frame, blue_lower, blue_upper)
y_blue, x_blue = np.where(mask_blue > 0)
if len(x_blue) > 0:
    print(f"Blue car pixels found: {len(x_blue)}")
else:
    print("Blue cars not found.")
