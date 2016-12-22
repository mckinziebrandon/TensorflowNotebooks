---
title: Playing around in OpenAI Gym in Jupyter
layout: post
---

## First, Figure out Jupyter Notebook Stuff

[This tutorial](http://nbviewer.jupyter.org/github/patrickmineault/xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb) helped a lot.


```python
# The typical imports
import gym
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))
```

### Simple Cartpole Example in Jupyter


```python
env = gym.make('CartPole-v0')

# Run a demo of the environment
observation = env.reset()
cum_reward = 0
frames = []
for t in range(5000):
    # Render into buffer. 
    frames.append(env.render(mode = 'rgb_array'))
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        break
env.render(close=True)
display_frames_as_gif(frames)
```

## OpenAI Gym - Documentation

Working through [this entire page](https://gym.openai.com/docs) on starting with the gym. First, we again show their cartpole snippet but with the Jupyter support added in by me.


```python
env = gym.make('CartPole-v0')
cum_reward = 0
frames = []
num_episodes=40
for i_episode in range(num_episodes):
    observation = env.reset()
    for t in range(500):
        # Render into buffer. 
        frames.append(env.render(mode = 'rgb_array'))
        action = env.action_space.sample() # random action
        observation, reward, done, info = env.step(action)
        if done:
            print("\rEpisode {}/{} finished after {} timesteps".format(i_episode, num_episodes, t+1), end="")
            break
env.render(close=True)
display_frames_as_gif(frames)
```

    [2016-12-21 18:10:06,922] Making new env: CartPole-v0
    

    Episode 12/40 finished after 15 timesteps


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-7-03477f9e2d21> in <module>()
          7     for t in range(500):
          8         # Render into buffer.
    ----> 9         frames.append(env.render(mode = 'rgb_array'))
         10         action = env.action_space.sample()
         11         observation, reward, done, info = env.step(action)
    

    c:\users\bmcki\appdata\local\programs\python\python35\lib\site-packages\gym\core.py in render(self, mode, close)
        190             raise error.UnsupportedMode('Unsupported rendering mode: {}. (Supported modes for {}: {})'.format(mode, self, modes))
        191 
    --> 192         return self._render(mode=mode, close=close)
        193 
        194     def close(self):
    

    c:\users\bmcki\appdata\local\programs\python\python35\lib\site-packages\gym\envs\classic_control\cartpole.py in _render(self, mode, close)
        147         self.poletrans.set_rotation(-x[2])
        148 
    --> 149         return self.viewer.render(return_rgb_array = mode=='rgb_array')
    

    c:\users\bmcki\appdata\local\programs\python\python35\lib\site-packages\gym\envs\classic_control\rendering.py in render(self, return_rgb_array)
        102             arr = arr.reshape(buffer.height, buffer.width, 4)
        103             arr = arr[::-1,:,0:3]
    --> 104         self.window.flip()
        105         self.onetime_geoms = []
        106         return arr
    

    c:\users\bmcki\appdata\local\programs\python\python35\lib\site-packages\pyglet\window\win32\__init__.py in flip(self)
        309     def flip(self):
        310         self.draw_mouse_cursor()
    --> 311         self.context.flip()
        312 
        313     def set_location(self, x, y):
    

    c:\users\bmcki\appdata\local\programs\python\python35\lib\site-packages\pyglet\gl\win32.py in flip(self)
        222 
        223     def flip(self):
    --> 224         wgl.wglSwapLayerBuffers(self.canvas.hdc, wgl.WGL_SWAP_MAIN_PLANE)
        225 
        226     def get_vsync(self):
    

    KeyboardInterrupt: 


### Environments

Environments all descend from the [Env](https://github.com/openai/gym/blob/master/gym/core.py#L14) base class. You can view a list of all environments via:

```python
from gym import envs
print(envs.registry.all())
```

Important environment functions/properties:

* __step__: Returns info regarding what our actions are doing to the environment at each step. The return values:
    * observation (object)
    * reward (float)
    * done (boolean)
    * info (dict)
* __reset__: returns an initial observation. 
* __Space objects__: two objects (below) that describe the valid actions and observations.
    * action_space [returns _Discrete(2)_ for cartpole]. Example usage of Discrete:
    ```python
    from gym import spaces
    space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
    x = space.sample()
    assert space.contains(x)
    assert space.n == 8
    ```
    * observation_space [returns _Box(4)_ for cartpole]


```python

```
