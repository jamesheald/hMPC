from matplotlib import animation
import matplotlib.pyplot as plt
import os

def save_frames_as_gif(frames, path, filename):

    plt.figure(figsize = (frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
      
      patch.set_data(frames[i])
    
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 50)
    
    os.makedirs(os.path.dirname(path), exist_ok = True)
    anim.save(path + filename, writer = 'imagemagick', fps = 60)

    plt.close()

    return None