import json

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

from utils import FullState, get_project_dir

class Visualizer():
    def __init__(self):
        self.agent_states_by_idx = []
        self.object_states_by_idx = []
        self.objects_to_track = []
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    def _plot_object_bounding_box(self, ax, object_state, color, alpha=0.5):
        x, y, z = object_state.bbox_center
        x_size, y_size, z_size = object_state.bbox_extents

        # print(f"Object {object_state.object_id} has bbox center {object_state.bbox_center} and bbox extents {object_state.bbox_extents}")
        # print(f"\tVelocity: {object_state.velocity}")

        vertices = [
            (x - x_size, y - y_size, z - z_size),
            (x + x_size, y - y_size, z - z_size),
            (x + x_size, y + y_size, z - z_size),
            (x - x_size, y + y_size, z - z_size),
            (x - x_size, y - y_size, z + z_size),
            (x + x_size, y - y_size, z + z_size),
            (x + x_size, y + y_size, z + z_size),
            (x - x_size, y + y_size, z + z_size)
        ]

        # Update the plot limits
        # self.min_x, self.min_y, self.min_z = min(self.min_x, vertices[0][0]), min(self.min_y, vertices[0][1]), min(self.min_z, vertices[0][2])
        # self.max_x, self.max_y, self.max_z = max(self.max_x, vertices[6][0]), max(self.max_y, vertices[6][1]), max(self.max_z, vertices[6][2])

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]], 
            [vertices[0], vertices[1], vertices[5], vertices[4]], 
            [vertices[2], vertices[3], vertices[7], vertices[6]], 
            [vertices[0], vertices[3], vertices[7], vertices[4]], 
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]

        prism = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=color)
        ax.add_collection3d(prism)

        return prism
    
    def _visualize_objects(self):

        # Reset plot limits
        self.min_x, self.min_y, self.min_z = -4, -1, -4
        self.max_x, self.max_y, self.max_z = 4, 4, 4

        print(self.visualization_index)

        prisms = []
        for obj_idx, object_id in enumerate(self.objects_to_track):
            prism = self._plot_object_bounding_box(self.ax, self.object_states_by_idx[self.visualization_index][object_id], self.colors[obj_idx % len(self.colors)], alpha=0.5)
            prisms.append(prism)

        margin = 0
        self.ax.set_xlim(self.min_x - margin, self.max_x + margin)
        self.ax.set_ylim(self.min_y - margin, self.max_y + margin)
        self.ax.set_zlim(self.min_z - margin, self.max_z + margin)

        legend_elements = [Line2D([0], [0], color=self.colors[i % len(self.colors)], lw=4, label=self.objects_to_track[i]) for i in range(len(self.objects_to_track))]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        plt.title(f"State {self.visualization_index}", fontsize=16)

    def _update_visualization(self, event):
        if event.key == "right":
            self.visualization_index = min(self.visualization_index + 1, len(self.agent_states_by_idx) - 1)
            self.ax.clear()
            self._visualize_objects()

        elif event.key == "left":
            self.visualization_index = max(self.visualization_index - 1, 0)
            self.ax.clear()
            self._visualize_objects()

    def visualize(self, trace, objects_to_track, start_idx=0):
        
        self.objects_to_track = objects_to_track
        self.visualization_index = start_idx

        replay = trace['replay']
        replay_len = int(len(replay))

        # Stores the most recent state of the agent and of each object
        most_recent_agent_state = None
        most_recent_object_states = {}

        initial_object_states = {}
        
        # Start by recording the states of objects we want to track
        for idx, state in tqdm(enumerate(replay), total=replay_len, desc=f"Processing replay", leave=False):
            state = FullState.from_state_dict(state)

            # Track changes to the agent
            if state.agent_state_changed:
                most_recent_agent_state = state.agent_state

            # And to objects
            objects_with_initial_rotations = []
            for obj in state.objects:
                if obj.object_id not in initial_object_states:
                    initial_object_states[obj.object_id] = obj

                obj = obj._replace(initial_rotation=initial_object_states[obj.object_id].rotation)
                objects_with_initial_rotations.append(obj)
                most_recent_object_states[obj.object_id] = obj

            self.agent_states_by_idx.append(most_recent_agent_state)
            self.object_states_by_idx.append(most_recent_object_states)

        # Plot figure
        self.fig = plt.figure()
        cid = self.fig.canvas.mpl_connect('key_press_event', self._update_visualization)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self._visualize_objects()
        plt.show()
        



if __name__ == "__main__":
    trace_path = "./reward-machine/traces/throw_ball_to_bin_unique_positions-rerecorded.json"
    trace = json.load(open(trace_path, 'r'))

    objects = ["Dodgeball|+00.19|+01.13|-02.80", "Floor|+00.00|+00.00|+00.00"]
    Visualizer().visualize(trace, objects_to_track=objects, start_idx=1500)