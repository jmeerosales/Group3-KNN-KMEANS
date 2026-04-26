import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np

# --- 1. SETUP DATA (Based on Patient #670 Analysis) ---
test_patient_coords = np.array([0.6399, -0.7777]) 
test_label = "Patient 670 (Test)"

training_data = [
    {"index": 60,  "coords": [-0.5264, -1.2569], "dist": 2.7662, "outcome": 0},
    {"index": 535, "coords": [0.0777, 0.3388],   "dist": 2.6366, "outcome": 1},
    {"index": 718, "coords": [-0.8285, -0.4590], "dist": 2.7588, "outcome": 0},
    {"index": 306, "coords": [1.8901, 1.3029],   "dist": 3.1781, "outcome": 1},
    {"index": 90,  "coords": [-0.8285, -1.3899], "dist": 3.4668, "outcome": 0},
    {"index": 231, "coords": [0.6819, 0.4053],   "dist": 3.5545, "outcome": 1},
    {"index": 346, "coords": [-0.8285, 0.5715],   "dist": 3.7066, "outcome": 0},
    {"index": 618, "coords": [1.5880, -0.3261],  "dist": 3.7615, "outcome": 1},
    {"index": 340, "coords": [-0.8285, 0.2723],   "dist": 3.9636, "outcome": 0},
    {"index": 294, "coords": [-1.1305, 1.3029],  "dist": 3.9831, "outcome": 0}
]

# Identify Top 5 Nearest Neighbors for the static graph
nearest_5 = sorted(training_data, key=lambda x: x['dist'])[:5]

# --- 2. PLOT INITIALIZATION ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), gridspec_kw={'height_ratios': [1.2, 1]})
# Adjusted top and hspace to ensure labels and titles don't mix or cut off
plt.subplots_adjust(bottom=0.12, top=0.80, hspace=0.5) 

# Helper function to draw points
def plot_points(target_ax, data_list, alpha_val=0.6):
    for pt in data_list:
        color = '#ff4c4c' if pt['outcome'] == 1 else '#00ff7f'
        target_ax.scatter(pt['coords'][0], pt['coords'][1], color=color, s=80, edgecolors='white', alpha=alpha_val, zorder=3)
        target_ax.text(pt['coords'][0]+0.06, pt['coords'][1]+0.06, f"#{pt['index']}", fontsize=8, color='white', alpha=alpha_val)

# ================= TOP GRAPH (ANIMATED SCANNER) =================
plot_points(ax1, training_data)
ax1.scatter(test_patient_coords[0], test_patient_coords[1], color='cyan', marker='X', s=150, label=test_label, zorder=5)
line, = ax1.plot([], [], color='cyan', linestyle='--', linewidth=1.5, alpha=0.9, zorder=4)

# ADJUSTED LIMITS: Ensures all points and labels are visible
ax1.set_xlim(-1.5, 2.5)
ax1.set_ylim(-2.0, 1.8)

# Titles and Highlights
ax1.text(0.5, 1.35, "K-NEAREST NEIGHBORS: EUCLIDEAN DISTANCE ANALYSIS", 
         transform=ax1.transAxes, color='white', fontsize=14, fontweight='bold', ha='center')

dist_text = ax1.text(0.5, 1.18, '', transform=ax1.transAxes, 
                    color='#c6df3b', fontweight='bold', fontsize=11, 
                    ha='center', va='center', 
                    bbox=dict(facecolor='black', alpha=0.8, edgecolor='#c6df3b', pad=8))

ax1.set_title("Real-Time Neighbor Scanning", color='#aaaaaa', fontsize=10, loc='left', pad=10)
ax1.set_ylabel("Glucose (Scaled)")
ax1.legend(loc='lower right', fontsize='small', framealpha=0.3)
ax1.grid(alpha=0.1)

# ================= BOTTOM GRAPH (STATIC CLASSIFICATION) =================
ax2.set_title("Classification Result: Top 5 Nearest Neighbors", color='white', fontsize=12, pad=15)

# ADJUSTED LIMITS: Matches the top graph for consistency
ax2.set_xlim(-1.5, 2.5)
ax2.set_ylim(-2.0, 1.8)

plot_points(ax2, training_data, alpha_val=0.15) 
plot_points(ax2, nearest_5, alpha_val=1.0)
ax2.scatter(test_patient_coords[0], test_patient_coords[1], color='cyan', marker='X', s=150, zorder=5)

for pt in nearest_5:
    ax2.plot([test_patient_coords[0], pt['coords'][0]], 
             [test_patient_coords[1], pt['coords'][1]], 
             color='white', linestyle='-', linewidth=1, alpha=0.4, zorder=2)

ax2.set_xlabel("Pregnancies (Scaled)")
ax2.set_ylabel("Glucose (Scaled)")
ax2.grid(alpha=0.1)

# --- 3. ANIMATION & INTERACTION ---
is_running = True

def update(frame):
    target = training_data[frame % len(training_data)]
    line.set_data([test_patient_coords[0], target['coords'][0]], 
                  [test_patient_coords[1], target['coords'][1]])
    dist_text.set_text(f"CURRENT SCAN: Patient #{target['index']} | DISTANCE: {target['dist']:.4f}")
    return line, dist_text

ani = FuncAnimation(fig, update, frames=len(training_data), blit=False, interval=1500)

ax_button = plt.axes([0.46, 0.01, 0.1, 0.04])
btn_pause = Button(ax_button, 'Pause', color='#222', hovercolor='#444')
btn_pause.label.set_color('white')

def toggle_animation(event):
    global is_running
    if is_running:
        ani.event_source.stop()
        btn_pause.label.set_text('Play')
    else:
        ani.event_source.start()
        btn_pause.label.set_text('Pause')
    is_running = not is_running
    plt.draw()

btn_pause.on_clicked(toggle_animation)
plt.show()