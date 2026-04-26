import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ---------------- DATASET ----------------
data = [ [1,120,2,30,20,1.2,'Saver'], [2,150,2,40,25,0.8,'Saver'], [3,180,2,50,30,1.5,'Saver'], [4,200,3,60,35,2.0,'Saver'], [5,220,3,70,40,1.0,'Saver'], [6,250,3,80,45,1.8,'Balanced'], [7,280,3,90,50,2.2,'Balanced'], [8,300,3,100,60,2.5,'Balanced'], [9,320,4,110,70,1.6,'Balanced'], [10,350,4,120,75,2.8,'Balanced'], [11,380,4,130,80,3.0,'Spender'], [12,400,4,140,85,3.2,'Spender'], [13,420,5,150,90,2.0,'Spender'], [14,450,5,160,100,2.5,'Spender'], [15,480,5,170,110,3.5,'Spender'], [16,500,5,180,120,4.0,'Spender'], [17,130,2,35,20,1.0,'Saver'], [18,160,2,45,25,1.3,'Saver'], [19,210,3,65,35,1.7,'Saver'], [20,260,3,85,50,2.1,'Balanced'], [21,290,3,95,55,2.4,'Balanced'], [22,310,4,105,65,2.7,'Balanced'], [23,340,4,115,70,2.9,'Balanced'], [24,390,4,125,80,3.1,'Spender'], [25,410,5,135,85,3.3,'Spender'], [26,430,5,145,95,3.6,'Spender'], [27,460,5,155,105,3.8,'Spender'], [28,490,5,165,115,4.1,'Spender'], [29,520,6,180,130,4.5,'Spender'], [30,550,6,190,140,5.0,'Spender'] ]

df = pd.DataFrame(data, columns=["ID","Budget","Meals","Snacks","Drinks","Distance","Class"])

# ---------------- NEW POINT ----------------
new_point = np.array([85, 50])

# ---------------- DISTANCE ----------------
df["Dist"] = np.sqrt((df["Snacks"] - new_point[0])**2 + (df["Drinks"] - new_point[1])**2)
neighbors = df.sort_values("Dist").head(3)

# ---------------- COLORS ----------------
color_map = {"Saver": "#00ff7f", "Balanced": "#ffa500", "Spender": "#ff4c4c"}

plt.style.use("dark_background")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.4, bottom=0.2)

# ===================== TOP GRAPH =====================
for cls in df["Class"].unique():
    sub = df[df["Class"] == cls]
    ax1.scatter(sub["Snacks"], sub["Drinks"], color=color_map[cls], alpha=0.7, label=cls)

ax1.scatter(new_point[0], new_point[1], color="cyan", marker="x", s=120, label="New Data")
ax1.set_title("K-NN Search: Calculating Distance to All Points", pad=15)
ax1.set_xlabel("Snack Spending")
ax1.set_ylabel("Drinks Spending")

# Prepare lines and text labels for every point
lines = [ax1.plot([], [], color="white", linestyle="--", linewidth=1, alpha=0.3)[0] for _ in range(len(df))]
dist_texts = [ax1.text(0, 0, "", color="white", fontsize=8, alpha=0.8) for _ in range(len(df))]

def update(frame):
    # Reset everything at the start of a loop
    if frame == 0:
        for l, t in zip(lines, dist_texts):
            l.set_data([], [])
            t.set_text("")
    
    # Update current frame
    if frame < len(df):
        target_x = df.iloc[frame]["Snacks"]
        target_y = df.iloc[frame]["Drinks"]
        distance = df.iloc[frame]["Dist"]
        
        # Update line
        lines[frame].set_data([new_point[0], target_x], [new_point[1], target_y])
        
        # Update text label next to the dot
        dist_texts[frame].set_position((target_x + 1, target_y + 1))
        dist_texts[frame].set_text(f"{distance:.1f}")
        
    return lines + dist_texts

total_dots = len(df)
# blit=False is safer when updating text labels dynamically
ani = FuncAnimation(fig, update, frames=total_dots + 10, interval=150, blit=False, repeat=True)
ani.event_source.stop()

# ===================== BOTTOM GRAPH =====================
ax2.set_title("Top 3 Nearest Neighbors (Result)", pad=9)
ax2.set_xlabel("Snack Spending")
ax2.set_ylabel("Drinks Spending")

for cls in df["Class"].unique():
    sub = df[df["Class"] == cls]
    ax2.scatter(sub["Snacks"], sub["Drinks"], color=color_map[cls], alpha=0.2)

for _, row in neighbors.iterrows():
    ax2.scatter(row["Snacks"], row["Drinks"], color=color_map[row["Class"]], s=120, edgecolors="white")

ax2.scatter(new_point[0], new_point[1], color="cyan", marker="x", s=120)

for _, row in neighbors.iterrows():
    ax2.plot([new_point[0], row["Snacks"]], [new_point[1], row["Drinks"]], 
             color="white", linestyle="--", alpha=0.8)
    # The text line that was here is now deleted

# ===================== BUTTONS =====================
ax_play = plt.axes([0.35, 0.05, 0.1, 0.05])
ax_pause = plt.axes([0.55, 0.05, 0.1, 0.05])

btn_play = Button(ax_play, "▶ Play", color="#222", hovercolor="#444")
btn_pause = Button(ax_pause, "⏸ Pause", color="#222", hovercolor="#444")

btn_play.label.set_color("white")
btn_pause.label.set_color("white")

def play_handler(event):
    ani.event_source.start()

def pause_handler(event):
    ani.event_source.stop()

btn_play.on_clicked(play_handler)
btn_pause.on_clicked(pause_handler)

plt.show()