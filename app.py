import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout="wide")
st.title("🧠 Social Curiosity Agent")

# =========================
# layout
# =========================
left, right = st.columns([1, 2])

# =========================
# LEFT: CONTROLS
# =========================
with left:
    st.markdown("## 🎛 Agent Parameters")

    temperature = st.slider( "Exploration Temperature", 0.1, 5.0, 1.0, help="Controls randomness of movement. High = more random exploration." ) 
    beta = st.slider( "Space Curiosity", 0.5, 5.0, 2.0, help="Weight of spatial exploration drive." ) 
    gamma = st.slider( "Social Curiosity", 0.5, 5.0, 2.0, help="Weight of social prediction error signal." ) 
    decay = st.slider( "Boredom Decay", 0.1, 2.0, 0.5, help="How fast repeated visits lose novelty." ) 
    lr = st.slider( "Emotion Learning Rate", 0.05, 1.0, 0.3, help="How fast agent updates belief about other's emotion." ) 

    st.markdown("## 🎭 Environment") 

    volatility = st.slider( "Social Volatility", 0.05, 1.0, 0.4, help="How rapidly the other's emotion changes over time." ) 
    perception_range = st.slider( "Emotion perception Range", 0.5, 5.0, 2.0, help="Distance sensitivity for perceiving social agent." ) 
    social_drive = st.slider( "Social Drive", 0.0, 0.2, 0.05, help="Intrinsic motivation to seek social interaction." ) 
    grid_size = st.slider( "World Size", 5, 12, 7, help="Size of the spatial environment grid." )

    start = st.button("▶ Start Simulation")
    reset = st.button("🔄 Reset")

# =========================
# RIGHT: VISUAL AREA
# =========================
with right:
    heatmap_placeholder = st.empty()
    info_placeholder = st.empty()
    plot_placeholder = st.empty()

# =========================
# INIT
# =========================
if "init" not in st.session_state or st.session_state.init != grid_size:

    st.session_state.init = grid_size

    st.session_state.visit_count = np.zeros((grid_size, grid_size))
    st.session_state.visited = np.zeros((grid_size, grid_size))

    st.session_state.pos = [grid_size // 2, grid_size // 2]

    center = grid_size // 2
    st.session_state.social_pos = [
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
    ]

    st.session_state.true_emotion = 0.5
    st.session_state.pred_emotion = 0.5
    st.session_state.t = 0
    st.session_state.time_since_seen = 0

    st.session_state.pe_space_hist = []
    st.session_state.pe_social_hist = []

# =========================
# RESET
# =========================
if reset:
    st.session_state.visited *= 0
    st.session_state.visit_count *= 0
    st.session_state.pos = [grid_size // 2, grid_size // 2]

    center = grid_size // 2
    st.session_state.social_pos = [
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
        np.clip(center + np.random.randint(-1, 2), 1, grid_size - 2),
    ]

    st.session_state.pred_emotion = 0.5
    st.session_state.t = 0
    st.session_state.time_since_seen = 0

    st.session_state.pe_space_hist = []
    st.session_state.pe_social_hist = []

# =========================
# MODEL
# =========================
def get_moves(pos):
    x, y = pos
    moves = []
    if x > 0: moves.append([x-1, y])
    if x < grid_size-1: moves.append([x+1, y])
    if y > 0: moves.append([x, y-1])
    if y < grid_size-1: moves.append([x, y+1])
    return moves

def pe_space(x, y):
    v = st.session_state.visit_count[x, y]
    return np.exp(-decay * v)

def update_true_emotion():
    st.session_state.t += 1
    st.session_state.true_emotion = 0.5 + volatility * np.sin(0.2 * st.session_state.t)

def choose(moves):

    qs = []

    for x, y in moves:

        s_pe = pe_space(x, y)

        dist = abs(x - st.session_state.social_pos[0]) + abs(y - st.session_state.social_pos[1])
        visibility = np.exp(-dist / perception_range)

        e_pe = abs(st.session_state.pred_emotion - st.session_state.true_emotion) * visibility

        uncertainty_bonus = social_drive * st.session_state.time_since_seen

        q = beta * s_pe + gamma * (e_pe + uncertainty_bonus)
        qs.append(q)

    qs = np.array(qs)
    probs = np.exp((qs - np.max(qs)) / temperature)
    probs /= np.sum(probs)

    return moves[np.random.choice(len(moves), p=probs)]

def step():

    pos = st.session_state.pos
    moves = get_moves(pos)
    new = choose(moves)

    x, y = new

    st.session_state.visit_count[x, y] += 1
    st.session_state.visited[x, y] = 1

    update_true_emotion()

    dist = abs(x - st.session_state.social_pos[0]) + abs(y - st.session_state.social_pos[1])
    visibility = np.exp(-dist / perception_range)

    # ⭐ 修改：始终根据 visibility 更新
    obs = st.session_state.true_emotion
    pred = st.session_state.pred_emotion

    pe_e = abs(pred - obs) * visibility

    # ⭐ 关键：学习率也乘 visibility（距离越远更新越弱）
    st.session_state.pred_emotion += lr * visibility * (obs - pred)

    if visibility > 0.01:
        st.session_state.time_since_seen = 0
    else:
        st.session_state.time_since_seen += 1

    pe_s = pe_space(x, y)

    st.session_state.pe_space_hist.append(pe_s)
    st.session_state.pe_social_hist.append(pe_e)

    st.session_state.pos = new

# =========================
# RUN
# =========================
if start:

    for _ in range(grid_size * grid_size * 3):

        step()

        # =========================
        # HEATMAP
        # =========================
        fig, ax = plt.subplots(figsize=(2, 2))

        data = st.session_state.visit_count.copy()

        data[data == 0] = np.nan

        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="white")

        ax.imshow(data, cmap=cmap)

        ax.scatter(st.session_state.pos[1], st.session_state.pos[0], c="black", s=80)

        ax.scatter(st.session_state.social_pos[1], st.session_state.social_pos[0], c="blue", s=80)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Heatmap", fontsize=9)

        heatmap_placeholder.pyplot(fig, use_container_width=False)

        # =========================
        # INFO
        # =========================
        info_placeholder.write(
            f"Space PE: {st.session_state.pe_space_hist[-1]:.2f} | "
            f"Social PE: {st.session_state.pe_social_hist[-1]:.2f} | "
            f"Pred emotion: {st.session_state.pred_emotion:.2f} | "
            f"True emotion: {st.session_state.true_emotion:.2f}"
        )

        # =========================
        # PE PLOT
        # =========================
        fig2, ax2 = plt.subplots(figsize=(4, 1.5))

        ax2.plot(st.session_state.pe_space_hist, label="space PE", linewidth=0.8)
        ax2.plot(st.session_state.pe_social_hist, label="social PE", linewidth=0.8)

        ax2.set_title("Prediction Errors", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=7)

        plot_placeholder.pyplot(fig2, use_container_width=False)

        time.sleep(0.1)
