# Robot Boxing — Reinforcement Learning Game

Human vs AI boxing game for **Machine Learning Lab** (Assignment 04). A DQN agent trained in a custom Gymnasium environment plays in real time over WebSockets.

## Project structure

```
Robot Game/
├── rl/                        # Layer 1 — RL core
│   ├── robot_boxing_env.py    # RobotBoxing-v1 environment
│   └── dqn.py                 # DQN network & agent
├── train_boxer.py             # Offline training (2000 episodes)
├── robot_boxer.pt             # Trained weights
├── training_performance.png   # Reward vs episode (report)
├── server.py                  # Layer 2 — FastAPI + Socket.IO
├── index.html                 # Layer 3 — UI shell
├── app.js                     # Layer 3 — Canvas game client
└── requirements.txt
```

## Quick start

### 1. Install dependencies

```powershell
cd "c:\Users\DELL\OneDrive\Desktop\Robot Game"
pip install -r requirements.txt
```

### 2. Train the AI (if `robot_boxer.pt` is missing)

```powershell
python -u train_boxer.py
```

### 3. Start the web server

```powershell
uvicorn server:socket_app --host 0.0.0.0 --port 8000 --reload
```

### 4. Play

Open **http://localhost:8000** in your browser, click **Start Match**, and fight the robot.

| Control | Action |
|---------|--------|
| `A` / `←` | Step left |
| `D` / `→` | Step right |
| `Space` | Jab |
| `S` / `↓` | Block |

## Assignment checklist

- [x] Human vs AI playable game  
- [x] DQN reinforcement learning  
- [x] Intentional reward design (+10 hit, -5 hurt, -1 miss, +3 block)  
- [x] Score tracking (hits displayed on end screen)  
- [x] Web deployment (FastAPI + Socket.IO)  
- [x] Training graph: `training_performance.png`  

## Report assets

- **Algorithm:** DQN with experience replay and ε-greedy decay  
- **Training plot:** `training_performance.png`  
- **Demo:** Record gameplay at `http://localhost:8000`  
