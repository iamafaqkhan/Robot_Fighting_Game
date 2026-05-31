# Robot Boxing — Reinforcement Learning Game

Human vs AI boxing game for **Machine Learning Lab** (Assignment 04). A DQN agent trained in a custom Gymnasium environment plays in real time via **HTTP POST** to a Vercel serverless function.

## Project structure

```
Robot Game/
├── api/
│   └── predict.py             # Vercel serverless — POST /api/predict
├── rl/
│   ├── robot_boxing_env.py    # RobotBoxing-v1 environment
│   ├── dqn.py                 # DQN network & agent
│   └── inference.py           # Shared model cache + predict
├── train_boxer.py             # Offline training (2000 episodes)
├── robot_boxer.pt             # Trained weights (commit to Git!)
├── training_performance.png   # Reward vs episode (report)
├── server.py                  # Optional local dev server
├── vercel.json                # Vercel routes & function config
├── index.html                 # Frontend UI
├── app.js                     # Canvas game + fetch() sync loop
├── requirements.txt           # Vercel (CPU torch + numpy)
└── requirements-train.txt     # Local training extras
```

## Deploy to Vercel (one-click)

### 1. Push to GitHub

Ensure these files are committed (especially **`robot_boxer.pt`**):

```powershell
git add .
git commit -m "Robot Boxing — Vercel serverless deployment"
git push origin main
```

### 2. Import on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import your GitHub repository
3. Framework preset: **Other** (static + Python functions auto-detected)
4. Deploy

Your live URL will be `https://<project>.vercel.app`.

### 3. First request (cold start)

The first `/api/predict` call may take **10–30 seconds** while PyTorch loads. Later requests on a warm instance are fast. The UI shows **API Warming Up…** until the health check succeeds.

> **Note:** CPU-only PyTorch is ~150MB. If the deploy fails on size limits, upgrade to Vercel Pro or contact your instructor about an alternative host (Railway, Render).

## Local development

### Train the AI (if needed)

```powershell
pip install -r requirements-train.txt
python -u train_boxer.py
```

### Option A — Vercel CLI (matches production)

```powershell
npm i -g vercel
vercel dev
```

Open the URL shown (usually `http://localhost:3000`).

### Option B — FastAPI local server

```powershell
pip install -r requirements-train.txt
python server.py
```

Open **http://localhost:8000**.

## Controls

| Key | Action |
|-----|--------|
| `A` / `←` | Step left |
| `D` / `→` | Step right |
| `Space` | Jab |
| `S` / `↓` | Block |

## API

**POST** `/api/predict`

```json
{
  "player_x": 0.25,
  "ai_x": 0.75,
  "player_health": 100,
  "ai_health": 100,
  "player_stamina": 100,
  "ai_stamina": 100,
  "player_action": "jab"
}
```

**Response:** `{ "action": "block" }`

## Assignment checklist

- [x] Human vs AI playable game  
- [x] DQN reinforcement learning  
- [x] Intentional reward design  
- [x] Score tracking  
- [x] Web deployment (Vercel serverless)  
- [x] Training graph: `training_performance.png`  

## Report assets

- **Algorithm:** DQN with experience replay and ε-greedy decay  
- **Training plot:** `training_performance.png`  
- **Live demo:** Your Vercel deployment URL  
