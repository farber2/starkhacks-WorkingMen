# Physical Chess Robot Controller (Logic Layer)

This project provides the chess-logic side of a physical chess robot.

It currently handles:
- chess state management
- legal move validation
- Stockfish move selection
- translation from UCI moves to robot-ready command objects
- local beginner coaching via Ollama
- optional local speech output via Piper

It intentionally does **not** include:
- motor control
- hardware drivers
- vision, voice, or GUI

## Project Structure

```text
chess_robot/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── chess_logic/
│   ├── __init__.py
│   ├── coach.py
│   ├── tts.py
│   ├── game_manager.py
│   ├── engine_wrapper.py
│   ├── move_info.py
│   ├── move_translator.py
│   └── command_schema.py
├── tests/
│   ├── test_game_manager.py
│   ├── test_engine_wrapper.py
│   ├── test_move_translator.py
│   └── test_special_moves.py
└── examples/
    └── demo_turn_loop.py
```

## Setup

### 1. Create and activate a virtual environment (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Stockfish

Install a Stockfish binary and make sure it is either:
- available on your `PATH` as `stockfish`, or
- provided via environment variable:

```bash
export STOCKFISH_PATH=/absolute/path/to/stockfish
```

## Run the CLI loop

```bash
python main.py
```

The loop will:
1. prompt for your UCI move (example: `e2e4`)
2. validate and translate to a `RobotMoveCommand`
3. apply your move
4. request and translate an engine move
5. apply the engine move

## Run tests

```bash
python -m pytest -q
```

If your virtual environment is activated, this also works:

```bash
pytest -q
```

## Ollama + Piper tutor integration

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama locally

Run Ollama in a terminal:

```bash
ollama serve
```

Pull or warm up the default model used by the coach (`phi3`):

```bash
ollama run phi3
```

Your Ollama setup can use model storage on:
- `/Volumes/Resto Chesto/ollama_models`

If you need to set it in your shell:

```bash
export OLLAMA_MODELS="/Volumes/Resto Chesto/ollama_models"
```

Verify Ollama is reachable:

```bash
ollama list
```

### 3. Install Piper separately (local TTS)

Piper is a local binary and must be installed separately from `pip` dependencies.
On macOS, install it with Homebrew:

```bash
brew install piper
```

### 4. Add the Amy voice model files

Place both files in this folder:

```text
voice/
  en_US-amy-medium.onnx
  en_US-amy-medium.onnx.json
```

The code expects this default path:
- `voice/en_US-amy-medium.onnx`

### 5. Run the demo with coaching + speech output

```bash
export CHESS_TTS=1
python -m examples.demo_turn_loop
```

The demo keeps using your local chess engine for move generation, uses local Ollama for short beginner-friendly explanations, and can speak each explanation locally with Piper + Amy voice.

## Notes for Future Hardware Integration

The output of `translate_move(...)` is a structured `RobotMoveCommand`, designed so a future hardware module can consume commands without changing chess logic.

## Hackathon Web App (FastAPI + React)

This repo now includes:
- `backend/app.py` (FastAPI wrapper around your existing chess logic)
- `frontend/` (React single-page UI with board, help panel, history, and status)

### 1. Start backend API

```bash
source .venv/bin/activate
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Available endpoints:
- `GET /state`
- `POST /move` with `{"uci":"e2e4"}`
- `POST /help` with `{"speak": true|false}`
- `POST /speak` with `{"text":"..."}`
- `POST /reset`

### 2. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Open:
- `http://127.0.0.1:5173`

Optional API override:

```bash
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

### 3. Piece image assets

Drop a matching PNG set in:

```text
frontend/public/assets/pieces/
```

Expected names:
- `wp.png`, `wn.png`, `wb.png`, `wr.png`, `wq.png`, `wk.png`
- `bp.png`, `bn.png`, `bb.png`, `br.png`, `bq.png`, `bk.png`

If image files are missing, the UI automatically falls back to unicode piece glyphs.
