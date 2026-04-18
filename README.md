# Physical Chess Robot Controller (Logic Layer)

This project provides the chess-logic side of a physical chess robot.

It currently handles:
- chess state management
- legal move validation
- Stockfish move selection
- translation from UCI moves to robot-ready command objects

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

## Notes for Future Hardware Integration

The output of `translate_move(...)` is a structured `RobotMoveCommand`, designed so a future hardware module can consume commands without changing chess logic.
