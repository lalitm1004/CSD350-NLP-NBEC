# Running Localy
1. Clone Repository

2. Install Dependencies
```bash
uv sync
# or
pip3 install -r requirements.txt
```

3. Download Dataset
```bash
uv run download_dataset.py
# or
python3 download_dataset.py
```

4. Train Model
```bash
uv run src/train_model.py
# or
python3 src/train_model.py
```

5. Run REPL
```bash
uv run src/main.py
# or
python3 src/main.py
```