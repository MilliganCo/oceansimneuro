.PHONY: setup run plot train eval test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

run:
	ocean sim --steps 2000

run-gui:
	ocean sim --gui --delay 0.01

run-gui-no-dialog:
	ocean sim --gui --delay 0.01 --no-interactive

record:
	ocean record --steps 50000 --out replay.npz

train:
	python train_dqn.py

eval:
	python eval_policy.py

test:
	pytest -q

