Now its DPSGD with peer sampling (for a fair evaluation against IDCA)
In the project root dir

```
python -m venv .venv/decentralizepy
source .venv/decentralizepy/bin/activate
```
```
python tutorial/IDCA/download_MNIST_dataset.py
./tutorial/IDCA/run_IDCA.sh
```
logs in eval/data/<date-time>

if get zmq error -> port probably full, change offset in config