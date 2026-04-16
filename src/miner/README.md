# IOTA Miner

The IOTA miner participates in distributed training by processing activations on an assigned layer of a neural network. It connects to the orchestrator to receive work, communicates with peers via P2P, and reports telemetry.

## Quick Start

### From source

```bash
uv sync --project src/miner --frozen --dev
uv run --project src/miner python main_pool.py
```

### From pre-built bundle

Download the latest macOS bundle from [GitHub Releases](https://github.com/macrocosm-os/iota_mvp/releases) (tags prefixed with `miner-v`):

```bash
gh release download --repo macrocosm-os/iota_mvp --pattern '*macos-arm64.tar.gz'
tar -xzf miner-v*.tar.gz
./main_pool
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--auto-start` | Auto-start mining with saved configuration |
| `--wallet <name>` | Coldkey wallet name |
| `--hotkey <name>` | Hotkey name |
| `--payout-coldkey <key>` | Custom payout address or `creators` (default) |
| `--dashboard` / `--no-dashboard` | Toggle the terminal dashboard (on by default) |
| `--no-btcli` | Skip bittensor CLI integration (offline wallet mode) |

## Architecture

```
main_pool.py                  Entry point (CLI args, wallet config, GPU cleanup)
  └─ pool/miner.py            Pool miner (extends base Miner)
      └─ new_miner.py         Base Miner (registration, training loop, P2P)
          ├─ training/
          │   ├─ training.py           Forward/backward pass loop
          │   ├─ activation_queue.py   Fetches and queues activations
          │   ├─ activation_cache.py   In-memory activation cache
          │   └─ activation_publisher.py  Publishes outputs to S3
          ├─ pool/
          │   ├─ wallet_utils/         Wallet selection and btcli integration
          │   └─ dashboard/            Rich terminal UI
          ├─ telemetry/                Prometheus metrics and buffered reporting
          └─ utils/                    Partition merging, stats, P2P helpers
```

## How It Works

### 1. Registration

The miner fetches available training runs from the orchestrator, selects the best one, and registers with its P2P node ID. The orchestrator assigns it a layer of the model.

### 2. Training Loop

Once the orchestrator moves to the `TRAINING` phase:

- **Forward pass**: The miner receives activations (from the previous layer via P2P or S3), computes its layer's forward pass, and publishes outputs.
- **Backward pass**: Gradients flow back through the layer. The miner accumulates gradients and runs a local optimizer step when enough are collected.

### 3. Weight Synchronization

Periodically the orchestrator transitions through:

- **WEIGHTS_UPLOADING**: Miners upload their local weights and optimizer state to S3.
- **MERGING_PARTITIONS**: Miners download the merged weights and re-initialize their optimizer.

The loop then returns to training.

### 4. P2P Activation Sharing

Miners share activations directly via the Iroh P2P network, falling back to S3 when peers are unavailable. This reduces latency and orchestrator load.

## Building the Bundle

The miner is packaged as a standalone macOS executable using PyInstaller:

```bash
cd src/miner
uv run pyinstaller main_pool.spec --clean
```

The CI workflow (`.github/workflows/build-miner-bundle.yml`) automates this, including Apple code signing and artifact upload. It triggers on pushes to `main` that change files in `src/miner/`, `shared/common/`, or `shared/subnet/`.

### Versioning

The bundle version is read from `src/miner/pyproject.toml` and combined with the commit SHA to form a tag:

```
miner-v{version}+{short_sha}   e.g. miner-v0.1.0+e9561fb
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `common` | Shared utilities (workspace package) |
| `subnet` | Subnet protocol and API client (workspace package) |
| `psutil` | System and GPU monitoring |
| `speedtest-cli` | Network speed testing |
| `blessed` | Terminal control |
| `platformdirs` | Platform-specific directories |
| `prometheus_client` | Metrics collection |

## Health Check

The miner exposes a health endpoint (default port 9000) at `/health` returning its status, hotkey, assigned layer, and registration state.
