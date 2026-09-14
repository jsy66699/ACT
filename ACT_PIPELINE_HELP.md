# ACT Pipeline Help

This file is a Markdown version of the `python -m act.pipeline --help` output.

## Basic Command

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --help
```

Equivalent module form:

```bash
python -m act.pipeline --help
```

## Description

```text
ACT Pipeline: Inference-based whitebox fuzzing for neural networks
```

## Usage

```text
python -m act.pipeline [-h]
                       (--list | --search QUERY | --info NAME | --download NAME | --list-downloaded | --fuzz | --verify TARGET | --validate-verifier | --list-verifications)
                       [--creator {vnnlib,torchvision}] [--category CATEGORY] [--max-instances MAX_INSTANCES]
                       [--dataset DATASET] [--model MODEL] [--num-samples NUM_SAMPLES]
                       [--iterations ITERATIONS] [--timeout TIMEOUT] [--output OUTPUT] [--no-save]
                       [--report-interval REPORT_INTERVAL] [--strict-mode] [--trace-level {0,1,2,3}]
                       [--trace-sample N] [--trace-storage {hdf5,json}] [--trace-output TRACE_OUTPUT]
                       [--mode {counterexample,bounds,comprehensive}] [--networks NETWORKS]
                       [--solvers SOLVERS [SOLVERS ...]] [--tf-modes TF_MODES [TF_MODES ...]]
                       [--input-samples SAMPLES] [--per-neuron-config PRESET|ATOL,RTOL,TOPK]
                       [--batch-sizes B1,B2,...] [--ignore-errors] [--device {cpu,cuda,gpu,mps}]
                       [--dtype {float32,float64}]
```

## Main Options

| Option | Meaning |
|---|---|
| `-h`, `--help` | Show help message and exit. |
| `--list`, `-l` | List available datasets/categories. |
| `--search QUERY`, `-s QUERY` | Search for datasets/categories. |
| `--info NAME`, `-i NAME` | Show detailed information. |
| `--download NAME`, `-d NAME` | Download dataset/category. |
| `--list-downloaded` | List downloaded data-model pairs. |
| `--fuzz`, `-f` | Run ACTFuzzer. |
| `--verify TARGET` | Run verification tests: `act2torch`, `torch2act`, or `all`. |
| `--validate-verifier` | Run verifier validation: counterexample and bounds checking. |
| `--list-verifications` | List available verification tests. |
| `--creator {vnnlib,torchvision}`, `-c {vnnlib,torchvision}` | Spec creator. Default: `vnnlib`. |
| `--device {cpu,cuda,gpu,mps}` | Device to use for computation. Default: best available. |
| `--dtype {float32,float64}` | Default dtype for tensors. Default: `float64`. |

## VNNLIB Options

| Option | Meaning |
|---|---|
| `--category CATEGORY` | VNNLIB category to fuzz, for example `acasxu_2023`. |
| `--max-instances MAX_INSTANCES` | Max VNNLIB instances to load. Default: `10`. |

## TorchVision Options

| Option | Meaning |
|---|---|
| `--dataset DATASET` | TorchVision dataset to fuzz, for example `MNIST`. |
| `--model MODEL` | TorchVision model to fuzz, for example `simple_cnn`. |
| `--num-samples NUM_SAMPLES` | Number of samples to load. Default: `10`. |

## Fuzzing Options

| Option | Meaning |
|---|---|
| `--iterations ITERATIONS` | Max fuzzing iterations. Default: `10000`. |
| `--timeout TIMEOUT` | Timeout in seconds. Default: `3600`. |
| `--output OUTPUT` | Output directory. Default: `fuzzing_results`. |
| `--no-save` | Do not save counterexamples to disk. |
| `--report-interval REPORT_INTERVAL` | Report progress every N iterations. Default: `100`. |
| `--strict-mode` | Enable strict mode: raise errors on input/output constraint violations. Default: `False`. |

## Execution Tracing Options

| Option | Meaning |
|---|---|
| `--trace-level {0,1,2,3}` | Tracing detail level. `0` = disabled, `1` = basic, `2` = full, `3` = debug. |
| `--trace-sample N` | Capture every Nth iteration. Default: `1`, meaning all iterations. |
| `--trace-storage {hdf5,json}` | Storage backend. `json` is readable text, `hdf5` is binary/compressed. |
| `--trace-output TRACE_OUTPUT` | Custom trace output path. Default: `<output-dir>/traces.{hdf5|json}`. |

## Validation Options

| Option | Meaning |
|---|---|
| `--mode {counterexample,bounds,comprehensive}` | Validation mode. Default: `comprehensive`. |
| `--networks NETWORKS` | Comma-separated list of networks to validate. Default: all. |
| `--solvers SOLVERS [SOLVERS ...]` | Solvers for Level 1 validation. Default: `gurobi torchlp`. |
| `--tf-modes TF_MODES [TF_MODES ...]` | Transfer function modes for Level 2 bounds validation: `interval`, `hybridz`, `dual`. Default: `interval`. |
| `--input-samples SAMPLES` | Number of input samples for Level 2 bounds validation. Default: `10`. |
| `--per-neuron-config PRESET\|ATOL,RTOL,TOPK` | Per-neuron bounds preset: `default`, `strict`, `loose`, or a triplet such as `1e-6,0.0,15`. |
| `--batch-sizes B1,B2,...` | Batch sizes to validate, for example `1,4`. Use `none` for the network native batch. |
| `--ignore-errors` | Always exit with status 0, ignoring failures and errors for CI. |

## Examples

### List Available VNNLIB Categories

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --list
```

### Search Benchmarks

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --search acas
```

### Show Category Information

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --info acasxu_2023
```

### Download Data-Model Pairs

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --download acasxu_2023
```

### List Downloaded Pairs

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --list-downloaded
```

### Fuzz VNNLIB Benchmark

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000
```

### Fuzz TorchVision Dataset

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --fuzz --creator torchvision --dataset MNIST
```

### Run Verification Tests

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --verify act2torch --device cpu
```

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --verify torch2act --device cpu
```

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --verify all --device cpu
```

### Run Verifier Validation

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --validate-verifier --device cpu --dtype float64
```

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --validate-verifier --mode counterexample
```

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --validate-verifier --mode bounds --input-samples 20
```

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --validate-verifier --mode bounds --per-neuron-config strict
```

```powershell
& "D:\ProgramFile\Anaconda\envs\act-py312\python.exe" -m act.pipeline --validate-verifier --mode bounds --per-neuron-config 1e-6,0.0,15
```

## Notes

If Windows encoding causes YAML or Unicode decoding errors, set UTF-8 mode in the current PowerShell session:

```powershell
$env:PYTHONUTF8="1"
```

Then rerun the ACT command.
