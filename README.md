# RCDRL

Core implementation of **Deadline-Constrained Dynamic Workflow Scheduling Can be Cast as a Representation Learning Problem** (NeurIPS 2026).

RCDRL learns action-conditioned, deadline-aware representations for safe-first workflow scheduling and VM-cost optimization.

## Environment and integration

RCDRL uses the workflow-scheduling simulator and datasets from [GATES](https://github.com/yashenCS/GATES). During training and testing, each workflow's deadline is dynamically set following the paper's experimental setup.

## Training and evaluation

Once integrated, run from the environment project's root:

```bash
python -m algorithms.rcdrl.main --seed 10
python -m algorithms.rcdrl.eval --log_path <run_directory> --wf_size M
```

Configure experiments in [`workflow_scheduling_rcdrl.yaml`](algorithms/rcdrl/config/workflow_scheduling_rcdrl.yaml). Training selects a checkpoint using validation data; evaluation uses that checkpoint on S, M, or L workflows.
