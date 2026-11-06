from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GACEEncoder(nn.Module):


    DEFAULTS = {
        "output_dim": 128,
        "d_model": 16,
        "dropout": 0.1,
        "dag_node_fea_size": 10,
        "active_wf_fea_size": 12,
        "candidate_action_fea_size": 16,
        "gat_heads": 2,
        "att_heads": 2,
        "att_en_layers": 2,
        "d_ff": 128,
    }

    def __init__(self, config: dict):
        super().__init__()
        cfg = dict(self.DEFAULTS)
        cfg.update(config)
        self.cfg = cfg

        self.output_dim = int(cfg["output_dim"])
        self.d_model = int(cfg["d_model"])
        self.dropout_p = float(cfg.get("dropout", 0.1))
        self.dag_node_fea_size = int(cfg["dag_node_fea_size"])
        self.active_wf_fea_size = int(cfg["active_wf_fea_size"])
        self.candidate_action_fea_size = int(cfg["candidate_action_fea_size"])
        self.gat_heads = int(cfg["gat_heads"])
        self.att_heads = int(cfg["att_heads"])
        self.att_en_layers = int(cfg["att_en_layers"])
        self.d_ff = int(cfg["d_ff"])

        if self.d_model % self.att_heads != 0:
            raise ValueError(
                f"GACE d_model must be divisible by att_heads, got d_model={self.d_model}, "
                f"att_heads={self.att_heads}."
            )

        try:
            from torch_geometric.nn import GATConv
        except Exception as exc:
            raise ImportError(
                "GACE requires torch_geometric for the current-workflow DAG GAT block."
            ) from exc


        self.gat_dag_layer1 = GATConv(
            self.dag_node_fea_size,
            self.d_model,
            heads=self.gat_heads,
            concat=True,
            dropout=self.dropout_p,
        )
        self.gat_dag_layer2 = GATConv(
            self.gat_heads * self.d_model,
            self.d_model,
            heads=1,
            concat=False,
            dropout=self.dropout_p,
        )
        self.dag_pool = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
        )


        self.active_wf_item_encoder = nn.Sequential(
            nn.Linear(self.active_wf_fea_size, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
        )
        self.active_wf_pool = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
        )


        self.global_encoder = nn.Sequential(
            nn.LazyLinear(self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
        )


        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * self.d_model + self.candidate_action_fea_size, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model),
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.att_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_p,
            batch_first=True,
            activation="relu",
        )
        self.action_self_attention = nn.TransformerEncoder(encoder_layer, num_layers=self.att_en_layers)


        self.output_projector = nn.Sequential(
            nn.Linear(5 * self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_ff, self.output_dim),
        )

    @staticmethod
    def _as_float_tensor(value, device, name: str, ndim: Optional[int] = None) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device)
        if ndim is not None and tensor.dim() != ndim:
            raise ValueError(f"{name} must be {ndim}-D, but got shape {tuple(tensor.shape)}.")
        return tensor

    @staticmethod
    def _edge_index_from_dag(dag, device) -> torch.Tensor:

        edges = []
        for u, v in dag.edges():
            edges.append((int(u), int(v)))
            edges.append((int(v), int(u)))
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _fallback_dag_features(self, device, state) -> torch.Tensor:

        import networkx as nx

        dag = state.dag
        node_count = dag.number_of_nodes()
        adj = torch.tensor(nx.adjacency_matrix(dag).todense(), dtype=torch.float32, device=device)
        predecessor = adj.sum(dim=0)
        successor = adj.sum(dim=1)
        rows = []
        for node, data in dag.nodes(data=True):
            process_time = float(data.get("processTime", 0.0))
            task_size = float(data.get("size", 0.0))
            sub_deadline = float(data.get("sub_deadline", 0.0))
            scheduled = float(data.get("scheduled", 0.0))
            rows.append(
                [
                    process_time,
                    task_size,
                    float(predecessor[int(node)].item()) / max(node_count - 1, 1),
                    float(successor[int(node)].item()) / max(node_count - 1, 1),
                    1.0 if int(node) == int(state.node_id) else 0.0,
                    1.0 if int(node) == int(state.node_id) else 0.0,
                    1.0 if scheduled > 0.0 and int(node) != int(state.node_id) else 0.0,
                    0.0,
                    sub_deadline,
                    process_time,
                ]
            )
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _encode_dag(self, device, state) -> Dict[str, torch.Tensor]:


        dag_x_np = getattr(state, "gace_dag_node_features", None)
        if dag_x_np is None:
            dag_x = self._fallback_dag_features(device, state)
        else:
            dag_x = self._as_float_tensor(dag_x_np, device, "state.gace_dag_node_features", ndim=2)
        if dag_x.size(1) != self.dag_node_fea_size:
            raise ValueError(
                f"GACE DAG node feature dimension mismatch: expected {self.dag_node_fea_size}, "
                f"got {dag_x.size(1)}."
            )

        edge_index = self._edge_index_from_dag(state.dag, device)
        dag_h = F.elu(self.gat_dag_layer1(dag_x, edge_index))
        dag_h = F.elu(self.gat_dag_layer2(dag_h, edge_index))

        ready_idx = int(state.node_id)
        if ready_idx < 0 or ready_idx >= dag_h.size(0):
            raise ValueError(f"state.node_id={ready_idx} is outside DAG node range [0, {dag_h.size(0)}).")
        h_task = dag_h[ready_idx:ready_idx + 1]
        h_cur_wf = self.dag_pool(torch.cat([dag_h.mean(dim=0, keepdim=True), dag_h.max(dim=0, keepdim=True).values], dim=-1))
        return {"h_task": h_task, "h_cur_wf": h_cur_wf, "dag_data": {"x": dag_x, "edge_index": edge_index}}

    def _encode_active_workflows(self, device, state) -> torch.Tensor:


        active_np = getattr(state, "active_wf_features", None)
        if active_np is None:
            active_x = torch.zeros((1, self.active_wf_fea_size), dtype=torch.float32, device=device)
        else:
            active_x = self._as_float_tensor(active_np, device, "state.active_wf_features", ndim=2)
            if active_x.numel() == 0:
                active_x = torch.zeros((1, self.active_wf_fea_size), dtype=torch.float32, device=device)
        if active_x.size(1) != self.active_wf_fea_size:
            raise ValueError(
                f"GACE active workflow feature dimension mismatch: expected {self.active_wf_fea_size}, "
                f"got {active_x.size(1)}."
            )
        item_h = self.active_wf_item_encoder(active_x)
        pooled = torch.cat([item_h.mean(dim=0, keepdim=True), item_h.max(dim=0, keepdim=True).values], dim=-1)
        return self.active_wf_pool(pooled)

    def _encode_global(self, device, state) -> torch.Tensor:


        global_np = getattr(state, "global_features", None)
        if global_np is None:
            global_x = torch.zeros((1, 6), dtype=torch.float32, device=device)
        else:
            global_x = self._as_float_tensor(global_np, device, "state.global_features")
            if global_x.dim() == 1:
                global_x = global_x.unsqueeze(0)
        if global_x.dim() != 2 or global_x.size(0) != 1:
            raise ValueError(f"state.global_features must have shape [1, G] or [G], got {tuple(global_x.shape)}.")
        return self.global_encoder(global_x)

    def _candidate_action_features(self, device, state, action_num: int) -> torch.Tensor:


        cand_np = getattr(state, "candidate_action_features", None)
        if cand_np is None:
            ob = self._as_float_tensor(state.ob, device, "state.ob", ndim=2)
            vm = self._as_float_tensor(state.vm_features, device, "state.vm_features", ndim=2)
            zeros_needed = self.candidate_action_fea_size - min(ob.size(1) + vm.size(1), self.candidate_action_fea_size)
            candidate_x = torch.cat([vm, ob, torch.zeros((action_num, max(zeros_needed, 0)), device=device)], dim=-1)
            candidate_x = candidate_x[:, :self.candidate_action_fea_size]
        else:
            candidate_x = self._as_float_tensor(cand_np, device, "state.candidate_action_features", ndim=2)
        if candidate_x.size(0) != action_num:
            raise ValueError(
                "state.candidate_action_features and state.ob must have the same number of candidate actions, "
                f"but got {candidate_x.size(0)} and {action_num}."
            )
        if candidate_x.size(1) != self.candidate_action_fea_size:
            raise ValueError(
                f"GACE candidate action feature dimension mismatch: expected {self.candidate_action_fea_size}, "
                f"got {candidate_x.size(1)}."
            )
        return candidate_x

    def forward(self, device, state) -> Dict[str, torch.Tensor]:

        ob = self._as_float_tensor(state.ob, device, "state.ob", ndim=2)
        action_num = int(ob.size(0))

        dag_outputs = self._encode_dag(device, state)
        h_task = dag_outputs["h_task"]
        h_cur_wf = dag_outputs["h_cur_wf"]
        h_active = self._encode_active_workflows(device, state)
        h_global = self._encode_global(device, state)
        candidate_x = self._candidate_action_features(device, state, action_num)


        h_task_all = h_task.expand(action_num, -1)
        h_cur_wf_all = h_cur_wf.expand(action_num, -1)
        h_active_all = h_active.expand(action_num, -1)
        h_global_all = h_global.expand(action_num, -1)
        pair_input = torch.cat([h_task_all, h_cur_wf_all, h_active_all, h_global_all, candidate_x], dim=-1)
        action_tokens = self.pair_encoder(pair_input)


        contextual_tokens = self.action_self_attention(action_tokens.unsqueeze(0)).squeeze(0)


        output_input = torch.cat([contextual_tokens, h_task_all, h_cur_wf_all, h_active_all, h_global_all], dim=-1)
        z_all = self.output_projector(output_input)

        ready_task_overall = h_task
        return {
            "z_all": z_all,
            "dag_data": dag_outputs["dag_data"],
            "ready_task_overall": ready_task_overall,
        }
