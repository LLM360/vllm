# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    get_tensor_model_parallel_rank
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.layernorm import rms_norm, fused_add_rms_norm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def permute_to_xllm(x):
    return x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).reshape(*x.shape[:-1], -1)


def permute_to_hf(x):
    return x.reshape(*x.shape[:-1], -1, 2).transpose(-1, -2).reshape(*x.shape[:-1], -1)


@CustomOp.register("grouped_rms_norm")
class XllmRMSNorm(RMSNorm):
    def __init__(self,
                 hidden_size: int,
                 n_groups: int,
                 tp_size: int,
                 num_replicas: int = 1,
                 eps=1e-6):
        """
        XllmRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.n_groups = n_groups
        self.hidden_size = hidden_size
        assert hidden_size % n_groups == 0
        self.weight = nn.Parameter(torch.ones(hidden_size * tp_size // num_replicas))
        self.variance_epsilon = eps

        if tp_size > 1:
            self.tp_weight = self.weight.reshape(
                tp_size // num_replicas, self.hidden_size
            )[get_tensor_model_parallel_rank() // num_replicas]
        else:
            self.tp_weight = self.weight

        # assert self._forward_method == self.forward_native
        assert self._forward_method in [self.forward_native, self.forward_cuda]
        if get_tensor_model_parallel_rank() == 0:
            print(f'{self._forward_method=}')

    def forward_native(self, x, residual=None):
        x = x.reshape(*x.shape[:-1], self.n_groups, -1)
        if residual is not None:
            residual = residual.reshape(x.shape)

        x = self.forward_static(
            x,
            self.variance_epsilon,
            self.hidden_size // self.n_groups,
            x.dtype,
            None,
            residual,
            self.variance_size_override,
        )
        if residual is not None:
            x, residual = x

        x = x.reshape(*x.shape[:-2], -1)
        x = self.tp_weight.data * x

        if residual is None:
            return x
        else:
            residual = residual.reshape(x.shape)
            return x, residual

    def forward_cuda(self, hidden_states, residual=None):
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # if residual is not None:
        #     hidden_states = hidden_states + residual.to(torch.float32)
        #     residual = hidden_states.to(input_dtype)

        hidden_states = hidden_states.reshape(*hidden_states.shape[:-1], self.n_groups, -1)
        if residual is not None:
            residual = residual.reshape(hidden_states.shape)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if residual is not None:
            hidden_states, residual = fused_add_rms_norm(
                x=hidden_states,
                residual=residual,
                weight=torch.ones(
                    hidden_states.shape[-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype),
                variance_epsilon=self.variance_epsilon)
        else:
            hidden_states = rms_norm(
                x=hidden_states,
                weight=torch.ones(
                    hidden_states.shape[-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype),
                variance_epsilon=self.variance_epsilon)

        hidden_states = hidden_states.reshape(*hidden_states.shape[:-2], -1)
        hidden_states = self.tp_weight * hidden_states

        if residual is None:
            return hidden_states
        else:
            residual = residual.reshape(hidden_states.shape)
            return hidden_states, residual


class XllmMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class XllmSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=config.moe_gate_bias,
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            n_shared_experts=config.num_shared_experts,
            top_k=config.num_experts_per_tok,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func=config.router_score_func,
            e_score_correction_bias=self.gate.bias,
            routed_scaling_factor=config.router_scaling_factor,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            routing_method_type=None # RoutingMethodType.Renormalize,
        )

        self.num_shared_experts = config.num_shared_experts
        if config.num_shared_experts > 0:
            self.shared_experts = XllmMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "XllmSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        if self.num_shared_experts > 0:
            final_hidden_states = (
                    final_hidden_states + self.shared_experts(hidden_states))

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class XllmAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        query_key_norm: bool,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rope_head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.rope_head_dim = rope_head_dim or self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.num_kv_head_replicas = self.qkv_proj.num_kv_head_replicas

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.query_key_norm = query_key_norm
        if self.query_key_norm:
            # self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            # self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.q_norm = XllmRMSNorm(
                hidden_size=self.num_heads * self.head_dim,
                n_groups=self.num_heads,
                tp_size=tp_size,
                eps=rms_norm_eps)
            self.k_norm = XllmRMSNorm(
                hidden_size=self.num_kv_heads * self.head_dim,
                n_groups=self.num_kv_heads,
                tp_size=tp_size,
                num_replicas=self.num_kv_head_replicas,
                eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Add qk-norm
        if self.query_key_norm:
            # q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            # q_by_head = self.q_norm(q_by_head)
            # q = q_by_head.view(q.shape)
            #
            # k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            # k_by_head = self.k_norm(k_by_head)
            # k = k_by_head.view(k.shape)
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope_head_dim == self.head_dim:
            q, k = self.rotary_emb(positions, q, k)
        else:
            tp_size, tp_rank = self.tp_size, self.tp_rank
            q_ = tensor_model_parallel_all_gather(q.contiguous())
            k_ = tensor_model_parallel_all_gather(k.contiguous())
            q_ = q_.reshape(*q_.shape[:-1], self.total_num_heads, self.head_dim)
            k_ = k_.reshape(
                *k_.shape[:-1], self.total_num_kv_heads * self.num_kv_head_replicas, self.head_dim
            )[..., ::self.num_kv_head_replicas, :]

            q_rope, q_nope = torch.split(
                permute_to_xllm(q_),
                split_size_or_sections=[self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1)
            k_rope, k_nope = torch.split(
                permute_to_xllm(k_),
                split_size_or_sections=[self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1)

            q_rope, k_rope = self.rotary_emb(
                positions, permute_to_hf(q_rope), permute_to_hf(k_rope))

            q_ = permute_to_hf(torch.cat(
                [permute_to_xllm(q_rope), q_nope], dim=-1)).reshape(*q_.shape[:-2], -1)
            k_ = permute_to_hf(torch.cat(
                [permute_to_xllm(k_rope), k_nope], dim=-1)).reshape(*k_.shape[:-2], -1)

            q = q_.split(q_.shape[-1] // tp_size, dim=-1)[tp_rank]
            k = k_.split(
                k_.shape[-1] // (tp_size // self.num_kv_head_replicas), dim=-1
            )[tp_rank // self.num_kv_head_replicas]

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class XllmDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = XllmAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            query_key_norm=config.query_key_norm,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_head_dim=getattr(config, "rope_head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = XllmSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = XllmMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        # self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm(
        #     config.hidden_size, eps=config.rms_norm_eps
        # )
        assert config.hidden_size % config.layernorm_num_groups == 0
        self.input_layernorm = XllmRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            tp_size=1,
            eps=config.rms_norm_eps)
        self.post_attention_layernorm = XllmRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            tp_size=1,
            eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class XllmModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: XllmDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        # self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        assert config.hidden_size % config.layernorm_num_groups == 0
        self.norm = XllmRMSNorm(
            hidden_size=config.hidden_size,
            n_groups=config.layernorm_num_groups,
            tp_size=1,
            eps=config.rms_norm_eps)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # Collect auxiliary hidden states if specified
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            # ".bias",
            # "_bias",
            # ".k_scale",
            # "_k_scale",
            # ".v_scale",
            # "_v_scale",
            # ".weight_scale",
            # "_weight_scale",
            # ".input_scale",
            # "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class XllmForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when XllmMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = XllmModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, XllmDecoderLayer)
            if isinstance(layer.mlp, XllmSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        # if example_layer is None:
        #     raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0

        if example_layer is not None:
            self.num_logical_experts = example_layer.n_logical_experts
            self.num_physical_experts = example_layer.n_physical_experts
            self.num_local_physical_experts = example_layer.n_local_physical_experts
            self.num_routed_experts = example_layer.n_routed_experts
            self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, XllmSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()