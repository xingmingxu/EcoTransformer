import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.models.gpt2.modeling_gpt2 import (GPT2Attention,
                                                    GPT2MLP,
                                                    GPT2Block,
                                                    GPT2SequenceSummary,
                                                    GPT2PreTrainedModel,
                                                    GPT2Model,
                                                    GPT2LMHeadModel,
                                                    GPT2PreTrainedModel,
                                                    GPT2ForSequenceClassification
                                                    )
from transformers.pytorch_utils import Conv1D

from typing import Callable, Optional, Tuple, Union

logger = logging.get_logger(__name__)

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.generation import GenerationMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.utils import logging

# Library functions

MAX_LENGTH=128

logger = logging.get_logger(__name__)


def generate_tensor_c(A, B):
  """
  Generates tensor C based on the given formula.

  Args:
    A: A torch tensor of shape (i, j, k, d).
    B: A torch tensor of shape (i, j, l, d).

  Returns:
    A torch tensor C of shape (i, k, l).
  """
  # Get dimensions of A and B
  i_dim, j_dim, k_dim, d_dim = A.shape
  difference = torch.abs(A.unsqueeze(3) - B.unsqueeze(2)) # Unsqueeze adds a dimension

  # Sum over the last dimension (d)
  C = torch.sum(difference, dim=-1)

  return C

def generate_tensor_c_batch(A, B, batch_size=8):

  # Get dimensions of A and B
  i_dim, j_dim, k_dim, d_dim = A.shape
  l_dim = B.shape[2]

  result = torch.empty(i_dim, j_dim, k_dim, l_dim, device=A.device, dtype=A.dtype)

  for i in range(0, k_dim, batch_size):
    end = min(i + batch_size, k_dim)
    A_batch = A[:,:,i:end,:]  # shape (i,j, batch, d)
    difference = torch.abs(A_batch[:, :, :, None, :] - B[:, :, None, :, :]) # (i, j, batch, l, d)
    dist_batch = torch.sum(difference, dim=-1)
    result[:, :, i:end, :] = dist_batch

  return result

L1_LAMBDA=-1

def L1_eager_attention_forward(module, query, key, value, attention_mask, lambd=-L1_LAMBDA, head_mask=None, **kwargs):

    """
    Performs forward for eager attention using L1 normalization. 
    
    Args:
    * lambd: tuning parameter lambda
    """

    attn_weights = generate_tensor_c_batch(query, key) * (lambd)
    #attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    ###causal mask is always used in decoder only transformer
    ####GPT-2 doesn't use encoder-decoder attention (unlike T5 or BART).
    ####It uses self-attention, meaning each token attends within the same input sequence.
    ####Therefore, query, key, and value vectors are all computed from the same sequence — making their lengths equal.

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
      # Apply the attention mask
      causal_mask = attention_mask[:, :, :, : key.shape[-2]]
      attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    # Mask heads if we want to
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


# Patch in model
class L1GPT2Attention(GPT2Attention):

    def __init__(self, config, layer_idx=None, is_cross_attention=False):

        # Inherit from GPT2Attention
        super().__init__(config)

        # Completely change the init function (for safety)
        self.config = config

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim: # Corrected self.nit to self.num_heads
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn


        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim) # mapping for the key, value, and l matrix (utility matrix)
        self.q_attn = Conv1D(self.embed_dim, self.embed_dim) # mapping for the query matrix

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

    # Moved forward method outside of __init__
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # FIX THIS LATER
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                if is_cross_attention:
                    past_key_value = past_key_value.cross_attention_cache
                else:
                    past_key_value = past_key_value.self_attention_cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs=cache_kwargs
            )

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"

        # Define the new attention_interface as the custom L1_eager_attention_forward
        attention_interface: Callable = L1_eager_attention_forward

        # XM: changing this to always be eager
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):

        using_eager = True
        logger.warning_once(
            "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
            'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
            # else:
            #     print("else")
            #     # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
            #     # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
            #     # not necessarily to eager (if mentioned options are provided).
            #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            # using eager
            print("using eager_attention_forward")
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            dropout = kwargs.pop('dropout', self.attn_dropout.p if self.training else 0.0)
            attn_output, attn_weights = attention_interface(
                self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                dropout=dropout,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


# Create L1 "clones" inheriting from GPT2 equivalents to patch in our changes to the forward function.

class L1GPT2Block(GPT2Block):

    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = L1GPT2Attention(config,
                                        layer_idx=layer_idx)

class L1GPT2Model(GPT2Model):

    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([L1GPT2Block(config, layer_idx=i)
        for i in range(config.num_hidden_layers)])


class L1GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.transformer = L1GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


class L1GPT2ForSequenceClassification(GPT2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = L1GPT2Model(config)