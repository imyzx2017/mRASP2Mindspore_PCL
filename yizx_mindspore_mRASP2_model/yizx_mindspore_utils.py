from typing import Dict, List, NamedTuple, Optional, Callable, Tuple, Any

import os
import sys
import importlib
import math
import warnings
import functools
import logging
import operator

from mindspore.common.tensor import Tensor
from mindspore.ops import Pow
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.nn as nn

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, XavierUniform, Normal, Constant
import mindspore.common.dtype as mstype

import numpy as np
import random
import uuid

from mindspore.ops import Fill, Zeros, NotEqual, CumSum, \
    Concat, matmul, Pad, ExpandDims, Add, Softmax, Dropout, \
    ReduceSum, ReLU, repeat_elements, Softmax, LogSoftmax, ReduceMean, Equal, ReduceAny

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)

logger = logging.getLogger(__name__)

def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        # "gelu_fast",  # deprecated
        # "gelu_accurate",
        "tanh",
        "linear",
    ]

def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor

class LayerNorm(nn.Cell):
    """
    Layer Normalization

    Args:
        normalized_shape: the corresponding shape of the normalized axes
        eps: epsilon, a small number avoiding zero division

    Inputs:
        x: input tensor

    Returns:
        rescaled_output: Tensor, returned tensor after layernorm
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape))
        self.beta = Parameter(initializer('zeros', normalized_shape))
        self.mean = P.ReduceMean(keep_dims=True)
        self.eps = eps

    def construct(self, x):
        mean = self.mean(x, -1)
        variance = self.mean(F.square(x - mean), -1)
        output = (x - mean) / F.sqrt(variance + self.eps)
        rescaled_output = output * self.gamma + self.beta
        return rescaled_output

def gelu():
    return nn.GELU()

def relu():
    return ReLU()

def softmax(x, dim=-1):
    softmax_ops = Softmax(axis=dim)
    return softmax_ops(x)

def log_softmax(x, dim=-1):
    logsoftmax_ops = LogSoftmax(axis=dim)
    return logsoftmax_ops(x)

def Linear(in_features, out_features, bias=True):
    m = nn.Dense(in_features, out_features, bias)
    # nn.init.xavier_uniform_(m.weight)
    # print(m.weight.asnumpy())
    m.weight = Parameter(initializer(XavierUniform(gain=1), [out_features, in_features]))
    if bias:
        m.bias = Parameter(initializer("zeros", out_features))
        # nn.init.constant_(m.bias, 0.0)
    return m

def masked_fill(weights, mask, unsqueeze_ops):
    mask = mask * 1
    unsqueeze_key_padding_mask = unsqueeze_ops(unsqueeze_ops(mask, 1), 2)
    Inversed_unsqueeze_key_padding_mask = 1 - unsqueeze_key_padding_mask

    # make inf matrix shape like attn_output_weights
    tmp_infMatrix = Tensor(np.ones(weights.shape), weights.dtype)
    # using -1e10 to replace -inf, because 0 * -inf = nan
    tmp_infMatrix_inf = fill_with_neg_inf(tmp_infMatrix)
    tmp_infMatrix_inf = tmp_infMatrix_inf * unsqueeze_key_padding_mask
    need2convertinf_attn_output_weights = weights * tmp_infMatrix_inf
    keep_attn_output_weights = weights * Inversed_unsqueeze_key_padding_mask
    attn_output_weights = need2convertinf_attn_output_weights + keep_attn_output_weights
    return attn_output_weights

def mindspore_linear(input, weight, bias=None):
    tens_ops = (input, weight)
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster
        tmp = matmul(input, weight.T)
        ret = bias + tmp
        # ret = torch.addmm(bias, input, weight.T)
    else:
        output = matmul(input, weight.T)
        if bias is not None:
            output += bias
        ret = output
    return ret

def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)

def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return relu()
    elif activation == "gelu":
        return gelu()
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.

    cumsum = CumSum()
    mask = (tensor == padding_idx)
    mask.set_dtype(mstype.int32)
    mask = 1 - mask
    # tmp = CumSum(mask, axis=1).type_as(mask) * mask
    tmp = cumsum(mask, 1) * mask + padding_idx
    return tmp

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    fill = Fill()
    # return fill(mstype.float32, t.shape, float("-inf"))
    # 0 * -inf = Nan !!!!!!!!!!!!
    return fill(mstype.float32, t.shape, float(-1e10))

def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim)
        m.embedding_table = Parameter(initializer(Normal(sigma=258 ** -0.5), [258, 1024]))
        if padding_idx is not None:
            m.embedding_table[padding_idx] = Parameter(initializer(Constant(value=0), [1, 1024]))
            # nn.init.constant_(m.weight[padding_idx], 0)
    else:
        assert 'ERROR'
    return m

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]

def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
            else:
                fairseq_rel_path = os.path.join(
                    os.path.dirname(__file__), "..", args.user_dir
                )
                if os.path.exists(fairseq_rel_path):
                    module_path = fairseq_rel_path
                else:
                    raise FileNotFoundError(module_path)

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            else:
                raise ImportError(
                    "Failed to import --user-dir={} because the corresponding module name "
                    "({}) is not globally unique. Please rename the directory to "
                    "something unique and try again.".format(module_path, module_name)
                )


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)  # , padding_idx)
        self.onnx_trace = False
        padding_idx = 1
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.vocab_size, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.vocab_size, 'Padding_idx must be within num_embeddings'
                padding_idx = self.vocab_size + padding_idx
        self.padding_idx = padding_idx
        if self.padding_idx is not None:
            self.max_positions = self.vocab_size - self.padding_idx - 1
        else:
            self.max_positions = self.vocab_size

    def construct(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."
        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX

                # positions = torch.zeros(
                #     (1, 1), device=input.device, dtype=input.dtype
                # ).fill_(int(self.padding_idx + input.size(1)))
                positions = Tensor(np.zeros(1, 1), dtype=input.dtype)
                positions = Fill(input.dtype, positions.shape, int(self.padding_idx + input.size(1)))

            else:
                positions = make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        tmp_embed = nn.Embedding(258, 1024, padding_idx=self.padding_idx)
        return tmp_embed(positions)


class MindsporeFairseqDropout(nn.Cell):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False
        self.training = True
    def construct(self, x, inplace: bool = False):
        if self.training or self.apply_during_inference:
            msDropoutLayer = nn.Dropout(keep_prob=self.p)
            msDropoutLayer.set_train()
            output = msDropoutLayer(x)
            return output

            # print(self.p)
            # msDropoutOp = Dropout(keep_prob=self.p)
            # out, mask = msDropoutOp(x)
            # return out
        else:
            return x


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state

def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls

@with_incremental_state
class MultiheadAttention(nn.Cell):
    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.training = True

        self.dropout_module = MindsporeFairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = nn.Dense(self.kdim, self.embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(self.vdim, self.embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim, has_bias=bias)

        if add_bias_kv:
            pass
            # self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            # self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            self.k_proj.weight = Parameter(initializer(XavierUniform(gain=1 / math.sqrt(2))
                                                       , [self.kdim, self.embed_dim]))
            self.v_proj.weight = Parameter(initializer(XavierUniform(gain=1 / math.sqrt(2))
                                                      , [self.vdim, self.embed_dim]))
            self.q_proj.weight = Parameter(initializer(XavierUniform(gain=1 / math.sqrt(2))
                                                      , [self.embed_dim, self.embed_dim]))
        else:
            assert 'ERROR reset params'

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def construct(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # print(query.shape, key.shape, key_padding_mask.shape, incremental_state, static_kv)
        static_k = None
        static_v = None

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.shape
        assert embed_dim == self.embed_dim
        assert list(query.shape) == [tgt_len, bsz, embed_dim]

        if (
            not self.onnx_trace
            and not self.tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
        ):
            assert key is not None and value is not None

            in_proj_weight = Tensor(np.array([]), mstype.float32)
            op_concat_axis0 = Concat()
            op_concat_axis1 = Concat(axis = 1)
            in_proj_bias = op_concat_axis0((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias))
            tens_ops = (query, key, value, in_proj_weight, in_proj_bias,
                        self.bias_k, self.bias_v,
                        self.out_proj.weight, self.out_proj.bias)
            tgt_len, bsz, embed_dim = query.shape
            embed_dim_to_check = self.embed_dim
            assert embed_dim == embed_dim_to_check
            # allow MHA to have different sizes for the feature dimension
            assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]

            use_separate_proj_weight = True
            if not use_separate_proj_weight:
                pass
            else:
                q_proj_weight_non_opt = self.q_proj.weight
                len1, len2 = q_proj_weight_non_opt.shape
                assert len1 == embed_dim and len2 == query.shape[-1]

                k_proj_weight_non_opt = self.k_proj.weight
                len1, len2 = k_proj_weight_non_opt.shape
                assert len1 == embed_dim and len2 == key.shape[-1]

                v_proj_weight_non_opt = self.v_proj.weight
                len1, len2 = v_proj_weight_non_opt.shape
                assert len1 == embed_dim and len2 == value.shape[-1]

                if in_proj_bias is not None:
                    q = mindspore_linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
                    k = mindspore_linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
                    v = mindspore_linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
                else:
                    q = mindspore_linear(query, q_proj_weight_non_opt, in_proj_bias)
                    k = mindspore_linear(key, k_proj_weight_non_opt, in_proj_bias)
                    v = mindspore_linear(value, v_proj_weight_non_opt, in_proj_bias)

            q = q * self.scaling
            unsqueeze_ops = ExpandDims()

            if attn_mask is not None:
                assert attn_mask.dtype == mstype.float32 or attn_mask.dtype == mstype.float64 or \
                       attn_mask.dtype == mstype.float16 or attn_mask.dtype == mstype.uint8 or attn_mask.dtype == mstype.bool_, \
                    'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
                if attn_mask.dim() == 2:
                    attn_mask = unsqueeze_ops(attn_mask, 0)
                    if list(attn_mask.shape) != [1, query.shape[0], key.shape[0]]:
                        raise RuntimeError('The size of the 2D attn_mask is not correct.')


            # convert ByteTensor key_padding_mask to bool
            if key_padding_mask is not None and key_padding_mask.dtype == mstype.uint8:
                pass
            if self.bias_k is not None and self.bias_v is not None:

                k = op_concat_axis0([k, self.bias_k.repeat(1, bsz, 1)])
                v = op_concat_axis0([v, self.bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    ops_pad = Pad(((0, 0), (0, 1)))
                    attn_mask = ops_pad(attn_mask)
                if key_padding_mask is not None:
                    ops_pad = Pad(((0, 0), (0, 1)))
                    key_padding_mask = ops_pad(key_padding_mask)

            else:
                assert self.bias_k is None
                assert self.bias_v is None

            q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
            if k is not None:
                k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
            if v is not None:
                v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

            src_len = k.shape[1]
            if key_padding_mask is not None:
                assert key_padding_mask.shape[0] == bsz
                assert key_padding_mask.shape[1] == src_len

            if self.add_zero_attn:
                pass
                # src_len += 1
                # k = op_concat_axis1()

            attn_output_weights = matmul(q, k.transpose(0, 2, 1))
            assert list(attn_output_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

            if attn_mask is not None:
                # [1, 54, 54], 右上角-inf，左下角0矩阵。 dtype=float16
                if attn_mask.dtype == mstype.bool_:
                    assert 'attn_mask type error'
                    # attn_output_weights.masked_fill_(attn_mask, float('-inf'))
                else:
                    attn_output_weights += attn_mask

            if key_padding_mask is not None:
                attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
                ############################
                # add masked_fill ops
                # (attn_output_weights)     # [24, 16, 61, 61]
                # (key_padding_mask.shape)  # [24, 61]  --> unsqueeze [24, 1, 1, 61]

                ###########################################
                # print(key_padding_mask, key_padding_mask.dtype)   # bool Tensor

                # key_padding_mask = key_padding_mask * 1
                #
                # # print(key_padding_mask, key_padding_mask.shape, key_padding_mask.asnumpy().sum())
                # # tmp = key_padding_mask.asnumpy()
                # # print(tmp.sum())
                # # ############################################
                #
                # unsqueeze_key_padding_mask = unsqueeze_ops(unsqueeze_ops(key_padding_mask, 1), 2)
                # # #
                # Inversed_unsqueeze_key_padding_mask = 1 - unsqueeze_key_padding_mask
                # # print(unsqueeze_key_padding_mask.asnumpy().sum(), Inversed_unsqueeze_key_padding_mask.asnumpy().sum())
                #
                #
                # # make inf matrix shape like attn_output_weights
                # tmp_infMatrix = Tensor(np.ones(attn_output_weights.shape), attn_output_weights.dtype)
                # # using -1e10 to replace -inf, because 0 * -inf = nan
                # tmp_infMatrix_inf = fill_with_neg_inf(tmp_infMatrix)
                # tmp_infMatrix_inf = tmp_infMatrix_inf * unsqueeze_key_padding_mask
                # need2convertinf_attn_output_weights = attn_output_weights * tmp_infMatrix_inf
                # keep_attn_output_weights = attn_output_weights * Inversed_unsqueeze_key_padding_mask
                # attn_output_weights = need2convertinf_attn_output_weights + keep_attn_output_weights

                attn_output_weights = masked_fill(attn_output_weights, key_padding_mask, unsqueeze_ops)

                ##############################
                attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)


            softmax_ops = Softmax(axis=-1)
            attn_output_weights = softmax_ops(attn_output_weights)

            msDropoutLayer = nn.Dropout(keep_prob=self.dropout_p)
            if self.training:
                msDropoutLayer.set_train()
            attn_output_weights = msDropoutLayer(attn_output_weights)

            attn_output = matmul(attn_output_weights, v)
            assert list(attn_output.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]
            attn_output = attn_output.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)
            attn_output = mindspore_linear(attn_output, self.out_proj.weight, self.out_proj.bias)

            if need_weights:
                # average attention weights over heads
                attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
                sum_ops = ReduceSum(keep_dims=False)
                return attn_output, sum_ops(attn_output_weights, axis=1) / self.num_heads
            else:
                return attn_output, None

        else:
            unsqueeze_ops = ExpandDims()

            saved_state = None
            if self.self_attention:
                q = self.q_proj(query)
                k = self.k_proj(query)
                v = self.v_proj(query)
            elif self.encoder_decoder_attention:
                # encoder-decoder attention
                q = self.q_proj(query)
                if key is None:
                    assert value is None
                    k = v = None
                else:
                    k = self.k_proj(key)
                    v = self.v_proj(key)
            else:
                assert key is not None and value is not None
                q = self.q_proj(query)
                k = self.k_proj(key)
                v = self.v_proj(value)
            q *= self.scaling
            # print(self.bias_k)  None

            q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
            if k is not None:
                k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)
            if v is not None:
                v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(1, 0, 2)

            assert k is not None
            src_len = k.shape[1]

            # This is part of a workaround to get around fork/join parallelism
            # not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None

            if key_padding_mask is not None:
                assert key_padding_mask.shape[0] == bsz
                assert key_padding_mask.shape[1] == src_len

            # print(self.add_zero_attn) # False

            attn_weights = matmul(q, k.transpose(0, 2, 1))
            attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

            assert list(attn_weights.shape) == [bsz * self.num_heads, tgt_len, src_len]

            # print(attn_weights.shape)  # From DecoderLayer [384, 131, 61]
            if attn_mask is not None:
                attn_mask = unsqueeze_ops(attn_mask, 0)
                if self.onnx_trace:
                    attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                # don't attend to padding symbols
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                if not self.tpu:
                    attn_weights = masked_fill(attn_weights,
                        key_padding_mask,
                        unsqueeze_ops
                    )
                else:
                    attn_weights = attn_weights.transpose(0, 2)
                    attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                    attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if before_softmax:
                return attn_weights, v

            attn_weights_float = softmax(attn_weights, dim=-1)
            attn_weights.set_dtype(attn_weights_float.dtype)
            if not self.dropout_module is None:
                attn_probs = self.dropout_module(attn_weights)
            else:
                attn_probs = attn_weights   # [384, 131, 61]

            assert v is not None
            attn = matmul(attn_probs, v)
            # print(attn.shape) # [384, 131, 61]
            assert list(attn.shape) == [bsz * self.num_heads, tgt_len, self.head_dim]

            if self.onnx_trace and attn.shape[1] == 1:
                # when ONNX tracing a single decoder step (sequence length == 1)
                # the transpose is a no-op copy before view, thus unnecessary
                attn = attn.view(tgt_len, bsz, embed_dim)
            else:
                attn = attn.transpose(1, 0, 2).view(tgt_len, bsz, embed_dim)

            attn = self.out_proj(attn)
            attn_weights: Optional[Tensor] = None
            # print(need_weights, attn_weights)
            if need_weights:
                attn_weights = attn_weights_float.view(
                    bsz, self.num_heads, tgt_len, src_len
                ).transpose(1, 0, 2, 3)
                if not need_head_weights:
                    ops_mean = ReduceMean(keep_dims=False)
                    # average attention weights over heads
                    attn_weights = ops_mean(attn_weights, axis=0)
                    # raise ImportError

            # print(attn.shape, attn_weights.shape)   # [131, 24, 1024], [24, 131, 61]
            return attn, attn_weights


if __name__ == '__main__':
    tmp = LayerNorm(1024)
    tmp = gelu()

    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)

    # ############## Linear Layer Test ##########################
    # np.random.seed(0)
    # random.seed(0)
    # in_data = Tensor(np.random.randn(2, 2), mstype.float32)
    # # # test my linear:
    # # my_linear = Linear(2, 8, bias=True)
    # # print(my_linear.weight.asnumpy())
    # # print(my_linear.bias.asnumpy())
    # # print(my_linear, my_linear.weight)
    # # out_data1 = my_linear(in_data)
    # # print(out_data1.shape, out_data1)
    #
    #
    # # # test nn.Dense
    # # linear = nn.Dense(2, 8, weight_init=XavierUniform(gain=1))
    # # print(linear.weight.asnumpy())
    # # print(linear.bias.asnumpy())
    # # print(linear, linear.weight)
    # # out_data2 = linear(in_data)
    # # print(out_data2.shape, out_data2)
    # ##############################################################

    # in_data = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
    # res = fill_with_neg_inf(in_data)
    # print(res)
    # res = (in_data == 0)

    # res = make_positions(in_data, -2)

    # ###########################################################
    # # Test LearnedEmbeddings
    # input_tokens = Tensor(np.array([[64822,  4887, 31201,  1714,  4817,    94,   136,  1351,  3418,  7857,
    #       473,  1624,   436,    46,     4,   603,  9265,   948,   481,   663,
    #     20736,    81,  1144,   157,    70,    46,     4,   332,    35,  2240,
    #       515,    46,  2071,     4,  5740,   158,    29,  2102,  1598,    76,
    #      6752,   154,  1071,  5294,    99,   738,   132, 19877,   283,    38,
    #      2196,  6484, 12600,  4970,   766,  2857,    94, 14555,  2772,     5,
    #         2]]).astype(np.int32))
    # # print(input_tokens)
    # m = LearnedPositionalEmbedding(258, 1024)
    # padding_idx = 1
    # m.embedding_table = Parameter(initializer(Normal(sigma=258 ** -0.5), [258, 1024]))
    # # print(m.embedding_table.asnumpy()[padding_idx])
    # m.embedding_table[padding_idx] = Parameter(initializer(Constant(value=0), [1, 1024]))
    # print(m.embedding_table.asnumpy()[padding_idx])
    # res = m(input_tokens, None, None)
    # print(res.shape)
    # #####################################################

    # #############################################
    # # test msFairseqDropout Cell
    # MyDropout = MindsporeFairseqDropout(0.5)
    # x = Tensor((20, 16, 50, 50), mstype.float32)
    # res = MyDropout(x)
    # print(res)
    # #############################################

    # #########################################
    # # test Mindspore_linear function
    # input_x = Tensor(np.random.randn(2, 24, 1024))
    # weight = Tensor(np.random.randn(1024, 1024))
    # bias = Tensor(np.random.randn(1024))
    # res01 = mindspore_linear(input_x, weight, bias)
    # print(res01.shape)
    # #########################################

    ############################################
    # test Yizx_Mindspore_MultiHeadAttention
    net = MultiheadAttention(1024, 16, None, None, 0.1, True, False, False, True, False, 0, 8)
    # print(net)
    # src_tokens = Tensor(np.random.randint(0, 1024, (24, 61)), mstype.int32)
    src_tokens = Tensor(np.random.randint(0, 2, (24, 61)), mstype.int32)
    # after Transformer.forward_embedding(src_tokensforward_embedding, token_embeddings), return x,
    # encoder_embedding
    x = Tensor(np.random.randn(24, 61, 1024), mstype.float32)
    x = x.transpose(1, 0, 2)
    encoder_padding_mask = (src_tokens == 1)  # [24, 61]
    attn_mask = None

    res = net(x, x, x, encoder_padding_mask, attn_mask)
    print(res[0].shape)
    ##################################################




