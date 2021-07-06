from yizx_mindspore_utils import *
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from argparse import Namespace
import mindspore.common.dtype as mstype
import uuid

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal

from yizx_fairseq_dictionary import Dictionary

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

class MindsporeDecoder(nn.Cell):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return log_softmax(logits, dim=-1)
        else:
            return softmax(logits, dim=-1)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True



@with_incremental_state
class MindsporeIncrementDecoder(MindsporeDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None,
                **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None,
                         **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        pass

    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Main entry point for reordering the incremental state.

        Due to limitations in TorchScript, we call this function in
        :class:`fairseq.sequence_generator.SequenceGenerator` instead of
        calling :func:`reorder_incremental_state` directly.
        """
        for module in self.modules():
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, "_beam_size", -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if (
                    module != self
                    and hasattr(module, "set_beam_size")
                    and module not in seen
                ):
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size


class TransformerMsDecoderLayer(nn.Cell):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        if args.dropout > 0:
            self.dropout_module = MindsporeFairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            )
        else:
            self.dropout_module = None
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )

        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)

        if activation_dropout_p > 0:
            self.activation_dropout_module = MindsporeFairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
        else:
            self.activation_dropout_module = None

        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True
        self.training = True

        self.onnx_trace = False


    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if not q_noise > 0:
            return Linear(input_dim, output_dim)
        else:
            raise Exception

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if not q_noise > 0:
            return Linear(input_dim, output_dim)
        else:
            raise Exception

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def construct(
        self,
        x,
        encoder_out: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[Tensor]] = None,
        prev_attn_state: Optional[List[Tensor]] = None,
        self_attn_mask: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if need_head_weights:
            need_attn = True
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            pass

        y = x
        # print(x.shape, self_attn_padding_mask.shape, incremental_state, self_attn_mask.shape)

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                pass

            # print(x.shape, encoder_out.shape, encoder_padding_mask.shape,
            #       incremental_state, need_attn, need_head_weights)
            # print(self.encoder_attn)

            # print('encoder_attn')

            # need_attn F,F,F,F,F,T   __<>> Passed!
            # tmp = need_attn or (not self.training and self.need_attn)
            # print(need_attn)

            # print(need_attn or (not self.training and self.need_attn))

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            if not self.dropout_module is None:
                x = self.dropout_module(x)

            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        if not self.activation_dropout_module is None:
            x = self.activation_dropout_module(x)
        x = self.fc2(x)
        if not self.dropout_module is None:
            x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        if not self.activation_dropout_module is None:
            x = self.activation_dropout_module(x)
        x = self.fc2(x)
        if not self.dropout_module is None:
            x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            pass

        return x, attn, None


class TransformerMsDecoder(MindsporeIncrementDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerMsDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self._future_mask = Tensor(np.array([]), mstype.float32)

        self.dropout_module = MindsporeFairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_size
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        if not args.adaptive_input and args.quant_noise_pq > 0:
            pass
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        # print(self.embed_positions)

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            pass
        else:
            self.layers = nn.CellList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
                args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            pass
        elif self.share_input_output_embed:
            self.output_projection = Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = Linear(
                self.output_embed_dim,
                len(dictionary),
                bias=False,
            )
            self.output_projection.weight = Parameter(initializer(Normal(sigma=self.output_embed_dim ** -0.5), [len(dictionary), self.output_embed_dim]))

        # print(self.output_projection)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerMsDecoderLayer(args, no_encoder_attn)

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def construct(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        # print(x.shape, extra)  # [24, 131, 1024] {}
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(1, 0, 2)

        self_attn_padding_mask: Optional[Tensor] = None
        ops_equal = Equal()
        ops_any = ReduceAny(keep_dims=False)
        tmp = ops_equal(prev_output_tokens, self.padding_idx)
        if self.cross_self_attention or ops_any(tmp):
            self_attn_padding_mask = ops_equal(prev_output_tokens, self.padding_idx) #prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # print(x.shape, encoder_out.encoder_out.shape, encoder_out.encoder_padding_mask.shape,
            #        incremental_state, self_attn_mask.shape, bool((idx == alignment_layer)),
            #        bool((idx == alignment_layer)))

            # [131, 24, 1024], [61, 24, 1024], [24, 61], None, (131, 131), False, False

            # print(bool((idx == alignment_layer)))

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            # print(x.shape, layer_attn is None)  # layer_attn need to be: T,T,T,T,T,F !!!

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            ops_mean = ReduceMean(keep_dims=False)
            attn = ops_mean(attn, axis=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0, 2)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


    def buffered_future_mask(self, tensor):
        dim = tensor.shape[0]
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.shape[0] == 0
            # or (not self._future_mask.device == tensor.device)
            or self._future_mask.shape[0] < dim
        ):
            triu_layer = nn.Triu()
            tmp_tensor = Tensor(np.zeros([dim, dim]), mstype.float32)
            tmp_tensor = fill_with_neg_inf(tmp_tensor)
            self._future_mask = triu_layer(tmp_tensor, 1)
            # self._future_mask = torch.triu(
            #     fill_with_neg_inf(torch.zeros([dim, dim])), 1
            # )
        self._future_mask = self._future_mask #.to(tensor)
        return self._future_mask[:dim, :dim]






def build_embedding(args, dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()

    emb = nn.Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        pass
    return emb



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    args = Namespace(activation_dropout=0.0, activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08,
                     adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0,
                     all_gather_list_size=16384, arch='transformer_wmt_en_de_big', attention_dropout=0.1,
                     batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None,
                     broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='',
                     clip_norm=0.0, cpu=False, criterion='label_smoothed_cross_entropy', cross_self_attention=False,
                     curriculum=0, data='/userhome/jobs/NMTrans/mRASP-master/toy/data/pre-train', data_buffer_size=10,
                     dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=16, decoder_embed_dim=1024,
                     decoder_embed_path=None, decoder_ffn_embed_dim=4096, decoder_input_dim=1024, decoder_layerdrop=0,
                     decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True,
                     decoder_normalize_before=False, decoder_output_dim=1024, device_id=0, disable_validation=False,
                     distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=True,
                     distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP',
                     dropout=0.2, empty_cache_freq=0, encoder_attention_heads=16, encoder_embed_dim=1024,
                     encoder_embed_path=None, encoder_ffn_embed_dim=4096, encoder_layerdrop=0, encoder_layers=6,
                     encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False,
                     eval_bleu=False, eval_bleu_args=None, eval_bleu_detok='space', eval_bleu_detok_args=None,
                     eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False,
                     fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None,
                     fix_batches_to_gpus=False, fixed_validation_seed=None, fp16=True, fp16_init_scale=128,
                     fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test',
                     ignore_prefix_size=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1,
                     label_smoothing=0.1, layernorm_embedding=False, left_pad_source='True', left_pad_target='False',
                     load_alignments=False, localsgd_frequency=3, log_format=None, log_interval=5, lr=[0.0005],
                     lr_scheduler='inverse_sqrt', max_epoch=0, max_source_positions=256, max_target_positions=256,
                     max_tokens=4096, max_tokens_valid=4096, max_update=100000, maximize_best_checkpoint_metric=False,
                     memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=1e-09,
                     model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False,
                     no_last_checkpoints=False, no_progress_bar=True, no_save=False, no_save_optimizer_state=False,
                     no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False,
                     nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, optimizer='adam',
                     optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never',
                     pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None,
                     pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None,
                     pipeline_model_parallel=False, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8,
                     quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False,
                     required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=True,
                     reset_lr_scheduler=True, reset_meters=True, reset_optimizer=True,
                     restore_file='checkpoint_last.pt',
                     save_dir='/userhome/jobs/NMTrans/mRASP-master/pretrain/transformer_big', save_interval=1,
                     save_interval_updates=50, scoring='bleu', seed=1, sentence_avg=False, shard_id=0,
                     share_all_embeddings=True, share_decoder_input_output_embed=False,
                     skip_invalid_size_inputs_valid_test=True, slowmo_algorithm='LocalSGD', slowmo_momentum=None,
                     source_lang='src', stop_time_hours=0, target_lang='trg', task='translation',
                     tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None,
                     tpu=False, train_subset='train', truncate_source=False, update_freq=[1], upsample_primary=1,
                     use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', validate_after_updates=0,
                     validate_interval=1, validate_interval_updates=0, warmup_init_lr=1e-07, warmup_updates=4000,
                     weight_decay=0.0, zero_sharding='none')


    src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    trg_path = os.path.join(args.data, "dict.{}.txt".format(args.target_lang))
    trg_dictionary = Dictionary.load(trg_path)
    src_dictionary = Dictionary.load(src_path)

    # ############################################
    # # # test Decoder
    # decoder = MindsporeDecoder(src_dictionary)
    # print(decoder)
    # #############################################

    # ############################################
    # # # test DecoderLayer
    # decoder = TransformerMsDecoderLayer(args, False)
    #
    # # do forward / construct
    # x = Tensor(np.random.randn(131, 24, 1024), mstype.float32)
    # encoder_out = Tensor(np.random.randn(61, 24, 1024), mstype.float32)
    # encoder_padding_mask = Tensor(np.random.randint(0, 2, (24, 61)), mstype.bool_)
    # incremental_state = None
    # self_attn_mask = Tensor(np.random.randn(131, 131), mstype.float32)
    # self_attn_padding_mask = Tensor(np.random.randint(0, 2, (24, 131)), mstype.bool_)
    # need_attn = False
    # need_head_weights = False
    #
    # res = decoder(x, encoder_out, encoder_padding_mask, incremental_state, self_attn_mask=self_attn_mask,
    #               self_attn_padding_mask=self_attn_padding_mask, need_attn = need_attn, need_head_weights = need_head_weights)
    #
    # #############################################


    ###############################################
    # Test Decoder
    myembed = build_embedding(args, trg_dictionary, args.decoder_embed_dim, args.decoder_embed_path)
    no_encoder_attn = getattr(args, "no_cross_attention", False)
    mydecoder = TransformerMsDecoder(args, trg_dictionary, myembed, no_encoder_attn)
    print(mydecoder)


    # #######################################################################
    # # Test TransformerMsEncoderLayer
    # myEncoderLayer = TransformerMsEncoderLayer(args)
    # print(myEncoderLayer)
    # src_tokens = Tensor(np.random.randint(0, 1024, (24, 61)), mstype.int32)
    # # after Transformer.forward_embedding(src_tokensforward_embedding, token_embeddings), return x,
    # # encoder_embedding
    # x = Tensor(np.random.randn(24, 61, 1024), mstype.float32)
    # x = x.transpose(1, 0, 2)   # [61, 24, 1024]
    # encoder_padding_mask = (src_tokens == 1)  # [24, 61]
    #
    # res = myEncoderLayer(x, encoder_padding_mask, None)
    # print(res.shape)   # need sample shape as x  Passed!
    # ######################################################################


    # ##############################################
    # # Test MsTransformerEncoder
    # src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    # # trg_path = os.path.join(data_dir, "dict.{}.txt".format(target_lang))
    # src_dictionary = Dictionary.load(src_path)
    #
    # myembed = build_embedding(args, src_dictionary, args.encoder_embed_dim, args.encoder_embed_path)
    # # myembed = LearnedPositionalEmbedding(src_dictionary.__len__(), args.encoder_embed_dim)
    # # print(myembed)
    # mynet = TransformerMsEncoder(args, src_dictionary, myembed)
    # print(mynet)

