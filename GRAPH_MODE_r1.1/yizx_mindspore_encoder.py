from yizx_mindspore_utils import *
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from argparse import Namespace
import mindspore.common.dtype as mstype

from yizx_fairseq_dictionary import Dictionary

import mindspore.ops as ops
ops_transpose = ops.Transpose()

class MindsporeEncoder(nn.Cell):
    """Base class for encoders."""
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)


class TransformerMsEncoderLayer(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = MindsporeFairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        if float(activation_dropout_p) == 0:
            self.activation_dropout_module = None
        else:
            self.activation_dropout_module = MindsporeFairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

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

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
                Rename layer norm states from `...layer_norms.0.weight` to
                `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
                `...final_layer_norm.weight`
                """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def construct(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask * -1e8

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        if not self.activation_dropout_module is None:
            x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


def build_embedding(args, dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()

    emb = nn.Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        pass
    return emb

class TransformerMsEncoder(MindsporeEncoder):
    """
        Transformer encoder consisting of *args.encoder_layers* layers. Each layer
        is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)

        self.dropout_module = MindsporeFairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_size     # embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.quant_noise = None

        self.layers = nn.CellList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return TransformerMsEncoderLayer(args)

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def construct(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        # B x T x C -> T x B x C
        x = ops_transpose(x, (1, 0, 2))

        ops_eq = Equal()
        # compute padding mask
        encoder_padding_mask = ops_eq(src_tokens, self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # print(x.shape, encoder_padding_mask.shape)  # [61, 24, 1024], [24, 61]

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


if __name__ == '__main__':
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


    ##############################################
    # Test MsTransformerEncoder
    src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    # trg_path = os.path.join(data_dir, "dict.{}.txt".format(target_lang))
    src_dictionary = Dictionary.load(src_path)

    myembed = build_embedding(args, src_dictionary, args.encoder_embed_dim, args.encoder_embed_path)
    # myembed = LearnedPositionalEmbedding(src_dictionary.__len__(), args.encoder_embed_dim)
    # print(myembed)
    mynet = TransformerMsEncoder(args, src_dictionary, myembed)
    print(mynet)

