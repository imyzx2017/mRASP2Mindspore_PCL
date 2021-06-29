import mindspore
import mindspore.nn as nn
from mindspore import Tensor
from yizx_mindspore_utils import *
from yizx_mindspore_decoder import MindsporeDecoder, build_embedding, TransformerMsDecoder
from yizx_mindspore_encoder import MindsporeEncoder, TransformerMsEncoder

from argparse import Namespace

from yizx_fairseq_dictionary import Dictionary


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)

def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)


class BaseFairseqModel(nn.Cell):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        else:
            try:
                if net_output.dtype:
                    # syntactic sugar for simple models which don't have a decoder
                    # (e.g., the classification tutorial)
                    logits = net_output.float()
                    if log_probs:
                        return log_softmax(logits, dim=-1)
                    else:
                        return softmax(logits, dim=-1)
            except:
                raise NotImplementedError
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(self, state_dict, strict=True, args=None):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        pass
        # self.upgrade_state_dict(state_dict)
        # new_state_dict = prune_state_dict(state_dict, args)
        # return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        pass
        # self.upgrade_state_dict_named(state_dict, "")

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)


class MindsporeEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, MindsporeEncoder)
        # print('Encoder build Passed!')
        assert isinstance(self.decoder, MindsporeDecoder)
        # print('Decoder build Passed!')

    def construct(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class TransformerModel(MindsporeEncoderDecoderModel):
    """
        Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
        <https://arxiv.org/abs/1706.03762>`_.

        Args:
            encoder (TransformerEncoder): the encoder
            decoder (TransformerDecoder): the decoder

        The Transformer model provides the following named architectures and
        command-line arguments:

        .. argparse::
            :ref: fairseq.models.transformer_parser
            :prog:
        """
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        pass

    def construct(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        # print(src_tokens.shape, src_lengths, prev_output_tokens.shape, return_all_hiddens,
        #       features_only, alignment_layer, alignment_heads)
        # [24, 61], [24], [24, 131] True False None None

        # print(src_tokens.shape, src_lengths.shape, return_all_hiddens)

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out



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

    src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    trg_path = os.path.join(args.data, "dict.{}.txt".format(args.target_lang))
    trg_dictionary = Dictionary.load(trg_path)
    src_dictionary = Dictionary.load(src_path)

    EncoderEmbed = build_embedding(args, src_dictionary, args.encoder_embed_dim, args.encoder_embed_path)
    myEncoder = TransformerMsEncoder(args, src_dictionary, EncoderEmbed)

    DecoderEmbed = build_embedding(args, trg_dictionary, args.decoder_embed_dim, args.decoder_embed_path)
    myDecoder = TransformerMsDecoder(args, trg_dictionary, DecoderEmbed, no_encoder_attn=getattr(args, "no_cross_attention", False))

    ######################################################################
    # Test FairseqEncoderDecoderModel
    myTransformerModel = TransformerModel(args, myEncoder, myDecoder)
    # print(EnDecoderModel)
    #######################################################################

    src_tokens = Tensor(np.random.randint(0, 64871, (24, 61)).astype(np.int64))
    src_lens = Tensor(np.random.randint(60, 62, (24)).astype(np.int64))
    prev_output_tokens = Tensor(np.random.randint(0, 64871, (24, 131)).astype(np.int64))

    res, extra = myTransformerModel(src_tokens, src_lens, prev_output_tokens)#, True, False, None, None)
    # print(res.shape, extra['attn'][0].shape)  # [24, 131, 64871], [24, 131, 61]
