from yizx_utils import *
from fairseq import options
from yizx_fairseq_model import transformer_vaswani_wmt_en_de_big
import yizx_distributed_utils
from argparse import Namespace

import numpy as np
from fairseq import tasks

def set_torch_seed(seed):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    import_user_module(args)
    print(args)
    ##################################
    # args lack of
    args.max_tokens = 4096
    ##################################

    assert (
            args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    np.random.seed(args.seed)
    set_torch_seed(args.seed)

    if yizx_distributed_utils.is_master(args):
        yizx_distributed_utils.verify_checkpoint_directory(args.save_dir)

    # # args lack of param:
    args.left_pad_source = True
    args.left_pad_target = False
    args.source_lang = 'src'
    args.target_lang = 'trg'
    args.upsample_primary = 1
    args.max_source_positions = 256
    args.max_target_positions = 256
    args.load_alignments = False
    args.truncate_source = False
    args.num_batch_buckets = 0

    args.data = '/userhome/jobs/NMTrans/mRASP-master/toy/data/pre-train'

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # # args lack of param:
    args.arch = 'transformer_wmt_en_de_big'
    args.encoder_layers_to_keep = None
    args.decoder_layers_to_keep = None
    args.encoder_layerdrop = 0
    args.decoder_layerdrop = 0
    args.quant_noise_pq = 0
    args.encoder_learned_pos = True
    args.decoder_learned_pos = True

    # Build model and criterion
    model = task.build_model(args)
    # print(model)
    criterion = task.build_criterion(args)
    print(criterion)  # 其实就是个交叉熵




def cli_main():
    parser = options.get_training_parser()
    args = parser.parse_args()
    transformer_vaswani_wmt_en_de_big(args)
    # args = update_args()
    yizx_distributed_utils.call_main(args, main)

def update_args():
    # 过滤yizx脚本中缺乏的args参数
    # raw_args = dict(activation_dropout=0.0, activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, arch='transformer_wmt_en_de_big', attention_dropout=0.1, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0, cpu=False, criterion='label_smoothed_cross_entropy', cross_self_attention=False, curriculum=0, data='/userhome/jobs/NMTrans/mRASP-master/toy/data/pre-train', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=16, decoder_embed_dim=1024, decoder_embed_path=None, decoder_ffn_embed_dim=4096, decoder_input_dim=1024, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=1024, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=True, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.2, empty_cache_freq=0, encoder_attention_heads=16, encoder_embed_dim=1024, encoder_embed_path=None, encoder_ffn_embed_dim=4096, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, eval_bleu=False, eval_bleu_args=None, eval_bleu_detok='space', eval_bleu_detok_args=None, eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, fp16=True, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', ignore_prefix_size=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=False, left_pad_source='True', left_pad_target='False', load_alignments=False, localsgd_frequency=3, log_format=None, log_interval=5, lr=[0.0005], lr_scheduler='inverse_sqrt', max_epoch=0, max_source_positions=256, max_target_positions=256, max_tokens=4096, max_tokens_valid=4096, max_update=100000, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=1e-09, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=True, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=True, reset_meters=True, reset_optimizer=True, restore_file='checkpoint_last.pt', save_dir='/userhome/jobs/NMTrans/mRASP-master/pretrain/transformer_big', save_interval=1, save_interval_updates=50, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=True, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang='src', stop_time_hours=0, target_lang='trg', task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=False, update_freq=[1], upsample_primary=1, use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_init_lr=1e-07, warmup_updates=4000, weight_decay=0.0, zero_sharding='none')
    # my_args = dict(activation_dropout=0.0, activation_fn='relu', adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, arch=None, attention_dropout=0.0, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0, cpu=False, criterion='cross_entropy', cross_self_attention=False, curriculum=0, data_buffer_size=10, dataset_impl=None, ddp_backend='c10d', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layers=6, decoder_learned_pos=False, decoder_normalize_before=False, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layers=6, encoder_learned_pos=False, encoder_normalize_before=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, layernorm_embedding=False, localsgd_frequency=3, log_format=None, log_interval=100, lr=[0.25], lr_scheduler='fixed', max_epoch=0, max_tokens=None, max_tokens_valid=None, max_update=0, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_shards=1, num_workers=1, optimizer=None, optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, profile=False, quantization_config_path=None, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='checkpoints', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=False, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', update_freq=[1], use_bmuf=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, zero_sharding='none')
    # raw_items = []
    # for item in raw_args:
    #     raw_items.append(item)
    # differ_dict = {}
    # for item in raw_items:
    #     try:
    #         tmp = my_args[item]
    #         if tmp == raw_args[item]:
    #             pass
    #         else:
    #             differ_dict[item] = raw_args[item]
    #     except:
    #         differ_dict[item] = raw_args[item]
    #
    # for diff_item in differ_dict:
    #     args[diff_item] = differ_dict[diff_item]

    args = Namespace(activation_dropout=0.0, activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, arch='transformer_wmt_en_de_big', attention_dropout=0.1, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0, cpu=False, criterion='label_smoothed_cross_entropy', cross_self_attention=False, curriculum=0, data='/userhome/jobs/NMTrans/mRASP-master/toy/data/pre-train', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=16, decoder_embed_dim=1024, decoder_embed_path=None, decoder_ffn_embed_dim=4096, decoder_input_dim=1024, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=1024, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=True, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.2, empty_cache_freq=0, encoder_attention_heads=16, encoder_embed_dim=1024, encoder_embed_path=None, encoder_ffn_embed_dim=4096, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, eval_bleu=False, eval_bleu_args=None, eval_bleu_detok='space', eval_bleu_detok_args=None, eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, fp16=True, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', ignore_prefix_size=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=False, left_pad_source='True', left_pad_target='False', load_alignments=False, localsgd_frequency=3, log_format=None, log_interval=5, lr=[0.0005], lr_scheduler='inverse_sqrt', max_epoch=0, max_source_positions=256, max_target_positions=256, max_tokens=4096, max_tokens_valid=4096, max_update=100000, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=1e-09, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=True, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=True, reset_meters=True, reset_optimizer=True, restore_file='checkpoint_last.pt', save_dir='/userhome/jobs/NMTrans/mRASP-master/pretrain/transformer_big', save_interval=1, save_interval_updates=50, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=True, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang='src', stop_time_hours=0, target_lang='trg', task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=False, update_freq=[1], upsample_primary=1, use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_init_lr=1e-07, warmup_updates=4000, weight_decay=0.0, zero_sharding='none')
    return args


if __name__ == '__main__':
    # # 过滤yizx脚本中缺乏的args参数
    # raw_args = dict(activation_dropout=0.0, activation_fn='gelu', adam_betas='(0.9, 0.98)', adam_eps=1e-08, adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, arch='transformer_wmt_en_de_big', attention_dropout=0.1, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0, cpu=False, criterion='label_smoothed_cross_entropy', cross_self_attention=False, curriculum=0, data='/userhome/jobs/NMTrans/mRASP-master/toy/data/pre-train', data_buffer_size=10, dataset_impl=None, ddp_backend='no_c10d', decoder_attention_heads=16, decoder_embed_dim=1024, decoder_embed_path=None, decoder_ffn_embed_dim=4096, decoder_input_dim=1024, decoder_layerdrop=0, decoder_layers=6, decoder_layers_to_keep=None, decoder_learned_pos=True, decoder_normalize_before=False, decoder_output_dim=1024, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=True, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.2, empty_cache_freq=0, encoder_attention_heads=16, encoder_embed_dim=1024, encoder_embed_path=None, encoder_ffn_embed_dim=4096, encoder_layerdrop=0, encoder_layers=6, encoder_layers_to_keep=None, encoder_learned_pos=True, encoder_normalize_before=False, eval_bleu=False, eval_bleu_args=None, eval_bleu_detok='space', eval_bleu_detok_args=None, eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, fp16=True, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', ignore_prefix_size=0, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, label_smoothing=0.1, layernorm_embedding=False, left_pad_source='True', left_pad_target='False', load_alignments=False, localsgd_frequency=3, log_format=None, log_interval=5, lr=[0.0005], lr_scheduler='inverse_sqrt', max_epoch=0, max_source_positions=256, max_target_positions=256, max_tokens=4096, max_tokens_valid=4096, max_update=100000, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=1e-09, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=True, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_seed_provided=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_batch_buckets=0, num_shards=1, num_workers=1, optimizer='adam', optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, profile=False, quant_noise_pq=0, quant_noise_pq_block_size=8, quant_noise_scalar=0, quantization_config_path=None, report_accuracy=False, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=True, reset_lr_scheduler=True, reset_meters=True, reset_optimizer=True, restore_file='checkpoint_last.pt', save_dir='/userhome/jobs/NMTrans/mRASP-master/pretrain/transformer_big', save_interval=1, save_interval_updates=50, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=True, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=True, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang='src', stop_time_hours=0, target_lang='trg', task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', truncate_source=False, update_freq=[1], upsample_primary=1, use_bmuf=False, use_old_adam=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, warmup_init_lr=1e-07, warmup_updates=4000, weight_decay=0.0, zero_sharding='none')
    # my_args = dict(activation_dropout=0.0, activation_fn='relu', adaptive_input=False, adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, all_gather_list_size=16384, arch=None, attention_dropout=0.0, batch_size=None, batch_size_valid=None, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, checkpoint_shard_count=1, checkpoint_suffix='', clip_norm=0.0, cpu=False, criterion='cross_entropy', cross_self_attention=False, curriculum=0, data_buffer_size=10, dataset_impl=None, ddp_backend='c10d', decoder_attention_heads=8, decoder_embed_dim=512, decoder_embed_path=None, decoder_ffn_embed_dim=2048, decoder_input_dim=512, decoder_layers=6, decoder_learned_pos=False, decoder_normalize_before=False, decoder_output_dim=512, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', dropout=0.1, empty_cache_freq=0, encoder_attention_heads=8, encoder_embed_dim=512, encoder_embed_path=None, encoder_ffn_embed_dim=2048, encoder_layers=6, encoder_learned_pos=False, encoder_normalize_before=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', keep_best_checkpoints=-1, keep_interval_updates=-1, keep_last_epochs=-1, layernorm_embedding=False, localsgd_frequency=3, log_format=None, log_interval=100, lr=[0.25], lr_scheduler='fixed', max_epoch=0, max_tokens=None, max_tokens_valid=None, max_update=0, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_loss_scale=0.0001, min_lr=-1.0, model_parallel_size=1, no_cross_attention=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_save=False, no_save_optimizer_state=False, no_scale_embedding=False, no_token_positional_embeddings=False, nprocs_per_node=1, num_shards=1, num_workers=1, optimizer=None, optimizer_overrides='{}', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, profile=False, quantization_config_path=None, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', save_dir='checkpoints', save_interval=1, save_interval_updates=0, scoring='bleu', seed=1, sentence_avg=False, shard_id=0, share_all_embeddings=False, share_decoder_input_output_embed=False, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, stop_time_hours=0, task='translation', tensorboard_logdir=None, threshold_loss_scale=None, tie_adaptive_weights=False, tokenizer=None, tpu=False, train_subset='train', update_freq=[1], use_bmuf=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, zero_sharding='none')
    # raw_items = []
    # for item in raw_args:
    #     raw_items.append(item)
    # differ_dict = {}
    # for item in raw_items:
    #     try:
    #         tmp = my_args[item]
    #         if tmp == raw_args[item]:
    #             pass
    #         else:
    #             differ_dict[item] = raw_args[item]
    #     except:
    #         differ_dict[item] = raw_args[item]
    # print(len(differ_dict))
    # print(differ_dict)

    # # test args
    # parser = options.get_training_parser()
    # args = parser.parse_args()
    # transformer_vaswani_wmt_en_de_big(args)

    cli_main()


