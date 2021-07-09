from yizx_mindspore_utils import get_args, log_softmax, softmax
from yizx_mindspore_decoder import Dictionary

import struct
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor
import os

from yizx_dataset_load import load_dataset, BinMaxDataset, LanguagePairDataset, \
    numpy_seed, EpochBatchIterator
from yizx_mindspore_decoder import build_embedding
from yizx_mindspore_transformer import TransformerMsEncoder, TransformerMsDecoder, TransformerModel
from yizx_mindspore_criterions import LabelSmoothedCrossEntropyCriterion, \
    label_smoothed_nll_loss, mRASP2ModelWithLoss

# import moxing
from mindspore.train.callback import Callback, TimeMonitor
import mindspore.nn as nn
import time
import argparse

time_stamp_init = False
time_stamp_first = 0

def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))

def get_args_dataset_modelWithLoss():
    args = get_args()
    # download src / trg data from obs
    # raw_data_dir = 's3://pcl-verify/yizx/gpt2.6b/mRASP/pretrain'
    # args.data = '/tmp/pretrain/'

    # if moxing.file.is_directory(raw_data_dir):
    #     ret = moxing.file.list_directory(raw_data_dir)
    # for file_item in ret:
    #     file_path = raw_data_dir + '/' + file_item
    #     moxing.file.copy(file_path, args.data + file_item)
    
    src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    trg_path = os.path.join(args.data, "dict.{}.txt".format(args.target_lang))
    trg_dictionary = Dictionary.load(trg_path)
    src_dictionary = Dictionary.load(src_path)
    load_dataset(args, src_dictionary, trg_dictionary, 'valid', combine=False)

    #######################################################
    # 1.Data Ready
    #######################################################
    # load mRASP2 dataset
    src_datasets = []
    tgt_datasets = []
    # paths_list = [args.data + '/valid.src-trg.src', args.data + '/valid.src-trg.trg']
    paths_list = [args.data + '/train.src-trg.src', args.data + '/train.src-trg.trg']

    src_dataset = BinMaxDataset(paths_list[0])
    tgt_dataset = BinMaxDataset(paths_list[1])

    src_datasets.append(src_dataset)
    tgt_datasets.append(tgt_dataset)

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        pass
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    left_pad_source = True
    left_pad_target = False
    eos = None
    align_dataset = None
    num_buckets = 0
    shuffle = True
    pad_to_multiple = 1


    myPairDataset = LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dictionary,
        tgt_dataset,
        tgt_dataset_sizes,
        trg_dictionary,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )

    ################################################
    # Test BatchSampler
    max_tokens = args.max_tokens
    max_sentences = None
    required_batch_size_multiple = args.required_batch_size_multiple

    seed = 1
    with numpy_seed(seed):
        indices = myPairDataset.ordered_indices()
    # print(len(indices))  # same seed=1, len=37764

    batch_sampler = myPairDataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )
    #############################################
    # print(batch_sampler[0])
    # for index in batch_sampler[0]:
    #     print(myPairDataset[index])
    #####################################################
    # test epoch_iter for generate new batches
    epoch_iter = EpochBatchIterator(
        myPairDataset,
        myPairDataset.collater,
        batch_sampler
    )
    finaldataset = epoch_iter.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_iter.next_epoch_idx > args.curriculum),
    )
    ######################################################
    numpy_seed(1)

    mRASPDataset = ds.GeneratorDataset(
        source=finaldataset,
        column_names=['id', 'nsentences', 'ntokens', 'src_tokens', 'src_lengths',
                      'prev_output_tokens', 'target'],
        shuffle=False
    )
    ######################################################
    # 2.get model
    #######################################################
    EncoderEmbed = build_embedding(args, src_dictionary, args.encoder_embed_dim, args.encoder_embed_path)
    myEncoder = TransformerMsEncoder(args, src_dictionary, EncoderEmbed)

    DecoderEmbed = build_embedding(args, trg_dictionary, args.decoder_embed_dim, args.decoder_embed_path)
    myDecoder = TransformerMsDecoder(args, trg_dictionary, DecoderEmbed,
                                     no_encoder_attn=getattr(args, "no_cross_attention", False))

    ######################################################################
    # Test FairseqEncoderDecoderModel
    myTransformerModel = TransformerModel(args, myEncoder, myDecoder)
    ######################################################################

    ######################################################
    # 3.define modelWithloss
    #######################################################
    sentence_avg = False
    label_smoothing = 0.1
    ignore_prefix_size = 0
    report_accuracy = False
    myTransformerModelWithLoss = mRASP2ModelWithLoss(myTransformerModel, sentence_avg,
                                                     label_smoothing, trg_dictionary.pad())
    ##########################################
    
    return args, mRASPDataset, myTransformerModelWithLoss

def linear_warmup(warmup_steps, current_step):
    return min([1.0, float(current_step)/float(warmup_steps)])

def rsqrt_decay(warmup_steps, current_step):
    return float(max([current_step, warmup_steps])) ** -0.5

def rsqrt_hidden(hidden_size):
    return float(hidden_size) ** -0.5


def create_dynamic_lr(schedule, training_steps, learning_rate, warmup_steps, hidden_size,
                      start_decay_step=0, min_lr=0.):
    """
    Generate dynamic learning rate.
    """
    if start_decay_step < warmup_steps:
        start_decay_step = warmup_steps
    lr = []
    for current_step in range(1, training_steps+1):
        cur_lr = 1.0
        for name in schedule.split("*"):
            if name == "constant":
                cur_lr *= float(learning_rate)
            elif name == "rsqrt_hidden":
                cur_lr *= rsqrt_hidden(hidden_size)
            elif name == "linear_warmup":
                cur_lr *= linear_warmup(warmup_steps, current_step)
            elif name == "rsqrt_decay":
                cur_lr *= rsqrt_decay(warmup_steps, current_step-start_decay_step+warmup_steps)
            else:
                raise ValueError("unknown learning rate schedule")
        if warmup_steps < current_step < start_decay_step:
            cur_lr = lr[-1]
        if current_step > warmup_steps:
            cur_lr = max([cur_lr, min_lr])
        lr.append(cur_lr)
    return lr

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_id
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}, overflow: {}, loss_scale: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs[0].asnumpy()),
                str(cb_params.net_outputs[1].asnumpy()),
                str(cb_params.net_outputs[2].asnumpy())))
            f.write('\n')
            
# class TransformerTrainOneStepWithLossScaleCell(nn.TrainOneStepWithLossScaleCell):
#     """
#     Encapsulation class of Transformer network training.

#     Append an optimizer to the training network after that the construct
#     function can be called to create the backward graph.

#     Args:
#         network (Cell): The training network. Note that loss function should have been added.
#         optimizer (Optimizer): Optimizer for updating the weights.
#         scale_update_cell (Cell): Cell to do the loss scale. Default: None.
#     """
#     def __init__(self, network, optimizer, scale_update_cell=None):
#         super(TransformerTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
#         self.cast = P.Cast()
#         self.degree = 1
#         if self.reducer_flag:
#             self.degree = get_group_size()
#             self.grad_reducer = DistributedGradReducer(optimizer.parameters, False, self.degree)

#         self.loss_scale = None
#         self.loss_scaling_manager = scale_update_cell
#         if scale_update_cell:
#             self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32))
    
#     def construct(self,
#         sample,
#         reduce=True,
#     ):
#         src_tokens = sample['src_tokens']
#         src_lengths = sample['src_lengths']
#         prev_output_tokens = sample['prev_output_tokens']
        
#         weights = self.weights
        
#         net_output = self.model(src_tokens, src_lengths, prev_output_tokens)
#         loss, nll_loss = self.compute_loss(self.model, net_output, sample, reduce=reduce)
        
#         # grads = self.grad(self.network, weights)
#         return loss

def argparse_init():
    """
    Argparse init.
    """
    parser = argparse.ArgumentParser(description='transformer')
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    return parser

if __name__ == '__main__':
    from mindspore import context
    import mindspore.communication.management as D
    from mindspore.nn.optim import Adam
    import mindspore.common.dtype as mstype
    from mindspore.train.loss_scale_manager import DynamicLossScaleManager
    from mindspore.context import ParallelMode
    from mindspore.train.model import Model
    
    # device_id = int(os.getenv('DEVICE_ID'))
    # print(device_id)
    
#     parser = argparse_init()
#     args_device, _ = parser.parse_known_args()
#     device_id = args_device.device_id
#     print(device_id)

    
    args, dataset, netwithloss = get_args_dataset_modelWithLoss()
    args.device_target = 'GPU'  #'Ascend'
    args.device_num = 1
    args.epoch_size = 52
    args.start_decay_step = 8000
    args.enable_lossscale = True
    args.distribute = 'false'  # 'true'

    device_id = args.device_id
    
    if args.device_target == "Ascend":
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target, device_id=device_id)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    


    if args.distribute == 'true':
        if args.device_target == "Ascend":
            device_num = args.device_num
            D.init('hccl')
        rank_id = args.device_id % device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num)
        print('!!!!!!!!!!!!! init HCCL passed !!!!!!!!!!!!!')
    else:
        rank_id = args.device_id % args.device_num

#     rank_id = hccl.get_rank_id()
#     context.reset_auto_parallel_context()
#     context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
#                                       device_num=device_num)
    # rank_id = args.device_id % device_num
        
    hidden_size = 1024
    learning_rate = 2 
    
    lr = Tensor(create_dynamic_lr(schedule="constant*rsqrt_hidden*linear_warmup*rsqrt_decay",
                                  training_steps=dataset.get_dataset_size()*args.epoch_size,
                                  learning_rate=learning_rate,
                                  warmup_steps=args.warmup_updates,
                                  hidden_size=hidden_size,
                                  start_decay_step=args.start_decay_step,
                                  min_lr=args.min_lr), mstype.float32)
    optimizer = Adam(netwithloss.trainable_params(), lr)
    
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(rank_id=rank_id)]
    if args.enable_lossscale == True:
        scale_manager = DynamicLossScaleManager(init_loss_scale=1024,scale_factor=2,scale_window=2000)
        update_cell = scale_manager.get_update_cell()
        # netwithgrads = TransformerTrainOneStepWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
        
    netwithloss.set_train(True)
    model = Model(netwithloss, optimizer=optimizer)
    
    
    model.train(args.epoch_size, dataset, callbacks=callbacks, dataset_sink_mode=False)
    
    