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

import moxing

if __name__ == '__main__':
    args = get_args()
    # download src / trg data from obs
    raw_data_dir = 's3://pcl-verify/yizx/gpt2.6b/mRASP/pretrain'
    args.data = '/tmp/pretrain/'
    if moxing.file.is_directory(raw_data_dir):
        ret = moxing.file.list_directory(raw_data_dir)
    for file_item in ret:
        file_path = raw_data_dir + '/' + file_item
        moxing.file.copy(file_path, args.data + file_item)
    
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
        shuffle=True
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

    # ######################################################
    # # 3.end. Test model loss output scalar
    # #######################################################
    # for data in mRASPDataset.create_dict_iterator():
    #     # src_tokens = data['src_tokens']
    #     # src_lengths = data['src_lengths']
    #     # prev_output_tokens = data['prev_output_tokens']
    #     # target = data['target']
    #     #
    #     # print(src_tokens)
    #     # print()
    #     # print(target)
    #     # res, extra = myTransformerModel(src_tokens, src_lengths, prev_output_tokens)
    #
    #     loss, sample_size = myTransformerModelWithLoss(data)
    #     print(loss, sample_size)
    #
    #     # print(res.shape)
    #
    #     # epsilon = 0.1
    #     # res = log_softmax(res, dim=-1)
    #     # res = res.view(-1, res.shape[-1])
    #     # target = target.view(-1)
    #     # loss, nll_loss = label_smoothed_nll_loss(res, target, epsilon, ignore_index=1, reduce=True)
    #     # print(loss, nll_loss)  # float32
    #
    #     break
    # # thisBatch = finaldataset[0]
    # ############################################################

    ######################################################################
    # Do training

    from mindspore.train.model import Model
    from mindspore import context
    import mindspore.communication.management as D
    from mindspore.communication.management import get_rank
    from mindspore.context import ParallelMode
    import os

    args.device_target = 'Ascend'
    ###########################################
    # # Initialize
    # if args.device_target == "Ascend":
    #     device_num = args.device_num
    #     D.init('hccl')
    # else:
    #     D.init('nccl')
    #     device_num = D.get_group_size()
    #     rank = get_rank()
    #     args.device_id = rank
    # context.reset_auto_parallel_context()
    # context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
    #                                   device_num=device_num)
    # rank_id = args.device_id % device_num
    ##########################################


    device_id = 0  # int(os.getenv("DEVICE_ID"))
    rank_id = 0  # int(os.getenv("RANK_ID"))
    local_rank = rank_id
    print('local_rank:{}, device id:{} start to run...'.format(
        local_rank, device_id),
        flush=True)
    save_graphs_path = "/var/log/npu/slog/device-" + str(local_rank) + "/"
    context.set_context(save_graphs=False,
                        save_graphs_path=save_graphs_path,
                        mode=context.PYNATIVE_MODE,  # context.GRAPH_MODE
                        device_target=args.device_target,
                        device_id=device_id)
    # model = Model(myTransformerModelWithLoss)

    for data in mRASPDataset.create_dict_iterator():
        # print(data)
        loss, sample_size = myTransformerModelWithLoss(data)
        print(loss, sample_size)
        # raise ImportError
        break



