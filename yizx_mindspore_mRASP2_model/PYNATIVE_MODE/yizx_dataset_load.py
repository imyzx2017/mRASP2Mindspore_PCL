# code for translation datasetload
# From fairseq/tasks/translation
#######################################
import os
from typing import List

from yizx_mindspore_utils import get_args
from yizx_mindspore_decoder import Dictionary

import struct
import numpy as np
import mindspore.dataset as ds
from mindspore import Tensor

from functools import lru_cache
import contextlib
import math

MANIFOLD_PATH_SEP = "|"

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: float,
    7: np.double,
    8: np.uint16,
}

def _is_batch_full(num_sentences, num_tokens, max_tokens, max_sentences):
    if num_sentences == 0:
        return 0
    if max_sentences > 0 and num_sentences == max_sentences:
        return 1
    if max_tokens > 0 and num_tokens > max_tokens:
        return 1
    return 0

def batch_by_size_fast(
    indices,
    num_tokens_fn,
    max_tokens,
    max_sentences,
    bsz_mult,
):
    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    indices_view = indices

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert max_tokens <= 0 or sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(len(batch), num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches

def split_paths(paths: str) -> List[str]:
    return (
        paths.split(os.pathsep)
        if "://" not in paths
        else paths.split(MANIFOLD_PATH_SEP)
    )

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def load_dataset(args, src_dict, tgt_dict, split, epoch=1, combine=False, **kwargs):
    """Load a given dataset split.

            Args:
                split (str): name of the split (e.g., train, valid, test)
            """
    dataset = {}

    paths = split_paths(args.data)
    assert len(paths) > 0
    if split != getattr(args, "train_subset", None):
        # if not training data set, use the first shard for valid and test
        paths = paths[:1]
    data_path = paths[(epoch - 1) % len(paths)]

    # infer langcode
    src, tgt = args.source_lang, args.target_lang

    # dataset[split] = load_langpair_dataset(
    #         data_path,
    #         split,
    #         src,
    #         src_dict,
    #         tgt,
    #         tgt_dict,
    #         combine=combine,
    #         dataset_impl=args.dataset_impl,
    #         upsample_primary=args.upsample_primary,
    #         left_pad_source=args.left_pad_source,
    #         left_pad_target=args.left_pad_target,
    #         max_source_positions=args.max_source_positions,
    #         max_target_positions=args.max_target_positions,
    #         load_alignments=args.load_alignments,
    #         truncate_source=args.truncate_source,
    #         num_buckets=args.num_batch_buckets,
    #         shuffle=(split != "test"),
    #         pad_to_multiple=args.required_seq_len_multiple,
    #     )
    # return dataset


def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """

    # try:
    #     from data_utils_fast import (
    #         batch_by_size_fast,
    #         batch_fixed_shapes_fast,
    #     )
    # except ImportError:
    #     raise ImportError(
    #         "Please build Cython components with: `pip install --editable .` "
    #         "or `python setup.py build_ext --inplace`"
    #     )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple  # 8

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        return batch_by_size_fast(
            indices,
            num_tokens_fn,
            max_tokens,
            max_sentences,
            bsz_mult,
        )
    else:
        pass

def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    # print(len(values[0]))
    size = max(v.shape[0] for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    # print(len(values), size)  # 56, 59
    res = pad_idx * np.ones((len(values), size)).astype(np.int64)  # values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        if left_pad:
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    res[i][size - len(v) :][0] = v[-1]
                else:
                    res[i][size - len(v) :][0] = eos_idx
                res[i][size - len(v) :][1:] = v[:-1]
            else:
                res[i][size - len(v):] = v
        else:
            if move_eos_to_beginning:
                if eos_idx is None:
                    # if no eos_idx is specified, then use the last token in src
                    res[i][: len(v)][0] = v[-1]
                else:
                    res[i][: len(v)][0] = eos_idx
                res[i][: len(v)][1:] = v[:-1]
            else:
                res[i][: len(v)] = v
        # copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][: len(v)])
    # print(res, res.shape)  # 72, 54
    return res

def index_select_numpy(in_data, axis, sorted_indices):
    if axis == 0:
        return in_data[sorted_indices]
    elif axis == 1:
        return in_data[:, sorted_indices]
    else:
        raise ImportError

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )
    # id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # # sort by descending source length
    src_lengths = [(s["source"] != pad_idx).sum() for s in samples]
    src_lengths = np.sort(src_lengths)[::-1]
    sort_order = np.argsort(src_lengths)[::-1]

    # src_lengths, sort_order = src_lengths.sort(descending=True)

    id = np.array([s["id"] for s in samples])
    id = index_select_numpy(id, 0, sort_order)  # id = id.index_select(0, sort_order)
    # print(id)

    src_tokens = index_select_numpy(src_tokens, 0, sort_order)  # src_tokens = src_tokens.index_select(0, sort_order)
    prev_output_tokens = None
    target = None

    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )

        target = index_select_numpy(target, 0, sort_order)  #     target = target.index_select(0, sort_order)
        tgt_lengths = [(s["target"] != pad_idx).sum() for s in samples]
    #     tgt_lengths = torch.LongTensor(
    #         [s["target"].ne(pad_idx).long().sum() for s in samples]
    #     ).index_select(0, sort_order)
        ntokens = np.sum(tgt_lengths)#.sum().item()

    #

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        pass

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = index_select_numpy(prev_output_tokens, 0, sort_order)

    return batch



class EpochListening:
    """Mixin for receiving updates whenever the epoch increments."""

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.

        This needs to return ``False`` if the sample sizes can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return True

    def set_epoch(self, epoch):
        """Will receive the updated epoch number at the beginning of the epoch."""
        pass


class TorchBaseDataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class FairseqMsDataset(TorchBaseDataset, EpochListening):
    """A dataset that provides helpers for batching."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        raise NotImplementedError

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        raise NotImplementedError

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        raise NotImplementedError

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None


class BinMaxDataset(object):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()
            _warmup_mmap_file(path)
            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super(BinMaxDataset, self).__init__()
        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)
    
    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))
        # print(self._index[0])  # keep same as fairseq

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        # print(self._index[0])   # Same

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        # print(ptr, size)
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        # print(np_array.shape)  # [75, ]
        return np_array

    def __len__(self):
        return len(self._index)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False


class LanguagePairDataset(FairseqMsDataset):
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                    self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            raise NotImplementedError
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        return res


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )


    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        return batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=None,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        # print(len(self)) keep same, 37764
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                # tgt_min_len = np.argsort(self.tgt_sizes[indices])[0]
                # print(self.tgt_sizes[indices[tgt_min_len]])
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]


    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        # # # return src_item, tgt_item
        # example = {
        #     "id": index,
        #     "source": src_item,
        #     "target": tgt_item,
        # }
        # if self.align_dataset is not None:
        #     example["alignment"] = self.align_dataset[index]
        # if self.constraints is not None:
        #     example["constraints"] = self.constraints[index]
        # return example

        return index, src_item, tgt_item


    def __len__(self):
        return len(self.src)


class EpochBatchIterator(object):
    def __init__(self, dataset, collate_fn, batch_sampler, num_workers=1, seed=1, epoch=1, timeout=0):
        self.dataset = dataset
        self.epoch = max(epoch, 1)
        self.seed = seed
        self.timeout = timeout
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self._frozen_batches = (
            tuple(batch_sampler) if not callable(batch_sampler) else None
        )
        self.num_workers = num_workers
        self._next_epoch_itr = None
        self._cur_epoch_itr = None

    @property
    def frozen_batches(self):
        if self._frozen_batches is None:
            self._frozen_batches = tuple(self.batch_sampler(self.dataset, self.epoch))
        return self._frozen_batches

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        self.epoch = self.next_epoch_idx
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            if callable(self.batch_sampler):
                # reset _frozen_batches to refresh the next epoch
                self._frozen_batches = None
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle,
                fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.shuffle = shuffle
        return self._cur_epoch_itr

    @property
    def frozen_batches(self):
        if self._frozen_batches is None:
            self._frozen_batches = tuple(self.batch_sampler(self.dataset, self.epoch))
        # print(self._frozen_batches[0])
        return self._frozen_batches

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            with numpy_seed(seed):
                np.random.shuffle(batches)
            return batches
        if shuffle:
            batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            # print(self.seed + epoch, self.frozen_batches[1])
        else:
            batches = self.frozen_batches

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
        return finalDataset(self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
        )

        # # Create data loader
        # itr = torch.utils.data.DataLoader(
        #     self.dataset,
        #     collate_fn=self.collate_fn,
        #     batch_sampler=batches[offset:],
        #     num_workers=self.num_workers,
        #     timeout=self.timeout,
        # )

    def __len__(self):
        return int(math.ceil(len(self.frozen_batches) / float(self.num_shards)))


class finalDataset(object):
    def __init__(self, dataset, collate_fn, batch_sampler, num_workers=1, seed=1, timeout=0):
        self.dataset = dataset
        self.seed = seed
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.timeout = timeout

        self.sample_iter = iter(self.batch_sampler)
        self.__initialized = True


    def __len__(self):
        return len(self.batch_sampler)

    def __getitem__(self, index):
        idx_list = self.batch_sampler[index]
        samples = []
        for idx in idx_list:
            i, src_item, tgt_item = self.dataset[idx]
            tmp_item_dict = {'id': i, 'source': src_item, 'target': tgt_item}
            samples.append(tmp_item_dict)

        res = self.collate_fn(samples)
        return res['id'], res['nsentences'], res['ntokens'], res['net_input']['src_tokens'], \
               res['net_input']['src_lengths'], res['net_input']['prev_output_tokens'], res['target']







if __name__ == '__main__':
    args = get_args()
    src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    trg_path = os.path.join(args.data, "dict.{}.txt".format(args.target_lang))
    trg_dictionary = Dictionary.load(trg_path)
    src_dictionary = Dictionary.load(src_path)
    # load_dataset(args, src_dictionary, trg_dictionary, 'valid', combine=False)



    ############################
    # # # fairseq/data/indexed_dataset   MMapIndexedDataset(torch.utils.data.Dataset):
    # tmp_path = index_file_path(_path)
    # _index = Index(tmp_path)
    # _warmup_mmap_file(data_file_path(_path))
    #
    # _bin_buffer_mmap = np.memmap(
    #     data_file_path(_path), mode="r", order="C"
    # )
    # _bin_buffer = memoryview(_bin_buffer_mmap)
    #
    # print(_index)
    ###############################################

    ##########################################
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

    # print(len(src_dataset[0]), len(tgt_dataset[0])) 52, 41


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
    # print(myPairDataset)

    # src_item, tgt_item = myPairDataset[0]
    # print(src_item, tgt_item)

    # example = myPairDataset[0]
    # print(example)
    ###########################


    # myTask = FairseqMsTask(args)
    # # train loader params
    # ignore_invalid_inputs = True
    # num_shards = 1
    # shard_id = 0
    # epoch = 1
    # disable_iterator_cache = False

    # batch_iterator = myTask.get_batch_iterator(myPairDataset, ignore_invalid_inputs=ignore_invalid_inputs,
    #     num_shards=num_shards, shard_id=shard_id,
    #     epoch=epoch, disable_iterator_cache=disable_iterator_cache)

    ################################################
    # Test BatchSampler
    max_tokens = args.max_tokens
    max_sentences = None
    required_batch_size_multiple = args.required_batch_size_multiple

    seed = 1
    with numpy_seed(seed):
        indices = myPairDataset.ordered_indices()
    # print(len(indices))  # same seed=1, len=37764
    # print(src_dataset[indices[0]], tgt_dataset[indices[0]])  #MinLen: 2

    batch_sampler = myPairDataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )
    #############################################
    # print(len(batch_sampler[0]), len(batch_sampler[1]), len(batch_sampler[2]))  # 72, 240, 144
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
    # ######################################################
    for data in mRASPDataset.create_dict_iterator():
        print(data)
        break
    # thisBatch = finaldataset[0]










