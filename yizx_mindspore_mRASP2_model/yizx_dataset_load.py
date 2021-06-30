# code for translation datasetload
# From fairseq/tasks/translation
#######################################
import os
from typing import List

from yizx_mindspore_utils import get_args
from yizx_mindspore_decoder import Dictionary

import struct
import numpy as np

from functools import lru_cache

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

def split_paths(paths: str) -> List[str]:
    return (
        paths.split(os.pathsep)
        if "://" not in paths
        else paths.split(MANIFOLD_PATH_SEP)
    )

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


        # print(len(self._sizes))  # 12004

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

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
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

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        return src_item, tgt_item


    def __len__(self):
        return len(self.src)

if __name__ == '__main__':
    args = get_args()
    src_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
    trg_path = os.path.join(args.data, "dict.{}.txt".format(args.target_lang))
    trg_dictionary = Dictionary.load(trg_path)
    src_dictionary = Dictionary.load(src_path)
    load_dataset(args, src_dictionary, trg_dictionary, 'valid', combine=False)



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
    paths_list = [args.data + '/valid.src-trg.src', args.data + '/valid.src-trg.trg']

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
    # print(myPairDataset)

    src_item, tgt_item = myPairDataset[0]
    print(src_item, tgt_item)
    ###########################


