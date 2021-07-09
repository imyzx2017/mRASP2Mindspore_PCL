import os
from collections import Counter
from multiprocessing import Pool
import os
import shutil
from typing import List, Optional
try:
    from fvcore.common.file_io import PathManager as FVCorePathManager

except ImportError:
    FVCorePathManager = None

class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if FVCorePathManager:
            return FVCorePathManager.open(
                path=path,
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if FVCorePathManager:
            return FVCorePathManager.copy(
                src_path=src_path, dst_path=dst_path, overwrite=overwrite
            )
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        if FVCorePathManager:
            return FVCorePathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: str) -> bool:
        if FVCorePathManager:
            return FVCorePathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        if FVCorePathManager:
            return FVCorePathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def ls(path: str) -> List[str]:
        if FVCorePathManager:
            return FVCorePathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        if FVCorePathManager:
            return FVCorePathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if FVCorePathManager:
            return FVCorePathManager.rm(path)
        os.remove(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        if "manifold" not in path:
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) -> None:
        if FVCorePathManager:
            return FVCorePathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs
    ) -> None:
        if FVCorePathManager:
            return FVCorePathManager.copy_from_local(
                local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
            )
        return shutil.copyfile(local_path, dst_path)


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _load_meta(self, lines):
        return 0




if __name__ == '__main__':
    data_dir = '/userhome/jobs/NMTrans/mRASP-master/toy/data/pre-train'
    source_lang = 'src'
    target_lang = 'trg'
    src_path = os.path.join(data_dir, "dict.{}.txt".format(source_lang))
    trg_path = os.path.join(data_dir, "dict.{}.txt".format(target_lang))
    myDictionary = Dictionary()
    src_dictionary = Dictionary.load(src_path)
    trg_dictionary = Dictionary.load(trg_path)
    print(src_dictionary.__len__(), src_dictionary[0])