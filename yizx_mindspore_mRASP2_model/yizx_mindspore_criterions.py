import mindspore
import mindspore.nn as nn
from yizx_mindspore_utils import ExpandDims, ReduceSum, Equal, masked_fill, item
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
import numpy as np

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    # lprobs.shape [3144, 64871]; target.shape [3144]; epsilon=0.1
    ops_unsqueeze = ExpandDims()
    if target.dim() == lprobs.dim() - 1:
        target = ops_unsqueeze(target, -1)
    ops_gather = ops.GatherD()
    nll_loss = ops_gather(-lprobs, -1, target)          # -lprobs.gather(dim=-1, index=target)
    ops_sum_keep = ReduceSum(keep_dims=True)
    ops_sum = ReduceSum(keep_dims=False)
    smooth_loss = ops_sum_keep(-lprobs, axis=-1)             # -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        ops_equal = Equal()
        pad_mask = ops_equal(target, ignore_index)      # target.eq(ignore_index)
        nll_loss = masked_fill(nll_loss, pad_mask, ops_unsqueeze, 0)
        smooth_loss = masked_fill(smooth_loss, pad_mask, ops_unsqueeze, 0)
        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        ops_squeeze = ops.Squeeze(-1)
        nll_loss = ops_squeeze(nll_loss)            # nll_loss.squeeze(-1)
        smooth_loss = ops_squeeze(smooth_loss)      # smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = ops_sum(nll_loss)                # nll_loss.sum()
        smooth_loss = ops_sum(smooth_loss)             # smooth_loss.sum()
    eps_i = epsilon / lprobs.shape[-1]
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


class LabelSmoothedCrossEntropyCriterion(nn.Cell):
    def __init__(
        self,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__()
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def construct(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].shape[0] if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss,
            "nll_loss": nll_loss,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].shape[0],
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            pass
            # n_correct, total = self.compute_accuracy(model, net_output, sample)
            # logging_output["n_correct"] = item(n_correct)
            # logging_output["total"] = item(total)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :]
                target = target[:, self.ignore_prefix_size:]
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :]
                target = target[self.ignore_prefix_size:, :]
        return lprobs.view(-1, lprobs.shape[-1]), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss



if __name__ == '__main__':
    sentence_avg = False
    label_smoothing = 0.1
    ignore_prefix_size = 0
    report_accuracy = False
    myloss = LabelSmoothedCrossEntropyCriterion(sentence_avg, label_smoothing)


    # ########################################################################
    # # Test label_smoothed_nll_loss with fairseq_source_code
    # lprobs = Tensor(np.random.randn(3144, 64871), mstype.float32)
    # target = Tensor(np.random.randn(3144), mstype.int64)
    # epsilon = 0.1
    # loss, nll_loss = label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=1, reduce=True)
    # print(loss.dtype, nll_loss)  # float32
    # ##########################################################################




