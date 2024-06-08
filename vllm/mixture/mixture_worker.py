from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import MixtureSampler
from vllm.sequence import (ExecuteModelRequest, SamplerOutput,
                           SequenceGroupMetadata)

from vllm.spec_decode.util import nvtx_range
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase, WorkerBase

logger = init_logger(__name__)


def create_mix_worker(*args, **kwargs) -> "MixtureWorker":
    """Helper method that is the entrypoint for Executors which use
    WorkerWrapper. It constructs a MixtureWorker from the mixture config.
    """
    assert "mixture_config" in kwargs
    mixture_config = kwargs.get("mixture_config")
    assert mixture_config is not None

    target_worker = Worker(*args, **kwargs)

    mixin_worker_kwargs = kwargs.copy()

    mixin_worker_kwargs.update(
        model_config=mixture_config.draft_model_config,
        parallel_config=mixture_config.draft_parallel_config,
        # TODO allow base-model specific load config.
        #load_config=load_config,
    )

    mixture_worker = MixtureWorker.create_worker(
        scorer_worker=target_worker,
        draft_worker_kwargs=mixin_worker_kwargs,
    )

    return mixture_worker

class MixtureWorker(LoraNotSupportedWorkerBase):
    """Worker for mixing logits
    scorer <-- the main model
    draft <-- the one whose logits are getting mixed in
    """

    @classmethod
    def create_worker(
        cls,
        scorer_worker: WorkerBase,
        draft_worker_kwargs: Dict[str, Any],
    ) -> "MixtureWorker":

        proposer_worker = Worker(**draft_worker_kwargs)

        return MixtureWorker(
            proposer_worker,
            scorer_worker
        )
    def __init__(
        self,
        proposer_worker: WorkerBase,
        scorer_worker: WorkerBase,
    ):
        """
        Create a MixtureWorker.

        """
        self.proposer_worker = proposer_worker
        self.scorer_worker = scorer_worker

        #self.probs_dtype = self.rejection_sampler.probs_dtype
        #self.token_id_dtype = self.rejection_sampler.token_id_dtype


    def init_device(self) -> None:
        """Initialize both scorer and proposer models.
        """
        # The scorer worker model is initialized first in case the proposer
        # model has a smaller TP degree than the target worker.
        self.scorer_worker.init_device()
        self.proposer_worker.init_device()

        # NOTE(cade): load_model is not part of the WorkerBase interface.
        self.scorer_worker.load_model()
        self.proposer_worker.load_model()

        self.mixture_sampler = MixtureSampler(vocab_size=self._vocab_size)

        #TODO: how to set these?
        #self.probs_dtype = self.rejection_sampler.probs_dtype
        #self.token_id_dtype = self.rejection_sampler.token_id_dtype

        self._configure_model_sampler_for_mixture()

    def load_model(self, *args, **kwargs):
        pass

    def _configure_model_sampler_for_mixture(self):
        """Configure model sampler to emit GPU tensors. This allows spec decode
        to keep data on device without transferring to CPU and serializing,
        which significantly reduces overhead of rejection sampling.

        NOTE(cade): This breaks abstraction boundaries pretty badly. The better
        design is to have the "move to CPU and serialize" sampling decision be
        done outside of the model/sampler; this way the "last-mile" worker
        object which interfaces with the scheduler can serialize and incur the
        performance hit as necessary. This allows us to run the worker several
        iterations in a row without incurring the "move to CPU and serialize"
        performance penalty.

        Since this requires a large change to vLLM, we defer it to later and
        temporarily accept this broken abstraction boundary.

        NOTE(cade): This will require a special check if the proposer worker
        does not have a sampler (e.g. ngram speculation).
        """
        self.scorer_worker.model_runner.model.sampler.include_gpu_logits_tensor = True
        self.proposer_worker.model_runner.model.sampler.include_gpu_logits_tensor = True

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of cache blocks to use.

        This is done by profiling the scorer model (which is typically the
        larger of the two). Then the total memory which would be used by the
        scorer cache is divided evenly between the proposer and scorer model KV,
        such that the number of blocks is equal in both KV caches.
        """
        num_gpu_blocks, num_cpu_blocks = (
            self.scorer_worker.determine_num_available_blocks())

        scorer_cache_block_size_bytes = (
            self.scorer_worker.get_cache_block_size_bytes())
        proposer_cache_block_size_bytes = (
            self.proposer_worker.get_cache_block_size_bytes())

        new_num_gpu_blocks = split_num_cache_blocks_evenly(
            scorer_cache_block_size_bytes, proposer_cache_block_size_bytes,
            num_gpu_blocks)
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the cache engine of the scorer and proposer workers.
        """
        self.scorer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                            num_cpu_blocks=num_cpu_blocks)
        self.proposer_worker.initialize_cache(num_gpu_blocks=num_gpu_blocks,
                                              num_cpu_blocks=num_cpu_blocks)

    @torch.inference_mode()
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None
    ) -> List[SamplerOutput]:
        """Perform logit mixing on the input batch.
        """
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []

        if execute_model_req is None:
            # This signals that there's no more requests to process for now.
            # All workers are running infinite loop with broadcast_tensor_dict,
            # and it stops the loop when the driver broadcasts an empty input.
            # Send an empty input to notify all other workers to stop their
            # execution loop.
            broadcast_tensor_dict({}, src=0)
            return []

        return self._run_logit_mix_step(execute_model_req)


    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        """Execute model loop to perform speculative decoding
        in parallel worker."""
        while self._run_non_driver_rank():
            pass

    def _run_non_driver_rank(self) -> bool:
        """Run proposer and verifier model in non-driver workers. This is used
        for both speculation cases (num_lookahead_slots>0) and non-speculation
        cases (e.g. prefill).

        Returns True iff there are remaining sequences to process.
        """
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            return False


        # todo hunter: is this right?
        self.proposer_worker.execute_model()
        self.scorer_worker.execute_model()
        return True

    @nvtx_range("mixture_worker._run_logit_mix_step")
    def _run_logit_mix_step(
            self, execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        """Execute a single step of logit mixture.

        Returns a list of SamplerOutput, each containing a single token per
        sequence.
        """

        scorer_sample_output = self.scorer_worker.execute_model(execute_model_req)[0]
        proposer_sample_output = self.proposer_worker.execute_model(execute_model_req)[0]

        # recompute this...todo can i do this better?
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list
        (_, _, _, sampling_metadata,
         _, _, _) = self.proposer_worker.model_runner.prepare_input_tensors(seq_group_metadata_list)

        logits1 = scorer_sample_output.logits
        logits2 = proposer_sample_output.logits

        output = self.mixture_sampler(logits1, logits2, sampling_metadata=sampling_metadata)

        return [output]

    @cached_property
    def _vocab_size(self) -> int:
        """Get the vocab size of the model and make sure it's consistent between
        draft and target workers.
        """
        vocab_sizes = [
            worker.vocab_size
            for worker in [self.proposer_worker, self.scorer_worker]
        ]
        assert all(vocab_sizes[0] == vocab_size for vocab_size in vocab_sizes)
        return vocab_sizes[0]

    @property
    def rank(self):
        return self.scorer_worker.rank

    @property
    def device(self):
        return self.scorer_worker.device

    @property
    def _driver_rank(self) -> int:
        return 0

    def get_cache_block_size_bytes(self):
        """Return the size of a cache block in bytes.

        This function is only used to compose workers within a SpecDecodeWorker.
        We leave composing a SpecDecodeWorker within a SpecDecodeWorker
        undefined for now, although it could be implemented in the future.
        See https://arxiv.org/abs/2308.04623.
        """
        raise NotImplementedError


def split_num_cache_blocks_evenly(scorer_cache_block_size_bytes: int,
                                  proposer_cache_block_size_bytes: int,
                                  total_num_gpu_blocks: int) -> int:
    """Given total_num_gpu_blocks, the number of GPU blocks that could be
    allocate to the target model, this function calculates how many blocks
    should be given to the draft and target model.

    Note that usually the block size, in bytes, of each model is different,
    as it's a function of number of KV/layer, number of heads, and hidden
    dimension size.

    Since the target and draft models allocate the same number of blocks, we
    simply calculate the number of blocks where if allocated by both models,
    the total memory usage from KV cache is no larger than the number of
    blocks allocatable by the target model alone.
    """
    new_num_gpu_blocks = int(
        total_num_gpu_blocks * scorer_cache_block_size_bytes /
        (proposer_cache_block_size_bytes + scorer_cache_block_size_bytes))

    return new_num_gpu_blocks
