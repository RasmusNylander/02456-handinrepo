# Copyright (c) Meta, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test time evaluation, either using the original SDR from [Vincent et al. 2006]
or the newest SDR definition from the MDX 2021 competition (this one will
be reported as `nsdr` for `new sdr`).
"""

from concurrent import futures
import logging
from pathlib import Path
from typing import Optional

import torch
from numpy import ndarray
from torch import Tensor

from dora.log import LogProgress
import numpy as np
import museval
import torch as th

from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs import distrib
from demucs.utils import DummyPoolExecutor
from demucs.speech import get_librimix_wav_testset


logger = logging.getLogger(__name__)


def new_sdr(references: Tensor, estimates: Tensor) -> Tensor:
    """
    Compute the SDR according to the MDX challenge definition.
    Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
    """
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta: float = 1e-7  # avoid numerical errors
    numerator: Tensor = th.sum(th.square(references), dim=(2, 3))
    denominator: Tensor = th.sum(th.square(references - estimates), dim=(2, 3))
    numerator += delta
    denominator += delta
    scores = 10 * th.log10(numerator / denominator)
    return scores


def eval_track(references: ndarray, estimates: ndarray, win: int, hop: int) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    num_windows = (references.shape[2] - win + hop) // hop

    # references = references.transpose((2, -1))
    # estimates = estimates.transpose((-2, -1))

    sdr = np.empty((references.shape[0], references.shape[1], num_windows))
    isr = np.empty((references.shape[0], references.shape[1], num_windows))
    sir = np.empty((references.shape[0], references.shape[1], num_windows))
    sar = np.empty((references.shape[0], references.shape[1], num_windows))
    for i, (track_estimate, track_sources) in enumerate(zip(estimates, references)):
        sdr[i], isr[i], sir[i], sar[i], _ = museval.metrics.bss_eval(
            track_sources, track_estimate,
            compute_permutation=False,
            window=win,
            hop=hop,
            framewise_filters=False,
            bsseval_sources_version=False)
    return sdr, isr, sir, sar


def evaluate_speech(model, samplerate: int, segment_length: int, audio_channels: int, number_of_prints: int, number_of_workers: Optional[int], save_separated_wavs_path: Optional[Path]) -> dict[str, float]:
    """
    Evaluate the model on the speech dataset.
    """

    # audio_sources = model.sources
    audio_sources = ["noise", "human"]

    test_set_root = Path("/work3/projects/02456/project04/librimix/Libri2Mix/wav16k/max/test")

    test_set = get_librimix_wav_testset(test_set_root, samplerate, segment_length, audio_channels)

    eval_device = 'cuda'

    batch_size = 20
    indicies = range(distrib.rank * batch_size, len(test_set), distrib.world_size * batch_size)

    indicies = LogProgress(logger, indicies, updates=number_of_prints, name='Eval')

    pendings = []

    # pool = futures.ProcessPoolExecutor if number_of_workers else DummyPoolExecutor
    # with pool(number_of_workers) as pool:
    for index in indicies:
        tracks: Tensor = th.stack([test_set[i].cuda() for i in range(index, index + batch_size) if i < len(test_set)], 0)
        mix: Tensor = tracks[:, 0, ...]
        sources: Tensor = tracks[:, 1:, ...]
        estimates: Tensor = apply_model(model, mix, device=eval_device)[:, 2:4]

        if save_separated_wavs_path is not None:
            track_names = test_set.track_names[index:(index + batch_size)]
            save_directory = save_separated_wavs_path / "wav"
            save_batched_estimates(["mixture"] + audio_sources, torch.cat((mix.unsqueeze(1), estimates), 1), samplerate, save_directory, track_names)


        # pendings.append((test_set.metadata[index].name, pool.submit(eval_track, sources, estimates)))
        # for track_estimate, track_sources in zip(estimates, sources):
        #     pendings.append((test_set.metadata[index].name, pool.submit(eval_track_old, track_sources, track_estimate, int(2.0 * samplerate), int(1.5 * samplerate))))
        # pendings.append((test_set.metadata[index].name, pool.submit(eval_track_old, sources.detach().cpu(), estimates.detach().cpu(), int(4.0 * samplerate), int(2.5 * samplerate))))
        sources = sources.transpose(2, 3)
        estimates = estimates.transpose(2, 3)
        pendings.append((str(index), eval_track(sources.detach().cpu().numpy(), estimates.detach().cpu().numpy(), int(4.0 * samplerate), int(2.5 * samplerate))))

    # pendings = LogProgress(logger, pendings, updates=number_of_prints, name='Eval (BSS)')

    print("About to start evaluating")
    #------------
    track_results: dict[str, dict[str, dict[str, ndarray]]] = {}
    for batch_id, evaluation in pendings:
        evaluation_result: tuple[ndarray, ndarray, ndarray, ndarray] = evaluation
        sdr, isr, sir, sar = evaluation_result
        sdr[sdr == float('-inf')] = float('nan'); sdr[sdr == float('inf')] = float('nan')
        isr[isr == float('-inf')] = float('nan'); isr[isr == float('inf')] = float('nan')
        sir[sir == float('-inf')] = float('nan'); sir[sir == float('inf')] = float('nan')
        sar[sar == float('-inf')] = float('nan'); sar[sar == float('inf')] = float('nan')
        for batch_component in range(sdr.shape[0]):
            track_results[batch_id + str(batch_component)] = {}
            for idx, target in enumerate(audio_sources):
                track_results[batch_id + str(batch_component)][target] = {
                    "SDR": sdr[batch_component][idx],
                    "SIR": sir[batch_component][idx],
                    "ISR": isr[batch_component][idx],
                    "SAR": sar[batch_component][idx]
                }

    all_tracks: dict = {}
    for src in range(distrib.world_size):
        all_tracks.update(distrib.share(track_results, src))

    result: dict[str, float] = {}
    metric_names = next(iter(all_tracks.values()))[audio_sources[0]]
    for metric_name in metric_names:
        avg = 0
        avg_of_medians = 0
        for source in audio_sources:
            medians = [
                np.nanmedian(all_tracks[track][source][metric_name])
                for track in all_tracks.keys()]
            mean = np.mean(medians)
            median = np.median(medians)
            result[metric_name.lower() + "_" + source] = mean
            result[metric_name.lower() + "_med" + "_" + source] = median
            avg += mean / len(audio_sources)
            avg_of_medians += median / len(audio_sources)
        result[metric_name.lower()] = avg
        result[metric_name.lower() + "_med"] = avg_of_medians
    return result


def save_batched_estimates(audio_sources: list[str], estimates: Tensor, samplerate: int, save_directory: Path, track_names: list[str]) -> None:
    for name, track_estimate in zip(track_names, estimates):
        track_folder = save_directory / name
        track_folder.mkdir(exist_ok=True, parents=True)
        for audio_source, estimate in zip(audio_sources, track_estimate):
            save_audio(estimate, track_folder / (audio_source + ".wav"), samplerate)
