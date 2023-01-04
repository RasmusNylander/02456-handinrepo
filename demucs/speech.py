from math import floor
from pathlib import Path
from random import random

import torch
from demucs.audio import convert_audio
from torch import Tensor
import torch.nn.functional as F

import torchaudio as ta
class SpeechSet:
	def __init__(self, root: Path, track_names: list[str], sources: list[str], samplerate: int, segment: int, channels: int):
		self.root = root
		self.track_names = track_names
		self.sources = sources
		self.samplerate = samplerate
		self.segment = segment
		self.channels = channels

	def __len__(self) -> int:
		return len(self.track_names)

	def get_file(self, name: str, source: str) -> Path:
		source_to_path = {
			"mixture": "mix_single",
			"human": "s1",
			"noise": "noise",
		}

		return self.root / source_to_path[source] / f"{name}.wav"


	def __getitem__(self, index: slice | int) -> Tensor:
		if isinstance(index, slice):
			# metas = self.metadata[index]
			# filepaths = [self.get_file(meta.name, source) for meta in metas for source in ["mixture"] + self.sources]
			# wavs = [convert_audio(wav, wav_samplerate, self.samplerate, self.channels) for (wav, wav_samplerate) in
			# 	[str(ta.load(path)) for path in filepaths]]
			# tracks = torch.stack(wavs, dim=0)
			# tracks = tracks.reshape(len(metas), len(self.sources) + 1, self.channels, -1)
			return torch.stack([self[i] for i in range(*index.indices(len(self)))], dim=0)

		if (index < 0) or (index >= len(self)):
			raise IndexError

		track_name = self.track_names[index]

		files_paths = [self.get_file(track_name, source) for source in ["mixture"] + self.sources]
		wavs = [convert_audio(wav, wav_samplerate, self.samplerate, self.channels) for (wav, wav_samplerate) in
				[ta.load(str(path)) for path in files_paths]]

		example: Tensor = torch.stack(wavs)

		desired_length = int(self.segment * self.samplerate)
		example = example[..., :desired_length]
		example = F.pad(example, (0, desired_length - example.shape[-1]))
		return example

def get_librimix_wav_datasets(root: Path, sources: list[str], samplerate: int, segment: int, audio_channels: int, validation_percentage: float = 0.1) -> tuple[SpeechSet, SpeechSet]:
	assert 0 <= validation_percentage <= 1, f"validation_percentage must be between 0 and 1 but is {validation_percentage}"
	mixturePath: Path = root / "mix_single"
	# metadata: list[TrackMetaData] = [TrackMetaData(file.stat().st_size, file.stem) for file in mixturePath.iterdir()]
	track_names = [file.stem for file in mixturePath.iterdir()]

	# pick a random 10% of the data for validation
	desired_validation_size = int(floor(len(track_names) * validation_percentage))
	validation_tracks: list[str] = []
	while len(validation_tracks) < desired_validation_size:
		index = int(random() * len(track_names))
		validation_tracks.append(track_names.pop(index))

	train_set = SpeechSet(root, track_names, sources, samplerate=samplerate, segment=segment, channels=audio_channels)
	valid_set = SpeechSet(root, validation_tracks, sources, samplerate=samplerate, segment=segment, channels=audio_channels)

	return train_set, valid_set

def get_librimix_wav_testset(testset_root: Path, samplerate: int, segment: int, audio_channels: int) -> SpeechSet:
	return get_librimix_wav_datasets(testset_root, ["human", "noise"], samplerate=samplerate, segment=segment, audio_channels=audio_channels, validation_percentage=0.9)[0]
