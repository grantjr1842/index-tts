from typing import Any, Protocol, List, Optional

class NormalizeEmoVecFn(Protocol):
    def __call__(self, emo_vector: List[float], apply_bias: bool = True) -> List[float]:
        ...


class InferFn(Protocol):
    def __call__(
        self,
        *,
        spk_audio_prompt: Optional[str],
        text: str,
        output_path: Optional[str],
        emo_audio_prompt: Optional[str] = None,
        emo_alpha: float = 1.0,
        emo_vector: Optional[List[float]] = None,
        use_emo_text: bool = False,
        emo_text: Optional[str] = None,
        use_random: bool = False,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        **generation_kwargs: Any,
    ) -> Any:
        ...


class IndexTTS2Client(Protocol):
    normalize_emo_vec: NormalizeEmoVecFn
    infer: InferFn
