"""Tecnicas de pre-processamento de imagens."""

from .contrast_expansion import ContrastExpansion
from .histogram_equalization import HistogramEqualization


PREPROCESSING_OPTIONS = (
    "none",
    "contrast_expansion",
    "histogram_equalization",
)


def get_preprocessing_transform(name):
    """Retorna a transformacao escolhida ou None para preservar a imagem."""
    if name == "none":
        return None
    if name == "contrast_expansion":
        return ContrastExpansion()
    if name == "histogram_equalization":
        return HistogramEqualization()

    choices = ", ".join(PREPROCESSING_OPTIONS)
    raise ValueError(
        f"Pre-processamento nao suportado: {name!r}. Opcoes: {choices}"
    )
