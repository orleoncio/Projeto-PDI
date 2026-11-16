"""Equalizacao global do histograma para imagens PIL."""

from PIL import Image, ImageOps


class HistogramEqualization:
    """Equaliza o histograma por canal, mantendo a imagem em RGB."""

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("HistogramEqualization espera uma imagem PIL")

        if image.mode == "RGB":
            channels = image.split()
            equalized = [ImageOps.equalize(channel) for channel in channels]
            return Image.merge("RGB", equalized)

        return ImageOps.equalize(image)
