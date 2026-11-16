"""Expansao linear de contraste para imagens PIL."""

from PIL import Image, ImageEnhance


class ContrastExpansion:
    def __init__(self, factor=1.5):
        if factor < 0:
            raise ValueError("factor deve ser maior ou igual a zero")
        self.factor = factor

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("ContrastExpansion espera uma imagem PIL")
        return ImageEnhance.Contrast(image).enhance(self.factor)
