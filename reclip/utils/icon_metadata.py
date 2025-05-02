
from enum import Enum
from dataclasses import dataclass

class IconShape(str, Enum):
    SQUARE = "square"
    CIRCLE = "circle"
    ROUNDED = "rounded"

class VisualStyle(str, Enum):
    FLAT = "flat"
    LINE = "line"
    GLYPH = "glyph"
    FILLED = "filled"

class ColorMode(str, Enum):
    MONOCHROME = "monochrome"
    DUOTONE = "duotone"
    MULTICOLOR = "multicolor"

class ContainerType(str, Enum):
    ICON = "icon"
    BUTTON = "button"
    BADGE = "badge"
    AVATAR = "avatar"

class DetailLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"

@dataclass
class IconMetadata:
    tags: list[str]
    shape: IconShape = IconShape.SQUARE
    visual_style: VisualStyle = VisualStyle.FLAT
    color_mode: ColorMode = ColorMode.MULTICOLOR
    container: ContainerType = ContainerType.ICON
    detail: DetailLevel = DetailLevel.MINIMAL

    def to_prompt(self) -> str:
        tag_str = ", ".join(self.tags)
        return (
            f"A {self.detail.value}, {self.color_mode.value} {self.shape.value} {self.visual_style.value}-style "
            f"{self.container.value} showing {tag_str}."
        )
