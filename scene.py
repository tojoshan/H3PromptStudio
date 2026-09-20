from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class VideoMode(str, Enum):
    T2V = "t2v"
    I2V = "i2v"
    FIRST_LAST = "first_last"
    REFERENCE = "reference"

class AspectRatio(str, Enum):
    WIDE = "21:9"
    LANDSCAPE = "16:9"
    CLASSIC = "4:3"
    SQUARE = "1:1"
    PORTRAIT = "3:4"
    VERTICAL = "9:16"

class QualityPreset(str, Enum):
    PREVIEW = "Preview"
    MEDIUM = "Medium"
    NATIVE = "Native 768p"

RESOLUTION_PRESETS = {
    "21:9": {"Preview": (960, 416), "Medium": (1152, 512), "Native 768p": (1760, 768)},
    "16:9": {"Preview": (864, 480), "Medium": (1056, 608), "Native 768p": (1344, 768)},
    "4:3":  {"Preview": (640, 480), "Medium": (800, 608),  "Native 768p": (1024, 768)},
    "1:1":  {"Preview": (480, 480), "Medium": (608, 608),  "Native 768p": (768, 768)},
    "3:4":  {"Preview": (480, 640), "Medium": (608, 800),  "Native 768p": (768, 1024)},
    "9:16": {"Preview": (480, 864), "Medium": (608, 1056), "Native 768p": (768, 1344)},
}

def resolve_dimensions(aspect_ratio: str, quality: str) -> tuple[int, int]:
    try:
        return RESOLUTION_PRESETS[aspect_ratio][quality]
    except KeyError as exc:
        raise ValueError(f"Preset H3 inválido: {aspect_ratio} / {quality}") from exc

class Reference(BaseModel):
    path: str
    role: Literal["character", "style", "environment", "object", "other"] = "character"
    description: str = ""

class Subject(BaseModel):
    name: str = ""
    appearance: str = ""
    identity_constraints: list[str] = Field(default_factory=list)

class Camera(BaseModel):
    shot: str = ""
    movement: str = ""
    angle: str = ""
    lens_feel: str = ""

class SceneContent(BaseModel):
    subject: Subject = Field(default_factory=Subject)
    environment: str = ""
    action: str = ""
    secondary_action: str = ""
    style: str = ""
    lighting: str = ""
    camera: Camera = Field(default_factory=Camera)
    must_preserve: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)

class GenerationConfig(BaseModel):
    duration_seconds: float = Field(default=5.0, ge=1.0, le=15.0)
    fps: Literal[24] = 24
    aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE
    quality: QualityPreset = QualityPreset.NATIVE
    width: int | None = None
    height: int | None = None
    seed: int = -1

    @model_validator(mode="after")
    def calculate_resolution(self):
        self.width, self.height = resolve_dimensions(
            self.aspect_ratio.value, self.quality.value
        )
        return self

class SceneSpec(BaseModel):
    mode: VideoMode = VideoMode.T2V
    user_request: str
    content: SceneContent
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    first_frame: str | None = None
    last_frame: str | None = None
    references: list[Reference] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mode_assets(self):
        if self.mode == VideoMode.I2V and not self.first_frame:
            raise ValueError("i2v requiere first_frame")
        if self.mode == VideoMode.FIRST_LAST and (not self.first_frame or not self.last_frame):
            raise ValueError("first_last requiere first_frame y last_frame")
        if self.mode == VideoMode.REFERENCE and not self.references:
            raise ValueError("reference requiere al menos una referencia")
        return self
