"""
Automatically generating diagrams from models.
We use svg.py to generate the diagrams.
"""
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from dataclasses import dataclass
from ml_lib.misc.module_link import lazy_import
if TYPE_CHECKING:
    import svg
else: 
    svg = lazy_import("svg") #why the lazy import? I don't want to need svg.py 
    # everywhere I am using DiagrammableMixin
    # it should only be imported when needed

default_style = dict(
    direction="vertical", 
    gap=10, 
)

@dataclass
class DiagramStyle():
    """
        A class for defining the style of the diagram.
        Works like css.
    """

    direction: Literal["vertical", "horizontal"] = "vertical"
    """or "horizontal", "horizontal" is currently not supported"""

    gap: int =  10
    """gap between boxes, is equal to margin / padding (all assumed to be the same)"""

    font_size: int = 10

    default_box_height: int = 50

@dataclass(frozen=True)
class Id():
    path: tuple[str, ...]

    @classmethod
    def from_str(cls, id: str):
        return Id(tuple(id.split(".")))
    
    @classmethod
    def from_strings(cls, *subpaths):
        return cls.from_str(".".join(subpaths))        

    @classmethod
    def create(cls, other):
        match other:
            case Id(path): return cls(path)
            case str(path): return cls.from_str(path)
            case *paths: return cls.from_strings(paths)
            case _: raise ValueError(f"cannot create id from {other}")

    def to_string(self,) -> str:
        # yes, html ids can have dots. In css, they can be escaped.
        # https://stackoverflow.com/questions/12310090/css-selector-with-period-in-id
        return ".".join(self.path)

    def escaped_string(self) -> str:
        return r"\.".join(self.path)
    def __str__(self) -> str:
        return self.to_string()
    def href(self) -> str:
        return f"#{self}"

SvgType = TypeVar("SvgType", "svg.G", "svg.Use")

@dataclass
class Diagram(Generic[SvgType]):
    svg: "SvgType"

    width: int
    height: int

    def to_use(self, id: Id) -> "Diagram[svg.Use]":
        return Diagram(
            svg= svg.Use(href=id.href()), 
            width=self.width, 
            height=self.height, 
            )


class DiagrammableMixin():

    diagram_class = None
    diagram_inline_style = None

    def get_diagram(self, diagram_maker: "DiagramMaker", id: str) -> "Diagram[svg.G]":
        """
        Returns the diagram of the model.
        Default is to return a box containing the name of the model.
        """
        model_name = self.__class__.__name__
        raise NotImplementedError


class DiagramMaker():

    base_obj: Any
    
    style: DiagramStyle
    diagrams: "dict[Id, Diagram[svg.G]]"
    queue: set[Id]

    def make_diagram(self, obj, id) -> "Diagram[svg.G]":
        match obj:
            case DiagrammableMixin():
                return obj.get_diagram(self, id)
            case _:
                return self.default_diagram(obj, id)

    def __getitem__(self, id_: str|tuple[str, ...]|Id) -> "Diagram[svg.Use]":
        id = Id.create(id_)
        if id in self.diagrams:
            return self.diagrams[id].to_use(id)
        obj = self.get_obj(id)
        diagram = self.make_diagram(obj, id)
        self.diagrams[id] = diagram
        return diagram.to_use(id)
            

    def default_diagram(self, obj, id) -> "Diagram[svg.G]":
        from svg import Length
        model_name = obj.__class__.__name__
        font_size = self.style.font_size
        text = svg.Text(text=model_name, font_size=font_size, 
                        x=Length(50, "%"), y=Length(50, "%"), 
                        text_anchor="middle", dominant_baseline="middle")
        width = len(model_name) * self.style.font_size + 2 * self.style.gap
        height = self.style.default_box_height
        box = svg.Rect(
                x=0, 
                y=0, 
                width= width,
                height = height, 
                class_= ["box"], 
                elements=[text]
                )
        container = svg.G(elements=[box])
        return Diagram(container, width=width, height=height)
        
    def get_obj(self, id: Id):
        o = self.base_obj
        for attr in id.path:
            o = getattr(o, attr)
        return o



