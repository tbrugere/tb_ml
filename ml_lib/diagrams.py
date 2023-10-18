"""
Automatically generating diagrams from models.
We use svg.py to generate the diagrams.
"""

from dataclasses import dataclass
import svg

default_style = dict(
    direction="vertical", # or "horizontal", "horizontal" is currently not supported
    gap=10, # gap between boxes, is equal to margin / padding (all assumed to be the same)
)

class DiagramStyle():
    """
        A class for defining the style of the diagram.
        Works like css.
    """

@dataclass
class Diagram():
    
    svg: svg.SVG

class DiagrammableMixin():

    diagram_class = None
    diagram_inline_style = None

    def get_diagram(self, style=None) -> Diagram:
        """
        Returns the diagram of the model.
        Default is to return a box containing the name of the model.
        """
        model_name = self.__class__.__name__

        raise 

