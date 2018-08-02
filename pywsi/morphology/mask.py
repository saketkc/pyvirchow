from shapely.geometry import Polygon as shapelyPolygon
from shapely.geometry import Polygon, box
from shapely.affinity import scale

from matplotlib.patches import Polygon


def mpl_polygon_to_shapely_scaled(polygon, x0=0, y0=0, scale_factor=1):
    """Convert a given matploltib polygon to shapely

    Shapely allows more operations that we really ned
    while matplotlib.polygon is in use for legacy reasons

    Parameters
    ----------
    polygon: matplotlib.Polygon
             Input object
    x0, y0: int
            Origin for scaling
    scale_factor: float
                  How much to scale
    Returns
    -------
    shapely_polygon: Polygon
                     Scaled polygon

    """
    xy = polygon.get_xy()
    shapely_polygon = shapelyPolygon(xy)
    return scale(
        shapely_polygon,
        xfact=scale_factor,
        yfact=scale_factor,
        origin=(x0, y0))


def mpl_rect_to_shapely_scaled(rectangle, x0=0, y0=0, scale_factor=1):
    """Convert matploltb.Rectangle to shapely

    Parameters
    ----------
    rectangle: matplotlib.Rectangle
               Input object
    x0, y0: int
            Origin for scaling
    scale_factor: float
                  How much to scale

    Returns
    -------
    sbox : shapely.box
    """
    minx, miny = rectangle.get_xy()
    height = rectangle.get_height()
    width = rectangle.get_width()
    maxx = minx + width
    maxy = miny + height
    sbox = box(minx, miny, maxx, maxy)
    return scale(sbox, xfact=scale_factor, yfact=scale_factor, origin=(x0, y0))


def get_common_interior_polygons(polygon, list_of_polygons):
    """Check if polygon resides inside any polygon
    in the list_of_polygons.

    Parameters
    ----------
    polygon: matplotlib.Polygon

    Returns
    -------
    list_of_common_polygons: list
                             A filtered list of ids

    """
    if isinstance(polygon, Polygon):
        polygon = shapelyPolygon(polygon.get_xy()).buffer(0)
    list_of_common_polygons = []
    for index, outside_polygon in enumerate(list_of_polygons):
        if isinstance(outside_polygon, Polygon):
            outside_polygon = shapelyPolygon(outside_polygon.get_xy()).buffer(0)
        if polygon.is_valid and outside_polygon.is_valid:
            if polygon.within(outside_polygon):
                list_of_common_polygons.append(index)
    return list_of_common_polygons
