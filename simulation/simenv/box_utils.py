def line_intersection(p1, p2, p3, p4):
    """
    Finds the intersection of two line segments (if any), handling vertical and parallel cases.
    """
    x1, y1 = p1
    x2, y2 = p2

    x3, y3 = p3
    x4, y4 = p4

    # Compute denominator
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denom == 0:
        return None  # Parallel or coincident lines

    # Special case: if one of the lines is vertical
    if x1 == x2:  # Line 1 is vertical
        x = x1
        if x3 != x4:
            slope2 = (y4 - y3) / (x4 - x3)
            y = y3 + slope2 * (x - x3)
        else:
            return None  # Both lines vertical
    elif x3 == x4:  # Line 2 is vertical
        x = x3
        if x1 != x2:
            slope1 = (y2 - y1) / (x2 - x1)
            y = y1 + slope1 * (x - x1)
        else:
            return None  # Both lines vertical
    else:
        # Compute intersection normally
        x_num = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        y_num = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        x = x_num / denom
        y = y_num / denom

    # Check if intersection is within both segments
    if (
        min(x1, x2) <= x <= max(x1, x2)
        and min(y1, y2) <= y <= max(y1, y2)
        and min(x3, x4) <= x <= max(x3, x4)
        and min(y3, y4) <= y <= max(y3, y4)
    ):
        return (x, y)

    return None  # Intersection is outside the segments
