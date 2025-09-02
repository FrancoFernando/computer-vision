def trilinear_interpolation(grid_3d, x, y, z):
    """
    Perform trilinear interpolation on a 3D grid
    grid_3d: 3D numpy array containing the data
    x, y, z: coordinates in grid space (0 to grid_size-1)
    """
    
    # Get the dimensions of the grid
    depth, height, width = grid_3d.shape
    
    # Get integer parts (corners of the cell)
    x0, y0, z0 = int(x), int(y), int(z)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    z1 = min(z0 + 1, depth - 1)
    
    # Get fractional parts
    tx = x - x0
    ty = y - y0
    tz = z - z0
    
    # Get the eight corner values
    c000 = grid_3d[z0, y0, x0]
    c100 = grid_3d[z0, y0, x1]
    c010 = grid_3d[z0, y1, x0]
    c110 = grid_3d[z0, y1, x1]
    c001 = grid_3d[z1, y0, x0]
    c101 = grid_3d[z1, y0, x1]
    c011 = grid_3d[z1, y1, x0]
    c111 = grid_3d[z1, y1, x1]
    
    # First, perform bilinear interpolation on the front face (z0)
    e = bilinear_interpolate(tx, ty, c000, c100, c010, c110)
    
    # Then, perform bilinear interpolation on the back face (z1)
    f = bilinear_interpolate(tx, ty, c001, c101, c011, c111)
    
    # Finally, interpolate between the two faces along z-axis
    result = e * (1 - tz) + f * tz
    
    return result

def bilinear_interpolate(tx, ty, c00, c10, c01, c11):
    """Helper function for bilinear interpolation"""
    a = c00 * (1 - tx) + c10 * tx
    b = c01 * (1 - tx) + c11 * tx
    return a * (1 - ty) + b * ty
