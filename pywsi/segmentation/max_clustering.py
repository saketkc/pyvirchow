import numpy as np

UNIQUE_PRIME = 15331


def max_clustering(image, foreground_mask, radius):
    width, height = image.shape
    px, py = np.nonzero(foreground_mask)

    local_max_value = np.zeros([width, height], dtype=np.float)
    local_max_index = np.zeros([width, height], dtype=np.int)
    peak_found = np.zeros([width, height], dtype=np.int)
    lowest_pixel = np.nanmin(image)

    for x, y in zip(px, py):
        # Search in a circle of radius r
        peak_found_nearby = False
        nearby_max_x = x
        nearby_max_y = y
        for xr in np.arange(-radius, radius + 1):
            nearby_x = x + xr
            if nearby_x < 0 or nearby_x > width:
                continue
            for yr in np.arange(-radius, radius + 1):
                nearby_y = y + yr

                if nearby_y < 0 or nearby_y > height:
                    continue

                if (nearby_x**2 + nearby_y**2) > radius**2:
                    continue

                pixel_value_xy = image[x, y]
                nearby_pixel_value_xy = lowest_pixel

                # Is it part of the foreground?
                if foreground_mask[nearby_x, nearby_y]:
                    nearby_pixel_value_xy = image[nearby_x, nearby_y]

                if nearby_pixel_value_xy > pixel_value_xy:
                    peak_found_nearby = True
                    pixel_value_xy = nearby_pixel_value_xy
                    nearby_max_x = x
                    nearby_max_y = y

        local_max_index[x, y] = nearby_max_x * UNIQUE_PRIME + nearby_max_y
        local_max_value[x, y] = pixel_value_xy

        if not peak_found_nearby:
            # Assign peak at this x,y
            peak_found[x, y] = 1

    # Reassign points to their local maximums
    # We reassign each point to their local maxima
    # Traverse a path till you encounter a real peak
    # then assign the coordinates of these to all points
    # that followed
    maxpath = []
    for x, y in zip(px, py):
        end_position = 0
        end_x = x
        end_y = y
        end_index = y * UNIQUE_PRIME + x
        end_max_index = local_max_index[end_y, end_x]

        maxpath.append((end_x, end_y))

        while not peak_found[end_x, end_y]:
            end_position += 1
            end_index = end_max_index
            end_x = int(end_index / UNIQUE_PRIME)
            end_y = int(end_index % UNIQUE_PRIME)
            end_max_index = local_max_index[end_x, end_y]
            maxpath.append((end_x, end_y))
        for rx, ry in maxpath:
            # Assign them the index and value
            # of the last found peak (above)
            local_max_index[rx, ry] = end_max_index
            local_max_value[rx, ry] = local_max_value[end_x, end_y]
            peak_found[rx, ry] = 1
    return local_max_value, local_max_index
