import numpy as np
import skimage.measure as skm

UNIQUE_PRIME = 15331


def normalize(image):
    min_pixel = image.min()
    max_pixel = image.max()
    range_pixel = max_pixel - min_pixel
    normalized_image = (image - min_pixel) / range_pixel
    return normalized_image


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
        pixel_value_xy = image[x, y]
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
        end_index = x * UNIQUE_PRIME + y
        end_max_index = local_max_index[end_y, end_x]

        maxpath.append((end_x, end_y))

        while not peak_found[end_x, end_y]:
            end_position += 1
            end_index = end_max_index
            end_x = int(end_index // UNIQUE_PRIME)
            end_y = int(end_index % UNIQUE_PRIME)
            end_max_index = local_max_index[end_x, end_y]
            maxpath.append((end_x, end_y))
        for rx, ry in maxpath:
            # Assign them the index and value
            # of the last found peak (above)
            local_max_index[rx, ry] = end_max_index
            local_max_value[rx, ry] = local_max_value[end_x, end_y]
            peak_found[rx, ry] = 1
    #return local_max_value, local_max_index
    image_labelled = skm.label(foreground_mask & (image == local_max_value))

    image_normalized = normalize(image)
    obj_props = skm.regionprops(image_labelled, image_normalized)

    obj_props = [
        prop for prop in obj_props
        if np.isfinite(prop.weighted_centroid).all()
    ]

    num_labels = len(obj_props)

    # extract object seeds
    seeds = np.array(
        [obj_props[i].weighted_centroid for i in range(num_labels)])
    seeds = np.round(seeds).astype(np.int)

    # Fix seeds for non-convex objects
    for i in range(num_labels):

        x = seeds[i, 0]
        y = seeds[i, 1]

        if image_labelled[x, y] == obj_props[i].label:
            continue

        # find object point with closest manhattan distance to center of mass
        pts = obj_props[i].coords

        xdist = np.abs(pts[:, 0] - x)
        ydist = np.abs(pts[:, 1] - y)

        seeds[i, :] = pts[np.argmin(xdist + ydist), :]

        assert image_labelled[seeds[i, 0], seeds[i, 1]] == obj_props[i].label

    # get seed responses
    seed_pixels = image[seeds[:, 0], seeds[:, 1]]



    # set label of each foreground pixel to the label of its nearest peak
    image_labelled_flat = image_labelled.ravel()
    print('image_labelled: {}'.format(image_labelled.shape))
    print('foreground_mask: {}'.format(foreground_mask.shape))
    print('image_labelled_flat: {}'.format(image_labelled_flat.shape))
    pind = np.flatnonzero(foreground_mask)
    index_flat = local_max_index.ravel()
    print('index_flat: {}'.format(index_flat))
    image_labelled_flat[pind] = image_labelled_flat[index_flat[pind]]
    return image_labelled, seeds, seed_pixels
