# Function to generate positions along one axis
def generate_positions(start, max_value, slice_size):
    positions = []
    positions_set = set()
    # Center position
    if 0 <= start <= max_value - slice_size:
        positions.append(start)
        positions_set.add(start)
    # Move outwards from the center
    offset = slice_size
    while True:
        added = False
        # Left/Up position
        pos1 = start - offset
        if 0 <= pos1 <= max_value - slice_size and pos1 not in positions_set:
            positions.append(pos1)
            positions_set.add(pos1)
            added = True
        # Right/Down position
        pos2 = start + offset
        if 0 <= pos2 <= max_value - slice_size and pos2 not in positions_set:
            positions.append(pos2)
            positions_set.add(pos2)
            added = True
        if not added:
            break
        offset += slice_size
    return positions