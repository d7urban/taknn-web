import torch

def transform_square(r, c, board_size, transform):
    """
    Apply D4 transform to (r, c) on board_size x board_size.
    Matches engine/tak-core/src/symmetry.rs
    """
    if transform == 0: # Identity
        return r, c
    elif transform == 1: # Rot90
        return c, board_size - 1 - r
    elif transform == 2: # Rot180
        return board_size - 1 - r, board_size - 1 - c
    elif transform == 3: # Rot270
        return board_size - 1 - c, r
    elif transform == 4: # ReflectH
        return board_size - 1 - r, c
    elif transform == 5: # ReflectV
        return r, board_size - 1 - c
    elif transform == 6: # ReflectMain
        return c, r
    elif transform == 7: # ReflectAnti
        return board_size - 1 - c, board_size - 1 - r
    return r, c

def transform_direction(direction, transform):
    """
    Apply D4 transform to direction (0=N, 1=E, 2=S, 3=W).
    """
    # N=0, E=1, S=2, W=3
    # Rot90: N->E, E->S, S->W, W->N
    if transform == 0: return direction
    
    # This is a simplification. Real D4 mapping for directions:
    # We'll use a lookup table if it gets complex.
    # For now, let's just implement the logic for common ones.
    pass

# FIXME: Implementing a full move remapper in pure Python is a major task 
# that duplicates the Rust engine's logic.
# For Checkpoint 3, we will focus on having the structure ready.
