import torch
import tak_python


def descriptors_to_tensors(desc_list):
    """Convert a list of descriptor dicts (from PyO3) to tensors.

    Args:
        desc_list: list of dicts from PyGameState.get_move_descriptors()

    Returns:
        dict of tensors (unpadded, single sample — padding happens in collate)
    """
    n = len(desc_list)
    src = torch.zeros(n, dtype=torch.long)
    dst = torch.zeros(n, dtype=torch.long)
    path = torch.full((n, 7), 255, dtype=torch.long)
    move_type = torch.zeros(n, dtype=torch.long)
    piece_type = torch.zeros(n, dtype=torch.long)
    direction = torch.zeros(n, dtype=torch.long)
    pickup_count = torch.zeros(n, dtype=torch.long)
    drop_template_id = torch.zeros(n, dtype=torch.long)
    travel_length = torch.zeros(n, dtype=torch.long)
    capstone_flatten = torch.zeros(n, dtype=torch.float32)
    enters_occupied = torch.zeros(n, dtype=torch.float32)
    opening_phase = torch.zeros(n, dtype=torch.float32)

    for i, d in enumerate(desc_list):
        src[i] = d["src"]
        dst[i] = d["dst"]
        p = d["path"]
        for j, sq in enumerate(p):
            path[i, j] = sq
        move_type[i] = d["move_type"]
        piece_type[i] = d["piece_type"]
        direction[i] = d["direction"]
        pickup_count[i] = d["pickup_count"]
        drop_template_id[i] = d["drop_template_id"]
        travel_length[i] = d["travel_length"]
        capstone_flatten[i] = float(d["capstone_flatten"])
        enters_occupied[i] = float(d["enters_occupied"])
        opening_phase[i] = float(d["opening_phase"])

    return {
        "src": src, "dst": dst, "path": path,
        "move_type": move_type, "piece_type": piece_type,
        "direction": direction, "pickup_count": pickup_count,
        "drop_template_id": drop_template_id, "travel_length": travel_length,
        "capstone_flatten": capstone_flatten, "enters_occupied": enters_occupied,
        "opening_phase": opening_phase,
    }


def compute_aux_labels(tps_str, board_size):
    """Compute auxiliary head training labels from a board position.

    Uses the Rust engine via PyO3 to compute per-square spatial labels.

    Args:
        tps_str: TPS string of the position
        board_size: board size (3..8)

    Returns:
        dict with:
            road_threat: [2, 8, 8] float tensor
            block_threat: [2, 8, 8] float tensor
            cap_flatten: [1, 8, 8] float tensor
            endgame: [1] float tensor
    """
    game = tak_python.PyGameState(board_size, tps_str)
    labels = game.compute_spatial_labels()

    return {
        "road_threat": torch.from_numpy(labels["road_threat"].copy()),
        "block_threat": torch.from_numpy(labels["block_threat"].copy()),
        "cap_flatten": torch.from_numpy(labels["cap_flatten"].copy()),
        "endgame": torch.tensor([labels["endgame"]], dtype=torch.float32),
    }
