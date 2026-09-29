"""Ordered joint names shared by BrainCo policy transforms and artifact export."""

JOINT_GROUPS = {
    "left_arm": [f"left_arm_joint{i}" for i in range(1, 8)],
    "right_arm": [f"right_arm_joint{i}" for i in range(1, 8)],
    "left_hand": [
        f"left_{finger}_{joint}_joint"
        for finger in ("little", "ring", "middle", "index")
        for joint in ("MPR", "MCP", "PIP", "DIP")
    ] + [f"left_thumb_{joint}_joint" for joint in ("MCP", "PIP", "DIP", "CMP", "CMR")],
    "right_hand": [
        f"right_{finger}_{joint}_joint"
        for finger in ("little", "ring", "middle", "index")
        for joint in ("MPR", "MCP", "PIP", "DIP")
    ] + [f"right_thumb_{joint}_joint" for joint in ("MCP", "PIP", "DIP", "CMP", "CMR")],
}

JOINT_GROUP_DIMS = {key: len(names) for key, names in JOINT_GROUPS.items()}
