NAV2 = [("nav2_controller", "controller_server"), ("nav2_planner", "planner_server"),
        ("nav2_behaviors", "behavior_server"), ("nav2_bt_navigator", "bt_navigator")]
NODES = [exe for _, exe in NAV2]
MANAGER = "lifecycle_manager_navigation"
CONFIGURE, ACTIVATE = 1, 3


def next_transition(states, nodes=NODES):
    labels = [states.get(n, "unknown") for n in nodes]
    for n, s in zip(nodes, labels):
        if s == "finalized":
            return "fail", n
    for n, s in zip(nodes, labels):
        if s not in ("unconfigured", "inactive", "active"):
            return "wait", n
    for n, s in zip(nodes, labels):
        if s == "unconfigured":
            return "configure", n
    for n, s in zip(nodes, labels):
        if s == "inactive":
            return "activate", n
    return "done", None
