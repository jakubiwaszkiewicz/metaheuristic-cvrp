from dto import Instance, Node

def single_instance_loader(path_to_file: str) -> Instance:

    with open(path_to_file, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    instance_name = ""
    capacity = 0
    dimension = 0
    coords = {}
    demands = {}
    section = None

    # parser
    for line in lines:

        if line.startswith("NAME"):
            instance_name = line.split(":")[1].strip()

        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1].strip())

        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[1].strip())

        elif line.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue

        elif line.startswith("DEMAND_SECTION"):
            section = "demand"
            continue

        elif line.startswith("DEPOT_SECTION"):
            section = "depot"
            continue

        elif line.startswith("EOF"):
            break

        ## No. based lines (only coords, demand and depot lines has lines contains only numbers
        elif section == "coords":
            parts = line.split()
            if len(parts) == 3:
                node_id, x, y = map(float, parts)
                coords[int(node_id)] = (x, y)
            else:
                raise ValueError(line)

        elif section == "demand":
            parts = line.split()
            if len(parts) == 2:
                node_id, demand = map(int, parts)
                demands[node_id] = demand
            else:
                raise ValueError(line)

        elif section == "depot":
            depot_id = int(line)
            if depot_id == -1:
                section = None
    # end of parser

    nodes = [
        Node(id=node_id, x=coords[node_id][0], y=coords[node_id][1], demand=demands.get(node_id, 0))
        for node_id in range(1, dimension + 1)
    ]


    # depot is always node with id=1
    depot = nodes[0]
    customers = nodes[1:]

    instance = Instance(
        instance_name = instance_name,
        truck_capacity = capacity,
        customers = customers,
        depot=depot,
    )

    return instance
