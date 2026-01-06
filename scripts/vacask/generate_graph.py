#!/usr/bin/env python3
"""
Generate DOT graphs from VACASK ring oscillator circuit data.

Produces:
1. Circuit topology graph (nodes connected by devices)
2. Jacobian sparsity graph (equation/unknown dependencies)
3. Evaluation order graph (device evaluation sequence)
"""

import re
import sys
from collections import defaultdict

# Node name mapping (from VACASK output)
# Unknown indices 1-47 map to circuit nodes
NODE_NAMES = {
    0: "GND",
    1: "1",      # Ring node 1
    2: "2",      # Ring node 2
    3: "3",      # Ring node 3
    4: "4",      # Ring node 4
    5: "5",      # Ring node 5
    6: "6",      # Ring node 6
    7: "7",      # Ring node 7
    8: "8",      # Ring node 8
    9: "9",      # Ring node 9
    10: "vdd:br", # Voltage source branch
    11: "u1:mn:nf1",  # Internal noise nodes
    12: "vdd",
    13: "u1:mp:nf1",
    14: "u1:mp:nf2",
    # ... internal nodes continue
}

def parse_hierarchy(text):
    """Parse hierarchy section to extract instances and their relationships."""
    instances = []
    hierarchy = {}

    lines = text.split('\n')
    in_hierarchy = False
    indent_stack = [(-1, None)]

    for line in lines:
        if line.startswith('Hierarchy:'):
            in_hierarchy = True
            continue
        if in_hierarchy and line.startswith('Devices:'):
            break
        if not in_hierarchy:
            continue

        if not line.strip():
            continue

        # Count indent
        indent = len(line) - len(line.lstrip())
        content = line.strip()

        # Parse instance info
        match = re.match(r'(\S+)\s+\(model=(\S+),\s*device=(\S+)\)', content)
        if match:
            name, model, device = match.groups()
            instances.append({
                'name': name,
                'model': model,
                'device': device,
                'indent': indent
            })

    return instances

def parse_sparsity(text):
    """Parse sparsity section to extract Jacobian structure."""
    entries = []
    lines = text.split('\n')
    in_sparsity = False

    for line in lines:
        if line.startswith('Sparsity:'):
            in_sparsity = True
            continue
        if not in_sparsity:
            continue

        match = re.match(r'\s*\((\d+),\s*(\d+)\)\s*:\s*(\d+)', line)
        if match:
            row, col, idx = map(int, match.groups())
            entries.append((row, col, idx))

    return entries

def generate_circuit_graph(instances):
    """Generate DOT graph of circuit topology."""
    dot = ['digraph circuit {']
    dot.append('  rankdir=LR;')
    dot.append('  node [shape=ellipse];')
    dot.append('')

    # Define ring nodes
    dot.append('  // Ring nodes')
    dot.append('  subgraph cluster_ring {')
    dot.append('    label="Ring Nodes";')
    dot.append('    style=dashed;')
    for i in range(1, 10):
        dot.append(f'    n{i} [label="{i}"];')
    dot.append('  }')
    dot.append('')

    # Define supply nodes
    dot.append('  // Supply nodes')
    dot.append('  vdd [label="VDD" shape=box style=filled fillcolor=lightblue];')
    dot.append('  gnd [label="GND" shape=box style=filled fillcolor=lightgray];')
    dot.append('')

    # Define inverter stages
    dot.append('  // Inverter stages')
    for i in range(1, 10):
        inp = i
        out = (i % 9) + 1 if i == 9 else i + 1
        if i == 9:
            out = 1
        else:
            out = i + 1

        dot.append(f'  subgraph cluster_u{i} {{')
        dot.append(f'    label="u{i}";')
        dot.append(f'    style=rounded;')
        dot.append(f'    u{i}_mp [label="MP\\n(PMOS)" shape=box];')
        dot.append(f'    u{i}_mn [label="MN\\n(NMOS)" shape=box];')
        dot.append(f'  }}')
    dot.append('')

    # Connect inverters to nodes
    dot.append('  // Connections')
    for i in range(1, 10):
        inp = i
        out = 1 if i == 9 else i + 1

        # Gate connections (input)
        dot.append(f'  n{inp} -> u{i}_mp [label="G" color=blue];')
        dot.append(f'  n{inp} -> u{i}_mn [label="G" color=blue];')

        # Drain connections (output)
        dot.append(f'  u{i}_mp -> n{out} [label="D" color=red];')
        dot.append(f'  u{i}_mn -> n{out} [label="D" color=red];')

        # Supply connections
        dot.append(f'  vdd -> u{i}_mp [label="S,B" style=dashed];')
        dot.append(f'  gnd -> u{i}_mn [label="S,B" style=dashed];')

    dot.append('')
    dot.append('  // Current source trigger')
    dot.append('  i0 [label="i0\\n(ISRC)" shape=diamond];')
    dot.append('  gnd -> i0;')
    dot.append('  i0 -> n1;')

    dot.append('}')
    return '\n'.join(dot)

def generate_jacobian_graph(sparsity):
    """Generate DOT graph of Jacobian sparsity pattern."""
    dot = ['digraph jacobian {']
    dot.append('  rankdir=TB;')
    dot.append('  node [shape=circle width=0.5];')
    dot.append('')

    # Group nodes by type
    ring_nodes = set(range(1, 10))
    supply_nodes = {10, 12}  # vdd branch, vdd

    dot.append('  // Ring nodes (equations/unknowns)')
    dot.append('  subgraph cluster_ring {')
    dot.append('    label="Ring Nodes";')
    for i in range(1, 10):
        dot.append(f'    u{i} [label="{i}"];')
    dot.append('  }')
    dot.append('')

    dot.append('  // Supply nodes')
    dot.append('  u12 [label="VDD" shape=box];')
    dot.append('  u10 [label="Ibr" shape=box];')
    dot.append('')

    # Add edges for non-diagonal entries (dependencies)
    dot.append('  // Jacobian dependencies (row depends on col)')
    edges = set()
    for row, col, _ in sparsity:
        if row != col and row <= 12 and col <= 12:
            edges.add((col, row))  # col affects row

    for col, row in sorted(edges):
        dot.append(f'  u{col} -> u{row};')

    dot.append('}')
    return '\n'.join(dot)

def generate_eval_order_graph(instances):
    """Generate DOT graph showing evaluation order."""
    dot = ['digraph eval_order {']
    dot.append('  rankdir=TB;')
    dot.append('  node [shape=box];')
    dot.append('')

    # Filter to leaf instances (actual devices)
    leaf_instances = [i for i in instances if i['device'] != '__hierarchical__' and i['name'] != '__topinst__']

    # Group by model
    by_model = defaultdict(list)
    for inst in leaf_instances:
        by_model[inst['model']].append(inst['name'])

    dot.append('  // Evaluation order (top to bottom)')
    dot.append('')

    # Models are evaluated in order: isource, vsource, then psp103n, psp103p
    eval_order = ['isource', 'vsource', 'psp103n', 'psp103p']

    prev_cluster = None
    for model in eval_order:
        if model not in by_model:
            continue

        cluster_id = model.replace(':', '_')
        dot.append(f'  subgraph cluster_{cluster_id} {{')
        dot.append(f'    label="{model}";')
        dot.append(f'    style=rounded;')

        for i, name in enumerate(by_model[model]):
            node_id = name.replace(':', '_')
            dot.append(f'    {node_id} [label="{name}"];')

            # Chain within model
            if i > 0:
                prev_id = by_model[model][i-1].replace(':', '_')
                dot.append(f'    {prev_id} -> {node_id} [style=invis];')

        dot.append('  }')
        dot.append('')

        # Link between clusters
        if prev_cluster and by_model[model]:
            prev_last = by_model[prev_cluster][-1].replace(':', '_')
            curr_first = by_model[model][0].replace(':', '_')
            dot.append(f'  {prev_last} -> {curr_first} [ltail=cluster_{prev_cluster.replace(":", "_")} lhead=cluster_{cluster_id} label="next"];')

        prev_cluster = model

    dot.append('}')
    return '\n'.join(dot)

def generate_stamping_graph(sparsity):
    """Generate graph showing which devices stamp which matrix entries."""
    dot = ['digraph stamping {']
    dot.append('  rankdir=LR;')
    dot.append('  compound=true;')
    dot.append('')

    # Device to matrix entry mapping (simplified for ring)
    # Each transistor stamps entries for its connected nodes

    dot.append('  // Devices')
    dot.append('  subgraph cluster_devices {')
    dot.append('    label="Devices (eval order)";')
    dot.append('    i0 [label="i0" shape=diamond];')
    dot.append('    vdd_src [label="vdd" shape=diamond];')
    for i in range(1, 10):
        dot.append(f'    u{i}_mn [label="u{i}:mn" shape=box];')
        dot.append(f'    u{i}_mp [label="u{i}:mp" shape=box];')
    dot.append('  }')
    dot.append('')

    dot.append('  // Matrix entries (row, col)')
    dot.append('  subgraph cluster_matrix {')
    dot.append('    label="Jacobian Entries";')
    dot.append('    node [shape=rect width=0.3 height=0.3 fontsize=8];')

    # Show key matrix entries
    key_entries = [(1,1), (1,2), (2,1), (2,2), (2,12), (12,2), (12,12)]
    for r, c in key_entries:
        dot.append(f'    m_{r}_{c} [label="({r},{c})"];')
    dot.append('    etc [label="..." shape=none];')
    dot.append('  }')
    dot.append('')

    # Show stamping relationships
    dot.append('  // Stamping (device -> matrix entries it fills)')
    dot.append('  u1_mn -> m_1_1 [label="Gds"];')
    dot.append('  u1_mn -> m_2_2 [label="Gds"];')
    dot.append('  u1_mn -> m_1_2 [label="-Gm"];')
    dot.append('  u1_mn -> m_2_1 [label="Gm"];')
    dot.append('  u1_mp -> m_2_2 [label="Gds"];')
    dot.append('  u1_mp -> m_2_12 [label="-Gds"];')
    dot.append('  u1_mp -> m_12_2 [label="-Gds"];')
    dot.append('  u1_mp -> m_12_12 [label="Gds"];')

    dot.append('}')
    return '\n'.join(dot)

def main():
    # Read the data file
    with open('/tmp/ring_graph_data.txt', 'r') as f:
        data = f.read()

    # Parse data
    instances = parse_hierarchy(data)
    sparsity = parse_sparsity(data)

    # Generate graphs
    outdir = '/Users/roberttaylor/Code/ChipFlow/reference/VACASK/benchmark/ring/vacask'

    # 1. Circuit topology
    with open(f'{outdir}/graph_circuit.dot', 'w') as f:
        f.write(generate_circuit_graph(instances))
    print(f"Generated {outdir}/graph_circuit.dot")

    # 2. Jacobian sparsity
    with open(f'{outdir}/graph_jacobian.dot', 'w') as f:
        f.write(generate_jacobian_graph(sparsity))
    print(f"Generated {outdir}/graph_jacobian.dot")

    # 3. Evaluation order
    with open(f'{outdir}/graph_eval_order.dot', 'w') as f:
        f.write(generate_eval_order_graph(instances))
    print(f"Generated {outdir}/graph_eval_order.dot")

    # 4. Stamping relationships
    with open(f'{outdir}/graph_stamping.dot', 'w') as f:
        f.write(generate_stamping_graph(sparsity))
    print(f"Generated {outdir}/graph_stamping.dot")

    # Print summary
    print(f"\nSummary:")
    print(f"  Instances: {len(instances)}")
    print(f"  Jacobian entries: {len(sparsity)}")
    print(f"  Matrix size: {max(e[0] for e in sparsity)}x{max(e[1] for e in sparsity)}")

if __name__ == '__main__':
    main()
