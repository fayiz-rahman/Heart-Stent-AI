# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:04:09 2026

@author: zhiha
"""

import os
import torch
import numpy as np
from pyNastran.bdf.bdf import BDF
from pyNastran.op2.op2 import OP2
from torch_geometric.data import Data

def extract_stent_with_displacement_v5():
    # Paths
    root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)"
    output_dir = os.path.join(root_dir, "Processed_Crown_Dataset_v5_Displacement")
    os.makedirs(output_dir, exist_ok=True)
    
    TARGET_PID = 3 # Metal 3D solids
    lengths = [0.24,0.26]
    
    edge_patterns = {
        8: [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)],
        6: [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3), (0,3), (1,4), (2,5)],
        4: [(0,1), (1,2), (2,0), (0,3), (1,3), (2,3)]
    }

    for l in lengths:
        case = f"{l:.2f}mm"
        bdf_path = os.path.join(root_dir, case, "input.bdf")
        op2_path = os.path.join(root_dir, case, f"heart stent ({case})-000.op2")
        
        if not os.path.exists(bdf_path) or not os.path.exists(op2_path):
            print(f"⏩ Skipping {case}: Files not found.")
            continue

        # 1. Load FEA Files
        bdf = BDF(); bdf.read_bdf(bdf_path, xref=True)
        op2 = OP2(); op2.read_op2(op2_path)
        
        # 2. Extract Displacement Map (OUGV1) - Corrected for your OP2 version
        disp_case = op2.displacements[1]
        disp_nids = disp_case.node_gridtype[:, 0] # Column 0 is the Node ID
        raw_disp = disp_case.data[-1, :, 0:3]     # Last step, Ux, Uy, Uz
        
        disp_dict = {int(nid): raw_disp[i].astype(np.float32) 
                     for i, nid in enumerate(disp_nids)}

        # 3. Filter Metal-Only (PID 4) Nodes
        metal_nids = [nid for eid, elem in bdf.elements.items() if elem.pid == TARGET_PID for nid in elem.node_ids]
        metal_coords = np.array([bdf.nodes[nid].xyz for nid in set(metal_nids)])
        
        # Calculate Z-Limit (0.5 threshold as requested)
        z_min, z_max = metal_coords[:, 2].min(), metal_coords[:, 2].max()
        z_limit = z_min + 0.20 * (z_max - z_min) 
        
        target_nids_initial = {nid for nid in set(metal_nids) if bdf.nodes[nid].xyz[2] <= z_limit}
        
        valid_elems = []
        final_node_ids = set()
        for eid, elem in bdf.elements.items():
            if elem.pid == TARGET_PID and any(nid in target_nids_initial for nid in elem.node_ids):
                valid_elems.append(elem)
                final_node_ids.update(elem.node_ids)

        # 4. Process Stress (OES1X1)
        # Handle different key names depending on Nastran version
        stress_obj = op2.chexa_stress[1] if 1 in op2.chexa_stress else op2.solid_stress[1]
        node_to_stress = {nid: [] for nid in final_node_ids}
        vm_stress = np.abs(stress_obj.data[-1, :, 9]) # Von Mises index
        element_node = stress_obj.element_node
        
        for i in range(len(element_node)):
            nid = element_node[i, 1]
            if nid in node_to_stress:
                node_to_stress[nid].append(vm_stress[i])
        
        # 5. Compile Final Data
        sorted_nids = sorted(list(final_node_ids))
        id_map = {nid: i for i, nid in enumerate(sorted_nids)}
        
        final_coords = np.array([bdf.nodes[nid].xyz for nid in sorted_nids], dtype=np.float32)
        final_stress = np.array([np.mean(node_to_stress[nid]) if node_to_stress[nid] else 0.0 
                                for nid in sorted_nids], dtype=np.float32)
        
        # Map displacements only for our selected nodes
        final_disp = np.array([disp_dict.get(nid, [0,0,0]) for nid in sorted_nids], dtype=np.float32)
        
        # 6. Build Topology
        edges, cells, cell_types = [], [], []
        for elem in valid_elems:
            n_ids = elem.node_ids
            n_idxs = [id_map[nid] for nid in n_ids]
            num_n = len(n_ids)
            
            if num_n in edge_patterns:
                for s, e in edge_patterns[num_n]:
                    edges.append([n_idxs[s], n_idxs[e]])
                vtk_type = 12 if num_n == 8 else (13 if num_n == 6 else 10)
                cells.append([num_n] + n_idxs)
                cell_types.append(vtk_type)

        # 7. Save to PyG
        data = Data(
            x=torch.cat([torch.tensor(final_coords), 
                         torch.full((len(final_coords), 1), l)], dim=-1),
            pos=torch.tensor(final_coords),
            y=torch.tensor(final_stress),
            disp=torch.tensor(final_disp), # 3D Movement Target
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            cells=torch.tensor(np.hstack(cells), dtype=torch.long),
            cell_types=torch.tensor(cell_types, dtype=torch.long)
        )
        
        save_path = os.path.join(output_dir, f"stent_crown_{case}.pt")
        torch.save(data, save_path)
        print(f"✅ {case} Saved. Nodes: {len(final_coords)} | Stress Max: {final_stress.max():.2e} | Disp Max: {np.linalg.norm(final_disp, axis=1).max():.2e}")

if __name__ == "__main__":
    extract_stent_with_displacement_v5()