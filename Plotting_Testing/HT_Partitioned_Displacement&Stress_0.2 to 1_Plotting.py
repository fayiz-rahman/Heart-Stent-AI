# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:35:36 2026

@author: zhiha
"""

import torch
import pyvista as pv
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
import os
import glob
import numpy as np


# --- Ensure StentMultiPredictor is defined in your script ---
# --- 1. Multi-Output Model ---
class StentMultiPredictor(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=256, out_channels=4):
        super().__init__()
        # GraphSAGE layers handle the spatial relationships 
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Output: 1 (Stress) + 3 (Displacement XYZ) = 4
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.clone()
        x[:, :3] *= 1000.0 # Standardize coordinates to mm
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        
        return self.lin(x)
    
    
# %% plot static stress field
    

def plot_static_stress_comparison(case_len="0.28mm"):
    root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)"
    checkpoint_path = os.path.join(root_dir, "stent_gnn_multi_v1.pth")
    data_path = os.path.join(root_dir, "Processed_Crown_Dataset_v5_Displacement", f"stent_crown_{case_len}.pt")

    # 1. Load Multi-Target Checkpoint 
    if not os.path.exists(checkpoint_path):
        print("❌ Multi-target checkpoint not found.")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    stats = checkpoint['stats']
    y_mean, y_std = stats['y_mean'], stats['y_std']

    model = StentMultiPredictor(in_channels=4, hidden_channels=256, out_channels=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Inference
    data = torch.load(data_path, weights_only=False)
    with torch.no_grad():
        out = model(data)
        # Column 0 is Stress 
        ai_stress_final = (out[:, 0].numpy() * y_std) + y_mean
        
    fea_stress_final = data.y.cpu().numpy().flatten()

    # 3. Setup PyVista Grids
    # Using original positions (no displacement applied)
    grid_ai = pv.UnstructuredGrid(data.cells.numpy(), data.cell_types.numpy(), data.pos.numpy())
    grid_ai.point_data["Stress"] = ai_stress_final

    grid_fea = pv.UnstructuredGrid(data.cells.numpy(), data.cell_types.numpy(), data.pos.numpy())
    grid_fea.point_data["Stress"] = fea_stress_final

    # 4. Side-by-Side Plotting 
    p = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    p.set_background("black")
    
    # Your preferred working linear range
    clim = [1.0e9, 3.73e10]

    # Left: AI Prediction
    p.subplot(0, 0)
    p.add_text(f"AI PREDICTED STRESS ({case_len})", color="cyan")
    p.add_mesh(grid_ai.extract_surface(), scalars="Stress", cmap="turbo", 
               clim=clim, smooth_shading=True, interpolate_before_map=True)

    # Right: Actual FEA
    p.subplot(0, 1)
    p.add_text(f"ACTUAL FEA STRESS ({case_len})", color="white")
    p.add_mesh(grid_fea.extract_surface(), scalars="Stress", cmap="turbo", 
               clim=clim, smooth_shading=True, interpolate_before_map=True)

    p.link_views()
    p.view_isometric()
    p.show()

    print(f"✅ Comparison plotted for {case_len}")
    print(f"📊 AI Max Stress: {ai_stress_final.max():.2e} Pa")
    print(f"📊 FEA Max Stress: {fea_stress_final.max():.2e} Pa")

if __name__ == "__main__":
    plot_static_stress_comparison("0.28mm")
    
# %% plot displacement
    
import torch
import pyvista as pv
import numpy as np
import os

# --- Ensure StentMultiPredictor is defined ---

def animate_pulsing_stent(case_len="0.28mm", scale_factor=3.0):
    root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)"
    checkpoint_path = os.path.join(root_dir, "stent_gnn_multi_v1.pth")
    data_path = os.path.join(root_dir, "Processed_Crown_Dataset_v5_Displacement", f"stent_crown_{case_len}.pt")

    # 1. Load Checkpoint and Stats
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    stats = checkpoint['stats']
    y_m, y_s = stats['y_mean'], stats['y_std']
    d_m, d_s = stats['disp_mean'].numpy(), stats['disp_std'].numpy()

    model = StentMultiPredictor(in_channels=4, hidden_channels=256, out_channels=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Inference
    data = torch.load(data_path, weights_only=False)
    with torch.no_grad():
        out = model(data)
        ai_stress = (out[:, 0].numpy() * y_s) + y_m
        ai_disp = (out[:, 1:4].numpy() * d_s) + d_m

    # 3. Setup Grid and Plotter
    orig_pos = data.pos.numpy()
    grid = pv.UnstructuredGrid(data.cells.numpy(), data.cell_types.numpy(), orig_pos)
    
    p = pv.Plotter(window_size=[1200, 900])
    p.set_background("black")
    clim = [1.0e9, 3.73e10]

    # Set Zoomed-out Camera
    p.view_isometric()
    p.camera.position = (0.03, 0.03, 0.03) 
    p.camera.focal_point = (0, 0, 0.005)
    
    gif_filename = f"stent_pulsing_{case_len}.gif"
    p.open_gif(gif_filename)
    
    print(f"🎬 Generating Pulsing (0-50%) Animation for {case_len}...")

    # We use 100 frames to show multiple cycles of expansion/compression
    frames = 100 
    for i in range(frames):
        # f oscillates between 0 and 0.5 using a sine wave
        # np.sin goes -1 to 1 -> +1 makes it 0 to 2 -> / 4 makes it 0 to 0.5
        t = (i / frames) * 2 * np.pi * 2  # 2 full cycles
        f = (np.sin(t - np.pi/2) + 1) / 4 
        
        # Update Geometry and Stress
        grid.points = orig_pos + (ai_disp * f * scale_factor)
        grid.point_data["Stress"] = ai_stress * f
        
        # Extract surface and update
        surf = grid.extract_surface()
        p.add_mesh(
            surf, 
            scalars="Stress", 
            name="stent_mesh", 
            cmap="turbo", 
            clim=clim, 
            smooth_shading=True
        )
        
        p.write_frame()

    p.close()
    print(f"✅ Pulsing Animation Saved: {gif_filename}")

if __name__ == "__main__":
    animate_pulsing_stent("0.28mm", scale_factor=2.0)