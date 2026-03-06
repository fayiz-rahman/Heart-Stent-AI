import streamlit as st
import torch
import pyvista as pv
from stpyvista import stpyvista
import numpy as np
import os
from torch_geometric.nn import SAGEConv

# --- 1. MODEL ARCHITECTURE ---
class StentMultiPredictor(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_channels=256, out_channels=4):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.clone()
        x[:, :3] *= 1000.0  # Scale coordinates to mm
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        return self.lin(x)

# --- 2. ASSET LOADER ---
@st.cache_resource
def load_segmented_assets():
    root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)\Processed_Segments_v5"
    models, stats = {}, {}
    for seg_id in [0, 1, 2]:
        ckpt_name = f"stent_gnn_multi_Segment_{seg_id}.pth"
        ckpt_path = os.path.join(root_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model = StentMultiPredictor()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models[seg_id] = model
            stats[seg_id] = checkpoint['stats']
    return models, stats

# --- 3. UI & ENHANCED HIGH-CONTRAST THEME ---
st.set_page_config(page_title="Stent AI Analysis", layout="wide")

st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] { 
        background-color: #000000 !important; 
        border-right: 1px solid #333; 
    }
    
    /* FORCE WHITE TEXT ON ALL SIDEBAR LABELS AND HEADERS */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown p { 
        color: #FFFFFF !important; 
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Adjust Slider Labels */
    .stSlider label { color: #FFFFFF !important; }
    
    /* Metric Highlights */
    [data-testid="stMetricValue"] { color: #00FFCC !important; }
    [data-testid="stMetricLabel"] { color: #FFFFFF !important; opacity: 0.9; }

    /* Remove padding to maximize screen usage */
    .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

st.title(" Heart Stent: AI Enabled Structural Response")

# --- 4. CONTROL PANEL ---
with st.sidebar:
    st.header("🔧 Design Controls")
    link_len = st.slider("Link Length (mm)", 0.24, 0.44, 0.28, step=0.02)
    case_str = f"{link_len:.2f}mm"
    
    st.markdown("---")
    st.header("Visualization Settings")
    viz_style = st.selectbox("Render Style", ["Surface", "Wireframe", "Points"])
    show_edges = st.checkbox("Show Mesh Edges", value=False)
    smooth_shading = st.checkbox("Smooth Shading", value=True)
    
    st.markdown("---")
    st.header("Advanced Analysis")
    mark_peak = st.checkbox("Mark Peak Stress Node", value=False)
    coord_scale = st.slider("Z-Axis Stretching", 1.0, 3.0, 1.0)
    opacity = st.slider("Global Mesh Opacity", 0.1, 1.0, 1.0)

# --- 5. PREDICTION & DATA MERGING ---
models, all_stats = load_segmented_assets()
root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)\Processed_Segments_v5"

segment_meshes = []
peak_stress_val = 0
peak_node_pos = None

if len(models) < 3:
    st.error("Error: Could not load model segments from the Processed_Segments_v5 folder.")
    st.stop()

for seg_id in [0, 1, 2]:
    data_path = os.path.join(root_dir, f"Segment_{seg_id}", f"stent_seg{seg_id}_{case_str}.pt")
    if os.path.exists(data_path):
        data = torch.load(data_path, weights_only=False)
        with torch.no_grad():
            out = models[seg_id](data)
            s_m, s_s = all_stats[seg_id]['y_mean'], all_stats[seg_id]['y_std']
            pred_stress = (out[:, 0].numpy() * s_s) + s_m
        
        # Build segment mesh
        seg_pos = data.pos.numpy().copy()
        seg_pos[:, 2] *= coord_scale
        
        grid = pv.UnstructuredGrid(data.cells.numpy(), data.cell_types.numpy(), seg_pos)
        grid.point_data["Stress"] = pred_stress
        segment_meshes.append(grid)
        
        # Global Peak Tracking
        current_max = pred_stress.max()
        if current_max > peak_stress_val:
            peak_stress_val = current_max
            max_idx = np.argmax(pred_stress)
            peak_node_pos = seg_pos[max_idx]

# --- 6. 3D RENDERING ---
if segment_meshes:
    # Merge segments for a unified visualization
    full_stent = segment_meshes[0].merge(segment_meshes[1:])
    
    # Large window for full-screen fit
    plotter = pv.Plotter(window_size=[1400, 1000])
    plotter.set_background("#000000")
    
    clim = [1.0e9, 3.73e10] 
    style_map = {"Surface": "surface", "Wireframe": "wireframe", "Points": "points"}
    
    # Extraction logic for edge visibility
    display_mesh = full_stent.extract_surface() if viz_style == "Surface" else full_stent
    
    plotter.add_mesh(
        display_mesh,
        scalars="Stress",
        cmap="turbo",
        clim=clim,
        style=style_map[viz_style],
        show_edges=show_edges,
        edge_color="white" if show_edges else None,
        opacity=opacity,
        smooth_shading=smooth_shading,
        interpolate_before_map=True,
        scalar_bar_args={'title': "Stress (Pa)", 'color': 'white'}
    )
    
    if mark_peak and peak_node_pos is not None:
        plotter.add_mesh(pv.Sphere(radius=0.0005, center=peak_node_pos), color="red")
        plotter.add_point_labels([peak_node_pos], [f"Peak: {peak_stress_val/1e9:.1f} GPa"], 
                                 font_size=20, text_color="white")

    plotter.view_isometric()
    plotter.camera.zoom(1.5) # Centered and filling the large window
    
    stpyvista(plotter, key=f"high_viz_{link_len}_{coord_scale}_{viz_style}_{show_edges}")

# --- 7. METRICS ---
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Design Link Length", f"{link_len:.2f} mm")
c2.metric("Global Peak Stress", f"{peak_stress_val / 1e9:.2f} GPa")
c3.metric("Safety Factor (316L)", f"{0.205 / (peak_stress_val / 1e9):.2f}")

st.caption("AI prediction using Graph Neural Networks to analyze Heart Stent structural response.")