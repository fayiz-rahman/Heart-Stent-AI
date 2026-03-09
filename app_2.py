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

# --- 2. ASSET LOADER (All 5 Segments) ---
@st.cache_resource
def load_all_segmented_assets():
    root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)\Processed_Segments_v5"
    models, stats = {}, {}
    # Loading all 5 trained segments
    for seg_id in range(5):
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

# --- 3. UI & COMPACT LAYOUT THEME ---
st.set_page_config(page_title="Stent AI Analysis", layout="wide")

st.markdown("""
    <style>
    /* Force Title and Header to the very top */
    .main .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 0rem !important; 
    }
    .stApp { background-color: #000000; color: #FFFFFF; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #000000 !important; border-right: 1px solid #333; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p { color: #FFFFFF !important; font-weight: 600 !important; }
    
    /* Compact Header */
    h1 { margin-top: -30px !important; font-size: 1.8rem !important; }
    
    .stTabs [data-baseweb="tab-list"] { background-color: #000000; }
    .stTabs [data-baseweb="tab"] { color: white; }
    </style>
    """, unsafe_allow_html=True)

# Title moved up via CSS margin
st.write("# 🫀 Heart Stent: AI Enabled Structural Response")

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("🔧 Design Controls")
    link_len = st.slider("Link Length (mm)", 0.24, 0.44, 0.28, step=0.02)
    case_str = f"{link_len:.2f}mm"
    
    st.markdown("---")
    st.header("Visual Settings")
    viz_style = st.selectbox("Render Style", ["Surface", "Wireframe"])
    show_edges = st.checkbox("Show Mesh Edges", value=False)
    
    st.markdown("---")
    st.header("Advanced Analysis")
    mark_peak = st.checkbox("Mark Peak Stress Node", value=False)
    coord_scale = st.slider("Z-Axis Stretching", 1.0, 3.0, 1.0)
    opacity = st.slider("Global Mesh Opacity", 0.1, 1.0, 1.0)

# --- 5. PREDICTION & FULL MESH ASSEMBLY ---
models, all_stats = load_all_segmented_assets()
root_dir = r"D:\fz\fz\CFD-KJ\10. FEA AI-2026\Heart stent\Heart Stent (Vertical Link Length)\Processed_Segments_v5"

segment_grids = []
peak_stress_val = 0
peak_node_pos = None

# Ensure all 5 segments are loaded
if len(models) < 5:
    st.warning(f"Only {len(models)}/5 segments found. Ensure all .pth files are in the directory.")

for seg_id in range(5):
    data_path = os.path.join(root_dir, f"Segment_{seg_id}", f"stent_seg{seg_id}_{case_str}.pt")
    if os.path.exists(data_path) and seg_id in models:
        data = torch.load(data_path, weights_only=False)
        with torch.no_grad():
            out = models[seg_id](data)
            s_m, s_s = all_stats[seg_id]['y_mean'], all_stats[seg_id]['y_std']
            d_m, d_s = all_stats[seg_id]['disp_mean'].numpy(), all_stats[seg_id]['disp_std'].numpy()
            
            # Prediction Results
            pred_stress = (out[:, 0].numpy() * s_s) + s_m
            pred_disp = (out[:, 1:4].numpy() * d_s) + d_m
        
        # Build segment mesh
        seg_pos = data.pos.numpy().copy()
        seg_pos[:, 2] *= coord_scale
        
        grid = pv.UnstructuredGrid(data.cells.numpy(), data.cell_types.numpy(), seg_pos)
        grid.point_data["Stress"] = pred_stress
        grid.point_data["DispVec"] = pred_disp
        segment_grids.append(grid)
        
        # Track Global Peak Stress
        if pred_stress.max() > peak_stress_val:
            peak_stress_val = pred_stress.max()
            peak_node_pos = seg_pos[np.argmax(pred_stress)]

# --- 6. TABS & RENDERING ---
tab_static, tab_anim = st.tabs(["📊 Static Stress Analysis", "🔄 Pulsing Animation"])

with tab_static:
    if segment_grids:
        # Merge all segments 0-4 for full visibility
        full_stent = segment_grids[0].merge(segment_grids[1:])
        
        # Window adjusted to avoid scrolling
        plotter = pv.Plotter(window_size=[1200, 950])
        plotter.set_background("#000000")
        
        mesh_to_plot = full_stent.extract_surface() if viz_style == "Surface" else full_stent
        plotter.add_mesh(mesh_to_plot, scalars="Stress", cmap="turbo", clim=[1.0e9, 3.73e10],
                         style="surface" if viz_style=="Surface" else "wireframe", 
                         show_edges=show_edges, edge_color="white", opacity=opacity)
        
        if mark_peak and peak_node_pos is not None:
            plotter.add_mesh(pv.Sphere(radius=0.0005, center=peak_node_pos), color="red")
            plotter.add_point_labels([peak_node_pos], [f"Peak: {peak_stress_val/1e9:.1f} GPa"], 
                                     font_size=20, text_color="white")

        plotter.view_isometric()
        plotter.camera.zoom(1.6) # Tighter zoom to fill the high-profile window
        stpyvista(plotter, key=f"static_v5_{link_len}_{coord_scale}_{viz_style}")

with tab_anim:
    if segment_grids:
        gif_filename = f"pulse_full_{link_len:.2f}mm.gif"
        with st.spinner("Generating 5-segment pulsing loop..."):
            full_stent_anim = segment_grids[0].merge(segment_grids[1:])
            orig_points = full_stent_anim.points.copy()
            disp_vectors = full_stent_anim.point_data["DispVec"]
            stress_base = full_stent_anim.point_data["Stress"]

            plotter_gif = pv.Plotter(off_screen=True, window_size=[1000, 800])
            plotter_gif.set_background("black")
            plotter_gif.open_gif(gif_filename)

            for i in range(30):
                f = (np.sin((i / 30) * 2 * np.pi - np.pi/2) + 1) / 4 
                full_stent_anim.points = orig_points + (disp_vectors * f) # Scale factor 1.0
                full_stent_anim.point_data["Stress"] = stress_base * f
                
                plotter_gif.add_mesh(full_stent_anim.extract_surface(), scalars="Stress", 
                                     cmap="turbo", clim=[1.0e9, 3.73e10], name="stent")
                
                # Fixed requested Focal Point
                plotter_gif.camera.position = (0.03, 0.03, 0.03)
                plotter_gif.camera.focal_point = (0, 0, 0.002)
                plotter_gif.write_frame()
            
            plotter_gif.close()
        
        st.image(gif_filename, use_container_width=True)

# --- 7. METRICS ---
st.markdown("---")
c1, c2, c3 = st.columns(3)
c1.metric("Design Link Length", f"{link_len:.2f} mm")
c2.metric("Global Peak Stress", f"{peak_stress_val / 1e9:.2f} GPa")
c3.metric("Safety Factor", f"{0.205 / (peak_stress_val / 1e9):.2f}")