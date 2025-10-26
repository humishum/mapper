import os
import re
import glob
import numpy as np
import open3d as o3d
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

def parse_ply(filename, max_points=300000):
    """Parse a PLY file using Open3D and return vertices (N,3) and colors (N,3) as numpy arrays."""
    print(f"DEBUG: parse_ply called with filename: {filename}, max_points: {max_points}")
    
    try:
        # Read point cloud using Open3D
        print("DEBUG: Loading point cloud with Open3D...")
        pcd = o3d.io.read_point_cloud(filename)
        print(f"DEBUG: Point cloud loaded, has {len(pcd.points)} points")
        
        # Downsample if too many points
        if len(pcd.points) > max_points:
            print(f"DEBUG: Downsampling from {len(pcd.points)} to ~{max_points} points...")
            # Use uniform downsampling - take every nth point
            downsample_ratio = len(pcd.points) / max_points
            indices = np.arange(0, len(pcd.points), int(downsample_ratio))[:max_points]
            pcd = pcd.select_by_index(indices)
            print(f"DEBUG: After downsampling: {len(pcd.points)} points")
        
        # Get vertices as numpy array
        print("DEBUG: Converting points to numpy array...")
        xyz = np.asarray(pcd.points)
        print(f"DEBUG: Points shape: {xyz.shape}")
        
        # Get colors if available, otherwise use default white
        print("DEBUG: Checking for colors...")
        if pcd.has_colors():
            print("DEBUG: Point cloud has colors, extracting...")
            rgb = np.asarray(pcd.colors)
            print(f"DEBUG: Colors shape: {rgb.shape}")
        else:
            print("DEBUG: No colors found, using default white...")
            rgb = np.ones_like(xyz)
            print(f"DEBUG: Default colors shape: {rgb.shape}")
        
        print("DEBUG: parse_ply completed successfully")
        return xyz, rgb
        
    except Exception as e:
        print(f"ERROR in parse_ply: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_pointclouds(base_dir):
    """
    Returns a dict:
    {
        'scene1': {
            'sequences': {
                'sequence_1': {
                    'thresholds': [1.0, 2.0, ...],
                    'files': {'1.0': '/path/to/scene1/pointclouds/sequence_1/scene_thr1.0.ply', ...}
                },
                'sequence_2': {...},
                ...
            }
        },
        ...
    }
    """
    result = {}
    # Look for directories with pointclouds subdirectory
    for pointcloud_dir in glob.glob(os.path.join(base_dir, "*", "pointclouds")):
        scene_name = os.path.basename(os.path.dirname(pointcloud_dir))
        
        # Look for sequence_* subdirectories
        sequence_dirs = glob.glob(os.path.join(pointcloud_dir, "sequence_*"))
        
        if sequence_dirs:
            sequences = {}
            for seq_dir in sequence_dirs:
                seq_name = os.path.basename(seq_dir)
                ply_files = glob.glob(os.path.join(seq_dir, "*.ply"))
                
                thresholds = []
                files = {}
                for pf in ply_files:
                    m = re.search(r"_thr([0-9]+(?:\.[0-9]+)?)", pf)
                    if m:
                        thr_float = float(m.group(1))
                        thresholds.append(thr_float)
                        files[thr_float] = pf
                if thresholds:
                    thresholds = sorted(thresholds, reverse=True)  # Sort numerically, highest first
                    sequences[seq_name] = {
                        "thresholds": thresholds,
                        "files": files
                    }
            
            if sequences:
                result[scene_name] = {"sequences": sequences}
    
    return result

# Set your base directory here
BASE_DIR = "/home/ape/repos/mapper/data/output_102425/"

pointclouds = get_pointclouds(BASE_DIR)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Pointcloud Threshold Visualizer"),
    dbc.Row([
        dbc.Col([
            html.Label("Scene:"),
            dcc.Dropdown(
                id="scene-dropdown",
                options=[{"label": k, "value": k} for k in pointclouds.keys()],
                value=list(pointclouds.keys())[0] if pointclouds else None
            ),
        ], width=3),
        dbc.Col([
            html.Div([
                html.Label("Sequences (select multiple):"),
                html.Button("Select All", id="select-all-sequences-btn", n_clicks=0, 
                           className="btn btn-sm btn-outline-secondary", 
                           style={"marginLeft": "10px", "marginBottom": "5px"})
            ], style={"display": "flex", "alignItems": "center"}),
            dcc.Dropdown(
                id="sequence-dropdown",
                multi=True
            ),
        ], width=4),
        dbc.Col([
            html.Div([
                html.Label("Thresholds (select multiple):"),
                html.Button("Select All", id="select-all-thresholds-btn", n_clicks=0, 
                           className="btn btn-sm btn-outline-secondary", 
                           style={"marginLeft": "10px", "marginBottom": "5px"})
            ], style={"display": "flex", "alignItems": "center"}),
            dcc.Dropdown(
                id="threshold-dropdown",
                multi=True
            ),
        ], width=3),
        dbc.Col([
            html.Button("Load Pointclouds", id="load-btn", n_clicks=0, className="btn btn-primary mt-4")
        ], width=2)
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Toggle Visibility:"),
            dcc.Checklist(
                id="visibility-checklist",
                options=[],
                value=[],
                labelStyle={"display": "block", "margin": "5px"}
            )
        ], width=12)
    ], id="toggle-row", style={"display": "none"}),
    html.Hr(id="toggle-hr", style={"display": "none"}),
    html.Div(id="plots-container")
], fluid=True)

@app.callback(
    Output("sequence-dropdown", "options"),
    Output("sequence-dropdown", "value"),
    Input("scene-dropdown", "value"),
    Input("select-all-sequences-btn", "n_clicks"),
    State("sequence-dropdown", "options"),
    prevent_initial_call="partial"
)
def update_sequences(scene, select_all_clicks, current_options):
    ctx = dash.callback_context
    
    if not scene or scene not in pointclouds:
        return [], []
    
    sequences = pointclouds[scene]["sequences"]
    # Sort sequence names naturally (sequence_1, sequence_2, etc.)
    seq_names = sorted(sequences.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
    options = [{"label": s, "value": s} for s in seq_names]
    
    # Check what triggered the callback
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "select-all-sequences-btn":
            # Select all sequences
            return options, seq_names
    
    # Default: select first sequence (when scene changes)
    value = [seq_names[0]] if seq_names else []
    return options, value

@app.callback(
    Output("threshold-dropdown", "options"),
    Output("threshold-dropdown", "value"),
    Input("scene-dropdown", "value"),
    Input("sequence-dropdown", "value"),
    Input("select-all-thresholds-btn", "n_clicks"),
    State("threshold-dropdown", "options"),
    prevent_initial_call="partial"
)
def update_thresholds(scene, sequences, select_all_clicks, current_options):
    ctx = dash.callback_context
    
    if not scene or scene not in pointclouds:
        return [], []
    if not sequences or len(sequences) == 0:
        return [], []
    
    # Collect all unique thresholds across selected sequences
    all_thresholds = set()
    for seq in sequences:
        if seq in pointclouds[scene]["sequences"]:
            all_thresholds.update(pointclouds[scene]["sequences"][seq]["thresholds"])
    
    # Sort thresholds numerically, highest first
    thrs = sorted(list(all_thresholds), reverse=True)
    thrs_str = [str(t) for t in thrs]
    options = [{"label": str(t), "value": str(t)} for t in thrs]
    
    # Check what triggered the callback
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "select-all-thresholds-btn":
            # Select all thresholds
            return options, thrs_str
    
    # Default: select first two thresholds if available
    value = [str(thrs[0]), str(thrs[1])] if len(thrs) >= 2 else [str(thrs[0])] if thrs else []
    return options, value

@app.callback(
    Output("plots-container", "children"),
    Output("visibility-checklist", "options"),
    Output("visibility-checklist", "value"),
    Output("toggle-row", "style"),
    Output("toggle-hr", "style"),
    Input("load-btn", "n_clicks"),
    State("scene-dropdown", "value"),
    State("sequence-dropdown", "value"),
    State("threshold-dropdown", "value"),
    prevent_initial_call=True
)
def load_pointclouds(n_clicks, scene, sequences, thresholds):
    print(f"DEBUG: load_pointclouds called with n_clicks={n_clicks}, scene={scene}, sequences={sequences}, thresholds={thresholds}")
    
    if not scene or not sequences or not thresholds or len(sequences) < 1 or len(thresholds) < 1:
        print("DEBUG: Early return - missing scene, sequences or thresholds")
        return html.Div("Select a scene, at least one sequence, and at least one threshold."), [], [], {"display": "none"}, {"display": "none"}
    
    print(f"DEBUG: Processing scene '{scene}' with sequences {sequences} and thresholds {thresholds}")
    
    all_plots = []
    checklist_options = []
    checklist_values = []
    
    # Create plots for each combination of sequence and threshold
    for seq in sequences:
        if seq not in pointclouds[scene]["sequences"]:
            print(f"DEBUG: Skipping invalid sequence {seq}")
            continue
            
        files = pointclouds[scene]["sequences"][seq]["files"]
        
        for thr_str in thresholds:
            thr_float = float(thr_str)
            
            # Skip if this threshold doesn't exist for this sequence
            if thr_float not in files:
                print(f"DEBUG: Threshold {thr_str} not available for {seq}, skipping")
                continue
            
            ply_path = files.get(thr_float)
            plot_id = f"{seq}_thr{thr_str}"
            plot_label = f"{seq} - thr={thr_str}"
            
            print(f"DEBUG: Processing {plot_label}, path: {ply_path}")
            
            if ply_path and os.path.exists(ply_path):
                try:
                    print(f"DEBUG: Reading PLY file: {ply_path}")
                    xyz, rgb = parse_ply(ply_path)
                    print(f"DEBUG: PLY loaded - points: {xyz.shape}, colors: {rgb.shape}")
                    
                    print("DEBUG: Creating Plotly trace...")
                    trace = go.Scatter3d(
                        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
                        mode='markers',
                        marker=dict(
                            size=1.5,
                            color=rgb,
                            opacity=0.8
                        )
                    )
                    
                    print("DEBUG: Creating figure...")
                    fig = go.Figure(data=[trace])
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=30, b=0),
                        scene=dict(
                            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                            aspectmode='data'
                        ),
                        title=f"{scene} - {plot_label}"
                    )
                    
                    print("DEBUG: Creating graph component...")
                    # Wrap each plot in a div with unique ID for toggling
                    # Using flex-basis for responsive width
                    plot_div = html.Div([
                        dcc.Graph(figure=fig, style={"height": "600px"})
                    ], id={"type": "plot-div", "index": plot_id}, 
                    style={
                        "flex": "1 1 400px",  # Grow, shrink, min-width 400px
                        "minWidth": "400px",
                        "maxWidth": "100%",
                        "padding": "10px",
                        "boxSizing": "border-box"
                    })
                    
                    all_plots.append(plot_div)
                    checklist_options.append({"label": plot_label, "value": plot_id})
                    checklist_values.append(plot_id)  # All visible by default
                    
                    print(f"DEBUG: Successfully created plot for {plot_label}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to process PLY file {ply_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    all_plots.append(html.Div(f"Error loading PLY file for {plot_label}: {str(e)}"))
            else:
                print(f"DEBUG: PLY file not found or doesn't exist: {ply_path}")
                all_plots.append(html.Div(f"PLY file not found for {plot_label}"))
    
    # Calculate responsive column width based on number of plots
    num_plots = len(all_plots)
    if num_plots == 0:
        return html.Div("No valid pointcloud combinations found."), [], [], {"display": "none"}, {"display": "none"}
    
    # Create flex container that automatically reflows when plots are hidden
    plots_grid = html.Div(
        all_plots,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "justifyContent": "flex-start",
            "alignItems": "flex-start",
            "gap": "0px"
        }
    )
    
    print(f"DEBUG: Returning {num_plots} plots")
    return plots_grid, checklist_options, checklist_values, {"display": "block"}, {"display": "block"}

# Callback to toggle visibility of plots without reloading
@app.callback(
    Output({"type": "plot-div", "index": dash.dependencies.ALL}, "style"),
    Input("visibility-checklist", "value"),
    State({"type": "plot-div", "index": dash.dependencies.ALL}, "id"),
    prevent_initial_call=True
)
def toggle_plot_visibility(selected_plots, plot_ids):
    """Toggle visibility of plots based on checklist - instant with no reloading"""
    print(f"DEBUG: toggle_plot_visibility called with selected_plots={selected_plots}")
    
    styles = []
    for plot_id_dict in plot_ids:
        plot_id = plot_id_dict["index"]
        if plot_id in selected_plots:
            # Show the plot with flex properties
            styles.append({
                "flex": "1 1 400px",
                "minWidth": "400px",
                "maxWidth": "100%",
                "padding": "10px",
                "boxSizing": "border-box",
                "display": "block"
            })
        else:
            # Hide the plot completely (removed from flex layout)
            styles.append({"display": "none"})
    
    return styles

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
