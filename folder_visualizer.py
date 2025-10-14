import os
import re
import glob
import numpy as np
import open3d as o3d
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

def parse_ply(filename, max_points=100000):
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
            'thresholds': [0.1, 0.2, ...],
            'files': {'0.1': '/path/to/scene1_thr0.1.ply', ...}
        },
        ...
    }
    """
    result = {}
    for scene_dir in glob.glob(os.path.join(base_dir, "*", "pointclouds")):
        scene_name = os.path.basename(os.path.dirname(scene_dir))
        ply_files = glob.glob(os.path.join(scene_dir, "*.ply"))
        thresholds = []
        files = {}
        for pf in ply_files:
            m = re.search(r"_thr([0-9.]+)", pf)
            if m:
                thr = m.group(1)
                thresholds.append(thr)
                files[thr] = pf
        if thresholds:
            thresholds = sorted(thresholds)
            result[scene_name] = {
                "thresholds": thresholds,
                "files": files
            }
    return result

# Set your base directory here
BASE_DIR = "/home/ape/repos/mapper/data/output_100425/"

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
            html.Label("Thresholds (select two):"),
            dcc.Dropdown(
                id="threshold-dropdown",
                multi=True
            ),
        ], width=6),
        dbc.Col([
            html.Button("Show", id="show-btn", n_clicks=0, className="btn btn-primary mt-4")
        ], width=3)
    ]),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Div(id="plot1-container")
        ], width=6),
        dbc.Col([
            html.Div(id="plot2-container")
        ], width=6)
    ])
], fluid=True)

@app.callback(
    Output("threshold-dropdown", "options"),
    Output("threshold-dropdown", "value"),
    Input("scene-dropdown", "value")
)
def update_thresholds(scene):
    if not scene or scene not in pointclouds:
        return [], []
    thrs = pointclouds[scene]["thresholds"]
    options = [{"label": str(t), "value": str(t)} for t in thrs]
    # Default: select first two thresholds if available
    value = [str(thrs[0]), str(thrs[1])] if len(thrs) >= 2 else [str(thrs[0])] if thrs else []
    return options, value

@app.callback(
    Output("plot1-container", "children"),
    Output("plot2-container", "children"),
    Input("show-btn", "n_clicks"),
    State("scene-dropdown", "value"),
    State("threshold-dropdown", "value")
)
def update_plots(n_clicks, scene, thresholds):
    print(f"DEBUG: update_plots called with n_clicks={n_clicks}, scene={scene}, thresholds={thresholds}")
    
    if not scene or not thresholds or len(thresholds) < 1:
        print("DEBUG: Early return - missing scene or thresholds")
        return html.Div("Select a scene and at least one threshold."), ""
    
    print(f"DEBUG: Processing scene '{scene}' with thresholds {thresholds}")
    files = pointclouds[scene]["files"]
    print(f"DEBUG: Available files: {files}")
    
    plots = []
    for i in range(2):
        if i < len(thresholds):
            thr = thresholds[i]
            ply_path = files.get(thr)
            print(f"DEBUG: Processing threshold {thr}, path: {ply_path}")
            
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
                        title=f"{scene} - thr={thr}"
                    )
                    
                    print("DEBUG: Creating graph component...")
                    plots.append(dcc.Graph(figure=fig, style={"height": "70vh"}))
                    print(f"DEBUG: Successfully created plot for threshold {thr}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to process PLY file {ply_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    plots.append(html.Div(f"Error loading PLY file for threshold {thr}: {str(e)}"))
            else:
                print(f"DEBUG: PLY file not found or doesn't exist: {ply_path}")
                plots.append(html.Div(f"PLY file not found for threshold {thr}"))
        else:
            plots.append(html.Div("No threshold selected."))
    
    print("DEBUG: Returning plots...")
    return plots[0], plots[1]

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
