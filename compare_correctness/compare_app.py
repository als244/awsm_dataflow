import streamlit as st
import torch
import pandas as pd
import numpy as np
import io
import os
import tkinter as tk
from tkinter import filedialog

# --- 1. CONFIGURATION & CACHING ---
st.set_page_config(layout="wide", page_title="Tensor Debugger Pro")

@st.cache_resource(show_spinner="Loading large file into memory...")
def load_data_from_path(path):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location='cpu')
    except Exception as e:
        st.error(f"Error loading file from path: {e}")
    return None

@st.cache_resource(show_spinner="Reading uploaded file...")
def load_data_from_upload(uploaded_file):
    try:
        if uploaded_file is not None:
            return torch.load(io.BytesIO(uploaded_file.read()), map_location='cpu')
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
    return None

# --- 2. HELPER FUNCTIONS ---

def open_file_dialog():
    """Opens OS native file picker."""
    try:
        root = tk.Tk()
        root.withdraw() 
        root.wm_attributes('-topmost', 1) 
        file_path = filedialog.askopenfilename(
            title="Select PyTorch File",
            filetypes=[("Torch Files", "*.pt *.pth *.bin *.ckpt"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path
    except Exception:
        return ""

def get_item_at_path(data, path):
    """Traverses nested dictionaries/lists."""
    current = data
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, (list, tuple)):
            try:
                idx = int(key)
                if idx < len(current):
                    current = current[idx]
                else:
                    return None
            except:
                return None
        else:
            return None
    return current

def calculate_stats(tensor):
    t = tensor.float()
    return {
        "Shape": str(tuple(tensor.shape)),
        "Dtype": str(tensor.dtype).replace("torch.", ""),
        "Min": t.min().item(),
        "Max": t.max().item(),
        "Mean": t.mean().item(),
        "NaNs": torch.isnan(t).sum().item(),
    }

def apply_permutation(tensor, perm_str):
    if not perm_str.strip():
        return tensor
    try:
        dims = [int(x.strip()) for x in perm_str.split(",")]
        if len(dims) == tensor.ndim:
            return tensor.permute(*dims)
    except:
        pass
    return tensor

# --- 3. UI COMPONENTS ---

def render_directory_explorer(key_prefix):
    """
    Renders a directory browser.
    Returns: Selected file path (str) OR None (if still browsing)
    """
    # State for current browsing directory
    browse_key = f"browse_dir_{key_prefix}"
    if browse_key not in st.session_state:
        # Default to current working directory or user home
        st.session_state[browse_key] = os.getcwd()

    current_dir = st.session_state[browse_key]
    
    # Header with "Up" button
    c1, c2 = st.columns([0.1, 0.9])
    with c1:
        if st.button("⬆️", key=f"up_{key_prefix}", help="Go to parent directory"):
            parent = os.path.dirname(current_dir)
            if os.path.exists(parent):
                st.session_state[browse_key] = parent
                st.rerun()
    with c2:
        st.markdown(f"**Dir:** `{current_dir}`")

    # List contents
    try:
        items = os.listdir(current_dir)
        # Separate folders and supported files
        folders = sorted([d for d in items if os.path.isdir(os.path.join(current_dir, d))])
        files = sorted([f for f in items if os.path.isfile(os.path.join(current_dir, f))])
        
        # Filter for likely torch files to reduce clutter
        # You can remove this filter if you want to see all files
        torch_exts = ('.pt', '.pth', '.bin', '.ckpt', '.safetensors')
        files = [f for f in files if f.endswith(torch_exts)]
        
        # UI: Selectbox for Folders
        if folders:
            selected_folder = st.selectbox(
                "📁 Subdirectories", 
                options=[""] + folders, 
                key=f"subdirs_{key_prefix}"
            )
            if selected_folder:
                st.session_state[browse_key] = os.path.join(current_dir, selected_folder)
                st.rerun()

        # UI: Selectbox for Files
        if files:
            selected_file = st.selectbox(
                "📄 Files", 
                options=[""] + files, 
                key=f"files_{key_prefix}"
            )
            if selected_file:
                return os.path.join(current_dir, selected_file)
        else:
            st.caption("(No .pt/.pth files found in this folder)")
            
    except PermissionError:
        st.error("Permission denied accessing this directory.")
    except Exception as e:
        st.error(f"Error reading directory: {e}")
        
    return None


def render_input_method(label, key_prefix):
    st.subheader(f"{label} Source")
    
    # 1. Select Method
    method = st.radio(
        "Input Method", 
        ["Directory Explorer", "Local Path", "Browser Upload"], 
        horizontal=True, 
        key=f"method_{key_prefix}",
        label_visibility="collapsed"
    )
    
    data = None
    path_key = f"path_str_{key_prefix}"
    widget_key = f"p_{key_prefix}"
    loaded_key = f"is_loaded_{key_prefix}"
    nav_key = f"path_{key_prefix}"

    # Init State
    if path_key not in st.session_state:
        st.session_state[path_key] = ""
    if loaded_key not in st.session_state:
        st.session_state[loaded_key] = False
    
    # --- METHOD A: BROWSER UPLOAD ---
    if "Browser Upload" in method:
        # Clear path state if switching
        if st.session_state[path_key]:
            st.session_state[path_key] = ""
            st.session_state[widget_key] = ""
            st.session_state[loaded_key] = False
            if nav_key in st.session_state:
                st.session_state[nav_key] = []
            
        uploaded_file = st.file_uploader(f"Upload {label}", key=f"u_{key_prefix}")
        if uploaded_file:
            if nav_key in st.session_state:
                 st.session_state[nav_key] = []
            data = load_data_from_upload(uploaded_file)
            
    # --- METHOD B & C: LOCAL PATH / DIRECTORY EXPLORER ---
    else:
        # If "Directory Explorer" is chosen, render it to update the path
        if "Directory Explorer" in method:
            st.info("Navigate folders below. Select a file to lock it in.")
            explorer_selection = render_directory_explorer(key_prefix)
            
            # If user picked a file in the explorer, update the main state
            if explorer_selection:
                # Only update if changed to avoid loop
                if explorer_selection != st.session_state[path_key]:
                    st.session_state[path_key] = explorer_selection
                    st.session_state[widget_key] = explorer_selection
                    st.session_state[loaded_key] = False
                    st.rerun()

        # Render the standard Path UI (Shared by "Local Path" and "Dir Explorer")
        col_txt, col_btn, col_clr = st.columns([6, 1.5, 0.5], vertical_alignment="bottom")
        
        with col_btn:
            # "Browse" button is only needed for "Local Path" mode (Native OS Picker)
            # In "Directory Explorer" mode, the browsing happens inline above.
            if "Local Path" in method:
                if st.button("📂 Browse", key=f"btn_{key_prefix}"):
                    selected_path = open_file_dialog()
                    if selected_path:
                        st.session_state[path_key] = selected_path
                        st.session_state[widget_key] = selected_path
                        st.session_state[loaded_key] = False 
                        st.rerun()

        with col_clr:
            if st.session_state[path_key]:
                if st.button("❌", key=f"clr_{key_prefix}"):
                    st.session_state[path_key] = ""
                    st.session_state[widget_key] = ""
                    st.session_state[loaded_key] = False
                    if nav_key in st.session_state:
                        st.session_state[nav_key] = []
                    st.rerun()

        with col_txt:
            st.text_input(
                f"Selected Path", 
                value=st.session_state[path_key],
                key=widget_key, 
                disabled=True,
                placeholder="No file selected..."
            )

        # Loading Logic
        if st.session_state[path_key]:
            if not st.session_state[loaded_key]:
                st.info("File selected. Click Load to read into memory.")
                if st.button(f"🚀 Load {label}", key=f"trigger_{key_prefix}", type="primary"):
                    st.session_state[loaded_key] = True
                    st.session_state[nav_key] = [] 
                    st.rerun()
            else:
                data = load_data_from_path(st.session_state[path_key])
                if data is not None:
                     st.success("Loaded")
                else:
                     st.error("Failed to load.")
                     if st.button("Reset", key=f"reset_{key_prefix}"):
                         st.session_state[loaded_key] = False
                         st.rerun()
                
    return data

def render_navigator(side_name, data, state_key):
    if data is None:
        return None

    st.markdown(f"**{side_name} Navigation**")

    if state_key not in st.session_state:
        st.session_state[state_key] = []
    
    path = st.session_state[state_key]

    if path:
        cols = st.columns([0.15, 0.85])
        if cols[0].button("⬅", key=f"back_{side_name}"):
            st.session_state[state_key].pop()
            st.rerun()
        cols[1].code(f"{' / '.join(map(str, path))}", language="text")
    else:
        st.markdown("*Root Level*")

    current_obj = get_item_at_path(data, path)

    if current_obj is None and path:
        st.warning("⚠️ Previous navigation path is invalid. Resetting...")
        st.session_state[state_key] = []
        st.rerun()
        return None

    if isinstance(current_obj, dict):
        keys = list(current_obj.keys())
        disp_keys = keys[:1000]
        suffix = f"... ({len(keys)-1000} more)" if len(keys) > 1000 else ""
        
        selected_key = st.selectbox(
            f"Select Key ({len(keys)})", [""] + disp_keys, 
            key=f"sel_{side_name}_{len(path)}",
            format_func=lambda x: f"{x} {suffix}" if (x and x == disp_keys[-1] and suffix) else x
        )
        if selected_key:
            st.session_state[state_key].append(selected_key)
            st.rerun()
            
    elif isinstance(current_obj, (list, tuple)):
        indices = list(range(len(current_obj)))
        selected_idx = st.selectbox(
            f"Select Index ({len(indices)})", [""] + [str(i) for i in indices], 
            key=f"sel_{side_name}_{len(path)}"
        )
        if selected_idx:
            st.session_state[state_key].append(int(selected_idx))
            st.rerun()
            
    elif isinstance(current_obj, torch.Tensor):
        return current_obj
    else:
        st.warning(f"Leaf node reached (Type: {type(current_obj)}).")
        st.write(current_obj)
        
    return None

def render_permute_ui(tensor, side_name):
    st.markdown("---")
    st.caption(f"{side_name} Config")
    
    ndims = tensor.ndim
    default_perm = ", ".join(str(i) for i in range(ndims))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.text(f"Raw Shape: {tuple(tensor.shape)}")
    
    perm_str = st.text_input(f"Permute {side_name} Dims", value=default_perm, key=f"perm_{side_name}")
    
    try:
        new_tensor = apply_permutation(tensor, perm_str)
        with col2:
            st.text(f"New Shape: {tuple(new_tensor.shape)}")
        return new_tensor
    except Exception as e:
        st.error(f"Invalid Permutation: {e}")
        return tensor

# --- 4. MAIN LAYOUT ---

st.title("🔍 Interactive Tensor Debugger")

col1, col2 = st.columns(2)

final_ref = None
final_tgt = None

with col1:
    data_ref = render_input_method("Reference", "ref")
    st.divider()
    raw_ref = render_navigator("Ref", data_ref, "path_ref")
    
    if raw_ref is not None and isinstance(raw_ref, torch.Tensor):
        final_ref = render_permute_ui(raw_ref, "Ref")

with col2:
    data_tgt = render_input_method("Target", "tgt")
    
    if "path_ref" in st.session_state and len(st.session_state["path_ref"]) > 0:
        if st.button("🔗 Sync Navigation"):
            st.session_state["path_tgt"] = list(st.session_state["path_ref"])
            st.rerun()
            
    st.divider()
    raw_tgt = render_navigator("Tgt", data_tgt, "path_tgt")
    
    if raw_tgt is not None and isinstance(raw_tgt, torch.Tensor):
        final_tgt = render_permute_ui(raw_tgt, "Tgt")

# --- 5. COMPARISON ---

st.divider()

if final_ref is not None and final_tgt is not None:
    st.header("📊 Comparison")
    
    shape_match = final_ref.shape == final_tgt.shape
    if not shape_match:
        st.warning(f"⚠️ Shape Mismatch: {tuple(final_ref.shape)} vs {tuple(final_tgt.shape)}")
        st.info("Adjust the 'Permute Dims' settings above to align them.")
    
    run_btn = st.button("⚡ Run Analysis", type="primary", disabled=not shape_match)
    
    if run_btn and shape_match:
        with st.spinner("Calculating differences..."):
            t_ref_f = final_ref.float()
            t_tgt_f = final_tgt.float()
            
            stats_ref = calculate_stats(final_ref)
            stats_tgt = calculate_stats(final_tgt)
            
            if torch.equal(final_ref, final_tgt):
                st.success("✅ Tensors are IDENTICAL.")
                st.json(stats_ref)
            else:
                st.warning("⚠️ Differences Detected")
                
                diff = (t_ref_f - t_tgt_f).abs()
                numel = diff.numel()
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Reference**")
                    st.dataframe(pd.DataFrame([stats_ref]).T, use_container_width=True)
                with c2:
                    st.markdown("**Target**")
                    st.dataframe(pd.DataFrame([stats_tgt]).T, use_container_width=True)
                with c3:
                    st.markdown("**Diff Summary**")
                    diff_summary = {
                        "Max Abs Diff": diff.max().item(),
                        "Median Abs Diff": diff.median().item(),
                        "Mean Abs Diff": diff.mean().item(),
                        "Zero Count": (diff == 0).sum().item(),
                    }
                    max_diff_idx = torch.argmax(diff)
                    coords = torch.unravel_index(max_diff_idx, final_ref.shape)
                    diff_summary["Max Diff Coords"] = str(tuple(c.item() for c in coords))
                    
                    st.dataframe(pd.DataFrame([diff_summary]).T.astype(str), use_container_width=True)

                st.subheader("📉 Distribution")
                
                QUANTILE_LIMIT = 5_000_000
                
                try:
                    if numel > QUANTILE_LIMIT:
                        st.caption(f"ℹ️ Tensor too large for exact quantiles ({numel:,} elements). Sampling {QUANTILE_LIMIT:,} elements for estimation.")
                        indices = torch.randint(0, numel, (QUANTILE_LIMIT,))
                        diff_for_quant = diff.flatten()[indices]
                    else:
                        diff_for_quant = diff.flatten()
                        
                    q_values = torch.tensor([0.5, 0.9, 0.99, 0.999])
                    quantiles = torch.quantile(diff_for_quant, q_values)
                    
                    d1, d2 = st.columns(2)
                    with d1:
                        st.table(pd.DataFrame({
                            "Percentile": ["Median", "90%", "99%", "99.9%"],
                            "Diff Value": [f"{q:.2e}" for q in quantiles.tolist()]
                        }))
                    
                    with d2:
                        n_gt_1e2 = (diff > 1e-2).sum().item()
                        n_gt_1e3 = (diff > 1e-3).sum().item()
                        n_gt_1e4 = (diff > 1e-4).sum().item()
                        n_gt_1e5 = (diff > 1e-5).sum().item()
                        n_gt_1e6 = (diff > 1e-6).sum().item()
                        
                        st.table(pd.DataFrame({
                            "Threshold": ["Larger than 1e-2", "Larger than 1e-3", "Larger than 1e-4", "Larger than 1e-5", "Larger than 1e-6"],
                            "Count": [n_gt_1e2, n_gt_1e3, n_gt_1e4, n_gt_1e5, n_gt_1e6],
                            "Percentage": [f"{n_gt_1e2/numel*100:.2f}%", f"{n_gt_1e3/numel*100:.2f}%", f"{n_gt_1e4/numel*100:.2f}%", f"{n_gt_1e5/numel*100:.2f}%", f"{n_gt_1e6/numel*100:.2f}%"]
                        }))
                except Exception as e:
                    st.error(f"Error calculating distribution stats: {e}")

                st.subheader("Visualizations")
                
                # --- 0. PREPARATION (No Sampling) ---
                # We calculate on the FULL tensors for 100% accuracy.
                
                with st.spinner("Calculating full distribution..."):
                    f_ref = final_ref.float().flatten()
                    f_tgt = final_tgt.float().flatten()
                    
                    # Convert to numpy for histogram calculation (usually zero-copy if contiguous)
                    s_ref = f_ref.numpy()
                    s_tgt = f_tgt.numpy()
                    
                    # Global limits for the bins
                    global_max = max(abs(s_ref.max()), abs(s_tgt.max()))
                    global_max = max(global_max, 1e-9) # Prevent log(0) issues

                    import altair as alt

                    # --- CHART 1: VALUE DISTRIBUTION ---
                    st.markdown("#### 1. Value Distribution")
                    st.caption("Full population (exact). Log-spaced bins. Label indicates the **bin start**.")

                    # A. Create Log-Spaced Bins
                    # We calculate how many powers of 10 cover the data range
                    n_powers = int(np.ceil(np.log10(global_max))) - int(np.floor(np.log10(1e-9)))
                    n_powers = max(n_powers, 3) 
                    
                    # Generate edges: [ ... -0.01, -0.001, 0, 0.001, 0.01 ... ]
                    pos_edges = np.logspace(np.log10(1e-9), np.log10(global_max), num=15)
                    neg_edges = -np.flip(pos_edges)
                    edges = np.concatenate([neg_edges, [0], pos_edges])
                    
                    # B. Compute Histogram (On full data)
                    hist_ref, _ = np.histogram(s_ref, bins=edges)
                    hist_tgt, _ = np.histogram(s_tgt, bins=edges)
                    
                    # C. Format Labels (Start Value Only)
                    bin_labels = []
                    for i in range(len(edges)-1):
                        low = edges[i]
                        if abs(low) < 1e-9: 
                             lbl = "0" # The zero bin
                        elif abs(low) < 1e-3 or abs(low) > 1e3:
                             lbl = f"{low:.1e}" # Scientific notation
                        else:
                             lbl = f"{low:.3f}" # Regular float
                        bin_labels.append(lbl)

                    # D. Build DataFrame
                    source_data = []
                    for i, (count_r, count_t, lbl) in enumerate(zip(hist_ref, hist_tgt, bin_labels)):
                        # Filter out empty bins to keep chart clean
                        if count_r > 0 or count_t > 0:
                             source_data.append({"BinLabel": lbl, "SortIdx": i, "Count": count_r, "Type": "Reference"})
                             source_data.append({"BinLabel": lbl, "SortIdx": i, "Count": count_t, "Type": "Target"})
                    
                    df_chart = pd.DataFrame(source_data)

                    if not df_chart.empty:
                        c = alt.Chart(df_chart).mark_bar().encode(
                            # Sort by numerical index (SortIdx) so bins are in correct order
                            x=alt.X('BinLabel', 
                                    sort=alt.EncodingSortField(field="SortIdx", order='ascending'), 
                                    title='Bin Start Value (Log Scale)',
                                    axis=alt.Axis(labelAngle=-45)), 
                            y=alt.Y('Count', title='Frequency'),
                            color=alt.Color('Type', scale=alt.Scale(domain=['Reference', 'Target'], range=['#FF4B4B', '#1C83E1'])),
                            xOffset='Type', # Side-by-side bars
                            tooltip=['BinLabel', 'Count', 'Type']
                        ).interactive()
                        st.altair_chart(c, use_container_width=True)
                    else:
                        st.warning("No data found in standard ranges.")

                    st.divider()

                    # --- CHART 2: DIFFERENCES ---
                    st.markdown("#### 2. Difference Distribution")
                    
                    # Calculate Difference on full tensors
                    # Note: We do this in PyTorch first to save memory, then convert to numpy
                    diff_tensor = (final_ref.float() - final_tgt.float()).abs()
                    s_diff = diff_tensor.flatten().numpy()

                    # Log10 transform
                    plot_diff = np.log10(s_diff + 1e-12)
                    
                    # Histogram
                    hist_diff, bin_edges_diff = np.histogram(plot_diff, bins=50)
                    bin_centers = 0.5 * (bin_edges_diff[:-1] + bin_edges_diff[1:])
                    
                    df_diff_chart = pd.DataFrame({"LogValue": bin_centers, "Count": hist_diff})

                    c_diff = alt.Chart(df_diff_chart).mark_bar(color="#09AB3B").encode(
                        x=alt.X('LogValue', 
                                title='Difference Magnitude (Scientific Notation)',
                                # Custom label expression to show 1e-5 instead of -5
                                axis=alt.Axis(labelExpr="format(pow(10, datum.value), '.1e')")),
                        y='Count',
                        tooltip=[alt.Tooltip('LogValue', title="Log10 Value", format='.2f'), alt.Tooltip('Count')]
                    )
                    
                    st.altair_chart(c_diff, use_container_width=True)
                    st.caption("X-axis represents absolute difference (L1) on the **full** tensor.")

elif final_ref is None and final_tgt is None:
    st.info("Load files and navigate to tensors to begin.")