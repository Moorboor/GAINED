import os
import pandas as pd
import numpy as np
from pyvis.network import Network


# Placeholders for the 500-session population data.
# Replace these values with the actual population statistics.
def preprocess_df(df):
  '''Takes dataframe and drops columns'''
  df["name"] = df["filename"].apply(lambda x: x.lower().split()[0].replace(",", ""))
  df["date"] = df["filename"].apply(lambda x: x.lower().split()[3].replace(",", ""))
  df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
  df = df[["name", "date", "TCCS_C", "TCCS_SP", "ALLIANCE", "activation_mean", "engagement_mean", 'CTS_Behaviours', 'CTS_Cognitions', 'CTS_Discovery', 'CTS_Methods','HSCL_4_5']].copy()
  
  df_cts = df.filter(regex=r"^CTS", axis=1).map(lambda x: float(x) if type(x)==int else np.nan)
  df["cognitive_therapy"] = np.nanmean(df_cts, axis=1)
  df = df.drop(columns=['CTS_Behaviours', 'CTS_Cognitions', 'CTS_Discovery', 'CTS_Methods'])
  df = df.rename(columns={"TCCS_C": "Challenging",
                "TCCS_SP": "Supportive",
                "ALLIANCE": "Alliance",
                "activation_mean": "Activation",
                "engagement_mean": "Engagement",
                "cognitive_therapy": "Cognitive Therapy",
                "HSCL_4_5": "HSCL_4"})
  return df

df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath("")), "dag", "transcriptions", "combined_data_rationale.xlsx"))
df = preprocess_df(df)
POPULATION_MEAN = df[df.columns[2:]].apply(lambda x: np.nanmean(x))
POPULATION_STD =  df[df.columns[2:]].apply(lambda x: np.nanstd(x))



class DAG():
  ''' Creates a directed acyclic graph with 4 levels based on defined psychometrics'''
  
  def __init__(self):
      
    self.net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        directed=True,
        layout=True,
        cdn_resources="in_line"
    )
    
    self.node_names = [
        ["Challenging", 
        "Supportive",
        "Cognitive Therapy"],
        
        ["Alliance",
        "Engagement",
        "Activation"],
        
        ["HSCL_4"]
    ]
    
    self.edges = [
        ["Cognitive Therapy", "Activation"],
        ["Cognitive Therapy", "Engagement"],
        ["Challenging", "Alliance"],
        ["Supportive", "Engagement"],
        ["Alliance", "HSCL_4"],
        ["Engagement", "HSCL_4"],
        ["Activation", "HSCL_4"],
    ]
    self.net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "nodeSpacing": 150,
          "levelSeparation": 250
        }
      },
      "physics": {
        "enabled": false,
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "solver": "hierarchicalRepulsion",
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "fit": true
        }
      },
      "interaction": {
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "edges": {
        "scaling": {
          "min": 1,
          "max": 3,
          "label": { "enabled": true }
        },
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal",
          "roundness": 0.5
        }
      },
      "nodes": {
        "shape": "box",
        "margin": {
          "top": 12,
          "bottom": 12,
          "left": 16,
          "right": 16
        },
        "borderWidth": 0,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.1)",
          "size": 10,
          "x": 2,
          "y": 4,
          "opacity": 0.15
        }
      }
    }
    """)
    for i in range(len(self.node_names)):
      
      # Match Dash app color scheme with slightly softer variants if needed
      color = "#059669" # Success Green for level 1
      font_color = "#ffffff"
      shape = "ellipse" 
      
      if i==0:
        color = "#2563eb" # Primary Blue for level 0
      elif i==2:
        color = "#1f2937" # Dark Gray for final level 2 outcome
        
      for x in self.node_names[i]:       
        # Use a modern font like the rest of the dashboard. Force white font.
        self.net.add_node(x, level=i, title="Waiting for data...", color=color, shape=shape, 
                          font={"color": font_color, "size": 16, "face": "system-ui, -apple-system, sans-serif"})
            
    for src, dst in self.edges:
      self.net.add_edge(src, dst, color="#9ca3af", width=1.5) # Gray-400 for sleek, non-intrusive edges
                           
  def update_node_values(self, new_node_values):
    for key in new_node_values.keys():
      try:
          node = self.net.get_node(key)
          update_dict = new_node_values[key]
          for k2 in update_dict.keys():
            node[k2] = update_dict[k2]
      except Exception:
          pass # node not found in graph
        
  def get_new_node_values(self, df_z):
    for i in range(len(df_z)):
      new_node_values = df_z.iloc[i].to_dict()
      
      for key, value in new_node_values.items():

        # Handle NaN smoothly
        if pd.isna(value):
          d = {
              "font": {"size": 16, "color": "#ffffff"},
              "title": f"Data Missing (NaN)",
          }
        else:
          font_size = max(14, min(35, (22 + 10*value)))
          d = {
              "font": {"size": font_size, "color": "#ffffff"},
              "title": f"Z-Score: {value:.2f}",
          }
        if key == "Alliance" or key == "HSCL_4":
          d["shape"] = "box"
          
        new_node_values[key] = d
    
    return new_node_values


def get_z_values(df, p_mean_series, p_std_series, window=5):
  """
  Calculate the rolling mean for the subset of data (past 5 sessions).
  Then, compute z-score relative to the 500-session population mean.
  """
  res_df = df.copy()
  for col in res_df.select_dtypes(include=np.number).columns:
    ma = res_df[col].rolling(window=window, min_periods=1).mean()
    res_df[col] = ma
    
  # df is now the windowed rolling mean. Compute Z based on Population
  return (res_df - p_mean_series) / p_std_series


def create_session_dag_from_json(df):
    """
    Generate the DAG HTML string from the provided multidimensional session dataframe.
    """
    if df.empty:
        return ""
        
    # Standardize column names to match the DAG expected nodes
    rename_map = {
        "challenging": "Challenging",
        "supporting": "Supportive",
        "activation": "Activation",
        "engagement": "Engagement",
        "ALLIANCE": "Alliance", 
        "HSCL_4_5": "HSCL_4",
        "session_number": "session_number"
    }
    plot_df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Calculate Cognitive Therapy composite if the CTS fields exist
    cts_cols = ['cts_cognitions', 'cts_behaviours', 'cts_discovery', 'cts_methods']
    existing_cts = [c for c in cts_cols if c in plot_df.columns]
    if existing_cts:
        plot_df["Cognitive Therapy"] = plot_df[existing_cts].mean(axis=1, skipna=True)
    else:
        plot_df["Cognitive Therapy"] = np.nan
        
    # Ensure all required columns exist (fill with NaN if explicitly missing)
    required_cols = ["Challenging", "Supportive", "Cognitive Therapy", "Alliance", "Engagement", "Activation", "HSCL_4"]
    for c in required_cols:
        if c not in plot_df.columns:
            plot_df[c] = np.nan
            
    # Sort strictly by session number
    if "session_number" in plot_df.columns:
        plot_df = plot_df.sort_values("session_number").reset_index(drop=True)
        
    # Narrow down to just metrics
    metric_df = plot_df[required_cols].copy()
    
    # Convert Population Dictionaries to Series for vectorised math
    p_mean_series = pd.Series(POPULATION_MEAN)
    p_std_series = pd.Series(POPULATION_STD)

    # Calculate Z scores using rolling window of 5
    df_z = get_z_values(df=metric_df, p_mean_series=p_mean_series, p_std_series=p_std_series, window=5)
    print(metric_df)
    # Fill any missing/NaN z-scores with 0 (representing exactly the population mean) 
    # This prevents the UI from showing ugly NaN blocks for incomplete datasets.
    df_z = df_z.fillna(0)
    
    dag = DAG()
    # Apply standard z-score styling using the last row (representing the rolling window up to the last known session)
    new_node_values = dag.get_new_node_values(df_z=df_z.tail(1))
    dag.update_node_values(new_node_values)
    
    # Note: Returning HTML directly to embed inside iframe srcDoc
    return dag.net.generate_html()
