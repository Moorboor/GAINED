from pyvis.network import Network
import os
import pandas as pd
import numpy as np




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
          "sortMethod": "hubsize",
          "nodeSpacing": 200,
          "levelSeparation": 350
        }
      },
      "physics": {
        "enabled": false
      },
      "edges": {
        "scaling": {
          "min": 1,
          "max": 4,
          "label": { "enabled": true }
        },
        "smooth": false
      },
      "nodes": {
        "shape": "box",
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.3)",
          "size": 12,
          "x": 6,
          "y": 6,
          "opacity": 0.5
        }
      }
    }
    """)

    for i in range(len(self.node_names)):
      
      color = "#bbd8b1"
      font_color = "white"
      shape = "ellipse"
      
      if i==0:
        color = "#8fb3df"
        font_color = "black"
        shape = "ellipse"
      elif i==2:
        color = "#891b2a"
        shape = "box"
        
      for x in self.node_names[i]:       
        self.net.add_node(x, level=i, title="", color=color, shape=shape, font={"color": font_color})
            
    for src, dst in self.edges:
      self.net.add_edge(src, dst, color="black")
                           
  def update_node_values(self, new_node_values):
    
    for key in new_node_values.keys():
      node = self.net.get_node(key)
      update_dict = new_node_values[key]
      for k2 in update_dict.keys():
        node[k2] = update_dict[k2]
        
  def get_new_node_values(self, df_z):
    
    for i in range(len(df_z)):
      new_node_values = df_z.iloc[i].to_dict()
      
      for key, value in new_node_values.items():
          d = {"font": {"size": max(10, (30 + 20*value))},
              "title": f"{key}-Zscore: {value:.1f}"}
          if key == "Alliance":
            d["shape"] = "box"
          new_node_values[key] = d
    
    return new_node_values



def preprocess_df(df):
  '''Takes dataframe and drops columns'''
  df["name"] = df["filename"].apply(lambda x: x.lower().split()[0].replace(",", ""))
  df["date"] = df["filename"].apply(lambda x: x.lower().split()[3].replace(",", ""))
  df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
  df = df[["name", "date", "TCCS_C", "TCCS_SP", "ALLIANCE", "activation_mean", "engagement_mean", 'CTS_Behaviours', 'CTS_Cognitions', 'CTS_Discovery', 'CTS_Methods','HSCL_4_5']]
  
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


def get_z_values(df, p_mean, p_std, window=3):
  for col in df.select_dtypes(include=np.number).columns:
    ma = df[col].rolling(window=window, min_periods=1).mean()
    df[col] = ma
  return (df - p_mean)/p_std


def create_session_dag(df, name, session_id, window=3):
  
  df = preprocess_df(df)
  p_mean = df[df.columns[2:]].apply(lambda x: np.nanmean(x))
  p_std = df[df.columns[2:]].apply(lambda x: np.nanstd(x))
  df_patient = df[df["name"]==name][:session_id]
  df_z = get_z_values(df=df_patient.drop(columns=["name", "date"]), 
                      p_mean=p_mean, 
                      p_std=p_std, 
                      window=window)

  dag = DAG()
  new_node_values = dag.get_new_node_values(df_z=df_z.tail(1))
  dag.update_node_values(new_node_values)
  
  # dag.net.save_graph(os.path.join(ABS_PATH, "output", f"dag-session_id-{session_id}.html")) 
  return dag.net.generate_html()
