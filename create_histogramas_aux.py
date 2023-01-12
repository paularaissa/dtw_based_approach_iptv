import pandas as pd
from histograms import HistogramPipeline

pd.set_option('mode.chained_assignment', None)
import warnings

warnings.filterwarnings("ignore")

def histograms(df_map_videos_file):
    chunk = 1
    #path with the csv files
    base_path = "/IPTV_Datasets_28Jan/"

    hp = HistogramPipeline()
    histograms_list = []

    for idx, row in df_map_videos_file.iterrows():
        path_hist = base_path + row["foldername"] + "/histograms" + "/figures/"
        path_datasets = base_path + row["foldername"] + "/histograms" + "/datasets/"
        file_name = row["filename"].split(".")[0]  # remove .csv do nome do ficheiro
        print(row["filename"])
        hist_vpn = hp.build_histograms(file_name, base_path + row["foldername"] + "/", chunk, path_hist, path_datasets,
                                       file_name)
        hist_vpn['foldername'] = row["foldername"]
        hist_vpn['channel'] = row["channel"]
        hist_vpn['location'] = row["location"]
        hist_vpn['codec'] = row["codec"]
        hist_vpn['group_time'] = row["group_time"]
        hist_vpn['time'] = row['time']
        histograms_list.append(hist_vpn)

    return histograms_list
