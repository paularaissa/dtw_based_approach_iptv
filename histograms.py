import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from dateutil.parser import parse
import datetime
import math
pd.set_option('mode.chained_assignment',None)
import warnings
warnings.filterwarnings("ignore")

class HistogramPipeline:
    def csv_reader(self, filename):
        for line in open(filename, "r"):
            yield line

    def get_minute(self, line):
        date = parse(line)
        date_minute = date.minute
        return date_minute

    def get_date(self, line):
        try:
            date = parse(line)
        except:
            return None
        date_hour = date.hour
        date_minute = date.minute
        date_second = date.second
        timestamp = str(date_hour) + ":" + str(date_minute) + ":" + str(date_second)
        timestamp2 = datetime.datetime.strptime(timestamp, "%H:%M:%S")
        return timestamp2

    # Remove TCP ACK packets
    def remove_tcp_ack(self, df):
        df_new = df[["frame.time", "ip.len", "ip.src", "ip.dst"]]
        df_new['ip.len'] = pd.to_numeric(df_new['ip.len'])
        data = df_new.loc[df_new["ip.len"] != 52]
        data = data.dropna()
        return data

    # Function to define the direction of the packet
    def set_upstream_downstream_regions(self, df, col):
        nan_value = float("NaN")
        df.replace("", nan_value, inplace=True)
        df.dropna(subset=["ip.len"], inplace=True)

        G = nx.from_pandas_edgelist(df, 'ip.src', 'ip.dst')
        G = nx.DiGraph(G)

        # Heuristic Sum of Packet Sizes
        sum_ip_len = df.groupby('ip.dst')['ip.len'].aggregate('sum').reset_index()
        if sum_ip_len.empty == False:
            sum_ip_len = sum_ip_len.sort_values(by=['ip.len'], ascending=False)
            ip_ref = sum_ip_len['ip.dst'].iloc[0]
            upstream = [n for n in nx.traversal.bfs_tree(G, ip_ref, reverse=True) if n != ip_ref]
            downstream = [n for n in nx.traversal.bfs_tree(G, ip_ref) if n != ip_ref]
            upstream_df = df[df["ip.src"].isin(upstream)]
            downstream_df = df[df["ip.dst"].isin(downstream)]
            downstream_negative = downstream_df[col].abs() * -1
            downstream_df.loc[:, col] = downstream_negative
            #downstream_df[col] = downstream_negative ##corrigir aqui
            frames = [upstream_df, downstream_df]
            result_sum_ipLen = pd.concat(frames)
            result_sum_ipLen = result_sum_ipLen.sort_values(by=['frame.time'])

            return result_sum_ipLen

    # Generate Histogram Function with package lenght
    def generate_histogram(self, df, path_name, file_name, cont, chunk):
        if df is not None:
            df['ip.len'] = df['ip.len'].apply(pd.to_numeric, errors='coerce')
            plt.clf()
            a_plot = sns.histplot(df["ip.len"], stat='probability', bins=10,
                                    log_scale=(False, True))
            a_plot.set(xlim=(-1500, 1500))
            plt.ylabel('Normalized Packets')
            plt.xlabel('Size (bytes)')
            plt.savefig(path_name + "/" + file_name + "_hist_" + str(cont) + "chunk_" + str(chunk) + '.png')
            ax = plt.gca()
            p = a_plot.patches
            heights = [patch.get_height() for patch in p]
            widths = [patch.get_width() for patch in p]
            return heights

    def create_time_diff_col(self, df):
        df["frame.time_new"] = pd.to_datetime(df["frame.time"])
        df['time_diff'] = df['frame.time_new'].diff()
        df['time_diff'] = pd.to_datetime(df['time_diff'], format='%d.%m.%Y %H:%M:%S.%f', errors='ignore')
        df['time_diff'] = df['time_diff'].dt.total_seconds()
        df.loc[0, 'time_diff'] = 0
        return df

    # Function to create the dataset of histograms
    def create_histogram_dataset(self, df_hist, list_values, label, filename):
        if list_values is not None:
            list_values.append(label)
            list_values.append(filename)
            df_hist.append(list_values)
            return df_hist

    def compute_kl_divergence(self, p_probs, q_probs):
        """"KL (p || q)"""
        kl_div_1 = p_probs * np.log(p_probs / q_probs)
        kl_div_2 = q_probs * np.log(q_probs / p_probs)
        kl_div = (kl_div_1 + kl_div_2) / 2
        return np.sum(kl_div)

    def balanced_kl_divergence(self, p, q):
        np.seterr(divide='ignore', invalid='ignore')
        kl1_abs = abs(np.where(p != 0, p * np.log(p / q), 0))
        filtered_kl1 = [v for v in kl1_abs if not (math.isinf(v) or math.isnan(v))]
        kl2_abs = abs(np.where(q != 0, q * np.log(q / p), 0))
        filtered_kl2 = [v for v in kl2_abs if not (math.isinf(v) or math.isnan(v))]
        kl1 = np.sum(filtered_kl1)
        kl2 = np.sum(filtered_kl2)
        kl_balanced = (kl1 + kl2) / 2
        return kl_balanced

    def build_histograms_time_interval(self, file_name, csv_dir, chunk, path_name_histograms, histogram_dataset_path, label):
        List = []
        list_segments = []
        row_count = 0
        df_hist = []
        if os.path.getsize(csv_dir + file_name + ".csv") != 0:  # Verify if the csv is empty
            csv_gen = self.csv_reader(csv_dir + file_name + ".csv")
            for row in csv_gen:
                if row_count == 0:
                    cols_names = row.split('"')
                # Get initial frame time
                if row_count == 1:
                    cols_values = row.split('"')
                    date_min_ini = self.get_date(cols_values[0], )

                # Split blocks per time chunk
                if row_count > 0:
                    cols_values = row.split('"')
                    # Validation for incomplete rows
                    non_empty_row = len(cols_values) - cols_values.count("")
                    has_next = next(csv_gen, None)
                    if non_empty_row >= 3 and has_next is not None:
                        date_min_now = self.get_date(cols_values[0])
                        diff = (date_min_now - date_min_ini).total_seconds()
                        if (diff < float(chunk)):
                            List.append(cols_values)
                        if (diff > float(chunk)) or (diff < float(chunk) and has_next is None):
                            df = pd.DataFrame(List, columns=cols_names)
                            df = df.loc[df["ip.version"] != "4,4"]  # remove invalid rows
                            df = self.remove_tcp_ack(df)
                            df = self.create_time_diff_col(df)
                            upstream_downstream_ipCount, upstream_downstream_ipSum = self.set_upstream_downstream_regions(
                                df, 'time_diff')
                            list_segments.append([file_name, date_min_ini.time(), date_min_now.time(), df.shape[0]])
                            hist = self.generate_histogram_package_interval(upstream_downstream_ipSum,
                                                                            path_name_histograms, file_name,
                                                                            date_min_now, chunk)

                            hist_dataset = self.create_histogram_dataset(df_hist, hist, label, file_name)
                            List.clear()
                            date_min_ini = date_min_now
                row_count = row_count + 1

            df_segments = pd.DataFrame(list_segments)
            histogram_dataset = pd.DataFrame(hist_dataset)
            histogram_dataset.to_csv(histogram_dataset_path + "/" + file_name + "_" + str(chunk) + ".csv", index=False,
                                     header=True)
        return histogram_dataset, df_segments

    def build_histograms(self, file_name, csv_dir, chunk, path_name_histograms, histogram_dataset_path, label):
        List = []
        list_segments = []
        row_count = 0
        time_count = 0
        df_hist = []
        hist_dataset = []
        if os.path.getsize(csv_dir + file_name + ".csv") != 0:  # Verify if the csv is empty
            csv_gen = self.csv_reader(csv_dir + file_name + ".csv")
            for row in csv_gen:
                if row_count == 0:
                    cols_names = row.split('"')
                # Get initial frame time
                if row_count == 1:
                    cols_values = row.split('"')
                    date_min_ini = self.get_date(cols_values[0], )

                # Split blocks per time chunk
                if row_count > 0:
                    cols_values = row.split('"')
                    # Validation for incomplete rows
                    non_empty_row = len(cols_values) - cols_values.count("")
                    has_next = next(csv_gen, None)
                    if non_empty_row >= 3 and has_next is not None:
                        date_min_now = self.get_date(cols_values[0])
                        diff = (date_min_now - date_min_ini).total_seconds()
                        if (diff < float(chunk)):
                            List.append(cols_values)
                        if (diff > float(chunk)) or (diff < float(chunk) and has_next is None):
                            df = pd.DataFrame(List, columns=cols_names)
                            df = df.loc[df["ip.version"] != "4,4"]  # remove invalid rows
                            nan_value = float("NaN")
                            df.replace("", nan_value, inplace=True)
                            df.dropna(subset=["ip.len"], inplace=True) # remove linhas sem tamanho do pacote
                            df = self.remove_tcp_ack(df)
                            upstream_downstream_ipSum = self.set_upstream_downstream_regions(
                                df, "ip.len")
                            list_segments.append([file_name, date_min_ini.time(), date_min_now.time(), df.shape[0]])
                            hist = self.generate_histogram(upstream_downstream_ipSum, path_name_histograms, file_name,
                                                           date_min_now, chunk)
                            hist_dataset = self.create_histogram_dataset(df_hist, hist, label, file_name)
                            List.clear()
                            date_min_ini = date_min_now
                row_count = row_count + 1

            df_segments = pd.DataFrame(list_segments)
            histogram_dataset = pd.DataFrame(hist_dataset)
            histogram_dataset.to_csv(histogram_dataset_path + "/" + file_name + "_" + str(chunk) + ".csv", index=False,
                                     header=True)
        return histogram_dataset

    def row_count(self, input):
        with open(input) as f:
            for i, l in enumerate(f):
                pass
        return i

    def get_initial_date_global(self, csv_dir1, file_name1, csv_dir2, file_name2):
        with open(csv_dir1 + file_name1 + '.csv') as f:
            firstline_vpn1 = f.readlines()[1].rstrip()
            cols_values1 = firstline_vpn1.split('"')
            date_min_ini1 = self.get_date(cols_values1[0])
        with open(csv_dir2 + file_name2 + '.csv') as f2:
            firstline_vpn2 = f2.readlines()[1].rstrip()
            cols_values2 = firstline_vpn2.split('"')
            date_min_ini2 = self.get_date(cols_values2[0])
        if date_min_ini1 < date_min_ini2:
            date_min_ini_global = date_min_ini2
        elif date_min_ini2 < date_min_ini1:
            date_min_ini_global = date_min_ini1
        return date_min_ini_global

    def get_final_date_global(self, csv_dir1, file_name1, csv_dir2, file_name2):
        with open(csv_dir1 + file_name1 + '.csv') as f:
            lastline_vpn1 = f.readlines()[-1].rstrip()
            cols_values1 = lastline_vpn1.split('"')
            date_max_end1 = self.get_date(cols_values1[0])
            if (date_max_end1 is None):
                f.close()
                with open(csv_dir1 + file_name1 + ".csv") as f:
                    lastline_vpn1 = f.readlines()[-2].rstrip()
                    cols_values1 = lastline_vpn1.split('"')
                    date_max_end1 = self.get_date(cols_values1[0])

        with open(csv_dir2 + file_name2 + '.csv') as f:
            lastline_vpn2 = f.readlines()[-1].rstrip()
            cols_values2 = lastline_vpn2.split('"')
            date_max_end2 = self.get_date(cols_values2[0])
            if (date_max_end2 is None):
                f.close()
                with open(csv_dir2 + file_name2 + ".csv") as f:
                    lastline_vpn2 = f.readlines()[-2].rstrip()
                    cols_values2 = lastline_vpn2.split('"')
                    date_max_end2 = self.get_date(cols_values2[0])

        if date_max_end1 < date_max_end2:
            date_max_end_global = date_max_end1
        elif date_max_end2 < date_max_end1:
            date_max_end_global = date_max_end2
        return date_max_end_global

    def aligned_histograms(self, file_name, csv_dir, chunk, path_name_histograms, histogram_dataset_path, label,
                           date_min_ini_global, date_time_end):
        List = []
        list_segments = []
        row_count = 0
        df_hist = []
        if os.path.getsize(csv_dir + file_name + ".csv") != 0:  # Verify if the csv is empty
            csv_gen = self.csv_reader(csv_dir + file_name + ".csv")
            for row in csv_gen:
                if row_count == 0:
                    cols_names = row.split('"')
                # Get initial frame time
                if row_count == 1:
                    cols_values = row.split('"')
                    date_min_ini = self.get_date(cols_values[0], )
                # Split blocks per time chunk
                if row_count > 0:
                    cols_values = row.split('"')
                    date_min_now = self.get_date(cols_values[0])
                    if (date_min_ini_global - date_min_ini).total_seconds() == 0:
                        date_min_ini = date_min_ini_global
                    if (date_min_ini_global - date_min_ini).total_seconds() > 0:
                        if (date_min_now - date_min_ini_global).total_seconds() == 0:
                            # date_min_now = date_min_ini_global
                            date_min_ini = date_min_now
                    non_empty_row = len(cols_values) - cols_values.count("")
                    has_next = next(csv_gen, None)
                    if non_empty_row >= 3 and has_next is not None and date_min_ini >= date_min_ini_global:
                        diff = (date_min_now - date_min_ini).total_seconds()
                        if (diff < float(chunk)):
                            List.append(cols_values)
                        if (diff > float(chunk)) or (diff < float(chunk) and has_next is None and (
                                date_time_end - date_min_now).total_seconds() == 0):
                            df = pd.DataFrame(List, columns=cols_names)
                            df = df.loc[df["ip.version"] != "4,4"]  # remove invalid rows
                            df = self.remove_tcp_ack(df)
                            upstream_downstream_ipCount, upstream_downstream_ipSum = self.set_upstream_downstream_regions(
                                df, "ip.len")
                            list_segments.append([file_name, date_min_ini.time(), date_min_now.time(), df.shape[0]])
                            hist = self.generate_histogram(upstream_downstream_ipSum, path_name_histograms, file_name,
                                                           date_min_now, chunk)
                            hist_dataset = self.create_histogram_dataset(df_hist, hist, label, file_name)
                            List.clear()
                            date_min_ini = date_min_now
                row_count = row_count + 1

        df_segments = pd.DataFrame(list_segments)
        df_segments.to_csv(histogram_dataset_path + "/" + file_name + "_segments_aligned_" + str(chunk) + ".csv",
                           index=False, header=True)
        histogram_dataset = pd.DataFrame(hist_dataset)
        histogram_dataset.to_csv(histogram_dataset_path + "/" + file_name + "_aligned_" + str(chunk) + ".csv",
                                 index=False, header=True)
        return histogram_dataset