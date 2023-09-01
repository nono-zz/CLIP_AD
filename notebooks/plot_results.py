# import pandas as pd
# import matplotlib.pyplot as plt

# import os

# csv_dir = '/home/zhaoxiang/CLIP_AD/WinClip_zzx/result_winclip/csv'
# csv_files = [os.path.join(csv_dir, csv_file) for csv_file in os.listdir(csv_dir)]

# column_numbers = [0,1]

# for column_number in column_numbers:
#     plt.figure(figsize=(10, 6))
#     for csv_file in csv_files:
#         data = pd.read_csv(csv_file, index_col = 0)
#         column = data.columns[column_number]
#         # transposed_data = data.transpose()
        
        
#         plt.bar(data.index, data[column], label=os.path.basename(csv_file))
#         plt.title(f'{column}')
#         plt.xlabel('Categories')
#         plt.ylabel(column)
#         plt.xticks(rotation=45)
#         plt.legend()
#         plt.savefig(f'{column}_comparison.png')
#         # plt.savefig('mvtec-15-vv-indx-1.png')
        
        
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# csv_dir = '/home/zhaoxiang/CLIP_AD/WinClip_zzx/result_winclip/csv'
csv_dir = '/home/zhaoxiang/CLIP_AD/CLIP_Surgery_zzx/result_clipSurgery/csv'
csv_files = [os.path.join(csv_dir, csv_file) for csv_file in os.listdir(csv_dir)]

plot_save_dir = csv_dir.replace('csv', 'plot')
if not os.path.exists(plot_save_dir):
    os.mkdir(plot_save_dir)

column_numbers = [0, 1, 2, 3, 4, 5]
# column_numbers = [0, 1]

bar_width = 0.2  # Width of each bar

for column_number in column_numbers:
    plt.figure(figsize=(10, 6))
    for idx, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file, index_col=0)
        column = data.columns[column_number]    
        x_positions = np.arange(len(data.index))  # X-axis positions for the groups
        
        # Shift the x-coordinates for each group of bars
        x_shifted = x_positions + idx * bar_width
        plt.bar(x_shifted, data[column], width=bar_width, label=os.path.basename(csv_file))
        
        # create a line indicating the image-level normal/anomalou sample ratio
        if 'i' in column:
            ratios = data['a/n_i_ratio']
            for i, ratio in enumerate(ratios):
                plt.hlines(y=ratio*100, xmin=x_shifted[i] - bar_width / 2, xmax=x_shifted[i] + bar_width / 2, color='red', linestyle='--', linewidth=2)
                plt.hlines(y=(1-ratio)*100, xmin=x_shifted[i] - bar_width / 2, xmax=x_shifted[i] + bar_width / 2, color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Comparison for Column {column}')
    plt.xlabel('Categories')
    plt.ylabel(column)
    plt.xticks(x_positions + bar_width * (len(csv_files) - 1) / 2, data.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_save_dir, f'{column}_comparison.png'))
    