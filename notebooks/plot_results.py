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

csv_dir = '/home/zhaoxiang/CLIP_AD/WinClip_zzx/result_winclip/csv'
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
    
    plt.title(f'Comparison for Column {column}')
    plt.xlabel('Categories')
    plt.ylabel(column)
    plt.xticks(x_positions + bar_width * (len(csv_files) - 1) / 2, data.index, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(plot_save_dir, f'{column}_comparison.png'))

        
        


# # Load the CSV file into a DataFrame
# data = pd.read_csv('CLIP_AD/WinClip_zzx/result_winclip/csv/mvtec-15-vv-indx-1.csv')

# # Set the 'index' column as the index for the DataFrame
# data.set_index('Unnamed: 0', inplace=True)

# # Plotting
# data.plot(kind='bar', figsize=(10, 6))
# plt.title('Results Visualization')
# plt.xlabel('Categories')
# plt.ylabel('Scores')
# plt.xticks(rotation=45)
# plt.legend(title='Metrics')
# # plt.show()
# plt.savefig('mvtec-15-vv-indx-1.png')