# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # csv_dir = '/home/zhaoxiang/CLIP_AD/WinClip_zzx/result_winclip/csv'
# csv_dir = '/home/zhaoxiang/CLIP_AD/CLIP_Surgery_zzx/result_clipSurgery/csv/single_word'
# csv_files = [os.path.join(csv_dir, csv_file) for csv_file in os.listdir(csv_dir)]

# plot_save_dir = csv_dir.replace('csv', 'plot')
# if not os.path.exists(plot_save_dir):
#     os.mkdir(plot_save_dir)
    
# column_numbers = [0, 1, 2, 3, 4, 5]
# for csv_file in csv_files:
#     file_name = os.path.basename(csv_file)
#     classname = file_name.replace('.csv', '')
#     for column_number in column_numbers:
#         plt.figure(figsize=(10, 6))  # Set the figure size (optional)
#         data = pd.read_csv(csv_file, index_col=0)
#         column = data.columns[column_number]    
    
#         plt.title(f'Comparison for Column {column} on category {classname}')
#         plt.xlabel('words')
#         plt.ylabel(column)
#         plt.legend()
#         # plt.tight_layout()
        
#         plt.savefig(os.path.join(plot_save_dir, f'{classname}-{column}_comparison.png'))
        
        
import pandas as pd
import matplotlib.pyplot as plt
import os

# Specify the directory containing CSV files
csv_dir = '/home/zhaoxiang/CLIP_AD/CLIP_Surgery_zzx/result_clipSurgery/csv/single_word'

# Create a directory to save plots if it doesn't exist
plot_save_dir = csv_dir.replace('csv', 'plot')
os.makedirs(plot_save_dir, exist_ok=True)

# Specify the column numbers you want to plot
# column_numbers = [0, 1, 2, 3, 4, 5]
column_numbers = [0, 1]

# Rotate angle for x-axis tick labels
x_tick_rotation = 45

# Iterate through CSV files
for csv_file in os.listdir(csv_dir):
    if csv_file.endswith(".csv"):
        file_name = os.path.splitext(csv_file)[0]
        classname = file_name.replace('.csv', '')

        # Read the CSV file
        data = pd.read_csv(os.path.join(csv_dir, csv_file), index_col=0)

        # Iterate through the specified column numbers
        for column_number in column_numbers:
            column_name = data.columns[column_number]

            # Create a figure and plot the data as a bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(data.index, data[column_name], label=column_name)

            # Set titles and labels
            plt.title(f'Comparison for {column_name} in {classname}')
            plt.xlabel('Words')
            plt.ylabel(column_name)
            plt.legend()

            # Rotate x-axis tick labels to show them all
            plt.xticks(rotation=x_tick_rotation, fontsize=8)

            # Save the plot
            plot_filename = f'{classname}-{column_name}_comparison.png'
            plot_filepath = os.path.join(plot_save_dir, plot_filename)
            plt.tight_layout()
            plt.savefig(plot_filepath)
            plt.close()  # Close the figure to avoid memory leaks
