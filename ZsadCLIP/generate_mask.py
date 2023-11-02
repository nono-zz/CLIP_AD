# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)

# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Choose random coordinates within the salient object region
# random_coords = non_zero_coords[np.random.choice(len(non_zero_coords))]

# # Create an empty mask for the random shape
# random_shape_mask = np.zeros_like(salient_mask)

# # Generate a random shape at the chosen coordinates
# cv2.circle(random_shape_mask, tuple(random_coords[::-1]), 50, 255, thickness=cv2.FILLED)

# # Display the random shape mask
# cv2.imwrite('random_shape_mask.png', random_shape_mask)


# %% random poly
# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)

# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Shuffle the coordinates randomly
# np.random.shuffle(non_zero_coords)

# # Choose a random number of vertices for the polygon
# num_vertices = np.random.randint(3, 10)  # You can adjust the range as needed

# # Select the first 'num_vertices' coordinates
# random_vertices = non_zero_coords[:num_vertices]

# # Create an empty mask for the random shape
# random_shape_mask = np.zeros_like(salient_mask)

# # Draw the filled polygon on the empty mask
# cv2.fillPoly(random_shape_mask, [random_vertices], 255)

# cv2.imwrite('random_shape_mask.png', random_shape_mask)


# #%% random brush
# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# # salient_mask = cv2.imread('salient_mask.png', cv2.IMREAD_GRAYSCALE)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)


# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Shuffle the coordinates randomly
# np.random.shuffle(non_zero_coords)

# # Create an empty mask for the random brush
# random_brush_mask = np.zeros_like(salient_mask)

# # Choose parameters for the random brush
# num_strokes = np.random.randint(5, 20)  # Number of brushstrokes
# min_thickness = 5
# max_thickness = 15

# # Generate the random brushstrokes
# for _ in range(num_strokes):
#     start_point = tuple(non_zero_coords[np.random.randint(len(non_zero_coords))][::-1])
#     end_point = tuple(non_zero_coords[np.random.randint(len(non_zero_coords))][::-1])
#     thickness = np.random.randint(min_thickness, max_thickness + 1)
#     cv2.line(random_brush_mask, start_point, end_point, 255, thickness)

# cv2.imwrite('random_brush_mask.png', random_brush_mask)

# #%% connected brush
# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# # salient_mask = cv2.imread('salient_mask.png', cv2.IMREAD_GRAYSCALE)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)


# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Shuffle the coordinates randomly
# np.random.shuffle(non_zero_coords)

# # Create an empty mask for the connected random brush
# connected_random_brush_mask = np.zeros_like(salient_mask)

# # Choose parameters for the connected random brush
# num_strokes = np.random.randint(5, 20)  # Number of brushstrokes
# min_thickness = 5
# max_thickness = 15

# # Generate the connected random brushstrokes
# for i in range(num_strokes):
#     start_point = tuple(non_zero_coords[i][::-1])
#     end_point = tuple(non_zero_coords[(i + 1) % len(non_zero_coords)][::-1])
#     thickness = np.random.randint(min_thickness, max_thickness + 1)
#     cv2.line(connected_random_brush_mask, start_point, end_point, 255, thickness)

# # Save the connected random brush mask locally
# cv2.imwrite('connected_random_brush_mask.png', connected_random_brush_mask)

# # Display the saved image path
# print("Image saved as 'connected_random_brush_mask.png'")

# #%% fill 
# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# # salient_mask = cv2.imread('salient_mask.png', cv2.IMREAD_GRAYSCALE)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)


# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Shuffle the coordinates randomly
# np.random.shuffle(non_zero_coords)

# # Create an empty mask for the connected random brush
# connected_random_brush_mask = np.zeros_like(salient_mask)

# # Choose parameters for the connected random brush
# num_strokes = np.random.randint(4, 7)  # Number of brushstrokes
# min_thickness = 5
# max_thickness = 15

# # Generate the connected random brushstrokes
# for i in range(num_strokes):
#     start_point = tuple(non_zero_coords[i][::-1])
#     end_point = tuple(non_zero_coords[(i + 1) % len(non_zero_coords)][::-1])
#     thickness = np.random.randint(min_thickness, max_thickness + 1)
#     cv2.line(connected_random_brush_mask, start_point, end_point, 255, thickness)

# # Create a mask for the central region
# # central_region_mask = np.zeros_like(salient_mask)
# # cv2.fillPoly(central_region_mask, [non_zero_coords], 255)

# # central_region_mask = np.ones_like(salient_mask) * 255


# # Combine the connected random brush and central region masks
# final_mask = cv2.bitwise_or(connected_random_brush_mask, central_region_mask)

# # Save the final mask locally
# cv2.imwrite('final_mask.png', final_mask)

# # Display the saved image path
# print("Image saved as 'final_mask.png'")
# #%%
# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# # salient_mask = cv2.imread('salient_mask.png', cv2.IMREAD_GRAYSCALE)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)

# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Shuffle the coordinates randomly
# np.random.shuffle(non_zero_coords)

# # Create an empty mask for the connected random brush
# connected_random_brush_mask = np.zeros_like(salient_mask)

# # Choose parameters for the connected random brush
# num_strokes = np.random.randint(4, 7)  # Number of brushstrokes
# min_thickness = 5
# max_thickness = 15

# # Generate the connected random brushstrokes
# for i in range(num_strokes):
#     start_point = tuple(non_zero_coords[i][::-1])
#     end_point = tuple(non_zero_coords[(i + 1) % len(non_zero_coords)][::-1])
#     thickness = np.random.randint(min_thickness, max_thickness + 1)
#     cv2.line(connected_random_brush_mask, start_point, end_point, 255, thickness)

# # Create a mask for the central region
# central_region_mask = np.ones_like(salient_mask)

# # Combine the connected random brush and central region masks
# final_mask = cv2.bitwise_or(connected_random_brush_mask, central_region_mask)

# # Save the final mask locally
# cv2.imwrite('final_mask.png', final_mask)

# # Display the saved image path
# print("Image saved as 'final_mask.png'")


# #%%
# import numpy as np
# import cv2

# # Load your salient object mask (assuming it's a binary mask)
# salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)

# # Get the coordinates of non-zero pixels in the salient mask
# non_zero_coords = np.column_stack(np.where(salient_mask > 0))

# # Shuffle the coordinates randomly
# np.random.shuffle(non_zero_coords)

# # Create an empty mask for the random painted shape
# random_painted_shape_mask = np.zeros_like(salient_mask)

# # Choose parameters for the random painted shape
# num_points = np.random.randint(5, 20)  # Number of points for the irregular shape

# # Select the first 'num_points' coordinates
# random_points = non_zero_coords[:num_points]

# # Draw the filled polygon on the empty mask
# cv2.fillPoly(random_painted_shape_mask, [random_points], 255)

# # Create a mask for the central region
# central_region_mask = np.ones_like(salient_mask)

# # Ensure the central region is filled by setting random_painted_shape_mask to 255 in the central region
# random_painted_shape_mask[central_region_mask == 1] = 255

# # Save the final mask locally
# cv2.imwrite('random_painted_shape_mask.png', random_painted_shape_mask)

# # Display the saved image path
# print("Image saved as 'random_painted_shape_mask.png'")

#%%
import numpy as np
import cv2

# Load your salient object mask (assuming it's a binary mask)
salient_mask = cv2.imread('/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation/imgs/bottle/bottle/fore_00896.png', cv2.IMREAD_GRAYSCALE)

# Get the coordinates of non-zero pixels in the salient mask
non_zero_coords = np.column_stack(np.where(salient_mask > 100))

# Shuffle the coordinates randomly
np.random.shuffle(non_zero_coords)

# Create an empty mask for the random painted shape
random_painted_shape_mask = np.zeros_like(salient_mask)

# Choose parameters for the random painted shape
num_points = np.random.randint(3,6)  # Number of points for the irregular shape

# Select the first 'num_points' coordinates
random_points = non_zero_coords[:num_points]

# Draw the filled polygon on the empty mask
cv2.fillPoly(random_painted_shape_mask, [random_points], 255)

# # Create a mask for the central region
# central_region_mask = np.ones_like(salient_mask)

# # Ensure the central region is filled by setting random_painted_shape_mask to 255 only in the central region
# random_painted_shape_mask[central_region_mask == 1] = 255

# Save the final mask locally
cv2.imwrite('random_painted_shape_mask.png', random_painted_shape_mask)

# Display the saved image path
print("Image saved as 'random_painted_shape_mask.png'")
