import cv2

# 1. Define Board Parameters
# The units (meters, mm, etc.) don't affect the PNG generation directly but are stored in the board object.
square_length = 0.026  # checker size
marker_length = 0.019  # aruco size
squares_x = 6          # columns
squares_y = 4          # rows

# 2. Select the ArUco Dictionary (4x4)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# 3. Create the ChArUco Board Object
# Note: In newer OpenCV (4.7+), use cv2.aruco.CharucoBoard((x, y), ...)
# In older versions, use cv2.aruco.CharucoBoard_create(x, y, ...)
board = cv2.aruco.CharucoBoard((squares_y, squares_x), square_length, marker_length, aruco_dict)

# 4. Generate the Image
# Define the output image size in pixels (e.g., 1000x1500 for a 4x6 ratio)
image_size = (400, 600)
charuco_image = board.generateImage(image_size)

# 5. Save to PNG
cv2.imwrite("charuco_board_6x4.png", charuco_image)

# Optional: Display the board
cv2.imshow("ChArUco Board", charuco_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
