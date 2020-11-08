import cv2

# loads the handwriting
img = cv2.imread("phrase_handwritten.png")

# img_rot = img[::-1]

img_rot = cv2.rotate(img, cv2.ROTATE_180)

cv2.imshow("Rotated Image", img_rot)

# cv2.imshow("inverted image", img_rot)

# create a copy of the image
img_copy = img_rot.copy()

# convert image to grayscale
gray_img = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

# read image in binary and display
thresh, binary_img = cv2.threshold(gray_img, 0, 250, cv2.THRESH_OTSU)
cv2.imshow("Binary images", binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Find a contour on the handwriting image
contours_list, hierarchy = cv2.findContours(
    binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
# print(hierarchy)

for cnt in contours_list:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_rot, (x, y), (x + w, y + h), (0, 255, 255), 2)
cv2.imshow("Contours marked on RGB image", img_rot)
cv2.imwrite("contours.jpg", img_rot)
cv2.waitKey(0)
cv2.destroyAllWindows()

# reading and creating reference image
ref_gray = cv2.imread("typed_B.png", cv2.IMREAD_GRAYSCALE)
ret, ref_binary_image = cv2.threshold(ref_gray, 0, 255, cv2.THRESH_OTSU)
cv2.imshow("Reference image", ref_binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Here i am")

# check if reference image contains more than one countours
ref_contour_list, ref_hierarchy = cv2.findContours(
    ref_binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
if len(ref_contour_list) >= 1:
    ref_contour = ref_contour_list[0]
else:
    import sys

    print("Reference image contains more than 1 contour. Please check!")
    sys.exit()


# find matching countours of the from the reference image
counter = 0

# save location of the matched contours in the tuple
distance_list = []
for cnt in contours_list:
    retval = cv2.matchShapes(cnt, ref_contour, cv2.CONTOURS_MATCH_I1, 0)
    # print(dist_list)
    distance_list.append(retval)
    counter = counter + 1

min_distance = min(distance_list)
print(
    "The minimum distance of the reference contour with a contour in the main image is "
    + str(min_distance)
)
ind_min_dist = distance_list.index(min_distance)

required_cnt = contours_list[ind_min_dist]
x, y, w, h = cv2.boundingRect(required_cnt)
cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Detected B", img_copy)
cv2.imwrite("dectected_b.jpg", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()