import os
import argparse
import sys
import getopt
import cv2
from faceswap.components.landmark_detection import detect_landmarks
from faceswap.components.convex_hull import find_convex_hull
from faceswap.components.delaunay_triangulation import find_delauney_triangulation
from faceswap.components.affine_transformation import apply_affine_transformation
from faceswap.components.clone_mask import merge_mask_with_image

#pip install Keras==2.3.1
#pip install opencv-python==4.1.2.30
#sudo apt install libgl1-mesa-glx
#sudo apt-get install -y libsm6 libxext6 libxrender-dev

def swap_images(src_img, dst_img, output):
    
    # Read images
    src_img = cv2.imread("mixed_images/style-mixing.png")
    dst_img = cv2.imread("raw_images/ipman.jpg")
    
    img_1 = src_img
    img_2 = dst_img

    # find the facial landmarks which return the key points of the face
    # localizes and labels areas such as eyebrows and nose
    # we are using the first face found no matter what in this case, could be expanded for multiple faces here
    landmarks_1 = detect_landmarks(img_1)[0]
    landmarks_2 = detect_landmarks(img_2)[0]
    
    # create a convex hull around the points, this will be like a mask for transferring the points
    # essentially this circles the face, swapping a convex hull looks more natural than a bounding box
    # we need to pass both sets of landmarks here because we map the convex hull from one face to another
    hull_1, hull_2 = find_convex_hull(landmarks_1, landmarks_2, img_1, img_2)

    # divide the boundary of the face into triangular sections to morph
    delauney_1 = find_delauney_triangulation(img_1, hull_1)
    delauney_2 = find_delauney_triangulation(img_2, hull_2)

    # warp the source triangles onto the target face
    img_1_face_to_img_2 = apply_affine_transformation(delauney_1, hull_1, hull_2, img_1, img_2)
    img_2_face_to_img_1 = apply_affine_transformation(delauney_2, hull_2, hull_1, img_2, img_1)

    swap_1 = merge_mask_with_image(hull_2, img_1_face_to_img_2, img_2)
    swap_2 = merge_mask_with_image(hull_1, img_2_face_to_img_1, img_1)
    cv2.imwrite(output, swap_1);
    

def main():
    parser = argparse.ArgumentParser(description='Mix real-world images from StyleGAN2 latent space')
    parser.add_argument('src_img', help='Source Image')
    parser.add_argument('dst_img', help='Destination Image')
    parser.add_argument('output', help='Output image filename')
    parser.add_argument('--src_col', default='4', help='source of column style feature from 0-17')
    parser.add_argument('--dst_col', default='8', help='dest of column style from 0-17')
    parser.add_argument('--network_pkl', default='models/generator_yellow-stylegan2-config-f.pkl', help='Path to local generator_yellow-stylegan2-config-f.pkl')
    args, other_args = parser.parse_known_args()
    swap_images(args.src_img,args.dst_img,args.output)    

#python swap_images.py raw_images/love.jpg raw_images/wood.jpg raw_images/output.png
if __name__ == '__main__':
    main()
   