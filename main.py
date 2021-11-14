from typing import Optional
from fastapi import FastAPI, BackgroundTasks, Request
import wget
import os
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import shutil
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import projector
import dataset_tool
from training import dataset
from training import misc
import requests
import pickle
import PIL.Image
import pretrained_networks
from encoder.generator_model import Generator
import getopt
import cv2
from faceswap.components.landmark_detection import detect_landmarks
from faceswap.components.convex_hull import find_convex_hull
from faceswap.components.delaunay_triangulation import find_delauney_triangulation
from faceswap.components.affine_transformation import apply_affine_transformation
from faceswap.components.clone_mask import merge_mask_with_image
from google.cloud import storage
import uuid
from pydantic import BaseModel

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
NETWORK_MODEL = "models/generator_yellow-stylegan2-config-f.pkl"
def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

app = FastAPI()

def project_image(proj, src_file, dst_dir, tmp_dir, video=False):

    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    shutil.copy(src_file, image_dir + '/')
    dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    print('Projecting image "%s"...' % os.path.basename(src_file))
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)
    if video:
        video_dir = '%s/video' % tmp_dir
        os.makedirs(video_dir, exist_ok=True)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if video:
            filename = '%s/%08d.png' % (video_dir, proj.get_cur_step())
            misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
    np.save(filename, proj.get_dlatents()[0])


@app.get("/")
def read_root(request: Request):
    client_host = request.client.host
    return {"Hello": client_host}

class BuildRequest(BaseModel):
    id: str
    image_url: str
    callback: str

def prep_project():
    _G, _D, Gs = pretrained_networks.load_networks(NETWORK_MODEL)
    proj = projector.Projector(
        vgg16_pkl             = "models/vgg16_zhang_perceptual.pkl",
        num_steps             = 1000,
        initial_learning_rate = 0.1,
        initial_noise_factor  = 0.05,
        verbose               = True
    )
    proj.set_network(Gs)
    return proj

def prep_landmark():
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    return landmarks_model_path

def build_projection(proj: projector.Projector, landmarks_model_path: str, item_id: str, image_url: str, callback_url: str):

    RAW_IMAGES_DIR = "raw_images/" + item_id
    ALIGNED_IMAGES_DIR = "aligned_images/" + item_id
    GENERATED_IMAGES_DIR = "generated_images/" + item_id
    if not os.path.exists(RAW_IMAGES_DIR):
        os.makedirs(RAW_IMAGES_DIR)
    wget.download(image_url, out=RAW_IMAGES_DIR  + "/source.png")
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)

    tmp_dir = ".stylegan2-tmp"

    src_files = sorted([os.path.join(ALIGNED_IMAGES_DIR, f) for f in os.listdir(ALIGNED_IMAGES_DIR) if f[0] not in '._'])
    for src_file in src_files:
        project_image(proj, src_file, GENERATED_IMAGES_DIR, tmp_dir, False)

    shutil.rmtree(tmp_dir)
    print(callback_url)
    f = open(GENERATED_IMAGES_DIR + "/success.txt","w+")
    f.write(item_id)
    f.close() 
    request_callback = requests.get(callback_url)
    print(request_callback)

@app.post("/build")
async def build(item: BuildRequest):
    try:
      item_id = item.id
      image_url = item.image_url
      callback_url = item.callback
      landmarks_model_path = prep_landmark()
      proj = prep_project()
      build_projection(proj, landmarks_model_path, item_id, image_url, callback_url)
      return {"canvas_id": item_id, "face_id": item_id, "image_url": image_url, "callback_url": callback_url}
    except:
      return {"canvas_id": "", "status":"error"}


def draw_style_mixing_figure(png, Gs,synthesis_kwargs, w, h, src_dlatents, dst_dlatents, columns):
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)      
    for row, dst_image in enumerate(list(dst_images)):
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_dlatents))
        for n in columns:
          row_dlatents[:, [n][row]] = src_dlatents[:, [n][row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            PIL.Image.fromarray(image, 'RGB').save(png)
    return png

def swap_images(src_img, dst_img, output):
    
    # Read images
    img_1 = cv2.imread(src_img)
    img_2 = cv2.imread(dst_img)

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


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    GCP_CRED_PATH = os.environ['GCP_CRED_PATH']
    storage_client = storage.Client.from_service_account_json(GCP_CRED_PATH)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    blob.make_public()
    return blob.public_url

@app.get("/face/{face_id}")
def read_root(face_id: str):
    try:    
      src_latent = "generated_images/" + face_id + "/source_01.npy"
      src_done = "generated_images/" + face_id + "/success.txt"
      src_failed = "generated_images/" + face_id + "/failed.txt"
      if os.path.exists(src_done):
          return {"id": face_id , "status": "ready"}
      elif os.path.exists(src_failed):
          return {"id": face_id , "status": "failed"}
      elif os.path.exists(src_latent):
          return {"id": face_id , "status": "pending"}
      else:
          return {"id": face_id , "status": "none"}
    except:
      return {"id": "", "status":"error"}

@app.get("/mix/face/{face_id}/canvas/{canvas_id}")
async def mix_items(face_id: str, canvas_id: str, columns: Optional[str] = "4,5,6,7,8,9"):
    try:
      uid = uuid.uuid1()
      unique_id = str(uid)
      output_filename = unique_id + ".png"
      temp = "mixed_images/" + unique_id + ".temp.png"
      RAW_IMAGES_DIR = "raw_images/" + canvas_id
      src_latent = "generated_images/" + face_id + "/source_01.npy"
      dst_latent = "generated_images/" + canvas_id + "/source_01.npy"
      output = "mixed_images/" + output_filename
      generator_network, discriminator_network, Gs_network = pretrained_networks.load_networks(NETWORK_MODEL)
      tflib.init_tf()
      synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=1)
      _Gs_cache = dict()
      src_npy = np.load(src_latent)
      dst_npy = np.load(dst_latent)
      cols = []
      for c in columns.split(','):
          cols.append(int(c))
      draw_style_mixing_figure(temp, Gs_network, synthesis_kwargs, w=1024, h=1024, src_dlatents=src_npy.reshape((-1, 18, 512)), dst_dlatents=dst_npy.reshape((-1, 18, 512)), columns=cols)
      for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f[0] not in '._']:
          raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
          swap_images(temp, raw_img_path, output)
      bucket_name = os.environ['GCP_UPLOAD_BUCKET']
      url = upload_blob(bucket_name, output, output_filename)
      return {"image_url": url}
    except:
      return {"image_url": "", "status":"error"}