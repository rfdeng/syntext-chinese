# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
import _pickle as cp
import matplotlib.pyplot as plt
from math import *
import cv2
import glob

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print()
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print (colorize(Color.RED,'Data not found and have problems downloading.',bold=True))
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')


def generate_TR_perspective(res, imgid, vertical=0):
  ninstance = len(res)
  for i in range(ninstance):
    img = res[i]['img']
    img_vis = img.copy()
    cross_flag = 0

    H, W, _ = img.shape
    lbl_candi = []
    texts = iter(res[i]['txt'])

    for id in range(res[i]['wordBB'].shape[2]):
      p1, p2, p3, p4 = [res[i]['wordBB'][:, j, id] for j in range(4)]

      # visualize wordBB
      polypts = res[i]['wordBB'][:,:,id].transpose().reshape(-1,1,2).astype(np.int32)
      cv2.polylines(img_vis, [polypts], True, (0,0,255), thickness=2)

      # judge valid
      if (p1[0] - p2[0]) * (p4[0] - p3[0]) < 0 or (p1[1] - p4[1]) * (p2[1] - p3[1]) < 0:
        cross_flag = 1
        break

      if len(lbl_candi) == 0:
        lbl_candi = next(texts).split('\n')
      lbl = lbl_candi[0]
      lbl_candi.pop(0)

      # preprocess for perspective transformation
      W_bb = 0.5 * (np.linalg.norm(p1-p2) + np.linalg.norm(p3-p4))
      H_bb = 0.5 * (np.linalg.norm(p1-p4) + np.linalg.norm(p2-p3))

      H_new = int(H_bb + 0.5)
      W_new = int(W_bb + 0.5)

      # perspective transformation
      tarpts = np.float32([[0, 0], [W_new, 0], [W_new, H_new], [0, H_new]])
      M = cv2.getPerspectiveTransform(res[i]['wordBB'][:,:,id].transpose(), tarpts)
      tarimg = cv2.warpPerspective(img, M, (W_new, H_new))

      # save results
      outpath = osp.join(output_dir, 'Recog', str(imgid//100))
      if not osp.exists(outpath):
        os.makedirs(outpath)
      outname = osp.join(outpath,str(imgid)+'_'+str(i) +'_'+str(id)+'_' +lbl + '.jpg')
      if vertical:
        stat = cv2.imwrite(outname, np.rot90(tarimg[:,:,::-1], -1))
      else:
        stat = cv2.imwrite(outname, tarimg[:,:,::-1])
      if not stat:
        print(colorize(Color.RED,'Failed to write the image..',bold=True))
        print('corresponding corners are: ', p1, p2, p3, p4)
        print('the label is: ', lbl)
        os.remove(outname)

    if cross_flag:
      invalid_list = glob.glob(osp.join(output_dir, 'Recog', str(imgid//100), str(imgid)+'_'+str(i) +'*.jpg'))
      for ele in invalid_list:
        os.remove(ele)

    Origin_outpath = osp.join(output_dir, 'Origin', str(imgid//1000))
    if not osp.exists(Origin_outpath):
      os.makedirs(Origin_outpath)
    if vertical:
      cv2.imwrite(osp.join(Origin_outpath, str(imgid) + '_' + str(i) + '.jpg'), np.rot90(img_vis[:, :, ::-1], -1))
    else:
      cv2.imwrite(osp.join(Origin_outpath, str(imgid) + '_' + str(i) + '.jpg'), img_vis[:,:,::-1])



def generate_TR_rotation(res, vertical=0):
  ninstance = len(res)
  for i in range(ninstance):
    img = res[i]['img']
    H, W, _ = img.shape
    lbl_candi = []
    texts = iter(res[i]['txt'])

    for id in range(res[i]['wordBB'].shape[2]):
      flag = 0
      p1,p2,p3,p4 = [res[i]['wordBB'][:, j, id] for j in range(4)]
      if len(lbl_candi) == 0:
        lbl_candi = next(texts).split('\n')
      lbl = lbl_candi[0]
      lbl_candi.pop(0)

      # calculate the rotation matrix
      rad = 0.5*(atan2(p2[1]-p1[1], p2[0]-p1[0]) + atan2(p3[1]-p4[1], p3[0]-p4[0]))
      rotateMtrix = cv2.getRotationMatrix2D((W/2, H/2), degrees(rad), 1)
      Hnew = int(W * fabs(sin(rad)) + H * fabs(cos(rad)))
      Wnew = int(H * fabs(sin(rad)) + W * fabs(cos(rad)))

      rotateMtrix[0,2] += (Wnew - W) / 2
      rotateMtrix[1,2] += (Hnew - H) / 2
      imgRotated = cv2.warpAffine(img, rotateMtrix, (Wnew, Hnew), borderValue=(0,0,0))

      # the four vertex of the new rect
      [[p1[0]], [p1[1]]] = np.dot(rotateMtrix, np.array([[p1[0]], [p1[1]], [1]]))
      [[p3[0]], [p3[1]]] = np.dot(rotateMtrix, np.array([[p3[0]], [p3[1]], [1]]))
      [[p2[0]], [p2[1]]] = np.dot(rotateMtrix, np.array([[p2[0]], [p2[1]], [1]]))
      [[p4[0]], [p4[1]]] = np.dot(rotateMtrix, np.array([[p4[0]], [p4[1]], [1]]))

      if p1[1] > p4[1]:  # handling the reversed cases
        p1,p4 = p4,p1
        p2,p3 = p3,p2
        cropped = imgRotated[int(max(0, min(p1[1], p2[1]))):min(Hnew, int(max(p3[1], p4[1]))),
                             int(max(0, min(p1[0], p4[0]))):min(Wnew, int(max(p2[0], p3[0])))][::-1,:,:]
        flag = 1
      else:
        cropped = imgRotated[int(max(0, min(p1[1], p2[1]))):min(Hnew, int(max(p3[1], p4[1]))),
                             int(max(0, min(p1[0], p4[0]))):min(Wnew, int(max(p2[0], p3[0])))]

      if p1[0] > p2[0] and p2[1] > p3[1]:  # there is an unreasonable case(rare) that I just skip it.
        continue

      outname = osp.join(output_dir, lbl + '.jpg')
      stat = cv2.imwrite(outname, cropped[:,:,::-1])
      if not stat:
        print('wrong cropping:', p1, p2, p3, p4)
        print(lbl)
        print(imgRotated.shape)
        print(flag)

def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    db['data'][dname].attrs['txt'] = res[i]['txt']
    #L = res[i]['txt']
    #L = [n.encode("utf-8", "ignore") for n in L]
    #db['data'][dname].attrs['txt'] = L


## Define some configuration variables:
NUM_IMG = 200 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = None #max time per image in seconds

vertical_FLAG = 1 # added by ruifeng, whether to create vertical lines.

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = osp.join(DATA_PATH,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'

##### declared by ruifeng#####
im_dir = './data/bg_img'
depth_dir = './data/depth.h5'
seg_dir = './data/seg.h5'
filtered_dir = 'imnames.cp'

output_dir = './data/generated/'
########### end ##############




def main(viz=False):
  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  ##db = get_data()
  ############## added by ruifeng##############

  depth_db = h5py.File(depth_dir, 'r')
  seg_db = h5py.File(seg_dir, 'r')

  imnames = sorted(depth_db.keys())

  with open(filtered_dir, 'rb') as f:
    filtered_imnames = set(cp.load(f))

  ################## end ######################
  print (colorize(Color.BLUE,'\t-> done',bold=True))

  # open the output h5 file:
  #out_db = h5py.File(OUT_FILE,'w')
  #out_db.create_group('/data')
  print (colorize(Color.GREEN,'Storing the output in: '+output_dir, bold=True))

  # get the names of the image files in the dataset:
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  for i in range(start_idx,end_idx):
    imname = imnames[i]
    # ignore if not in filetered list:
    if imname not in filtered_imnames: continue

    try:
      # get the image:
      #img = Image.fromarray(db['image'][imname][:])
      img = Image.open(osp.join(im_dir, imname)).convert('RGB')

      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = depth_db[imname][:].T
      depth = depth[:, :, 0]

      # get segmentation:
      seg = seg_db['mask'][imname][:].astype('float32')
      area = seg_db['mask'][imname].attrs['area']
      label = seg_db['mask'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz, Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz, Image.NEAREST))

      if vertical_FLAG:
        depth = np.rot90(depth)
        seg = np.rot90(seg)
        img = np.rot90(img)

      print (colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        generate_TR_perspective(res, i, vertical_FLAG)
        #add_res_to_db(imname,res,out_db)
      # visualize the output:
      if viz:
        if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      continue
  depth_db.close()
  seg_db.close()
  #out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)
