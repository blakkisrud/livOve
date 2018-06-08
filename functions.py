
#=================================================================
#
# Import functions
#
#=================================================================

from glob import glob
import os
import dicom as dc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

#=================================================================
#
# Custom functions
#
#=================================================================


def listdir_nohidden(path):
  
  return glob(path + "*")

def load_scan(path):
  
  slices = [dc.read_file(s) for s in listdir_nohidden(path)]
  slices.sort(key = lambda x: int(x.InstanceNumber))
  
  try:
      slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
      
  except:
      slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
      
  for s in slices:
      s.SliceThickness = slice_thickness
      
  return slices

def get_pixels_hu(scans):
  
  image = np.stack([s.pixel_array for s in scans])
  
  image = image.astype(np.int16)
  
  image[image == -2000] = 0
  
  intercept = scans[0].RescaleIntercept
  slope = scans[0].RescaleSlope
  
  if slope != 1:
      
      image = slope*image.astype(np.float64)
      image = image.astype(np.int16)
      
  image += np.int16(intercept)
  
  return np.array(image, dtype = np.int16)


def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
  fig,ax = plt.subplots(rows,cols,figsize=[12,12])
  for i in range(rows*cols):
      ind = start_with + i*show_every
      ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
      ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
      ax[int(i/rows),int(i % rows)].axis('off')
  plt.show()

def resample(image, scan, new_spacing=[1,1,1]):
  # Determine current pixel spacing
  spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
  spacing = np.array(list(spacing))

  resize_factor = spacing / new_spacing
  new_real_shape = image.shape * resize_factor
  new_shape = np.round(new_real_shape)
  real_resize_factor = new_shape / image.shape
  new_spacing = spacing / real_resize_factor
  
  image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
  
  return image, new_spacing

