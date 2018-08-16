def histeq(im,nbr_bins=256):
  data = im.flatten()
  imhist,bins = histogram(data, nbr_bins, normed=True)
  cdf = imhist.cumsum()
  cdf = 255 * cdf
  im2 = interp(im.flatten(), bins[:-1], cdf)
  return im2.reshape(im.shape)


from PIL import Image
 
def fn(filt):
  img1 = Image.open("test.png").transpose(Image.FLIP_TOP_BOTTOM)
  rgb2xyz = (
    1, 0, 0, 0,
    1, 0, 0, 0,
    1, 0, 0, 0 )
  img1 = img1.convert("RGB", rgb2xyz)
  img1 = array(img1.convert('L'))
  img1 = histeq(img1)
 
  img1_f = 1 * (img1 > filt)
 
  return img1_f


from numpy import *
from pylab import *
from scipy.ndimage import measurements,morphology
 
def count_items(img1_f, it):
  im_open = morphology.binary_opening( \
    img1_f,ones((9,5)), iterations=it)
  labels_open, nbr_objects_open = measurements.label(im_open)
  return labels_open


def fig(filt, it):
  clf()
  imshow(Image.open("test.png"))
 
  gray()
  contour(count_items(fn(filt), it), origin='image')
  show()
 
fig(90, 3)
