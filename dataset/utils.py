import SimpleITK as sitk
import numpy as np
import glob
from PIL import Image
import scipy.ndimage as nd
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import numpy as np

"""
def dense_crf(probs, img, n_iters=10, sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    b, w, h ,c = probs.shape
    probs = probs[0].transpose(2, 0, 1).copy(order='C')
    #print("num classes", b,w,h,c, img.shape)
    U = -np.log(probs).reshape([c, -1]).astype(np.float32)
    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((c, h, w)).transpose(1, 2, 0)
    #print("perds shape: ", preds.shape)
    return np.expand_dims(preds, 0)
"""
def evaluate(img, label):
    np.set_printoptions(suppress=True)
    img_temp = np.array(img)
    label_temp = np.array(label)
    dice=[0,0,0,0,0]
    dice_score = 0
    ppv_score = 0
    sens_score = 0
    Complete_list = []
    Core_list = []
    Enhanc_list = []
    hist = np.zeros((2, 2))
    for i in range(img.shape[0]):
        im = img_temp[i]
        la = label_temp[i]
        im[im>0] = 1
        la[la>0] = 1
        hist += fast_hist(la.flatten(), im.flatten(), 2)
    dice_score = np.sum(np.diag(hist)[1:])*2/float(np.sum(hist.sum(1)[1:])+np.sum(hist.sum(0)[1:]))
    ppv_score = np.sum(np.diag(hist)[1:])/float(np.sum(np.diag(hist)[1:])+hist[1][0])
    sens_score = np.sum(np.diag(hist)[1:])/float(np.sum(np.diag(hist)[1:])+hist[0][1])
    Complete_list.append(dice_score)
    Complete_list.append(ppv_score)
    Complete_list.append(sens_score)
    # for core
    hist = np.zeros((2, 2))
    img_temp = np.array(img)
    label_temp = np.array(label)
    for i in range(img.shape[0]):
        im = img_temp[i]
        la = label_temp[i]
        im[im==1] = 1
        im[im==3] = 1
        im[im==4] = 1
        im[im==2] = 0
        la[la==1] = 1
        la[la==3] = 1
        la[la==4] = 1
        la[la==2] = 0
        hist += fast_hist(la.flatten(), im.flatten(), 2)
    dice_score = np.sum(np.diag(hist)[1:])*2/float(np.sum(hist.sum(1)[1:])+np.sum(hist.sum(0)[1:]))
    ppv_score = np.sum(np.diag(hist)[1:])/float(np.sum(np.diag(hist)[1:])+hist[1][0])
    sens_score = np.sum(np.diag(hist)[1:])/float(np.sum(np.diag(hist)[1:])+hist[0][1])
    Core_list.append(dice_score)
    Core_list.append(ppv_score)
    Core_list.append(sens_score)
    # for enhacing
    hist = np.zeros((2, 2))
    img_temp = np.array(img)
    label_temp = np.array(label)
    for i in range(img.shape[0]):
        im = img_temp[i]
        la = label_temp[i]
        im[im==1] = 0
        im[im==4] = 1
        im[im==2] = 0
        im[im==3] = 0
        la[la==1] = 0
        la[la==4] = 1
        la[la==2] = 0
        la[la==3] = 0
        hist += fast_hist(la.flatten(), im.flatten(), 2)
    dice_score = np.sum(np.diag(hist)[1:])*2/float(np.sum(hist.sum(1)[1:])+np.sum(hist.sum(0)[1:])+1e-10)
    ppv_score = np.sum(np.diag(hist)[1:])/float(np.sum(np.diag(hist)[1:])+hist[1][0]+1e-10)
    sens_score = np.sum(np.diag(hist)[1:])/float(np.sum(np.diag(hist)[1:])+hist[0][1]+1e-10)
    Enhanc_list.append(dice_score)
    Enhanc_list.append(ppv_score)
    Enhanc_list.append(sens_score)

    return np.array(Complete_list), np.array(Core_list), np.array(Enhanc_list)

def remove(im, im_bin):
    x = {}
    M = -1
    ind = 0
    result = np.zeros((im.shape[0], im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im_bin[i, j] not in x:
                x[im_bin[i, j]] = 0
            else:
                x[im_bin[i, j]] += 1
    """
    all_slices = []
    for i in x:
        all_slices.append((i, x[i]))
    all_slices = sorted(all_slices, reverse=True, key=lambda x: x[1])
    if (len(all_slices) > 1):
        print(all_slices[1][1])
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                if x[im_bin[i, j]]  == all_slices[1][1]:
                    result[i][j] = int(im[i, j])
                else:
                    result[i][j] = 0
    # threshold on pixel value
    """
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if x[im_bin[i, j]] > 50:
                result[i][j] = int(im[i, j])
            else:
                result[i][j] = 0
    return result

def cc(result):
    im = sitk.GetImageFromArray(result)
    ccf = sitk.ConnectedComponentImageFilter()
    im_bin = ccf.Execute(im)
    im = result
    im_bin = sitk.GetArrayFromImage(im_bin)
    output = remove(im, im_bin)
    return output

def postprocessing(im):
  # 155 240 240
  check_len = 3
  connected = [[]]
  group = 0
  for i in range(0, 155):
    flag = 0
    if np.count_nonzero(im[i]) > 0:
      connected[group].append(i)
    else:
      connected.append([])
      group += 1
  maxg = 0
  max_g = []
  for g in connected:
    if len(g) > maxg:
      maxg = len(g)
      max_g = g
  for i in range(155):
    if i not in max_g:
      im[i] = np.zeros((240, 240))
  return im

def norm_image_by_patient(imname):
	im = sitk.GetArrayFromImage(sitk.ReadImage(imname)).astype(np.float32)
	return (im - im.mean()) / im.std()

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.uint8) + b[k], minlength=n**2).reshape(n, n)

def eval_single(pred, labels , num_class):
    #pred: [b, w, h, num_class]
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(labels.flatten(), pred.flatten(), num_class)
    return hist

def writeMedicalImage(im, filename, depth):
    im = sitk.GetImageFromArray(im[:,:,0])
    newImage_sitk = sitk.Cast(sitk.RescaleIntensity(im), sitk.sitkUInt8)
    #newImage = sitk.GetArrayFromImage(newImage_sitk)
    sitk.WriteImage(newImage_sitk, filename)

def writeImage(image, filename):
    """ store label data to colored image """
    Sky = [0,0,0]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road, Road_marking, Pavement])
    for l in range(0,6):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)
"""
All histogram matching methods are borrow from https://github.com/jupito/dwilib/blob/master/dwi/standardize.py

"""
def landmark_scores(img, pc, landmarks, thresholding, mask=None):
    """Get scores at histogram landmarks, ignoring nans and infinities.
    ref: https://github.com/jupito/dwilib/blob/master/dwi/standardize.py
    Parameters
    ----------
    img : ndarray
        Model used for fitting.
    pc : pair of numbers
        Minimum and maximum percentiles.
    landmarks : iterable of numbers
        Landmark percentiles.
    thresholding : 'none' or 'mean' or 'median'
        Threshold for calculating landmarks.
    mask : ndarray
        Image foreground mask for calculating landmarks (before thresholding).
    Returns
    -------
    p : pair of floats
        Minimum and maximum percentile scores.
    scores : tuple of floats
        Landmark percentile scores.
    """
    mask = img > 0
    img = img[mask]
    img = img[np.isfinite(img)]
    if thresholding == 'none':
        threshold = None
    elif thresholding == 'mean':
        threshold = np.mean(img)
    elif thresholding == 'median':
        threshold = np.median(img)
    else:
        raise ValueError('Invalid parameter: {}'.format(thresholding))
    if threshold:
        img = img[img > threshold]
    p = tuple(np.percentile(img, pc))
    scores = tuple(np.percentile(img, landmarks))
    return p, scores


def map_onto_scale(p1, p2, s1, s2, v):
    """Map value v from original scale [p1, p2] onto standard scale [s1, s2].
    Parameters
    ----------
    p1, p2 : number
        Minimum and maximum percentile scores.
    s1, s2 : number
        Minimum and maximum intensities on the standard scale.
    v : number
        Value to map.
    Returns
    -------
    r : float
        Mapped value.
    """
    assert p1 <= p2, (p1, p2)
    assert s1 <= s2, (s1, s2)
    if p1 == p2:
        assert s1 == s2, (p1, p2, s1, s2)
        return s1
    f = (v-p1) / (p2-p1)
    r = f * (s2-s1) + s1
    return r


def transform(img, p, scores, scale, mapped_scores):
    """Transform image onto standard scale.
    Parameters
    ----------
    img : ndarray
        Image to transform.
    p : pair of numbers
        Minimum and maximum percentile scores.
    scores : iterable of numbers
        Landmark percentile scores.
    scale : pair of numbers
        Minimum and maximum intensities on the standard scale.
    mapped_scores : iterable of numbers
        Standard landmark percentile scores on the standard scale.
    Returns
    -------
    r : ndarray of integers
        Transformed image.
    """
    p1, p2 = p
    s1, s2 = scale
    scores = [p1] + list(scores) + [p2]
    mapped_scores = [s1] + list(mapped_scores) + [s2]
    r = np.zeros_like(img, dtype=np.int16)
    for pos, v in np.ndenumerate(img):
        # Select slot where to map.
        slot = sum(v > s for s in scores)
        slot = np.clip(slot, 1, len(scores)-1)
        r[pos] = map_onto_scale(scores[slot-1], scores[slot],
                                mapped_scores[slot-1], mapped_scores[slot], v)
    r = np.clip(r, s1-1, s2+1, out=r)
    return r

def get_stats(pc, scale, landmarks, img, thresholding):
    """Gather info from single image."""
    p, scores = landmark_scores(img, pc, landmarks, thresholding)
    p1, p2 = p
    s1, s2 = scale
    mapped_scores = [map_onto_scale(p1, p2, s1, s2, x) for x in
                     scores]
    mapped_scores = [int(x) for x in mapped_scores]
    return dict(p=p, scores=scores, mapped_scores=mapped_scores)

def transform_image(img_list, default_configuration, mode=""):
    data = []
    landmarks = default_configuration['landmarks']
    pc = default_configuration['pc']
    scale = default_configuration['scale']
    thresholding = default_configuration['thresholding']

    for i in img_list:
        im = sitk.GetArrayFromImage(sitk.ReadImage(i))
        d = get_stats(pc, scale, landmarks, im, thresholding)
        data.append(d)
    mapped_scores = np.array([x['mapped_scores'] for x in data], dtype=np.int)
    mapped_scores = np.mean(mapped_scores, axis=0, dtype=mapped_scores.dtype)
    mapped_scores = list(mapped_scores)
    
    print(mode, mapped_scores)

    for i in img_list:
        im = sitk.GetArrayFromImage(sitk.ReadImage(i))
        p, scores = landmark_scores(im, pc, landmarks, thresholding)
        transformed_image = transform(im, p, scores, scale, mapped_scores)
        sitk.WriteImage(sitk.GetImageFromArray(transformed_image), i.replace(".nii", "NEW.nii"))

def normalizePerSequence():
    thresholding = 'mean'
    default_configuration = dict(
        pc=(0., 99.8),  # Min, max percentiles.
        landmarks=tuple(range(10, 100, 10)),  # Landmark percentiles.
        scale=(1, 100),  # Min, max intensities on standard scale.
        thresholding="none"
    )
    trainData = "/data/CVPR_Release/Brats17TrainingData/"
    validationData = "/data/CVPR_Release/Brats17ValidationData/"
    folderHGG = glob.glob(trainData + 'HGG/*')
    folderLGG = glob.glob(trainData + 'LGG/*')
    folder_train = folderHGG + folderLGG
    print(len(folder_train))
    folder_val = glob.glob(validationData + '*')
    print(len(folder_val))
    # get training data
    flair = []
    t2 = []
    t1 = []
    t1c = []
    for index, i in enumerate(folder_train+folder_val):
        flair += glob.glob(i + '/*flair.nii')
        t2 += glob.glob(i + '/*t2.nii')
        t1 += glob.glob(i + '/*t1.nii')
        t1c += glob.glob(i + '/*t1ce.nii')
    print(len(flair), len(t2), len(t1), len(t1c))
    # 210 + 75 + 46 = 331
    assert len(flair) == len(t2) == len(t1) == len(t1c)
    #transform_image(flair, default_configuration, mode="flair")
    #print('done flair...')
    #transform_image(t2, default_configuration, mode="t2")
    #print('done t2...')
    #transform_image(t1c, default_configuration, mode="t1c")
    #print('done t1c...')
    transform_image(t1, default_configuration, mode="t1")
    print('done t1...')

def refine():
    filenames = glob.glob("/data/CVPR_Release/v2/best3/*.nii")
    for imname in filenames:
        if ('Brats17_TCIA_195_1' in imname or 'Brats17_TCIA_248_1' in imname):
            im = sitk.GetArrayFromImage(sitk.ReadImage(imname))
            im[im==4] = 1
            image = sitk.GetImageFromArray(im)
            sitk.WriteImage(image, imname)
            print('done')


if __name__ == '__main__':
	#normalizePerSequence()
    refine()