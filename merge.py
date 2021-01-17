import cv2
import numpy as np
import dlib
import scipy as sp
from scipy import ndimage, misc


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def steps(img_in, img_match="profil.jpg"):
    x1_, y1_, x2_, y2_, pts_match = profil_points(img_match)
    x1, y1, x2, y2, pts = profil_points(img_in)
    h_ = abs(y2_-y1_)
    w_ = abs(x2_-x1_)
    center_ = np.average(pts_match, axis=0)
    d_ = np.average(np.abs(pts_match-center_))
    corners = np.array( ((x1, y1), (x2, y2)) )
    pts = pts @ R
    h = abs(y2-y1)
    w = abs(x2-x1)
    sw = w_/w
    sh = h_/h
    s = (sh+sw)/2
    center = np.average(pts, axis=0)
    d = np.average(np.abs(pts-center))
    s = d_/d
    scaled_pts = s*pts
    scaled_corners = s*corners
    trans = np.average(pts_match-scaled_pts, axis=0)
    centered_pts = scaled_pts+trans
    centered_corners = scaled_corners+trans
    show_points(pts, corners)
    show_points(scaled_pts, scaled_corners)
    show_points(centered_pts, centered_corners)


def show_points(pts, corners=None, image_name="profil.jpg"):
    global detector, predictor 
    img = cv2.imread(image_name)# Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)# Use detector to find landmarks
    faces = detector(gray)
    face = faces[0]
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point
    # Create landmark object
    landmarks = predictor(image=gray, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(img=img, center=(int(pts[n,0]), int(pts[n,1])), radius=3, color=(255, 0, 0), thickness=-1)
    cv2.circle(img=img, center=(x1, y1), radius=3, color=(255, 255, 0), thickness=-1)
    cv2.circle(img=img, center=(x2, y2), radius=3, color=(255, 255, 0), thickness=-1)
    if not corners is None:
        for n in range(0, 2):
            cv2.circle(img=img, center=(int(corners[n,0]), int(corners[n,1])), radius=3, color=(0, 0, 255), thickness=-1)
    cv2.imshow(winname="Face", mat=img)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

def profil_points(image_name="profil.jpg"):
    global detector, predictor
    pts = np.zeros( (68,2), np.float)
    img = cv2.imread(image_name)# Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)# Use detector to find landmarks
    faces = detector(gray)
    face = faces[0]
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point
    # Create landmark object
    landmarks = predictor(image=gray, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        pts[n] = (x,y)
        #cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
    return (x1, y1, x2, y2, pts)

def points(img):
    global detector, predictor
    pts = np.zeros( (68,2), np.float)

    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)# Use detector to find landmarks
    faces = detector(gray)
    face = faces[0]
    x1 = face.left() # left point
    y1 = face.top() # top point
    x2 = face.right() # right point
    y2 = face.bottom() # bottom point
    # Create landmark object
    landmarks = predictor(image=gray, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        pts[n] = (x,y)
        #cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
    return (x1, y1, x2, y2, pts)

def center(img):
    x1, y1, x2, y2, pts = points(img)
    center = np.average(pts, axis=0)
    return center

def center_points(image_name="profil.jpg"):
    x1, y1, x2, y2, pts = profil_points(image_name)
    center = np.average(pts, axis=0)
    center = pts[0]
    return center

def rotation(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def loss(pts, pts_match):
    diff = pts-pts_match
    sq_diff = diff**2
    return np.sum(sq_diff)

def loss_angle(pts, pts_match, angle=10):
    R = rotation(angle)
    pts = pts @ R
    center_ = np.average(pts_match, axis=0)
    d_ = np.average(np.abs(pts_match-center_))
    center = np.average(pts, axis=0)
    d = np.average(np.abs(pts-center))
    s = d_/d
    scaled_pts = s*pts
    trans = np.average(pts_match-scaled_pts, axis=0)
    centered_pts = scaled_pts+trans
    return loss(centered_pts, pts_match)

def best_angle(pts, pts_match):
    a = 0
    l_min = loss_angle(pts, pts_match, a)
    a+=0.5
    l = loss_angle(pts, pts_match, a)
    while l<=l_min:
        l_min = l
        a+=0.5
        l = loss_angle(pts, pts_match, a)
    a-=0.5
    l = loss_angle(pts, pts_match, a)
    while l<=l_min:
        l_min = l
        a-=0.5
        l = loss_angle(pts, pts_match, a)
    return a

def params(img_in, img_match="profil.jpg"):
    x1_, y1_, x2_, y2_, pts_match = profil_points(img_match)
    x1, y1, x2, y2, pts = profil_points(img_in)
    a = best_angle(pts, pts_match)
    R = rotation(a)
    rotated_pts = pts @ R
    center_ = np.average(pts_match, axis=0)
    d_ = np.average(np.abs(pts_match-center_))
    center = np.average(rotated_pts, axis=0)
    d = np.average(np.abs(rotated_pts-center))
    s = d_/d
    scaled_pts = s*rotated_pts
    v = np.average(pts_match-scaled_pts, axis=0)
    centered_pts = scaled_pts+v
    #show_points(centered_pts)
    M = s*R
    #show_points( (pts@M + v ) )
    return a, s, v, R, M


def fit_image(image_name, img_match="profil.jpg"):
    a, s, v, R, M = params(image_name, img_match)
    img = cv2.imread(image_name)
    img_ = cv2.imread(img_match)
    rotated_img = ndimage.rotate(img, a, reshape=False)
    shape = rotated_img.shape[:2]
    scaled_shape = (int(s*shape[1]), int(s*shape[0]))
    scaled_img = cv2.resize(rotated_img, dsize=scaled_shape, interpolation=cv2.INTER_CUBIC)
    v = center(img_) - center(scaled_img)#recalculate center
    centered_img = ndimage.shift(scaled_img, shift=(v[1], v[0], 0))
    shape_match = img_.shape
    croped_img = np.zeros(shape_match)
    shape = min(centered_img.shape[0], shape_match[0]), min(centered_img.shape[1], shape_match[1])
    croped_img[:shape[0],:shape[1], :] = centered_img[:shape[0],:shape[1], :]/255

    # cv2.imshow(winname="Face", mat=img)
    # cv2.waitKey(delay=0)
    # cv2.imshow(winname="Face", mat=rotated_img)
    # cv2.waitKey(delay=0)
    # cv2.imshow(winname="Face", mat=scaled_img)
    # cv2.waitKey(delay=0)
    # cv2.imshow(winname="Face", mat=centered_img)
    # cv2.waitKey(delay=0)
    # cv2.destroyAllWindows()
    
    return croped_img


def merge(image_names):
    fitted_imgs = [fit_image(image_name) for image_name in image_names]
    merged_img = np.average( fitted_imgs, axis=0)
    # for fitted_img in fitted_imgs:
    #     cv2.imshow(winname="Face", mat=fitted_img)
    #     cv2.waitKey(delay=0)
    # cv2.imshow(winname="Face", mat=merged_img)
    # cv2.waitKey(delay=0)
    # cv2.destroyAllWindows()
    return merged_img

def merge_images(imagein_names, imageout_name):
    merged_img = merge( imagein_names )
    cv2.imwrite( imageout_name, merged_img*255)



if __name__=="__main__":
    #merge_images( ("aiden.png", "profil.jpg") )
    merge_images( ("aiden.png", "paul.jpg"), "aiden_paul.jpg" )
    merge_images( ("paul.jpg", "aiden.png"), "paul_aiden.jpg" )

