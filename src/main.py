from globals import *
import matplotlib.image as mpimg
from camera import Camera
# from preprocess_helpers import *
from image_processor import *
import numpy as np
import os
from moviepy.editor import VideoFileClip
from line_history import LineHistory


# configure settings here
CFG['history_length'] = 6  # in frame count
CFG['ym_per_pix'] = 30./720  # meters per pixel in y dimension
CFG['xm_per_pix'] = 3.7/700  # meters per pixel in x dimension
CFG['camera_calib_file'] = '../camera_calib.json'
CFG['calibration_image_path'] = '../input/camera_cal/calibration*.jpg'
CFG['road_images'] = '../input/road_images/'
CFG['calibration_test_target_dir'] = '../output/image_test_undistortion/'
CFG['perspective_calib_img'] = '../input/road_images/straight_lines2.jpg'
CFG['perspective_calib_pts'] = np.float32([[217, 719], [593, 450], [689, 450], [1112, 719]])
CFG['videos_dir'] = '../input/videos/'
# out dirs
CFG['road_images_out_dir'] =  '../output/road_images/'
CFG['videos_out_dir'] = '../output/videos/'


# because it is not possible to pass data to callbacks, ugly globals are needed
GLOBAL['camera'] = None
GLOBAL['frame_cnt'] = 0


def process_still_images():
    if not os.path.exists(CFG['road_images_out_dir']):
        os.mkdir(CFG['road_images_out_dir'])
    for image_name in os.listdir(CFG['road_images']):
        path_name = CFG['road_images'] + image_name
        print('Processing image:', path_name)
        source = mpimg.imread(path_name)
        #ImageProcessor.reset()
        processed = ImageProcessor.do(source, show_dbg=False)
        mpimg.imsave(CFG['road_images_out_dir'] + image_name, processed)


def process_video(video_path):
    print('Processing video:', video_path, '...')
    if not os.path.exists(CFG['videos_out_dir']):
        os.mkdir(CFG['videos_out_dir'])

    basename = os.path.basename(video_path)
    outpath = CFG['videos_out_dir'] + basename
    clip = VideoFileClip(video_path, audio=False).subclip(38, 43)  ################################ TODO: use full videos
    #clip = VideoFileClip(video_path, audio=False)

    # Reinit for each video of possible different image sizes
    histories = (LineHistory('left'), LineHistory('right'))
    ImageProcessor.init((clip.w, clip.h), GLOBAL['camera'], histories, show_dbg=True)
    result_clip = clip.fl_image(ImageProcessor.do)
    result_clip.write_videofile(outpath)
    result_clip.close()
    clip.close()
    print('Saved to:', outpath)


def main():
    GLOBAL['camera'] = Camera(CFG['calibration_image_path'],
                              pattern_size=(9, 6),
                              save_path=CFG['camera_calib_file'],
                              persp_path=CFG['perspective_calib_img'],
                              persp_ref_points=CFG['perspective_calib_pts'],
                              show_dbg=True,
                              )
    # Uncomment this to generate camera calibration test images:
    #GLOBAL['camera'].generate_test_images(CFG['calibration_image_path'], CFG['calibration_test_target_dir'])

    img_source = mpimg.imread('../input/road_images/test4.jpg')
    ImageProcessor.init((img_source.shape[1], img_source.shape[0]), GLOBAL['camera'], show_dbg=True)

    #process_still_images()
    process_video(CFG['videos_dir'] + 'project_video.mp4')
    #process_video(CFG['videos_dir'] + 'challenge_video.mp4')
    #process_video(CFG['videos_dir'] + 'harder_challenge_video.mp4')


if __name__ == '__main__':
    main()
