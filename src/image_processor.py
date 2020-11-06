from preprocess_helpers import *
from line import Line
from line_history import LineHistory
from globals import *


class ImageProcessor:
    """Singleton-like. All methods are static. Do not instantiate."""
    camera = None
    frame_count = 0
    txt = ""
    left_history = None
    right_history = None

    @classmethod
    def init(cls, size, camera, line_histories = None, show_dbg=False):
        cls.camera = camera
        cls.frame_count = 0
        cls.txt1 = ""
        if line_histories:
            cls.left_history = line_histories[0]
            cls.right_history = line_histories[1]

    @classmethod
    def reset(cls):
        cls.frame_count = 0
        #cls.mask = cls.default_mask

    @classmethod
    def has_history(cls):
        return cls.left_history and cls.right_history

    @classmethod
    def sanity_checks(cls, left_line, right_line, left_fit_px, right_fit_px):
        ok_parallel, deviations = left_line.is_sane_parallel(right_line)
        ok_distance, min, max = left_line.is_sane_other_dist(right_line)
        disp_results = np.array([ok_parallel, ok_distance])
        results = np.array([ok_parallel, ok_distance])
        strs = [
            'Parallel {} {}'.format(
                'OK' if ok_parallel else 'INSANE',
                'deviation between min and max: {:6d}px'.format(int(deviations))
            ),
            'Distance {} {}'.format(
                'OK' if ok_distance else 'INSANE',
                'min:{:6d}px max:{:6d}px'.format(int(min), int(max))
            ),
        ]
        ok_laverage, ok_raverage = 1, 1
        if cls.has_history():
            ok_laverage, lavg, ldiffs = cls.left_history.is_sane_avg(left_fit_px)
            ok_raverage, ravg, rdiffs = cls.right_history.is_sane_avg(right_fit_px)
            disp_results = np.hstack((disp_results, [ok_laverage, ok_raverage]))
            results = np.hstack((results, [ok_laverage and ok_raverage]))
            strs += [
                'Left history {} {}'.format(
                    'OK' if ok_laverage else 'INSANE',
                    'coeff diffs: {:8.7f}, {:5.3f}, {:5.1f}'.format(ldiffs[0], ldiffs[1], ldiffs[2])
                ),
                'Right history {} {}'.format(
                    'OK' if ok_raverage else 'INSANE',
                    'coeff diffs: {:8.7f}, {:5.3f}, {:5.1f}'.format(rdiffs[0], rdiffs[1], rdiffs[2])
                ),
            ]
        ok_count = disp_results.sum()
        display_res_merged = np.vstack((disp_results, strs)).T

        return results, len(disp_results), display_res_merged

    @classmethod
    def sanity_to_img(cls, img, details):
        x, y, dy = 0, 75, 25
        colors = [(255, 128, 128), (128, 255, 128)]
        for is_ok, txt in details:
            cv2.putText(img, txt, (x, y), cv2.QT_FONT_NORMAL, 1, colors[int(is_ok)])
            y += dy



    @classmethod
    def do(cls, img, show_dbg=False):
        verbose = True
        # orig perspective
        img_undistorted = cls.camera.undistort(img)
        img_extd = extend(img_undistorted)
        bin_undistorted = combined_threshold_4(img_extd, show_dbg and False)

        # warp to 2D
        bin_2d = warp(bin_undistorted, cls.camera.perspective_matrix, show_dbg and False)
        if (  # prior polys can be used
            cls.left_history and cls.right_history
            and cls.left_history.history[0] and cls.left_history.history[0].quality == 5
            and cls.right_history.history[0] and cls.right_history.history[0].quality == 5
        ):
            left_fit, right_fit = cls.left_history.history[0].coeffs, cls.right_history.history[0].coeffs
            leftx, lefty, rightx, righty, img_win = find_lane_pixels_around_poly(bin_2d, left_fit, right_fit, show_dbg and False)
        else:
            leftx, lefty, rightx, righty, img_win = find_lane_pixels_sliding_window(bin_2d, show_dbg and False)

        left_fit_px, right_fit_px, left_fit_m, right_fit_m = fit_polynomial(leftx, lefty, rightx, righty, img_win, show_dbg and False)

        # use these for road-space and screen-space overlays
        img_overlay_for_unwarp = np.zeros_like(img_extd)  # this will be warped back to the road surface
        img_overlay_screen = np.zeros_like(img)      # this will stay in screen-space

        # texts
        left_r_m, right_r_m, center_dist_m = measure_radius_m(left_fit_m, right_fit_m, img_extd.shape)
        if cls.txt1 == None or cls.frame_count % 1 == 0:
            cls.txt1 = "Center Offset:{:3d}cm, Radius:{:5d}m, {:5d}m".format(int(center_dist_m*100), int(left_r_m), int(right_r_m))
        cv2.putText(img_overlay_screen, cls.txt1, (0, 25), cv2.QT_FONT_NORMAL, 1, color=(255, 255, 255))

        # sanity checks
        lline = Line(cls.frame_count, 'left', img.shape[0], left_fit_px, 1)
        rline = Line(cls.frame_count, 'right', img.shape[0], right_fit_px, 1)
        sanity_res, check_total_count, sanity_details = cls.sanity_checks(lline, rline, left_fit_px, right_fit_px)
        sanity_count = np.count_nonzero(sanity_res)
        sanity_max = len(sanity_res)

        if verbose:
            cls.sanity_to_img(img_overlay_screen, sanity_details)
            cv2.putText(img_overlay_screen, 'Frame: {:d}'.format(cls.frame_count), (0, 50), cv2.QT_FONT_NORMAL, 1, color=(255, 255, 255))

        # apply prior info
        enable_history = True  ######################################################################################### enable / disable history here
        has_left_line, has_right_line = True, True
        if enable_history and cls.left_history and cls.right_history:
            if sanity_res[0] and sanity_res[1]:  # parallel and dist ok
                lline.quality, rline.quality = 5, 5
            else:
                lline.quality, rline.quality = 2, 2

            has_left_line, left_fit_px = cls.left_history.update(lline, 4)  # use avg and update
            has_right_line, right_fit_px = cls.right_history.update(rline, 4)

        # display lane polys
        if has_left_line and has_right_line:
            draw_polys_inplace(left_fit_px, right_fit_px, img_overlay_for_unwarp, show_dbg and True)
        elif verbose:
            cv2.putText(img_overlay_screen, 'Lines: missing', (830, 25), cv2.QT_FONT_NORMAL, 1, color=(255, 255, 255))

        # render the overlays
        img_persp_overlay = warp(img_overlay_for_unwarp, cls.camera.perspective_inv_matrix, show_dbg and False)
        img_persp_overlay = crop_ref(img_persp_overlay)
        combined = cv2.addWeighted(img, 1.0, img_persp_overlay, 1.0, 0.)

        if verbose:
            # add raw pixel data to screen overlay
            dst_size = (img_overlay_screen.shape[1], img_overlay_screen.shape[0] // 3)
            img_win_shrinked = cv2.resize(img_win, dst_size)
            height = 180
            upper_padding = np.zeros((height, dst_size[0], 3), dtype=np.uint8)
            lower_padding = np.zeros((dst_size[1] * 2 - height , dst_size[0], 3), dtype=np.uint8)
            img_win_shrinked = np.vstack((upper_padding, img_win_shrinked, lower_padding))
            combined = cv2.addWeighted(img_win_shrinked, 1.0, combined, 0.6, 0.)

        combined = cv2.addWeighted(combined, 1.0, img_overlay_screen, 1.0, 0.)

        cls.frame_count += 1
        return combined

    @classmethod
    def show(cls, img, title=None):
        colormap = None if len(img.shape) > 2 and img.shape[2] > 1 else 'gray'
        if title:
            plt.title(title)
        plt.imshow(img, cmap=colormap)
        plt.show()
