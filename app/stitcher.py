import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict


class ProgressiveStitcher:
    def __init__(
        self,
        feature: str = "ORB",
        n_features: int = 3000,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 4.0,
        resize: float = 1.0,
        feather: bool = True,
    ) -> None:
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.resize = resize
        self.feather = feather

        # Feature detector/descriptor and matcher
        if feature.upper() == "ORB":
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Fallback to AKAZE if requested feature is unsupported
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Panorama buffers
        self.panorama: Optional[np.ndarray] = None  # float32 [0..1]
        self.mask: Optional[np.ndarray] = None      # uint8 0/255
        self.offset: Tuple[float, float] = (0.0, 0.0)  # world->canvas translation

        # Keyframes store
        self.keyframes: List[Dict] = []

    @staticmethod
    def _ensure_color(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _prep(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.resize != 1.0:
            img = cv2.resize(img, None, fx=self.resize, fy=self.resize, interpolation=cv2.INTER_AREA)
        img = self._ensure_color(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    def _detect(self, gray: np.ndarray):
        kps, desc = self.detector.detectAndCompute(gray, None)
        return kps, desc

    def _match(self, d1, d2):
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            return []
        matches = self.matcher.knnMatch(d1, d2, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        return good

    @staticmethod
    def _homography_from_matches(kps_src, kps_dst, matches, ransac_thresh):
        if len(matches) < 6:
            return None, None
        src_pts = np.float32([kps_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        return H, inliers

    @staticmethod
    def _project_corners(H: np.ndarray, w: int, h: int) -> np.ndarray:
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        return proj

    def _world_bounds_from_canvas(self) -> Tuple[float, float, float, float]:
        if self.panorama is None:
            return 0.0, 0.0, 0.0, 0.0
        Hc, Wc = self.panorama.shape[:2]
        ox, oy = self.offset
        return -ox, -oy, Wc - ox, Hc - oy

    def _ensure_canvas(self, world_corners: np.ndarray) -> None:
        # Expand canvas if needed to fit new world corners
        if self.panorama is None:
            mins = world_corners.min(axis=0)
            maxs = world_corners.max(axis=0)
            minx, miny = mins
            maxx, maxy = maxs
            # Offset so that min corner is at least 0,0
            ox = -min(0.0, minx)
            oy = -min(0.0, miny)
            W = int(np.ceil(maxx + ox) + 4)
            H = int(np.ceil(maxy + oy) + 4)
            self.panorama = np.zeros((H, W, 3), dtype=np.float32)
            self.mask = np.zeros((H, W), dtype=np.uint8)
            self.offset = (ox, oy)
            return

        minx0, miny0, maxx0, maxy0 = self._world_bounds_from_canvas()
        minx1, miny1 = world_corners.min(axis=0)
        maxx1, maxy1 = world_corners.max(axis=0)

        new_minx = min(minx0, minx1)
        new_miny = min(miny0, miny1)
        new_maxx = max(maxx0, maxx1)
        new_maxy = max(maxy0, maxy1)

        # If bounds unchanged, keep canvas
        if (new_minx == minx0 and new_miny == miny0 and new_maxx == maxx0 and new_maxy == maxy0):
            return

        old_pan = self.panorama
        old_mask = self.mask
        old_ox, old_oy = self.offset

        new_ox = -min(0.0, new_minx)
        new_oy = -min(0.0, new_miny)
        new_W = int(np.ceil(new_maxx + new_ox) + 4)
        new_H = int(np.ceil(new_maxy + new_oy) + 4)

        new_pan = np.zeros((new_H, new_W, 3), dtype=np.float32)
        new_msk = np.zeros((new_H, new_W), dtype=np.uint8)

        # Copy old canvas into new with translation shift
        shift_x = int(round(new_ox - old_ox))
        shift_y = int(round(new_oy - old_oy))

        y1 = max(0, shift_y)
        x1 = max(0, shift_x)
        y2 = min(new_H, shift_y + old_pan.shape[0])
        x2 = min(new_W, shift_x + old_pan.shape[1])

        oy1 = max(0, -shift_y)
        ox1 = max(0, -shift_x)
        oy2 = oy1 + (y2 - y1)
        ox2 = ox1 + (x2 - x1)

        new_pan[y1:y2, x1:x2] = old_pan[oy1:oy2, ox1:ox2]
        new_msk[y1:y2, x1:x2] = old_mask[oy1:oy2, ox1:ox2]

        self.panorama = new_pan
        self.mask = new_msk
        self.offset = (new_ox, new_oy)

    def _blend_into(self, warped: np.ndarray, warped_mask: np.ndarray) -> None:
        if self.panorama is None:
            return
        pano = self.panorama
        mask_old = self.mask

        if not self.feather:
            # Simple overwrite with alpha=1 for new pixels, average on overlaps
            overlap = (mask_old > 0) & (warped_mask > 0)
            only_new = (mask_old == 0) & (warped_mask > 0)
            pano[only_new] = warped[only_new]
            pano[overlap] = 0.5 * pano[overlap] + 0.5 * warped[overlap]
            mask_old[warped_mask > 0] = 255
            return

        # Feather blending via distance transform weights
        m_old = (mask_old > 0).astype(np.uint8) * 255
        m_new = (warped_mask > 0).astype(np.uint8) * 255

        if np.count_nonzero(m_old) == 0:
            pano[m_new > 0] = warped[m_new > 0]
            mask_old[m_new > 0] = 255
            return

        # Distance transforms
        d_old = cv2.distanceTransform(m_old, cv2.DIST_L2, 3)
        d_new = cv2.distanceTransform(m_new, cv2.DIST_L2, 3)

        eps = 1e-6
        w_new = d_new / (d_new + d_old + eps)
        w_old = 1.0 - w_new

        # Only compute where either mask is present
        region = (m_old > 0) | (m_new > 0)
        w_new3 = np.dstack([w_new, w_new, w_new])
        w_old3 = np.dstack([w_old, w_old, w_old])

        pano[region] = w_old3[region] * pano[region] + w_new3[region] * warped[region]
        mask_old[m_new > 0] = 255

    def add_image(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], bool, Optional[str]]:
        img, gray = self._prep(img_bgr)
        h, w = img.shape[:2]

        if len(self.keyframes) == 0:
            # Initialize panorama and first keyframe
            H_world = np.eye(3, dtype=np.float64)
            world_corners = self._project_corners(H_world, w, h)
            self._ensure_canvas(world_corners)

            ox, oy = self.offset
            T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)
            warped = cv2.warpPerspective(img.astype(np.float32) / 255.0, T @ H_world, (self.panorama.shape[1], self.panorama.shape[0]))
            warped_mask = (cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, T @ H_world, (self.panorama.shape[1], self.panorama.shape[0])) > 0).astype(np.uint8) * 255

            self.panorama = warped.copy()
            self.mask = warped_mask.copy()

            kps, desc = self._detect(gray)
            self.keyframes.append({
                "img": img,
                "gray": gray,
                "kps": kps,
                "desc": desc,
                "H_world": H_world,
            })
            return (np.clip(self.panorama * 255.0, 0, 255).astype(np.uint8), True, None)

        # Match against last keyframe (simple and fast)
        ref = self.keyframes[-1]
        kps_new, desc_new = self._detect(gray)

        matches = self._match(desc_new, ref["desc"])  # new -> ref
        if len(matches) < 6:
            return (np.clip(self.panorama * 255.0, 0, 255).astype(np.uint8), False, "Not enough matches")

        H_new_to_ref, inliers = self._homography_from_matches(kps_new, ref["kps"], matches, self.ransac_reproj_thresh)
        if H_new_to_ref is None:
            return (np.clip(self.panorama * 255.0, 0, 255).astype(np.uint8), False, "Homography failed")

        # Compose to world frame: H_new_world maps new image -> world
        H_new_world = ref["H_world"] @ H_new_to_ref

        # Expand canvas if needed
        world_corners = self._project_corners(H_new_world, w, h)
        self._ensure_canvas(world_corners)

        # Warp into canvas (world->canvas via offset T)
        ox, oy = self.offset
        T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)
        H_canvas = T @ H_new_world

        warped = cv2.warpPerspective(img.astype(np.float32) / 255.0, H_canvas, (self.panorama.shape[1], self.panorama.shape[0]))
        warped_mask = (cv2.warpPerspective(np.ones((h, w), dtype=np.uint8) * 255, H_canvas, (self.panorama.shape[1], self.panorama.shape[0])) > 0).astype(np.uint8) * 255

        # Blend
        self._blend_into(warped, warped_mask)

        # Optionally promote to keyframe by motion/coverage heuristic
        self.keyframes.append({
            "img": img,
            "gray": gray,
            "kps": kps_new,
            "desc": desc_new,
            "H_world": H_new_world,
        })

        return (np.clip(self.panorama * 255.0, 0, 255).astype(np.uint8), True, None)

    def get_panorama(self) -> Optional[np.ndarray]:
        if self.panorama is None:
            return None
        return np.clip(self.panorama * 255.0, 0, 255).astype(np.uint8)
