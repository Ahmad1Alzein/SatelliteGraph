"""Image-evidence rules extracted from the frozen study, without study inputs/audits."""
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from .segmentation import parcels as segment_v2
CODES = {"new_building": 1, "extension": 2, "demolished": 3, "tree_loss": 4, "tree_gain": 5, "new_road": 6}
DEFAULT_PARAMS = {
    "PRESENT": 0.5, "ABSENT": 0.2, "absent_window_px": 5,
    "object_fraction_new": 0.6, "extension_min_m2": 40.0, "building_min_m2": 40.0,
    "tree_min_m2": 150.0, "road_min_m2": 100.0, "building_buffer_px": 4,
    "reliable_min_m2": 60.0, "reliable_min_conf": 0.6,
    # added after the visual audit of the first run (see audit/ and README):
    "shadow_gray": 20,              # pixels darker than this carry no evidence (deep shadow)
    "new_building_high_m2": 100.0,  # new area needed for a high-confidence new building
    "extension_high_m2": 100.0,     # new area needed for a high-confidence extension
    "region_high_min_m2": 100.0,    # ...and it must come as ONE connected piece of this size
                                    # (second audit: thin/small extension pieces were mostly false)
}

class EvidenceRules:
    def __init__(self,pixel_area):
        self.pixel_area=pixel_area
        self.params=dict(DEFAULT_PARAMS)
        pixel=np.sqrt(pixel_area)
        self.params["absent_window_px"]=max(1,round(3.75/pixel))
        self.params["building_buffer_px"]=max(1,round(3/pixel))

    def absent(self, prob_max):
        w = self.params['absent_window_px']
        return ndimage.maximum_filter(prob_max, size=w) <= self.params['ABSENT']

    def roof_remap(self, arr, fdy, fdx, nearest=False):
        """Sample a 2013 array at q - d(q): moves 2013 roofs into the 2023 roof frame."""
        h, w = arr.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = (xx - fdx[None, :]).astype(np.float32)
        map_y = (yy - fdy[None, :]).astype(np.float32)
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(arr.astype(np.float32), map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)

    def components(self, mask, min_m2):
        lab, n = ndimage.label(mask, structure=segment_v2.FOUR)
        if n == 0:
            return lab
        sizes = np.bincount(lab.ravel())
        keep = sizes * self.pixel_area >= min_m2
        keep[0] = False
        return np.where(keep[lab], lab, 0)

    def detect(self, ev13, ev23, lab13, n13, lab23, n23, fdy, fdx, img13=None, img23=None):
        """Return change_map (codes), per-node tables for both years, and change regions."""
        P = self.params
        is_b13 = np.isin(lab13, n13.index[n13.object_type == 'building'])
        is_b23 = np.isin(lab23, n23.index[n23.object_type == 'building'])
        b13_2x_al = self.roof_remap(ev13['building_2x'], fdy, fdx)
        b13_max_al = self.roof_remap(ev13['building_max'], fdy, fdx)
        lab13_al = self.roof_remap(lab13, fdy, fdx, nearest=True).astype(np.int64)
        is_b13_al = np.isin(lab13_al, n13.index[n13.object_type == 'building'])
        dark13 = dark13_al = dark23 = np.zeros(lab23.shape, bool)
        if img13 is not None:
            up = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), lab23.shape[::-1], interpolation=cv2.INTER_LINEAR)
            dark13 = up(img13) < P['shadow_gray']
            dark13_al = self.roof_remap(dark13.astype(np.float32), fdy, fdx) > 0.5
            dark23 = up(img23) < P['shadow_gray']
        gain = is_b23 & (ev23['building_2x'] >= P['PRESENT']) & self.absent(b13_max_al) & ~dark13_al
        loss = is_b13_al & (b13_2x_al >= P['PRESENT']) & self.absent(ev23['building_max']) & ~dark23
        gain = ndimage.binary_opening(gain, structure=segment_v2.FOUR)
        loss = ndimage.binary_opening(loss, structure=segment_v2.FOUR)
        change = np.zeros(lab23.shape, np.uint8)
        rows23, rows13 = ([], [])
        for nid in n23.index[n23.object_type == 'building']:
            m = lab23 == nid
            area = m.sum() * self.pixel_area
            g = (gain & m).sum() * self.pixel_area
            status = 'unchanged'
            if g >= P['object_fraction_new'] * area and area >= P['building_min_m2']:
                status = 'new_building'
                change[m] = CODES['new_building']
            elif g >= P['extension_min_m2']:
                status = 'extension'
                change[gain & m] = CODES['extension']
            node = n23.loc[nid]
            reliable = node.area_m2 >= P['reliable_min_m2'] and node.mean_conf >= P['reliable_min_conf']
            need = P['new_building_high_m2'] if status == 'new_building' else P['extension_high_m2']
            confidence = '' if status == 'unchanged' else 'high' if reliable and g >= need else 'low'
            rows23.append({'node_id': int(nid), 'object_type': 'building', 'change': status, 'confidence': confidence, 'new_area_m2': round(g, 1), 'area_m2': round(area, 1), 'p2013_building_max': round(float(b13_max_al[m].mean()), 3), 'p2023_building': round(float(ev23['building_2x'][m].mean()), 3), 'reliability': 'high' if reliable else 'low'})
        for nid in n13.index[n13.object_type == 'building']:
            m = lab13_al == nid
            if not m.any():
                continue
            area = m.sum() * self.pixel_area
            l = (loss & m).sum() * self.pixel_area
            status = 'unchanged'
            if l >= P['object_fraction_new'] * area and area >= P['building_min_m2']:
                status = 'demolished'
                change[m & (change == 0)] = CODES['demolished']
            node = n13.loc[nid]
            reliable = node.area_m2 >= P['reliable_min_m2'] and node.mean_conf >= P['reliable_min_conf']
            rows13.append({'node_id': int(nid), 'object_type': 'building', 'change': status, 'confidence': 'unverified' if status != 'unchanged' else '', 'lost_area_m2': round(l, 1), 'area_m2': round(area, 1), 'p2013_building': round(float(b13_2x_al[m].mean()), 3), 'p2023_building_max': round(float(ev23['building_max'][m].mean()), 3), 'reliability': 'high' if reliable else 'low'})
        near_b = ndimage.binary_dilation(is_b13 | is_b23 | is_b13_al, iterations=P['building_buffer_px']) | dark23 | dark13
        smooth = lambda a: ndimage.uniform_filter(a, size=self.params["absent_window_px"])
        t13, t23 = (smooth(ev13['tree_2x']), smooth(ev23['tree_2x']))
        tree_loss = self.components((t13 >= 0.6) & self.absent(ev23['tree_max']) & ~near_b, P['tree_min_m2'])
        tree_gain = self.components((t23 >= 0.6) & self.absent(ev13['tree_max']) & ~near_b, P['tree_min_m2'])
        r13, r23 = (smooth(ev13['road_2x']), smooth(ev23['road_2x']))
        new_road = self.components((r23 >= 0.6) & self.absent(ev13['road_max']) & ~is_b23, P['road_min_m2'])
        for code, lab in ((CODES['tree_loss'], tree_loss), (CODES['tree_gain'], tree_gain), (CODES['new_road'], new_road)):
            change[(lab > 0) & (change == 0)] = code
        return (change, pd.DataFrame(rows23), pd.DataFrame(rows13))
