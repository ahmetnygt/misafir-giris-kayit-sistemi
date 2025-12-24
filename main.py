import cv2
import pytesseract
import os
import numpy as np
import re

# ==========================================
# AYARLAR
# ==========================================
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSDATA_PATH = r'C:\Program Files\Tesseract-OCR\tessdata'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH

class IdentityScanner:
    def __init__(self):
        self.init_tracker()
        self.is_tracking = False
        self.tracked_bbox = None
        self.current_approx = None 

    def init_tracker(self):
        try:
            self.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT.create()

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def get_warped_card(self, frame, approx):
        if approx is None:
            x, y, w, h = self.tracked_bbox
            pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
        else:
            pts = approx.reshape(4, 2).astype("float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        if maxHeight > maxWidth:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped

    def find_best_contour(self, frame, roi_bbox=None):
        if roi_bbox:
            x, y, w, h = [int(v) for v in roi_bbox]
            x1, y1 = max(0, x-20), max(0, y-20)
            x2, y2 = min(frame.shape[1], x+w+20), min(frame.shape[0], y+h+20)
            work_img = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_img = frame
            offset = (0, 0)
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_approx = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30000: continue 
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 1.3 < aspect_ratio < 1.9:
                    if area > max_area:
                        max_area = area
                        best_approx = approx + offset
        return best_approx

    def preprocess_region(self, region):
        """Harfleri koruyan ve gürültüleri silen jilet gibi temizlik."""
        # Tesseract büyük harf sever, x3 büyütüyoruz
        region = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Keskinliği koruyan gürültü silme
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Harfleri yemeyen Threshold (C=12 yaptık)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 13
        )
        
        # Kritik: Erode işlemini sildim, harfler artık kopmayacak!
        # Etrafına 20px beyaz boşluk ekleyerek Tesseract'ı rahatlatıyoruz
        thresh = cv2.copyMakeBorder(thresh, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        return thresh

    def scan_roi(self, card_img):
        h, w = card_img.shape[:2]
        # KOORDİNATLAR GÜNCELLENDİ (Etiket çizgilerini eledik)
        roi_map = {
            "tc_no":   [int(h*0.78), int(h*0.86), int(w*0.38), int(w*0.82)],
            "isim":    [int(h*0.58), int(h*0.67), int(w*0.38), int(w*0.95)],
            "soyisim": [int(h*0.68), int(h*0.77), int(w*0.38), int(w*0.95)]
            }
        
        results = {}
        for key, coords in roi_map.items():
            y1, y2, x1, x2 = coords
            region = card_img[y1:y2, x1:x2]
            if region.size == 0: continue
            
            processed = self.preprocess_region(region)
            cv2.imshow(f"KESILEN_{key}", processed) 

            # Tırnak hatası vermeyen temiz config
            conf = '--psm 7 --oem 3'
            
            raw_text = pytesseract.image_to_string(processed, lang='tur', config=conf).strip()
            print(f"DEBUG [{key}] Ham Metin: '{raw_text}'")

            if key == "tc_no":
                results[key] = "".join(re.findall(r'\d+', raw_text))
            else:
                # Sadece Türkçe harfler ve boşlukları al
                results[key] = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', raw_text).strip()
            
        cv2.waitKey(1)
        return results

def main():
    scanner = IdentityScanner()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break
        display_frame = frame.copy()
        key = cv2.waitKey(1)

        if key == ord('r') or key == ord('R'):
            scanner.is_tracking = False
            scanner.init_tracker()
            print("🔄 Resetlendi.")

        if scanner.is_tracking:
            success, bbox = scanner.tracker.update(frame)
            if success:
                scanner.tracked_bbox = bbox
                approx = scanner.find_best_contour(frame, bbox)
                if approx is not None:
                    scanner.current_approx = approx
                    cv2.drawContours(display_frame, [approx], -1, (0, 255, 0), 3)
                else:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                scanner.is_tracking = False
                scanner.init_tracker()

        else:
            approx = scanner.find_best_contour(frame)
            if approx is not None:
                x, y, w, h = cv2.boundingRect(approx)
                scanner.tracker.init(frame, (x, y, w, h))
                scanner.is_tracking = True
                scanner.current_approx = approx

        cv2.imshow("Smart ID Scanner", display_frame)

        if key == ord('s'):
            if scanner.is_tracking:
                warped = scanner.get_warped_card(frame, scanner.current_approx)
                data = scanner.scan_roi(warped)
                print("\n" + "="*30)
                print(f"TC NO   : {data.get('tc_no')}")
                print(f"İSİM    : {data.get('isim')}")
                print(f"SOYİSİM : {data.get('soyisim')}")
                print("="*30)
            else: print("⚠️ Kart kilitlenmedi!")
        elif key == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()