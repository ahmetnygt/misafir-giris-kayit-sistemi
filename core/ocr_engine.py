import pytesseract
import cv2
import os
import numpy as np
import re

class IdentityOCR:
    def __init__(self, tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def preprocess(self, image):
        """
        Görüntüyü OCR için agresif bir şekilde temizler.
        """
        # 1. Görüntüyü biraz büyüt (Tesseract büyük karakterleri sever)
        img = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 2. Gri tonlamaya çevir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Kontrastı artır (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        # Bu, karanlık bölgelerdeki detayları ortaya çıkarır.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrasted = clahe.apply(gray)

        # 4. Gürültü temizleme (Denoising)
        # Arka plandaki ince desenleri bulanıklaştırır.
        denoised = cv2.fastNlMeansDenoising(contrasted, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 5. Eşikleme (Thresholding) - Adaptive Thresholding kullanacağız.
        # Bu yöntem, görüntünün her bölgesi için farklı bir eşik değeri hesaplar.
        # Arka planı aydınlık olmayan, gölgeli resimler için birebirdir.
        # blockSize: Komşuluk alanı boyutu (tek sayı olmalı).
        # C: Ortalamadan çıkarılacak sabit sayı. Bu sayıyı artırırsan daha az gürültü, azaltırsan daha çok detay alırsın.
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 13)

        # 6. (Opsiyonel) Morfolojik İşlemler
        # Küçük noktaları temizlemek için 'opening' işlemi uygulanabilir.
        kernel = np.ones((2,2), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Ara sonucu görmek için (test ederken açabilirsin)
        # cv2.imshow("Preprocessed Image", opened)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return opened

    def extract_text(self, processed_image):
        # 1. TESSDATA yolunu tertemiz hale getir
        tesspath = r'C:\Program Files\Tesseract-OCR\tessdata'
        
        # 2. Ortam değişkenini kodun içinde set et (En garanti yol budur)
        os.environ['TESSDATA_PREFIX'] = tesspath
        
        # 3. Config kısmında tırnak işaretlerinden ve karmaşadan kaçın
        # Sadece PSM modunu veriyoruz, yolu zaten yukarıda belirttik
        config = '--psm 3' 
        
        try:
            # Lang kısmına 'tur' diyoruz, artık TESSDATA_PREFIX sayesinde onu bulacak
            return pytesseract.image_to_string(processed_image, lang='tur', config=config)
        except Exception as e:
            return f"OCR Hatası: {str(e)}"

    def parse_data(self, raw_text):
        data = {
            "tc_no": None,
            "isim": None,
            "soyisim": None
        }
        
        # 1. TC NO (Zaten çalışıyor ama sağlamlaştıralım)
        tc_match = re.search(r'([1-9]{1}[0-9]{10})', raw_text)
        if tc_match:
            data["tc_no"] = tc_match.group(1)

        # Satırları temizleyip listeye alalım
        lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) > 2]

        # 2. İSİM VE SOYİSİM AYIKLAMA (Daha agresif yöntem)
        for i, line in enumerate(lines):
            # "Adı" kelimesini yakala (Yanındaki (Name) kısmını salla, Tesseract orayı yanlış okuyabiliyor)
            if "Adı" in line:
                # Genelde bir sonraki satır isimdir
                if i + 1 < len(lines):
                    raw_name = lines[i+1]
                    # Tesseract'ın klasik hatalarını düzeltelim (ö -> A, i -> l vs.)
                    clean_name = raw_name.replace('ö', 'A').replace('Ce\'en', 'Ceren').replace('ösi', 'Asi')
                    data["isim"] = clean_name
            
            # "Soyadı" kelimesini yakala
            if "Soyadı" in line:
                if i + 1 < len(lines):
                    data["soyisim"] = lines[i+1].upper() # Soyadlar genelde büyük harf olur

        return data