import cv2
import pytesseract
import numpy as np

# Tesseract yolunu buraya tekrar yaz (Senin bilgisayarındaki yol)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def check_system():
    try:
        # Boş bir siyah resim oluştur
        img = np.zeros((100, 300), dtype=np.uint8)
        # Üzerine beyaz bir yazı yaz
        cv2.putText(img, "TEST", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255), 3)
        
        # Tesseract'a bu resmi okut
        text = pytesseract.image_to_string(img).strip()
        
        if "TEST" in text:
            print("✅ Başarılı: Tesseract ve OpenCV el sıkıştı!")
        else:
            print(f"⚠️ Uyarı: Tesseract çalışıyor ama yazıyı tam okuyamadı. Okunan: '{text}'")
    except Exception as e:
        print(f"❌ Hata: Sistemde bir şeyler eksik -> {e}")

check_system()