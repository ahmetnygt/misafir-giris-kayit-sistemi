import re

def test_parser(raw_text):
    # Basit bir TC no yakalama regex'i
    tc_match = re.search(r'\b[1-9]{1}[0-9]{10}\b', raw_text)
    tc_no = tc_match.group(0) if tc_match else "Bulunamadı"
    
    # İsim çekme (Bu kısım kimlik formatına göre çok değişir, şimdilik basit tut)
    # Genelde TC'den sonraki satırlarda olur.
    print(f"Test Sonucu -> Bulunan TC: {tc_no}")

# Sanki Tesseract okumuş gibi kirli bir metin veriyoruz
kirli_metin = "TÜRKİYE CUMHURİYETİ KİMLİK KARTI \n TC NO: 12345678901 \n SOYADI: YILMAZ \n ADI: AHMET"
test_parser(kirli_metin)