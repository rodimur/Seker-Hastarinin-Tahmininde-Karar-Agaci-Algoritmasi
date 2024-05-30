# -- coding: utf-8 --
"""
Created on Wed May 29 17:49:55 2024

@author: tunah
"""

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import graphviz
import matplotlib.pyplot as plt

# Veriyi yükle ve işleme
veriler = pd.read_csv('diabetes.csv')

# Korelasyon matrisini hesapla
korelasyon_matrisi = veriler.corr()
# Tüm korelasyon matrisini göster
print(korelasyon_matrisi)

# NaN olan değerleri sütunun ortalaması ile doldur
veriler[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = veriler[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
veriler[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = imputer.fit_transform(veriler[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]])

x = veriler.iloc[:, :-1]
y = veriler.iloc[:, -1]

# Veriyi eğitim ve test setlerine ayır
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Veriyi standardize etme
sc = StandardScaler()
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)

# Karar Ağacı Modeli
agac = DecisionTreeClassifier(random_state=0)
agac.fit(x_train, y_train)


# Karar ağacını görselleştir ve PDF olarak kaydet
dot_data = export_graphviz(agac, out_file=None, feature_names=veriler.columns[:-1], class_names=["0", "1"], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("karar_agaci")
graph.view()

# RandomForest Modeli
model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10, criterion='gini')
model.fit(x_train, y_train)

# GUI Kısmı
def kullanici_girdisi_al():
    try:
        kullanici_girdisi = [float(girdi.get()) for girdi in girisler]
        kullanici_girdisi = np.array(kullanici_girdisi).reshape(1, -1)
        kullanici_girdisi = imputer.transform(kullanici_girdisi)  # Eksik değerleri doldur
        kullanici_girdisi = sc.transform(kullanici_girdisi)  # Standartlaştır
        tahmin = model.predict(kullanici_girdisi)
        sonuc = "Şeker hastası olabilirsiniz." if tahmin[0] == 1 else "Şeker hastası değilsiniz."
        messagebox.showinfo("Sonuç", sonuc)
    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm alanlara geçerli sayılar giriniz.")


def karmaşıklık_matrisi_oranlarını_hesapla(karmaşıklık_matrisi):
    sınıf_sayısı = karmaşıklık_matrisi.shape[0]
    sınıf_oranları = np.zeros_like(karmaşıklık_matrisi, dtype=float)
    for i in range(sınıf_sayısı):
        toplam_örnekler = np.sum(karmaşıklık_matrisi[i])
        if toplam_örnekler > 0:
            sınıf_oranları[i] = karmaşıklık_matrisi[i] / toplam_örnekler
    return sınıf_oranları

# Model değerlendirme
agac_tahmini = agac.predict(x_test)
rf_tahmini = model.predict(x_test)
agac_doğruluk = accuracy_score(y_test, agac_tahmini)
rf_doğruluk = accuracy_score(y_test, rf_tahmini)

print(f"Karar Ağacı Doğruluk: {agac_doğruluk}")
print(f"Random Forest Doğruluk: {rf_doğruluk}")

# Karar Ağacı için karmaşıklık matrisi
agac_km = confusion_matrix(y_test, agac_tahmini)
print("Karar Ağacı Karmaşıklık Matrisi:")
print(agac_km)

# Random Forest için karmaşıklık matrisi
rf_km = confusion_matrix(y_test, rf_tahmini)
print("Random Forest Karmaşıklık Matrisi:")
print(rf_km)

# Karar Ağacı için komşuluk matrisi
agac_sınıf_oranları = karmaşıklık_matrisi_oranlarını_hesapla(agac_km)
print("Karar Ağacı Komşuluk Matrisi:")
print(agac_sınıf_oranları)

# Random Forest için komşuluk matrisi
rf_sınıf_oranları = karmaşıklık_matrisi_oranlarını_hesapla(rf_km)
print("Random Forest Komşuluk Matrisi:")
print(rf_sınıf_oranları)

# Ana pencereyi oluştur
pencere = tk.Tk()
pencere.title("Şeker Hastalığı Kontrolü")

# Kullanıcıdan 8 veri girişi için giriş kutuları oluştur
etiketler = ["Hamilelik", "Glikoz", "Kan Basıncı", "Cilt Kalınlığı", "Insulin", "Vücut Kitle Endeksi", "Diyabet Soy Geçmişi", "Yaş"]
girisler = []

for i, etiket in enumerate(etiketler):
    tk.Label(pencere, text=f"{etiket}:").grid(row=i, column=0)
    girdi = tk.Entry(pencere)
    girdi.grid(row=i, column=1)
    girisler.append(girdi)

# Kontrol Et butonu oluştur
tk.Button(pencere, text="Kontrol Et", command=kullanici_girdisi_al).grid(row=len(etiketler), column=0, columnspan=2)

# Pencereyi çalıştır
pencere.mainloop()

# Doğruluk Grafiği
modeller = ['Karar Ağacı', 'Random Forest']
doğruluklar = [agac_doğruluk, rf_doğruluk]

plt.figure(figsize=(10, 5))
plt.bar(modeller, doğruluklar, color=['blue', 'green'])
plt.xlabel('Modeller')
plt.ylabel('Doğruluk Oranı')
plt.title('Karar Ağacı ve Random Forest Doğruluk Karşılaştırması')
plt.ylim([0, 1])
plt.show()

# Korelasyon grafiği
plt.figure(figsize=(10, 8))
plt.imshow(korelasyon_matrisi, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.title('Özellikler Arasındaki Korelasyon Matrisi')
plt.xticks(range(len(korelasyon_matrisi)), korelasyon_matrisi.columns, rotation=90)
plt.yticks(range(len(korelasyon_matrisi)), korelasyon_matrisi.columns)
plt.show()


