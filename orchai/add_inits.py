"""
Klasör yapısında eksik __init__.py dosyalarını otomatik oluşturur.
"""
import os

root = os.path.dirname(os.path.abspath(__file__))

for dirpath, dirnames, filenames in os.walk(root):
    # __pycache__ ve .git klasörlerini atla
    if "__pycache__" in dirpath or ".git" in dirpath:
        continue
    if "app" not in dirpath and dirpath != root:
        continue

    init_file = os.path.join(dirpath, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")
        print(f"Oluşturuldu: {init_file}")
    else:
        print(f"Mevcut:      {init_file}")

print("\nTamamlandı.")
