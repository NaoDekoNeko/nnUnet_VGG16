import os
import random
import matplotlib.pyplot as plt
import cv2

# Directorios de imágenes
original_dir = 'mini_test'  # Carpeta con imágenes originales (MRI)
mask_dir = 'mini_test_output'  # Máscaras del modelo nnU-Net
merged_dir_nnunet = 'mini_test_output_merged'  # Superpuestas con nnU-Net
mask_dir_vgg16 = 'mini_test_output_vgg16'  # Máscaras de nnU-Net con VGG-16
merged_dir_vgg16 = 'mini_test_output_merged_vgg16'  # Superpuestas con nnU-Net + VGG-16

# Obtener lista de archivos ordenados
original_files = sorted(os.listdir(original_dir))
mask_files = sorted(os.listdir(mask_dir))
merged_files_nnunet = sorted(os.listdir(merged_dir_nnunet))
mask_files_vgg16 = sorted(os.listdir(mask_dir_vgg16))
merged_files_vgg16 = sorted(os.listdir(merged_dir_vgg16))

# Seleccionar 3 imágenes aleatorias
random_indices = random.sample(range(len(original_files)), 3)

# Visualizar imágenes de 5 en 5
def plot_images(index):
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    
    # Cargar imágenes
    original = cv2.imread(os.path.join(original_dir, original_files[index]))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(os.path.join(mask_dir, mask_files[index]), cv2.IMREAD_GRAYSCALE)

    merged_nnunet = cv2.imread(os.path.join(merged_dir_nnunet, merged_files_nnunet[index]))
    merged_nnunet = cv2.cvtColor(merged_nnunet, cv2.COLOR_BGR2RGB)

    mask_vgg16 = cv2.imread(os.path.join(mask_dir_vgg16, mask_files_vgg16[index]), cv2.IMREAD_GRAYSCALE)

    merged_vgg16 = cv2.imread(os.path.join(merged_dir_vgg16, merged_files_vgg16[index]))
    merged_vgg16 = cv2.cvtColor(merged_vgg16, cv2.COLOR_BGR2RGB)

    # Mostrar imágenes
    axes[0].imshow(original)
    axes[0].set_title("Original (MRI)")

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Máscara (nnU-Net)")

    axes[2].imshow(merged_nnunet)
    axes[2].set_title("Superpuesta (nnU-Net)")

    axes[3].imshow(mask_vgg16, cmap='gray')
    axes[3].set_title("Máscara (VGG16-nnU-Net)")

    axes[4].imshow(merged_vgg16)
    axes[4].set_title("Superpuesta (VGG16-nnU-Net)")

    # Eliminar ejes
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Iterar sobre 3 imágenes seleccionadas
for i in random_indices:
    print(f"Mostrando imágenes para el índice {i + 1}")
    plot_images(i)
