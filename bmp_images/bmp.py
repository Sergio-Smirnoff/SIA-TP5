import struct
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import json
import pygame

def load_emojis(path="emojis.json"):
    with open(path, "r") as f:
        data = json.load(f)

    emojis = data["emojis"]
 
    # Convertir a NumPy array: (20, 7, 5)
    arr = np.array(emojis).astype(np.float32)

    # Aplanar cada imagen 7x5 → 35
    arr = arr.reshape(arr.shape[0], 35)

    return arr

# ------------------------------------------------------
#   Load the JSON with 7x5 emojis
# ------------------------------------------------------
def load_emojis_json(path="emojis.json"):
    with open(path, "r") as f:
        data = json.load(f)

    emojis = np.array(data["emojis"]).astype(int)   # (20, 7, 5)
    return emojis


# ------------------------------------------------------
#   Pygame drawing of a single emoji (scaled)
# ------------------------------------------------------
def draw_emoji(surface, emoji_matrix, x, y, pixel_size=20):
    rows, cols = emoji_matrix.shape

    for r in range(rows):
        for c in range(cols):
            val = emoji_matrix[r, c]
            color = (255, 255, 255) if val == 1 else (0, 0, 0)
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(
                    x + c * pixel_size,
                    y + r * pixel_size,
                    pixel_size,
                    pixel_size
                )
            )


# ------------------------------------------------------
#   Build the collage and save
# ------------------------------------------------------
def plot_all_emojis(emojis=None ,output_filename="emojis_collage.png"):

    BASE_PATH = "bmp_images/bit_output/"

    if emojis is None:
        emojis = load_emojis_json()    # shape (20, 7, 5)

    # Grid layout: 5 emojis per row
    per_row = 5
    total = len(emojis)
    rows = (total + per_row - 1) // per_row

    pixel_size = 20     # Scale factor for each bit
    emoji_w = 5 * pixel_size
    emoji_h = 7 * pixel_size

    margin = 10
    width = per_row * emoji_w + (per_row + 1) * margin
    height = rows * emoji_h + (rows + 1) * margin

    # Initialize pygame
    pygame.init()
    surface = pygame.Surface((width, height))

    # Background
    surface.fill((50, 50, 50))

    # Draw each emoji in grid
    for idx, emoji in enumerate(emojis):
        r = idx // per_row
        c = idx % per_row

        x = margin + c * (emoji_w + margin)
        y = margin + r * (emoji_h + margin)

        draw_emoji(surface, emoji, x, y, pixel_size)

    # Save PNG
    pygame.image.save(surface, BASE_PATH + output_filename)
    print(f"Saved emoji collage as: {output_filename}")

    # Display on screen
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Emoji Collage Preview")
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    print("Press any key to exit...")
    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
    #             running = False

    pygame.quit()

class BMPParser:
    """Parser para archivos BMP que extrae los datos de píxeles para usar en VAE"""
    
    def __init__(self):
        self.header = {}
        self.dib_header = {}
        self.pixel_data = None
        
    def parse_bmp(self, filepath: str) -> np.ndarray:
        """
        Lee un archivo BMP y retorna los datos de píxeles como array numpy
        
        Args:
            filepath: Ruta al archivo BMP
            
        Returns:
            numpy array con shape (height, width, channels)
        """
        with open(filepath, 'rb') as f:
            # Leer BMP Header (14 bytes)
            bmp_header = f.read(14)
            if bmp_header[0:2] != b'BM':
                raise ValueError("No es un archivo BMP válido")
            
            self.header['file_size'] = struct.unpack('<I', bmp_header[2:6])[0]
            self.header['pixel_data_offset'] = struct.unpack('<I', bmp_header[10:14])[0]
            
            # Leer DIB Header (primeros 4 bytes para obtener tamaño)
            dib_size = struct.unpack('<I', f.read(4))[0]
            f.seek(14)  # Volver al inicio del DIB header
            dib_header = f.read(dib_size)
            
            # Extraer información importante del DIB header
            self.dib_header['size'] = dib_size
            self.dib_header['width'] = struct.unpack('<i', dib_header[4:8])[0]
            self.dib_header['height'] = struct.unpack('<i', dib_header[8:12])[0]
            self.dib_header['bits_per_pixel'] = struct.unpack('<H', dib_header[14:16])[0]
            self.dib_header['compression'] = struct.unpack('<I', dib_header[16:20])[0]
            
            if self.dib_header['compression'] != 0:
                raise ValueError("Solo se soportan BMP sin compresión")
            
            # Leer datos de píxeles
            f.seek(self.header['pixel_data_offset'])
            
            width = abs(self.dib_header['width'])
            height = abs(self.dib_header['height'])
            bits_per_pixel = self.dib_header['bits_per_pixel']
            
            # Calcular bytes por fila (con padding a múltiplo de 4)
            bytes_per_pixel = bits_per_pixel // 8
            row_size = ((bits_per_pixel * width + 31) // 32) * 4
            
            # Leer todos los píxeles
            pixel_data = []
            for _ in range(height):
                row = f.read(row_size)
                # Extraer solo los bytes válidos (sin padding)
                valid_bytes = row[:width * bytes_per_pixel]
                pixel_data.append(valid_bytes)
            
            # Convertir a numpy array
            pixel_array = np.frombuffer(b''.join(pixel_data), dtype=np.uint8)
            
            # Reshape según el número de canales
            if bits_per_pixel == 24:  # BGR
                pixel_array = pixel_array.reshape((height, width, 3))
                # Convertir BGR a RGB
                pixel_array = pixel_array[:, :, ::-1]
            elif bits_per_pixel == 32:  # BGRA
                pixel_array = pixel_array.reshape((height, width, 4))
                # Convertir BGRA a RGBA
                pixel_array = pixel_array[:, :, [2, 1, 0, 3]]
            elif bits_per_pixel == 8:  # Escala de grises
                pixel_array = pixel_array.reshape((height, width, 1))
            else:
                raise ValueError(f"Bits por píxel no soportado: {bits_per_pixel}")
            
            # BMP almacena las filas de abajo hacia arriba, voltear
            if self.dib_header['height'] > 0:
                pixel_array = np.flipud(pixel_array)
            
            self.pixel_data = pixel_array
            return pixel_array
    
    def normalize_for_vae(self, pixel_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normaliza los datos de píxeles al rango [0, 1] para el VAE
        
        Args:
            pixel_data: Array de píxeles (opcional, usa self.pixel_data si no se proporciona)
            
        Returns:
            numpy array normalizado con valores entre 0 y 1
        """
        if pixel_data is None:
            pixel_data = self.pixel_data
            
        if pixel_data is None:
            raise ValueError("No hay datos de píxeles para normalizar")
        
        return pixel_data.astype(np.float32) / 255.0
    
    def get_shape(self) -> Tuple[int, int, int]:
        """Retorna las dimensiones de la imagen (height, width, channels)"""
        if self.pixel_data is None:
            raise ValueError("No se han cargado datos de píxeles")
        return self.pixel_data.shape
    
    def get_info(self) -> dict:
        """Retorna información del archivo BMP"""
        return {
            'header': self.header,
            'dib_header': self.dib_header,
            'shape': self.get_shape() if self.pixel_data is not None else None
        }
    
    def create_bmp_from_array(self, pixel_data: np.ndarray, output_path: str, 
                              denormalize: bool = True):
        """
        Crea un archivo BMP a partir de un array numpy (salida del VAE)
        
        Args:
            pixel_data: Array numpy con shape (height, width, channels) o (height, width)
            output_path: Ruta donde guardar el archivo BMP
            denormalize: Si True, asume valores [0,1] y los convierte a [0,255]
        """
        # Procesar datos
        if denormalize:
            # Convertir de [0,1] a [0,255]
            pixel_data = np.clip(pixel_data, 0, 1)
            pixel_data = (pixel_data * 255).astype(np.uint8)
        else:
            pixel_data = pixel_data.astype(np.uint8)
        
        # Asegurar que tenga 3 dimensiones
        if len(pixel_data.shape) == 2:
            # Escala de grises, convertir a RGB
            pixel_data = np.stack([pixel_data] * 3, axis=-1)
        
        height, width, channels = pixel_data.shape
        
        # Determinar bits por píxel
        if channels == 1:
            bits_per_pixel = 8
        elif channels == 3:
            bits_per_pixel = 24
        elif channels == 4:
            bits_per_pixel = 32
        else:
            raise ValueError(f"Número de canales no soportado: {channels}")
        
        # Convertir RGB a BGR (formato BMP)
        if channels == 3:
            pixel_data = pixel_data[:, :, ::-1]
        elif channels == 4:
            pixel_data = pixel_data[:, :, [2, 1, 0, 3]]
        
        # Voltear imagen (BMP guarda de abajo hacia arriba)
        pixel_data = np.flipud(pixel_data)
        
        # Calcular padding
        bytes_per_pixel = bits_per_pixel // 8
        row_size = ((bits_per_pixel * width + 31) // 32) * 4
        padding_size = row_size - (width * bytes_per_pixel)
        
        # Calcular tamaños
        pixel_data_size = row_size * height
        dib_header_size = 40  # BITMAPINFOHEADER
        pixel_data_offset = 14 + dib_header_size
        file_size = pixel_data_offset + pixel_data_size
        
        with open(output_path, 'wb') as f:
            # ===== BMP Header (14 bytes) =====
            f.write(b'BM')  # Signature
            f.write(struct.pack('<I', file_size))  # File size
            f.write(struct.pack('<H', 0))  # Reserved 1
            f.write(struct.pack('<H', 0))  # Reserved 2
            f.write(struct.pack('<I', pixel_data_offset))  # Pixel data offset
            
            # ===== DIB Header (40 bytes - BITMAPINFOHEADER) =====
            f.write(struct.pack('<I', dib_header_size))  # DIB header size
            f.write(struct.pack('<i', width))  # Width
            f.write(struct.pack('<i', height))  # Height (positivo = bottom-up)
            f.write(struct.pack('<H', 1))  # Planes
            f.write(struct.pack('<H', bits_per_pixel))  # Bits per pixel
            f.write(struct.pack('<I', 0))  # Compression (0 = no compression)
            f.write(struct.pack('<I', pixel_data_size))  # Image size
            f.write(struct.pack('<i', 2835))  # X pixels per meter (72 DPI)
            f.write(struct.pack('<i', 2835))  # Y pixels per meter (72 DPI)
            f.write(struct.pack('<I', 0))  # Colors in palette
            f.write(struct.pack('<I', 0))  # Important colors
            
            # ===== Pixel Data =====
            padding = b'\x00' * padding_size
            for row in pixel_data:
                f.write(row.tobytes())
                if padding_size > 0:
                    f.write(padding)
        
        print(f"Imagen BMP guardada en: {output_path}")
        print(f"Dimensiones: {width}x{height}, {bits_per_pixel} bits por píxel")


def load_bmp_dataset(directory: str, normalize: bool = True) -> np.ndarray:
    """
    Carga múltiples archivos BMP desde un directorio
    
    Args:
        directory: Ruta al directorio con archivos BMP
        normalize: Si True, normaliza los valores al rango [0, 1]
        
    Returns:
        numpy array con shape (n_images, height, width, channels)
    """
    parser = BMPParser()
    images = []
    
    path = Path(directory)
    bmp_files = sorted(path.glob("*.bmp"))
    
    if not bmp_files:
        raise ValueError(f"No se encontraron archivos BMP en {directory}")
    
    for bmp_file in bmp_files:
        try:
            img = parser.parse_bmp(str(bmp_file))
            if normalize:
                img = parser.normalize_for_vae(img)
            images.append(img)
            print(f"Cargado: {bmp_file.name} - Shape: {img.shape}")
        except Exception as e:
            print(f"Error al cargar {bmp_file.name}: {e}")
    
    return np.array(images)


def save_vae_output_as_bmp(vae_output: np.ndarray, output_path: str, 
                           original_shape: tuple = None):
    """
    Función auxiliar para guardar la salida del VAE como BMP
    
    Args:
        vae_output: Array numpy con la salida del VAE (normalizado 0-1)
                   Puede ser: (height, width, channels) o (height*width*channels,)
        output_path: Ruta donde guardar el BMP
        original_shape: Shape original si vae_output es 1D (height, width, channels)
    """
    parser = BMPParser()
    
    # Si es 1D, reshape al formato original
    if len(vae_output.shape) == 1:
        if original_shape is None:
            raise ValueError("Se necesita original_shape para reshape de array 1D")
        vae_output = vae_output.reshape(original_shape)
    
    parser.create_bmp_from_array(vae_output, output_path, denormalize=True)


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo 1: Cargar una imagen individual
    parser = BMPParser()
    
    try:
        # Reemplaza con la ruta a tu archivo BMP
        img_data = parser.parse_bmp("imagen.bmp")
        print(f"\nDimensiones de la imagen: {img_data.shape}")
        print(f"Tipo de datos: {img_data.dtype}")
        print(f"Rango de valores: [{img_data.min()}, {img_data.max()}]")
        
        # Normalizar para VAE
        normalized = parser.normalize_for_vae()
        print(f"\nDatos normalizados:")
        print(f"Rango: [{normalized.min()}, {normalized.max()}]")
        print(f"Shape: {normalized.shape}")
        
        # Ver información del archivo
        info = parser.get_info()
        print(f"\nInformación del BMP:")
        print(f"Ancho: {info['dib_header']['width']}")
        print(f"Alto: {info['dib_header']['height']}")
        print(f"Bits por píxel: {info['dib_header']['bits_per_pixel']}")
        
    except FileNotFoundError:
        print("Archivo no encontrado. Actualiza la ruta en el código.")
    
    # Ejemplo 2: Cargar un dataset completo
    try:
        # Reemplaza con la ruta a tu directorio de imágenes
        dataset = load_bmp_dataset("./imagenes_bmp", normalize=True)
        print(f"\n\nDataset cargado:")
        print(f"Cantidad de imágenes: {len(dataset)}")
        print(f"Shape del dataset: {dataset.shape}")
        print(f"Listo para entrenar el VAE!")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nNo se pudo cargar el dataset: {e}")
    
    # Ejemplo 3: Generar imagen desde salida del VAE
    print("\n" + "="*50)
    print("Ejemplo: Guardar salida del VAE como BMP")
    print("="*50)
    
    # Simular salida del VAE (valores normalizados entre 0 y 1)
    # En tu caso real, esto vendría de vae.forward() o vae.generate()
    altura, ancho, canales = 64, 64, 3
    
    # Simulación: crear una imagen de gradiente como ejemplo
    vae_output = np.zeros((altura, ancho, canales))
    for i in range(altura):
        for j in range(ancho):
            vae_output[i, j, 0] = i / altura  # Canal rojo
            vae_output[i, j, 1] = j / ancho   # Canal verde
            vae_output[i, j, 2] = 0.5         # Canal azul constante
    
    # Guardar como BMP
    parser = BMPParser()
    parser.create_bmp_from_array(vae_output, "vae_generada.bmp", denormalize=True)
    
    # También puedes usar la función auxiliar
    save_vae_output_as_bmp(vae_output, "vae_generada_2.bmp")
    
    # Si tu VAE devuelve un array 1D (flattened), puedes hacer:
    vae_output_flat = vae_output.flatten()
    save_vae_output_as_bmp(vae_output_flat, "vae_generada_3.bmp", 
                           original_shape=(altura, ancho, canales))