import zipfile
import shutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,
Dropout
from tensorflow.keras.optimizers import Adam
# === Step 1: Paths to ZIP files ===
nofil_zip_path = r'C:\Users\PMLS\Pictures\Nofil.zip'
notnofil_zip_path = r'C:\Users\PMLS\Downloads\Not nofil.zip'
# === Step 2: Temp & Final Paths ===
temp_dir = 'temp_extract'
final_dataset_dir = 'processed_dataset'
nofil_final = os.path.join(final_dataset_dir, 'Nofil')
notnofil_final = os.path.join(final_dataset_dir, 'NotNofil')
# Clean and re-create dirs
for path in [temp_dir, final_dataset_dir, nofil_final, notnofil_final]:
if os.path.exists(path):
shutil.rmtree(path)
os.makedirs(path, exist_ok=True)
# === Step 3: Extract ZIP files ===
with zipfile.ZipFile(nofil_zip_path, 'r') as zip_ref:
zip_ref.extractall(os.path.join(temp_dir, 'nofil'))
with zipfile.ZipFile(notnofil_zip_path, 'r') as zip_ref:
zip_ref.extractall(os.path.join(temp_dir, 'notnofil'))
# === Step 4: Copy ONLY IMAGES to final dataset folders ===
def copy_images(src, dst):
if not os.path.exists(src):
print(f"Source folder not found: {src}")
return
for file in os.listdir(src):
if file.lower().endswith(('.jpg', '.jpeg', '.png')):
shutil.copy(os.path.join(src, file), os.path.join(dst, file))
# Your face images
copy_images(os.path.join(temp_dir, 'nofil', 'my images'), nofil_final)
# Not you images
copy_images(os.path.join(temp_dir, 'notnofil', 'images'), notnofil_final)
# === Step 5: Setup Data Generator ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
train_data = datagen.flow_from_directory(
final_dataset_dir,
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode='binary',
subset='training'
)
val_data = datagen.flow_from_directory(
final_dataset_dir,
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode='binary',
subset='validation'
)
# === Step 6: Build CNN Model ===
model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0],
IMG_SIZE[1], 3)),
MaxPooling2D(2, 2),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(2, 2),
Conv2D(128, (3, 3), activation='relu'),
  MaxPooling2D(2, 2),
Flatten(),
Dropout(0.5),
Dense(128, activation='relu'),
Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.0001),
loss='binary_crossentropy',
metrics=['accuracy'])
# === Step 7: Train the Model ===
history = model.fit(
train_data,
epochs=EPOCHS,
validation_data=val_data
)
# === Step 8: Save Model ===
model.save("nofil_face_recognizer.keras")
# === Step 9: Plot Accuracy ===
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
