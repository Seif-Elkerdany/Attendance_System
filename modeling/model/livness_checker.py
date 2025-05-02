import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)

def compute_fft_magnitude(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude

def compute_lbp_histogram(image, P=8, R=1):
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def is_real_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"[Blur Score] Laplacian Variance: {blur_score:.2f}")
    if blur_score < 10:
        print("→ Rejected due to blur")
        return False

    fft_mag = compute_fft_magnitude(gray)
    h, w = gray.shape
    center_region = fft_mag[h//2 - 20:h//2 + 20, w//2 - 20:w//2 + 20]
    high_freq_energy = np.mean(center_region)
    print(f"[FFT] High Frequency Energy: {high_freq_energy:.2f}")
    if high_freq_energy > 180:
        print("→ Rejected due to high frequency energy")
        return False

    hist = compute_lbp_histogram(gray)
    uniformity_score = hist[0] + hist[-1]
    print(f"[LBP] Uniformity Score: {uniformity_score:.2f}")
    if uniformity_score > 0.7:
        print("→ Rejected due to flat texture")
        return False

    print("Face passed all spoof checks")
    return True


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        exit()

    print("Press 'c' to check liveness")
    print("Press 'q' to quit")

    result = None  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        if result is not None:
            label = "Real" if result else "Fake"
            color = (0, 255, 0) if result else (0, 0, 255)
            cv2.putText(display_frame, f"Liveness: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Liveness Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):

            face_img = mtcnn(frame, save_path=None)

            if face_img is not None:
                face_np = face_img.permute(1, 2, 0).mul(255).byte().numpy()  
                face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                face_np = cv2.resize(face_np, (160, 160))

                result = is_real_face(face_np)
            else:
                print("No face detected")

    cap.release()
    cv2.destroyAllWindows()
