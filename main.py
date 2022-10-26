import face_recognition
import cv2
import numpy as np


# buka webcam
video_capture = cv2.VideoCapture(0)

# load gambar yang dikenalin
michel_image = face_recognition.load_image_file("michel/michel.jpg")
michel_face_encoding = face_recognition.face_encodings(michel_image)[0]

hasna_image = face_recognition.load_image_file("hasna/hasna.jpg")
hasna_face_encoding = face_recognition.face_encodings(hasna_image)[0]

hamzah_image = face_recognition.load_image_file("hamzah/hamzah.jpg")
hamzah_face_encoding = face_recognition.face_encodings(hamzah_image)[0]

royan_image = face_recognition.load_image_file("royan/royan.jpg")
royan_face_encoding = face_recognition.face_encodings(royan_image)[0]

farhan_image = face_recognition.load_image_file("farhan/farhan.jpg")
farhan_face_encoding = face_recognition.face_encodings(farhan_image)[0]


# buat label dari wajah yang dikenali
known_face_encodings = [
    michel_face_encoding,
    hasna_face_encoding,
    hamzah_face_encoding,
    royan_face_encoding,
    farhan_face_encoding
]
known_face_names = [
    "michel",
    "hasna",
    "hamzah",
    "royan",
    "farhan"
]

# inisialisasikan variabel
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # buka kamera
    ret, frame = video_capture.read()

    # ubah ukuran bingkai video ke ukuran 1/4 untuk pemrosesan pengenalan wajah yang lebih cepat
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # convert gambar dr rgb
    rgb_small_frame = small_frame[:, :, ::-1]

    # Hanya proses setiap frame video lainnya untuk menghemat waktu
    if process_this_frame:
        # Temukan semua wajah dan penyandian wajah dalam bingkai video saat ini
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # wajah tidak terdeteksi
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Kamu siapa"

            # wajah dikenali
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Dtampilkan hasil
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Skalakan kembali lokasi wajah karena bingkai yang kami deteksi diskalakan ke ukuran 1/4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # buat kotak pada wajah
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # beri label
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)

    # tampilkan hasil
    cv2.imshow('Video', frame)

    # keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
