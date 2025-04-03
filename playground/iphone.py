
import cv2
import time


def use_virtual_webcam(camera_index=0):
    """
    Use a virtual webcam created by apps like EpocCam, DroidCam, or Camo.
    These apps typically create a virtual webcam device on your computer.
    """
    # Set up video capture
    cap = cv2.VideoCapture(camera_index)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    # Get and print camera properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened successfully at index {camera_index}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Display the resulting frame
        cv2.imshow('iPhone Camera', frame)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


def list_all_cameras():
    """
    List all available camera devices.
    """
    max_cameras = 10  # Adjust as needed

    print("Checking available cameras:")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"Camera {i}: Available, Resolution: {width}x{height}, FPS: {fps}")
            else:
                print(f"Camera {i}: Available but can't read frame")
            cap.release()
        else:
            print(f"Camera {i}: Not available")


if __name__ == "__main__":
    # First, list all available cameras
    list_all_cameras()

    # Then try to use the virtual webcam
    # The index might vary depending on your system and app used
    camera_index = int(input("\nEnter the camera index to use: "))
    use_virtual_webcam(camera_index)
