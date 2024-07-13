import cv2
import numpy as np

class PolygonDrawer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.polygon_points = []
        self.cap = None
        self.frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_points.append((x, y))
            print(f"Point Added: (X: {x}, Y: {y})")

    def load_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError("Cannot open video file")

    def get_frame(self):
        ret, self.frame = self.cap.read()
        if not ret:
            raise EOFError("Cannot read frame from video")
        self.frame = cv2.resize(self.frame, (1920, 1080))

    def draw_polygon(self):
        if len(self.polygon_points) > 1:
            cv2.polylines(self.frame, [np.array(self.polygon_points)], 
                          isClosed=False, color=(0, 255, 0), thickness=2)

    def run(self):
        self.load_video()
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.mouse_callback)

        while True:
            try:
                self.get_frame()
                self.draw_polygon()
                cv2.imshow('Frame', self.frame)

                key = cv2.waitKey(0)
                if key == 27:  # Esc key
                    break
            except EOFError:
                print("End of video reached")
                break

        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        if self.cap:
            self.cap.release()

    def print_polygon_points(self):
        print("Polygon Points:")
        for point in self.polygon_points:
            print(f"X: {point[0]}, Y: {point[1]}")

def main():
    video_path = r'C:\Users\Admin\Desktop\computervision videos/carsvid.mp4'
    drawer = PolygonDrawer(video_path)
    drawer.run()
    drawer.print_polygon_points()

if __name__ == "__main__":
    main()