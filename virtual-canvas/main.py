import cv2 as cv
import numpy as np
import math

def process_frame(frame):

    global prev_tip, canvas

    frame = cv.resize(frame, (1080, 740))

    blur = cv.GaussianBlur(frame, (5, 5), 0)

    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 30, 50], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    x1, y1 = 500, 50
    x2, y2 = 1000, 700

    roi_mask = np.zeros_like(mask)
    roi_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

    contour, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    valid_contour = []

    for c in contour:
        area = cv.contourArea(c)

        if area > 500:
            valid_contour.append(c)

    if not valid_contour:
        return frame

    hand_contour = max(valid_contour, key=cv.contourArea)

    epsilon = 0.005 * cv.arcLength(hand_contour, True)

    approx = cv.approxPolyDP(
        hand_contour,
        epsilon,
        True
    )

    hull = cv.convexHull(hand_contour, returnPoints=False)

    fingertip = get_fingertips(hand_contour, frame)

    if len(hull) <= 3:
        return frame
        
    defects = cv.convexityDefects(hand_contour, hull)

    if defects is not None:

        valid_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])

            depth = d / 256.0

            a = np.linalg.norm(np.array(start) - np.array(far))
            b = np.linalg.norm(np.array(start) - np.array(end))
            c = np.linalg.norm(np.array(end) - np.array(far))

            if b == 0 or c == 0:
                continue

            angle = math.degrees(
                math.acos(
                    np.clip((b*b + c*c - a*a) / (2*b*c),
                        -1, 1)
                    )
                )

            if depth > 15 and angle < 90:

                valid_defects += 1

                cv.line(frame, start, end, (255, 0, 0), 2)
                cv.circle(frame, far, 7, (0, 0, 255), -1)

    if fingertip is not None:

        if prev_tip is not None:

            alpha = 0.02

            smooth_x = int(alpha * fingertip[0] + (1-alpha) * prev_tip[0])
            smooth_y = int(alpha * fingertip[1] + (1-alpha) * prev_tip[1])

            max_distance = 70

            distance = np.linalg.norm(
                np.array(fingertip) - np.array(prev_tip)
            )

            if distance < max_distance:
                cv.line(canvas, prev_tip, (smooth_x, smooth_y), (255, 255, 255), 5)

        prev_tip = fingertip

    else:
        prev_tip = None

    frame = cv.addWeighted(frame, 1, canvas, 1, 0)
        
    cv.drawContours(frame, [approx], -1, (0, 255, 0), 2)

    hull_point = cv.convexHull(approx, returnPoints=True)
    cv.polylines(frame, [hull_point], True, (255, 0, 255), 2)
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv.putText(frame, f"defects: {valid_defects}", 
               (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame

def get_fingertips(hand_contour, frame):

    hull = cv.convexHull(hand_contour, returnPoints=True)
    hull_points = hull.squeeze()

    if len(hull_points) == 1:
        hull_points = hull_points.reshape(-1, 2)

    fingertips = []

    M = cv.moments(hand_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        palm_center = (cx, cy)

        for point in hull_points:

            x, y = point
            distance = np.linalg.norm(np.array([x, y]) - np.array(palm_center))

            if y < (cy - 20) and distance > 30:

                fingertips.append((x, y))

            if fingertips:

                tip = min(fingertips, key=lambda point: point[1])
                cv.circle(frame, tip, 8, (0, 255, 255), -1)

    return tip

canvas = np.zeros((740, 1080, 3), dtype=np.uint8)
prev_tip = None

cap = cv.VideoCapture(0)

def main():
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)

        frame= process_frame(frame)

        cv.imshow('display', frame)

        if cv.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()