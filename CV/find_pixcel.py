import cv2

# 클릭 이벤트 함수
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"클릭 좌표: ({x}, {y})")

        # 화면에도 표시
        img = param.copy()
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"({x},{y})", (x+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.imshow("image", img)


if __name__ == "__main__":
    image_path = "./testData/center_test.png"  

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("이미지 못찾음")

    cv2.imshow("image", img)

    # 클릭 이벤트 등록
    cv2.setMouseCallback("image", click_event, img)

    print("이미지 클릭하면 좌표 나옴. ESC 누르면 종료")

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()