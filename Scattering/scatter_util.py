import numpy as np
import time
import cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import math
from Robot_env.scattering_path import *


def find_neighboring_obj(seg, target, angle, w):

    # = 세그맨테이션 정보와 현재 타겟 정보, 각도를 받아와서
    # . 모든 세그맨테이션을 팽창시킨뒤
    # . 단축의 각도에 해당하는 방향으로 물체 간섭 확인
    # . 물체가 있다면, SVM 을 실행하여
    # . 물체를 잡아도 됨을 반환
    delated_seg = np.copy(seg)
    delated_seg2 = np.copy(seg)
    binary_image_array = np.zeros(shape=(720, 1280), dtype=np.uint8)

    target_list = np.argwhere(np.array(seg) == target)

    print("%d angle & short axis : %f, %f" % (target, angle, w))

    [binary_image_array.itemset((y, x), 255) for [y, x] in target_list]

    # 얼마나 팽창 연산을 수행할지를 결정하는 파라미터들
    # if target in []:   # Big size objects
    if target in [5, 8, 9, 14]:  # Big size objects
        # kernel = np.ones((5, 5), np.uint8)  # 5x5 Filter size
        # target_dilation = np.array(cv2.dilate(binary_image_array, kernel, iterations=6))
        kernel_size = 9
        kernel_half = math.trunc(kernel_size / 2)
        kernelorg = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel = cv2.ellipse(kernelorg, (kernel_half, kernel_half), (kernel_half + 1, kernel_half - 1),
                             angle, 0, 360, 1, -1)
        target_dilation = np.array(cv2.dilate(binary_image_array, kernel, iterations=3))

        # >> check
        cv2.namedWindow("kernel")
        img_k = kernel * 255
        cv2.imshow("kernel", img_k)
        cv2.moveWindow("kernel", 2560 - 720, 640)
        cv2.waitKey(2)
        cv2.imwrite("{}_kernel".format(target) + ".png", img_k)

        cv2.namedWindow("target_dilation")
        img_td = cv2.flip(target_dilation, 0)
        cv2.imshow("target_dilation", img_td)
        cv2.moveWindow("target_dilation", 2560 - 720, 680)
        cv2.waitKey(2)
        cv2.imwrite("{}_target_dilation".format(target) + ".png", img_td)

        # ekernel_size = 3
        # ekernel = np.ones((ekernel_size, ekernel_size), np.uint8)
        # target_mod = np.array(cv2.dilate(target_dilation, ekernel, iterations=1))
        target_mod = target_dilation

    ######
    # kernel_size = 5
    # kernelorg = np.ones((kernel_size, kernel_size), np.uint8)
    # target_mod = np.array(cv2.dilate(binary_image_array, kernelorg, iterations=6))

    else:
        # kernel = np.ones((3, 3), np.uint8)  # 3x3 Filter
        # target_dilation = np.array(cv2.dilate(binary_image_array, kernel, iterations=7))
        kernel_size = 7
        kernel_half = math.trunc(kernel_size / 2)
        kernelorg = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel = cv2.ellipse(kernelorg, (kernel_half, kernel_half), (kernel_half + 1, kernel_half - 1), angle, 0, 360,
                             1, -1)
        target_dilation = np.array(cv2.dilate(binary_image_array, kernel, iterations=4))

        # >> check
        cv2.namedWindow("kernel")
        img_k = kernel * 255
        cv2.imshow("kernel", img_k)
        cv2.moveWindow("kernel", 2560 - 720, 640)
        cv2.waitKey(2)
        cv2.imwrite("{}_kernel".format(target) + ".png", img_k)

        cv2.namedWindow("target_dilation")
        img_td = cv2.flip(target_dilation, 0)
        cv2.imshow("target_dilation", img_td)
        cv2.moveWindow("target_dilation", 2560 - 720, 680)
        cv2.waitKey(2)
        cv2.imwrite("{}target_dilation".format(target) + ".png", img_td)

        # ekernel_size = 3
        # ekernel = np.ones((ekernel_size, ekernel_size), np.uint8)
        # target_mod = np.array(cv2.dilate(target_dilation, ekernel, iterations=1))
        target_mod = target_dilation

    ######
    # kernel_size = 3
    # kernelorg = np.ones((kernel_size, kernel_size), np.uint8)
    # target_mod = np.array(cv2.dilate(binary_image_array, kernelorg, iterations=7))

    # #>> check
    # cv2.namedWindow("fkernel")
    # #cv2.imshow("fkernel", ekernel*255)
    # cv2.moveWindow("fkernel", 2560 - 400, 640)
    # cv2.waitKey(2)
    # #cv2.imwrite("fkernel.png", ekernel*255)

    # cv2.namedWindow("target_mod")
    # cv2.imshow("target_mod", cv2.flip(target_mod, 0))
    # cv2.moveWindow("target_mod", 2560 - 400, 680)
    # cv2.waitKey(2)
    # cv2.imwrite("target_mod.png", cv2.flip(target_mod, 0))

    # ---------------------------------------------------
    binary_image_array2 = np.zeros(shape=(256, 256), dtype=np.uint8)

    # cv2.kmeans(mean_set_list, 1)
    target_list_mean = np.mean(target_list[:-1], axis=0)
    try:
        p1 = math.trunc(target_list_mean[0])
        p2 = math.trunc(target_list_mean[1])  # [math.trunc(target_list_mean[0]),math.trunc(target_list_mean[1])]
        binary_image_array2[p1, p2] = 255
    except:
        pass

    # -> k means 써서 물체 중심 찾고 중심을 255로 만들기...?
    # 아니면 깡 평균
    if target in [5, 8, 9, 14]:  # Big size objects
        kernel_size = 11
        kernel_half = math.trunc(kernel_size / 2)
        kernelorg = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel_1 = cv2.ellipse(kernelorg, (kernel_half, kernel_half), (kernel_half, 0), angle, 0, 360, 1, 1)
        target_dilation2 = np.array(cv2.dilate(binary_image_array2, kernel_1, iterations=4))

        cv2.namedWindow("kernel_1")
        img_k = kernel_1 * 255
        cv2.imshow("kernel_1", img_k)
        cv2.moveWindow("kernel_1", 2560 - 720 + 256, 640)
        cv2.waitKey(2)
        cv2.imwrite("test_kernel_1".format(target) + ".png", img_k)

        kernel_size2 = 11
        kernel_half2 = math.trunc(kernel_size2 / 2)
        kernelorg2 = np.zeros((kernel_size2, kernel_size2), np.uint8)
        kernel_2 = cv2.ellipse(kernelorg2, (kernel_half2, kernel_half2), (kernel_half2 - 1, kernel_half2 - 1), angle, 0,
                               360, 1, -1)
        target_mod2 = np.array(cv2.dilate(target_dilation2, kernel_2, iterations=1))

        cv2.namedWindow("kernel_2")
        img_k2 = kernel_2 * 255
        cv2.imshow("kernel_2", img_k2)
        cv2.moveWindow("kernel_2", 2560 - 720 + 256 + 128, 640)
        cv2.waitKey(2)
        cv2.imwrite("test_kernel_2".format(target) + ".png", img_k2)

        cv2.namedWindow("target_mod2")
        img_td2 = cv2.flip(target_mod2, 0)
        cv2.imshow("target_mod2", img_td2)
        cv2.moveWindow("target_mod2", 2560 - 720 + 256 + 128, 680)
        cv2.waitKey(2)
        cv2.imwrite("{}_target_mod2".format(target) + ".png", img_td2)

    else:
        kernel_size = 9
        kernel_half = math.trunc(kernel_size / 2)
        kernelorg = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel_1 = cv2.ellipse(kernelorg, (kernel_half, kernel_half), (kernel_half, 0), angle, 0, 360, 1, 1)
        target_dilation2 = np.array(cv2.dilate(binary_image_array2, kernel_1, iterations=3))

        cv2.namedWindow("kernel_1")
        img_k = kernel_1 * 255
        cv2.imshow("kernel_1", img_k)
        cv2.moveWindow("kernel_1", 2560 - 720 + 256, 640)
        cv2.waitKey(2)
        cv2.imwrite("test_kernel_1".format(target) + ".png", img_k)

        kernel_size = 9
        kernel_half = math.trunc(kernel_size / 2)
        kernelorg = np.zeros((kernel_size, kernel_size), np.uint8)
        kernel_2 = cv2.ellipse(kernelorg, (kernel_half, kernel_half), (kernel_half - 1, kernel_half - 1), angle, 0, 360,
                               1, -1)
        target_mod2 = np.array(cv2.dilate(target_dilation2, kernel_2, iterations=1))

        cv2.namedWindow("kernel_2")
        img_k2 = kernel_2 * 255
        cv2.imshow("kernel_2", img_k2)
        cv2.moveWindow("kernel_2", 2560 - 720 + 256 + 128, 640)
        cv2.waitKey(2)
        cv2.imwrite("test_kernel_2".format(target) + ".png", img_k2)

        cv2.namedWindow("target_mod2")
        img_td2 = cv2.flip(target_mod2, 0)
        cv2.imshow("target_mod2", img_td2)
        cv2.moveWindow("target_mod2", 2560 - 720 + 256 + 128, 680)
        cv2.waitKey(2)
        cv2.imwrite("{}_target_mod2".format(target) + ".png", img_td2)
    # ---------------------------------------------------

    no0 = binary_image_array.copy()  # 원 물체 세그
    no1 = target_dilation.copy()  # 팽창 후 물체
    no2 = target_mod2.copy()  # 검사 할 영역

    # box_list = np.argwhere(box_img == 32)
    # boxedseg_list = np.argwhere(box_img == 255)
    # dilated_list = np.argwhere(no1 == 255)
    bl_img = np.zeros(shape=(256, 256), dtype=np.uint8)

    bl_img = bl_img + (no1 / 255) * 128
    bl_img = bl_img + (no0 / 255) * 127

    # con_img1, contour1, _ = cv2.findContours(no0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con_img1, contour1, _ = cv2.findContours(no0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour1:
        rect = cv2.minAreaRect(cnt)

        box_cont = np.int0(cv2.boxPoints(rect))
        box_img = cv2.drawContours(bl_img, [box_cont], 0, 32, 1)

    no2_list = np.argwhere(no2 == 255)

    for [x, y] in no2_list:
        box_img[x, y] = 64

    cv2.namedWindow("final_input")
    img_fi = cv2.flip(box_img, 0)
    cv2.imshow("final_input", img_fi / 255)
    cv2.moveWindow("final_input", 2560 - 720 - 256, 0)
    cv2.waitKey(2)
    cv2.imwrite("{}_final_input".format(target) + ".png", img_fi)

    close_obj_index_list2 = []
    n_obj_temp = []
    dilated_target_pointList2 = np.argwhere(target_mod2 == 255)
    for [y, x] in dilated_target_pointList2:  # 디텍션 영역의 좌표에서
        label2 = seg[y, x]  # 해당좌표의 세그맨테이션 된 이미지 정보를 라벨로 지정

        if label2 != target and label2 != 16:  # 라벨과 타겟이 다를거나, 라벨이 배경일 경우
            close_obj_index_list2.append(label2)  # 근처 오브젝트를 포함 시킴
            delated_seg2[y, x] = target  # 원 세그멘테이션에 라벨이 타겟&배경과 다른경우를 해당 위치를 타겟값으로 저장

    if close_obj_index_list2.__len__() != 0:  # 가까운 오브젝트가 존재할 경우
        dilated_cls2 = np.unique(delated_seg2)  # 디텍트부분을 타겟으로 확장 후 저장된 포인트 들 중 중복 제거
        seg_cls = np.unique(seg)  # 세그멘테이션 원 데이터 포인트 들 중 중복 제거

        n_obj2 = np.unique(close_obj_index_list2)  # 근처 오브젝트 중 중복 제거

        for cls in dilated_cls2:
            seg_cls = np.delete(seg_cls, np.argwhere(seg_cls == cls))  # 세그멘테이션 된 데이터와 다이얼레이션 에서 검출된 데이터를 비교, 같은것을 제거

        # -> 원래 아이템이랑, 디텍트 확장시켰을때 아이템이랑 서로 지웠을때  뭐가남는다? = 먹혔다.
        if seg_cls.size != 0:  # 먹힌게 존재하면
            for cls in seg_cls:  # 원래 새그멘테이션 정보 중에서 가까운 물체를 삭제
                n_obj2 = np.delete(n_obj2, np.argwhere(cls == n_obj2))

        # ->
        for obj in n_obj2:
            exist_pt = np.argwhere(delated_seg2 == obj)  # 가까운 오브젝트들 중에서 확장된 세그맨테이션이랑 겹치는것

            if exist_pt.size != 0:  # 안먹히고 남은 부분이 존재하면 그것은 가까운 오브젝트이다.
                n_obj_temp = n_obj2  # 인덱스 0 = 스케터링 X ,
    # ---- ---- ---- ----

    dilated_target_pointList = np.argwhere(target_mod == 255)  # 팽창된 이미지 중에서 255값에 해당하는 포인트만 추출

    #  = (원코드)
    #  = 근접 물체 파악 및 경로 생성시 어떤 포인트 들을 쓸건지 저장
    close_obj_index_list = []
    for [y, x] in dilated_target_pointList:
        label = seg[y, x]  # 세그맨테이션 된 이미지에서 추출된 포인트에 해당하는것을 라벨로 지정

        if label != target and label != 16:  # 라벨과 타겟이 다를경우, 라벨이 배경일 경우
            close_obj_index_list.append(label)  # 근처 오브젝트를 포함 시킴
            delated_seg[y, x] = target  # 타겟에 해당하는 것들의 좌표를 저장

    if close_obj_index_list.__len__() != 0:  # 가까운 오브젝트 인덱스 리스트.길이가 0이 아닐경우
        dilated_cls = np.unique(delated_seg)  # 다이얼레이션 후 저장된 포인트 들 중 중복 제거
        seg_cls = np.unique(seg)  # 세그멘테이션 원 데이터 포인트 들 중 중복 제거

        n_obj = np.unique(close_obj_index_list)  # 근처 오브젝트 중 중복 제거

        for cls in dilated_cls:
            seg_cls = np.delete(seg_cls, np.argwhere(seg_cls == cls))
        # 세그멘테이션 된 데이터와 다이얼레이션 에서 검출된 데이터를 비교, 같은것을 제거
        if seg_cls.size != 0:
            for cls in seg_cls:
                n_obj = np.delete(n_obj, np.argwhere(cls == n_obj))
        # 가까이있는 물체중에서 세그맨테이션 원 데이터 삭제

        # = (원 코드)
        for obj in n_obj:
            exist_pt = np.argwhere(delated_seg == obj)

            if exist_pt.size != 0:
                return delated_seg, delated_seg2, n_obj_temp, n_obj  # 인덱스 0 = 스케터링 X ,

        return None, None, None, None
    else:
        return None, None, None, None


def non_linear_scatter(seg_img, target_cls, angle, w):
    dilated_img, dilated_img2, neighbored_list, neighbored_list_org = find_neighboring_obj(seg_img, target_cls, angle,
                                                                                           w)
    if neighbored_list is None or neighbored_list == []:
        return None

    if np.all(np.unique(dilated_img) != np.unique(dilated_img2)):
        check_id = False
        for n_idx in np.unique(dilated_img):  # 1, 4, 5, 8, 14 , 16
            for s_idx in neighbored_list:  # 8, 10
                if n_idx == s_idx:
                    check_id = True
                else:
                    pass

        if check_id == False:
            return None

    # Non-Linear classification-
    target_pt = np.argwhere(dilated_img == target_cls).astype("float64")  # 타겟 물체의 세그멘테이션 픽셀들
    mean_tgt = np.mean(target_pt, axis=0)  # 타겟 물체의 중앙을 찾아냄

    n_size = target_pt.shape[0]  # 포인트 갯수

    b = np.zeros((1, 2))

    for idx in neighbored_list_org:  # 주변 물체들 픽셀 위치를 검사
        neighbor_pt = np.argwhere(dilated_img == idx).astype("float64")  # 세그맨테이션 픽셀들이 주변물체면 저장
        b = np.concatenate((b, neighbor_pt))  # 맨 앞 0,0 에 뒤로 붙임
        n_size += neighbor_pt.shape[0]  # 주변 물체들의 포인트 갯수를 타겟 포인트 갯수와 더함

    # # asdaf ...........
    # if np.unique(dilated_img) == dilated_img2 :
    # 	for idx in neighbored_list:
    # 		neighbor_pt = np.argwhere(dilated_img == idx).astype("float64")
    # 		b = np.concatenate((b, neighbor_pt))
    # 		n_size += neighbor_pt.shape[0]
    # else:
    # 	for idx in neighbored_list:
    # 		return None

    # Labeling
    label = np.zeros(n_size)  # 1렬 0 행렬 생성
    label[target_pt.shape[0]:] = 1  # -> target = 0, neighbor = 1
    # ㄴ 타겟 갯수만큼 건너뛴뒤 1로 저장

    data = np.concatenate((target_pt, b[1:]))  # 타겟 포인트와 주변물체 포인트를 합침
    temp = np.array([data[:, 1], data[:, 0]]).transpose()  # 모두 x,y 좌표에 맞게 변환

    # svm = SVC(kernel='rbf', random_state=0, gamma=0.0001, C=1000)   # svm 파라미터 설정
    svm = SVC(kernel='rbf', random_state=0, gamma=0.0001, C=1000.000)  # svm 파라미터 설정
    # if np.unique(label) == 0:
    # 	return None
    # if temp is None :
    # 	return
    svm.fit(temp, label)  # 해당 픽셀들로 svm을 실행

    # = ????
    # Path가 두 개 이상 생성되었을때
    l_path = plot_decision_regions(classifier=svm)
    l_dist = np.zeros(l_path.shape[0])

    for idx, path in enumerate(l_path):
        l_dist[idx] = np.min(np.linalg.norm((path - mean_tgt), axis=1))

    # = ???
    # 가장 가까운 거리의 Path 선정. => Target Path
    target_path = l_path[np.argmin(l_dist)]
    t_path_len = int(target_path.size / 2)

    is_loop = check_loop(target_path[0], target_path[-1])  # 순환하는지를 체크, 부울로 나옴

    # Target Path를 따라 생성되는 부분 공간 Path 찾기
    # temp_img = np.copy(seg_img)
    # for y, x in target_path:
    #     temp_img[y, x] = 255
    # plt.imshow(temp_img)
    # cv2.imshow("TEst viewer", temp_img)

    # Split Path
    space_idx, space_dist = split_path(seg_img, neighbored_list_org, target_path)

    # For Space Conncection
    head = target_path[space_idx[0][0]]
    tail = target_path[space_idx[-1][1]]

    # = 확인용
    non_L_path = np.copy(seg_img)
    for [y, x] in target_path:
        non_L_path[y, x] = 100

    non_L_path[head[0], head[1]] = 255
    non_L_path[tail[0], tail[1]] = 255
    plt.figure('non_L_path');
    plt.imshow(cv2.flip(non_L_path, 0))
    time.sleep(0)

    for [y, x] in dil(tail):
        try:
            if (y == head[0]) and (x == head[1]):
                new_s = space_idx[-1][0]
                target_path = np.concatenate((target_path[new_s:], target_path[:new_s]), axis=0)

                space_idx, space_dist = split_path(seg_img, neighbored_list, target_path)
                break
        except:
            pass

    # ???
    # 가장 넓은 공간 찾기.
    t_idx = int(np.argmax(space_dist).squeeze())

    if is_loop:
        start = space_idx[t_idx][1]
        end = space_idx[t_idx][0]

        if start > end:
            main_path = np.concatenate((target_path[start:], target_path[:end]), axis=0)
        else:
            main_path = target_path[start:end]
    else:
        start = space_idx[t_idx][0]
        end = space_idx[t_idx][1]

        if start == 0 and end != t_path_len - 1:
            start = np.copy(end)
            end = end
        elif start != 0 and end == t_path_len - 1:
            start = start
            end = start

        main_path = target_path[start:end]

    if is_loop:
        # Start 늘리기
        for pt in range(t_path_len - 1):
            t_st = start - pt

            if t_st < 0:
                t_st = t_path_len - t_st

            if t_st > t_path_len:
                return "linear"

            chk_surr = False

            # starting point 주변 픽셀이 배경인지 확인
            near_size = 5
            if target_cls in [5, 8, 9, 14, 11]:
                near_size = 7

            for [y, x] in dil(target_path[t_st], near_size):
                try:
                    if seg_img[y, x] != 16:
                        chk_surr = True
                        break
                except:
                    chk_surr = True
                    break

            if chk_surr:
                continue
            else:
                if t_st > start:
                    temp_path = np.concatenate((target_path[t_st:], target_path[:start]), axis=0)
                    main_path = np.concatenate((temp_path, main_path), axis=0)
                else:
                    main_path = np.concatenate((target_path[t_st:start], main_path), axis=0)

                break

        # End 늘리기
        for pt in range(t_path_len):
            t_et = (end + pt) % t_path_len
            chk_surr = False

            for [y, x] in dil(target_path[t_et], 1):
                try:
                    if seg_img[y, x] != 16:
                        chk_surr = True
                        break
                except:
                    chk_surr = True
                    break

            if chk_surr:
                continue
            else:
                if is_loop:
                    if t_et < end:
                        temp_path = np.concatenate((target_path[end:], target_path[:t_et]), axis=0)
                        main_path = np.concatenate((main_path, temp_path), axis=0)
                    else:
                        main_path = np.concatenate((main_path, target_path[end:t_et + 1]), axis=0)

                break

    else:
        # Start 줄이기
        for pt in range(t_path_len):
            t_st = start - pt
            chk_surr = False

            # starting point 주변 픽셀이 배경인지 확인
            near_size = 5
            if target_cls in [5, 8, 9, 14, 11]:
                near_size = 7

            for [y, x] in dil(target_path[t_st], near_size):
                try:
                    if seg_img[y, x] != 16:
                        chk_surr = True
                        break
                except:
                    chk_surr = True
                    break

            if chk_surr:
                continue
            else:
                main_path = target_path[t_st:start]
                break

        # End 늘리기
        for pt in range(t_path_len):
            t_et = end + pt

            chk_surr = False

            for [y, x] in dil(target_path[t_et], 2):
                try:
                    if seg_img[y, x] != 16:
                        chk_surr = True
                        break
                except:
                    chk_surr = False
                    break

            if chk_surr:
                continue
            else:
                main_path = np.concatenate((main_path, target_path[end:t_et]), axis=0)
                break

    # 경로 시각화
    # temp_img = np.copy(seg_img)
    # for y, x in main_path:
    #     temp_img[y, x] = 40
    #     cv2.imshow("tesT", temp_img)
    #     cv2.waitKey(1)
    #
    # plt.imshow(temp_img)

    return main_path


def linear_scatter(seg_img, target_cls, angle, w):
    delated_seg, delated_seg2, neighbored_list, neighbored_list_org = find_neighboring_obj(seg_img, target_cls, angle,
                                                                                           w)

    if neighbored_list is None or neighbored_list == []:
        return

    # Non-Linear classification # ...? 타겟 오브젝트를 seg 이미지에서 추출
    target_pt = np.argwhere(delated_seg == target_cls).astype("uint8")

    # =ch
    # dseg_img = np.copy(delated_seg)
    # for [y, x] in target_pt:
    # 	dseg_img[y, x] = 60
    #
    # plt.figure('target_pt'); plt.imshow(cv2.flip(dseg_img, 0))

    # pangchang
    n_size = int(target_pt.size / 2)  # ??? 왜 타겟 크기를 반으로 쪼갬 ????

    neighbor_obj = 0
    pt_max = 0

    for idx in neighbored_list:  # 주변 물체 아이디로 다이얼레이션 된 영역 검사
        temp_pt = np.argwhere(delated_seg == idx)
        if pt_max < temp_pt.size / 2:  # 그 검사된 크기/2가 임시 변수보다 작을때
            pt_max = temp_pt.size / 2  # 1/2크기를 임시변수에 저장
            neighbor_obj = idx  # 집중할 주변 오브젝트의 크기가 큰것을 선정

    # =ch
    # dseg_img2 = np.copy(delated_seg)
    # for [y, x] in temp_pt:
    # 	dseg_img2[y, x] = 60
    #
    # plt.figure('temp_pt');
    # plt.imshow(cv2.flip(dseg_img2, 0))

    # # asdf........
    # if np.unique(delated_seg) == delated_seg2 :
    # 	for idx in neighbored_list:
    # 		temp_pt = np.argwhere(delated_seg == idx)
    # 		if pt_max < temp_pt.size / 2:
    # 			pt_max = temp_pt.size / 2
    # 			neighbor_obj = idx
    # else:
    # 	for idx in neighbored_list:
    # 		None

    # asdf..........

    # label : 0 = neigbho
    label = np.zeros(int(n_size + pt_max))  # 타겟 크기반으로 쪼갠 것에 1/2크기를 추가하여 저장
    label[n_size:] = 1  # 그 중 반타겟크기 이후를 1로 저장 = 주변물체 1, 타겟 = 0
    # = 다이얼레이션 진행한 세그맨테이션에서 주변집중물체를 찾음, 그리고 타겟 오브젝트의 세그이미지에 합함
    data = np.concatenate((target_pt, np.argwhere(np.array(delated_seg) == neighbor_obj)))

    temp = np.array([data[:, 1], data[:, 0]]).transpose()  # 좌표로 변환

    temp_list = temp.tolist()
    binary_label_array = np.zeros(shape=(256, 256), dtype=np.uint8) + 127
    index = 0
    for [y, x] in temp_list:
        binary_label_array.itemset((y, x), label[index] * 255)
        index = index + 1

    cv2.imshow("binary_label_array", binary_label_array)
    cv2.waitKey(1)
    cv2.imwrite("{}_binary_label_array".format(neighbor_obj) + ".png", binary_label_array)

    # svm = SVC(kernel='linear', random_state=0, gamma=0.1, C=1)
    svm = SVC(kernel='linear', random_state=0, gamma=0.0001, C=1000)
    svm.fit(temp, label)

    # Target path, size
    target_path = plot_decision_regions(classifier=svm)

    print(target_path.shape)
    target_path = target_path[0]
    t_path_len = (target_path.shape[0])

    # ??????? <-------------------------------------------------------
    # start_idx = 0
    # end_idx = 0
    # f_start = False
    # sim_img = np.copy(seg_img)
    # for idx, [y, x] in enumerate(target_path):
    #
    # 	sim_img[y, x] = 100
    # 	# plt.figure(101); plt.imshow(cv2.flip(sim_img, 0))
    #
    # 	# if seg_img[y, x] == neighbor_obj and f_start == False:
    # 	if seg_img[y, x] == neighbor_obj and start_idx == 0:
    # 		start_idx = idx
    # 		f_start = True
    #
    # 	if seg_img[y, x] != neighbor_obj and f_start:
    # 		end_idx = idx
    # 		f_start = False
    #
    # 	# seg_img[y, x] = 255
    # 	# plt.show(seg_img)

    # test = target_path[start_idx:end_idx]
    #
    # for [y, x] in test:
    # 	sim_img[y, x] = 50
    #
    # plt.figure(102); plt.imshow(cv2.flip(sim_img, 0))

    # ------- ------------------------------------------------------->

    # = 윗 코드의 실제 적용
    # Find space
    non_neighbor_upper = []  # 타겟 패스의 시작부터 검사, 패스 윗부분(첫부분) 저장공간
    non_neighbor_under = []  # ", 패스 아랫부분 (끝부분) 저장공간

    upper_idx = 0  # 패스 윗부분의 첫 시작점
    upper_n_idx = 0
    under_idx = 0  # 패스 아랫부분의 첫 시작점
    under_n_idx = 0

    # = 패스 검사 시작
    is_through = True
    is_through_n = True
    for idx, [y, x] in enumerate(target_path):
        if seg_img[y, x] not in [neighbor_obj, 16]:  # 검사하는 이미지가 = 주변 물체 혹은 배경이 아니라면,
            non_neighbor_upper.append(seg_img[y, x])  # 리스트에 그 좌표를 추가
            if is_through:  # 처음 측정된것이 발견되면
                upper_idx = idx  # 첫 포인트로 저장
                is_through = False  # 더이상 첫 포인트로 저장하지 않음
        # break

        elif seg_img[y, x] in [neighbor_obj]:  # 검사하는 이미지가 = 주변 물체 라면,
            non_neighbor_upper.append(seg_img[y, x])  # 리스트에 좌표를 추가
            if is_through_n:
                upper_n_idx = idx
                is_through_n = False
            break

    # 패스 역방향 검사 시작
    is_through = True
    is_through_n = True
    for idx, [y, x] in enumerate(target_path[-1::-1]):
        if seg_img[y, x] not in [neighbor_obj, 16]:
            non_neighbor_under.append(seg_img[y, x])
            if is_through:
                under_idx = t_path_len - idx
                is_through = False
        # break

        elif seg_img[y, x] in [neighbor_obj]:
            non_neighbor_under = np.unique(non_neighbor_under)
            if is_through_n:
                under_n_idx = t_path_len - idx
                is_through_n = False
            break

    # = 패스에 다른물체들이 검출되지 않을경우, 주변물체가 첫 시작점
    if upper_idx == 0:
        upper_idx = upper_n_idx

    if under_idx == 0:
        under_idx = under_n_idx

    if upper_n_idx == 0:
        upper_n_idx = upper_idx

    if under_n_idx == 0:
        under_n_idx = under_idx

    # = 어떤 물체들이 걸렸나 저장
    non_neighbor_under = np.unique(non_neighbor_under)
    non_neighbor_upper = np.unique(non_neighbor_upper)

    # = 지나가는 패스에 물체가 얼마나 많은지 체크 후 결정
    if non_neighbor_upper.size < non_neighbor_under.size:
        start = upper_idx
        end = under_n_idx
    elif non_neighbor_upper.size > non_neighbor_under.size:
        start = under_idx
        end = upper_n_idx
    else:  # ????
        if abs(88 - target_path[upper_idx][0]) < (88 - target_path[under_idx][0]):
            start = upper_idx
            end = under_n_idx
        else:
            start = under_idx
            end = upper_n_idx

    if start > end:
        target_path = target_path[::-1]
        start, end = start, end

        start = t_path_len - start
        end = t_path_len - end

    # main Path
    chk_surr = False
    for pt in range(start):
        chk_cnt = 0
        for y, x in dil(target_path[start - pt], 2):
            try:
                if seg_img[y, x] != 16:
                    chk_surr = True
                    break
                chk_cnt += 1
            except:
                chk_surr = True
                break

        if chk_surr is True:
            chk_surr = False
            continue

        if chk_cnt == 8:
            start = (t_path_len + (start - pt)) % t_path_len
            break

    for pt in range(start):
        chk_cnt = 0
        t_img = np.copy(seg_img)
        for y, x in dil(target_path[start - pt], 2):
            try:
                if seg_img[y, x] != 16:
                    chk_surr = True
                    break
                chk_cnt += 1
            except:
                chk_surr = True
                break

        if chk_surr is True:
            chk_surr = False
            continue

        if chk_cnt == 8:
            start = (t_path_len + (start - pt)) % t_path_len
            break

    for pt in range(t_path_len - end):
        chk_cnt = 0
        for y, x in dil(target_path[end + pt], 2):
            try:
                if seg_img[y, x] != 16:
                    chk_surr = True
                    break

                chk_cnt += 1
            except:
                chk_surr = True
                break

        if chk_surr is True:
            chk_surr = False
            continue

        if chk_cnt == 8:
            end = int(((end + pt + 1) + end) / 2) % t_path_len
            break

    main_path = target_path[start - 6:end + 9]

    ## To Visualization,
    # t_img = np.copy(seg_img)
    # # sub line
    # for [y, x] in target_path:
    # 	t_img[y, x] = 15
    #
    # # main Path
    # for [y, x] in main_path:
    # 	t_img[y, x] = 30
    #
    # # # = 이걸 써야하는지... 흐음...
    # # if end > 255:
    # # 	end = 255
    #
    # t_img = cv2.circle(t_img, (target_path[start][1], target_path[start][0]), 2, 40, -1)
    # t_img = cv2.circle(t_img, (target_path[end][1], target_path[end][0]), 2, 40, -1)

    # plt.figure(90)
    # plt.imshow(t_img)

    # pre_angle = 0
    # angle = []

    # main_path = main_path[::3]
    #
    # for idx, _ in enumerate(main_path):
    #     if idx == (main_path.size / 2) -1:
    #         continue
    #
    #     ay = main_path[idx+1][0] - main_path[idx][0]
    #     ax = main_path[idx+1][1] - main_path[idx][1]
    #     # theta = np.round(np.arctan(ay), 4)  * 180/ np.pi
    #     theta_1 = int(np.arctan2(ay, ax)* 180/ np.pi)
    #
    #     print(theta_1)
    #     if idx != 0:
    #         angle.append(theta_1 - pre_angle)
    #         pre_angle = theta_1
    #     else:
    #         angle.append(theta_1)
    #         pre_angle = theta_1
    #
    # angle.append(0)

    return main_path  # , angle
