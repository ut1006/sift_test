import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_feature_matching(image1_path, image2_path):
    # 画像の読み込み (グレースケール)
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # SIFTインスタンスの作成
    sift = cv2.SIFT_create()

    # 特徴点と特徴量の検出
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 特徴点マッチング用のインスタンス (BFMatcher with L2 Norm)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # マッチングを実行
    matches = bf.match(descriptors1, descriptors2)

    # 距離でソート（距離が短いほど良いマッチング）
    matches = sorted(matches, key=lambda x: x.distance)

    # 上位N個のマッチングを選択
    num_matches = 50
    good_matches = matches[:num_matches]

    # マッチング結果を描画
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, good_matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 結果を表示
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches, cmap='gray')
    plt.title("Feature Matching with SIFT")
    plt.axis('off')
    plt.show()

# 使用する画像のパスを指定
image1_path = "l0014.png"  # 1枚目の画像
image2_path = "l0018.png"  # 2枚目の画像

# SIFT特徴量のマッチング
sift_feature_matching(image1_path, image2_path)
