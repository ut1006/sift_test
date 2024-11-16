import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import open3d as o3d

# カメラ内部パラメータ
fx, fy = 1400.6, 1400.6
cx1, cy = 1103.65, 574.575
baseline = 62.8749  # mm

def depth_to_3d(x, y, disparity):
    """
    左画像のSIFT特徴点 (x, y) と対応する視差から3次元座標を計算。
    """
    depth = (fx * baseline) / (disparity[int(y), int(x)] + 1e-6)  # mm
    X = (x - cx1) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z]) / 1000  # Convert to meters

def detect_and_match_sift(image1, image2):
    """
    SIFT特徴点を検出し、特徴量マッチングを実行。
    """
    sift = cv2.SIFT_create()
    print("SIFT特徴点の検出を開始します。")
    
    # 特徴点と特徴量の検出
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    print(f"画像1の特徴点数: {len(keypoints1)}, 画像2の特徴点数: {len(keypoints2)}")

    # 特徴量のマッチング
    print("特徴量のマッチングを実行中...")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"マッチング結果の数: {len(matches)}")
    # マッチング結果をスコア順にソート
    matches = sorted(matches, key=lambda x: x.distance)

    # マッチングを可視化
    img_matches = cv2.drawMatches(
        image1, keypoints1,  # 最初の画像とその特徴点
        image2, keypoints2,  # 次の画像とその特徴点
        matches[:200],      # 上位20個のマッチ
        None,              # マッチを描画する画像（新規作成）
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # 特徴点だけではなく対応を描画
    )

    # 結果を表示
    cv2.imshow('Feature Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return keypoints1, keypoints2, matches

def calculate_3d_points(keypoints1, keypoints2, matches, disp1, disp2):
    """
    SIFT特徴点のマッチング結果を利用して、3次元対応点を計算。
    """
    points1 = []
    points2 = []
    
    print("3次元対応点の計算を開始します。")
    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        x1, y1 = keypoints1[idx1].pt
        x2, y2 = keypoints2[idx2].pt

        # 左画像の視差情報から3次元座標を計算
        try:
            p1 = depth_to_3d(x1, y1, disp1)
            p2 = depth_to_3d(x2, y2, disp2)
            points1.append(p1)
            points2.append(p2)
        except IndexError: 
            # 視差画像の境界を超える特徴点をスキップ
            continue

    print(f"有効な3次元対応点数: {len(points1)}")
    return np.array(points1), np.array(points2)
def estimate_transformation_ransac(points1, points2):
    # Create Open3D point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd2.points = o3d.utility.Vector3dVector(points2)

    # Generate correspondences (assuming a direct mapping here, adjust as needed)
    correspondences = np.arange(len(points1)).reshape(-1, 2)
    corres = o3d.utility.Vector2iVector(correspondences)

    # Set up RANSAC
    distance_threshold = 0.05
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd1, pcd2, corres, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # ransac_n: number of points to sample
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    # 推定された変換行列を取得
    transformation = result.transformation

    # インライアの数を取得
    inlier_count = len(result.correspondence_set)

    print("推定された変換行列:")
    print(transformation)
    print(f"インライア数: {inlier_count}")
    return transformation, inlier_count

def main():
    # 画像ファイルと視差ファイル（深度情報が格納された.npy）
    image1_file = "l0014.png"
    image2_file = "l0018.png"
    disp1_file = "0014.npy"
    disp2_file = "0018.npy"

    print("データの読み込みを開始します。")
    # データ読み込み
    image1 = cv2.imread(image1_file, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_file, cv2.IMREAD_GRAYSCALE)
    disp1 = np.load(disp1_file)
    disp2 = np.load(disp2_file)
    print("データの読み込みが完了しました。")

    # SIFT特徴点の検出とマッチング
    keypoints1, keypoints2, matches = detect_and_match_sift(image1, image2)

    # 3次元対応点の計算
    points1, points2 = calculate_3d_points(keypoints1, keypoints2, matches, disp1, disp2)

    # 座標変換の推定
    transformation = estimate_transformation_ransac(points1, points2)

    print("推定された変換行列:")
    print(transformation)

if __name__ == "__main__":
    main()
