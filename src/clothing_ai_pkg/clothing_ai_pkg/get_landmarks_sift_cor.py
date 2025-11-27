#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from custom_interfaces_pkg.srv import GetLandmarksSiftCor
from sensor_msgs.msg import Image
import threading
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from custom_interfaces_pkg.msg import Landmark
import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage
from scipy.spatial import KDTree
import random


class GetLandmarksNode(Node):
    def __init__(self):
        super().__init__('get_landmarks_node')

        # Image update lock
        self.lock = threading.Lock()
        self.image = None
        self.bridge = CvBridge()
        
        # Load Model
        model_name = "model.pth"
        self.pkg_path = get_package_share_directory('clothing_ai_pkg')
        model_path = os.path.join(self.pkg_path, 'data', model_name)
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file not found at {model_path}")
            exit()
        
        self.num_landmarks = 25
        self.data_per_landmark = 3  # x and y, visibility
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 224
        self.model = self.load_model(model_path)
        
        # ROS2 Subscribers and Services
        self.sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        self.srv = self.create_service(
            GetLandmarksSiftCor,
            '/get_landmarks_sift_cor_srv',
            self.get_landmarks
        )
        
        self.keypoints = {1: None, 
                     2: None,
                     3: None,
                     4: None,
                     5: None}
        
        self.landmarks = {1: None,
                          2: None,
                          3: None,
                          4: None,
                          5: None}
        
        self.images = {1:None,
                       2:None, 
                       3:None,
                       4:None,
                       5:None}
        
        self.last_step = 0
        self.sift = cv2.SIFT_create()

        self.get_logger().info("get_landmarks node ready and providing /get_landmarks_sift_cor_srv.")

    def load_model(self, model_path):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_landmarks * self.data_per_landmark)
    
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Check if it's a checkpoint dict or just state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Remove 'backbone.' prefix if present
                if all(k.startswith('backbone.') for k in state_dict.keys()):
                    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                self.get_logger().info(f"✅ Loaded model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                self.get_logger().info(f"✅ Loaded model from {model_path}")
        except FileNotFoundError:
            self.get_logger().error(f"❌ Model not found at {model_path}")
            exit()
            
        model.to(self.device)
        model.eval()
        return model
        
    def image_callback(self, msg):
        with self.lock:
            self.image = msg
    
    def get_landmarks(self, request, response):
        cv_image = None
        cv_crop = None
        diff = 1280 - 720
        left_crop = int(1/4*diff)
        right_crop = int(3/4*diff)

        # --------- FETCH MOST RECENT IMAGE ----------
        with self.lock:
            if self.image is not None:
                cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')

                # Crop center 720 × 720
                cv_crop = cv_image[:, left_crop:1280-right_crop]

                # Rotate 180° (same as your inference input)
                cv_crop = cv2.rotate(cv_crop, cv2.ROTATE_180)

            else:
                self.get_logger().warning("No image available yet")
                return response

        # ============================================================
        # STEP 1  →  DETECT LANDMARKS + SAVE SIFT
        # ============================================================
        if request.step_number == 1:

            # 1) Run landmark inference on 180° rotated crop
            final_preds = self.run_inference(cv_crop)
            landmarks_list = [Landmark(x=float(p[0]), y=float(p[1])) for p in final_preds]

            # 2) Select only needed landmarks
            landmarks_list = [
                landmarks_list[14],
                landmarks_list[9],
                landmarks_list[8],
                landmarks_list[1],
                landmarks_list[5],
                landmarks_list[22],
                landmarks_list[21],
                landmarks_list[16]
            ]

            # 3) Convert coordinates BACK to full image coords with center adjustment
            crop_width = 720
            crop_height = 720
            landmarks_corrected = []
            landmarks_not_moved =[]

            for lm in landmarks_list:
                # Reverse 180 degree rotation
                x_cropped = crop_width - lm.x
                y_cropped = crop_height - lm.y

                landmarks_not_moved.append(Landmark(x=float(x_cropped + left_crop), y = float(y_cropped)))

                # Move points slightly towards center to avoid edge issues
                point = np.array([x_cropped, y_cropped])
                image_center = np.array([crop_width/2, crop_height/2])
                vec = (image_center - point)
                new_point = point + vec/np.linalg.norm(vec) * 80  # Slightly move points towards center

                # Add the left crop offset to get back to original image coordinates
                x_original = new_point[0] + left_crop
                y_original = new_point[1]

                landmarks_corrected.append(Landmark(x=float(x_original), y=float(y_original)))
            # Return and store
            response.landmarks = landmarks_corrected

            # 4) Compute SIFT on the rotated crop (same coordinate frame)
            kp, des = self.sift.detectAndCompute(cv_crop, None)
            self.keypoints[1] = (kp, des)
            self.landmarks[1] = landmarks_corrected
            self.last_step = 1
            self.images[1] = cv_crop

            new_image = cv_image.copy()
            # Visual output
            self.annotate_image(cv_image, landmarks_corrected, "landmarks_step1")

            self.annotate_image(new_image, landmarks_not_moved, "landmarks_step1_not_moved")
            return response

        # ============================================================
        # STEP N > 1  →  TRACK LANDMARKS WITH LOCAL HOMOGRAPHIES
        # ============================================================
        elif request.step_number == 5 and self.last_step > 0:
            response.landmarks = self.landmarks[1]
            return response
        
        elif self.last_step > 0:
            # Get SIFT in the NEW frame (on rotated crop)
            kp_curr, des_curr = self.sift.detectAndCompute(cv_crop, None)
            self.keypoints[request.step_number] = (kp_curr, des_curr)

            # Previous frame SIFT
            kp_prev, des_prev = self.keypoints[self.last_step]

            # ----------- CORRECT MATCH DIRECTION --------------------
            # Previous → Current
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des_prev, des_curr, k=2)

            # Lowe's ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # Need at least 8 matches for reliability
            if len(good) < 8:
                self.get_logger().warn("Not enough SIFT matches!")
                return response

            # Build matched coordinate sets
            pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
            pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good])
            
            # -------------------------------
            # Draw current frame keypoints
            # -------------------------------
            num_keypoints = len(kp_curr)
            image_name = f"keypoints_step{request.step_number}_n_{num_keypoints}"
            self.draw_keypoints(cv_crop, kp_curr, save_name=image_name)
                        
                        # KD-tree for local matching
            tree = KDTree(pts_prev)

            # -------- Convert stored full-frame landmarks → crop-rot frame ----------
            selected_px_crop_rot = []
            for lm in self.landmarks[self.last_step]:

                # Convert full → crop
                x_crop = lm.x - left_crop
                y_crop = lm.y

                # Convert crop → rotated crop
                x_rot = 720 - x_crop
                y_rot = 720 - y_crop

                selected_px_crop_rot.append((x_rot, y_rot))

            # -------- Track with local homography --------
            tracked_crop_rot = []
            for px in selected_px_crop_rot:
                new_px, ok = self.track_pixel_local_homography(px, pts_prev, pts_curr, tree, k=12)
                
                if ok:
                    # Ensure Python floats
                    new_x = float(new_px[0])
                    new_y = float(new_px[1])
                    tracked_crop_rot.append((new_x, new_y))
                else:
                    tracked_crop_rot.append((-1.0, -1.0))

            # -------- Convert back: rotated crop → full frame --------
            tracked_full = []
            for (x_rot, y_rot) in tracked_crop_rot:

                x_rot = float(x_rot)
                y_rot = float(y_rot)

                if x_rot < 0:
                    tracked_full.append(Landmark(x=float(-1), y=float(-1)))
                    continue

                x_crop = float(720 - x_rot - 1)
                y_crop = float(720 - y_rot - 1)

                x_full = float(x_crop + left_crop)
                y_full = float(y_crop)

                tracked_full.append(Landmark(x=x_full, y=y_full))


            # return
            
            # -------------------------------
            # Draw current frame keypoints
            # -------------------------------
            num_keypoints = len(kp_curr)
            image_name = f"keypoints_step{request.step_number}_n_{num_keypoints}"
            self.draw_keypoints(cv_crop, kp_curr, save_name=image_name)

            # Build KD-tree on current keypoints for local matching
            pts_curr_array = np.float32([kp.pt for kp in kp_curr])
            tree_curr = KDTree(pts_curr_array)

            # Draw landmarks and their 12 closest matched keypoints
            landmarks_for_draw = tracked_full
            self.draw_landmarks_with_matches(cv_crop, landmarks_for_draw, pts_curr_array, tree_curr, k=12,
                                            save_name=f"landmarks_and_matches_step{request.step_number}")

            # Draw all matches between previous and current frame
            self.draw_all_matches(self.images[self.last_step], cv_crop, kp_prev, kp_curr, good,
                                save_name=f"all_matches_step{request.step_number}")

                        
            self.last_step = request.step_number
            self.landmarks[request.step_number] = tracked_full
            self.images[request.step_number] = cv_crop
            response.landmarks = tracked_full
            name = f"landmarks_step{request.step_number}"
            self.annotate_image(cv_image, tracked_full,image_name=name)
            return response

        else:
            self.get_logger().warning("No previous frame available")
            return response

    
    def track_pixel_local_homography(self, px, pts1, pts2, tree, k=12):
        """
        px = (x, y)
        k = number of nearest SIFT matches used for local model
        """
        # find k nearest matched SIFT features
        dists, idxs = tree.query(px, k=k)

        local_src = pts1[idxs]   # matched points in img1
        local_dst = pts2[idxs]   # matched points in img2

        # need at least 4 points
        if len(local_src) < 4:
            return None, False

        # compute local homography
        H, mask = cv2.findHomography(local_src, local_dst, cv2.RANSAC, 3.0)
        
        if H is None:
            return None, False

        # apply H to pixel
        p = np.array([[px]], dtype=np.float32)  # shape (1,1,2)
        p2 = cv2.perspectiveTransform(p, H)[0][0]

        return (float(p2[0]), float(p2[1])), True
        
    def draw_landmarks_with_matches(self, image, landmarks, pts_curr, tree, k=12, save_name="landmarks_matches"):
        """
        Draws landmarks and their k nearest SIFT keypoints in the current frame.
        Each landmark and its keypoints get a unique color. Unused keypoints are gray.
        
        Args:
            image: np.ndarray (BGR)
            landmarks: list of Landmark objects
            pts_curr: np.ndarray of shape (N,2) keypoints in current frame
            tree: KDTree built on pts_curr
            k: number of nearest keypoints per landmark
            save_name: filename to save
        """
        vis = image.copy()
        used_idxs = set()
        
        # Generate unique colors for each landmark
        colors = [tuple(np.random.randint(0,255,3).tolist()) for _ in landmarks]

        for idx, lm in enumerate(landmarks):
            x, y = lm.x, lm.y
            color = colors[idx]

            # Draw landmark
            cv2.circle(vis, (int(x), int(y)), 6, color, -1)
            cv2.putText(vis, f"L{idx}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Find k nearest keypoints
            if tree is not None:
                dists, nn_idxs = tree.query([x, y], k=k)
                if k == 1:  # ensure iterable
                    nn_idxs = [nn_idxs]
                for kp_idx in nn_idxs:
                    used_idxs.add(kp_idx)
                    kp = pts_curr[kp_idx]
                    cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, color, -1)

        # Draw unused keypoints in gray
        for i, kp in enumerate(pts_curr):
            if i not in used_idxs:
                cv2.circle(vis, (int(kp[0]), int(kp[1])), 2, (150, 150, 150), -1)

        # Save
        save_path = os.path.join(os.path.expanduser("~"), "clothing_debug", save_name + ".png")
        cv2.imwrite(save_path, vis)
        print(f"Saved landmark+keypoint visualization → {save_path}")


    def draw_all_matches(self, img_prev, img_curr, kp_prev, kp_curr, matches, save_name="all_matches"):
        """
        Draws all matches between previous and current frame.
        """
        # Convert keypoints to cv2.KeyPoint objects if needed
        # matches: list of cv2.DMatch
        img_matches = cv2.drawMatches(
            img_prev, kp_prev, img_curr, kp_curr, matches, None,
            matchColor=(0,255,0), singlePointColor=(255,0,0), flags=2
        )

        save_path = os.path.join(os.path.expanduser("~"), "clothing_debug", save_name + ".png")
        cv2.imwrite(save_path, img_matches)
        print(f"Saved all matches visualization → {save_path}")

        
    def run_inference(self, image):
        # Convert OpenCV image (BGR) to PIL RGB
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(image)
        else:
            raise RuntimeError("Invalid image type passed to run_inference")
        
        # Transform and create tensor
        tfms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])
        img_tensor = tfms(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            preds = self.model(img_tensor).view(self.num_landmarks, self.data_per_landmark).cpu()

        # Extract only x, y coordinates (ignore visibility)
        preds_xy = preds[:, :2]

        # Scale model output [0-1] -> pixel coordinates
        final_preds = preds_xy.clone()
        final_preds[:, 0] *= pil_image.width
        final_preds[:, 1] *= pil_image.height

        landmarks = final_preds.tolist()
        return landmarks

    def annotate_image(self, image, landmarks, image_name):

        # -------------------------------
        # 1. Choose a dynamic save folder
        # -------------------------------
        save_dir = os.path.join(os.path.expanduser("~"), "clothing_debug")

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # -------------------------------
        # 2. Draw landmarks
        # -------------------------------
        for idx, lm in enumerate(landmarks):
            x = int(lm.x)
            y = int(lm.y)

            # Draw point
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # Coordinates
            cv2.putText(
                image, f"({x},{y})", (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
            )

            # Index
            cv2.putText(
                image, f"{idx}", (x-10, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        # -------------------------------
        # 3. Save image
        # -------------------------------
        save_path = os.path.join(save_dir, image_name + '.png')
        cv2.imwrite(save_path, image)

        self.get_logger().info(f"Saved visualization → {save_path}")
    
    def draw_keypoints(self, image, keypoints, save_name=None):
        """
        Draws a list of OpenCV KeyPoint objects on the image.

        Args:
            image (np.ndarray): The image to draw on (BGR).
            keypoints (list of cv2.KeyPoint): List of keypoints.
            save_name (str, optional): If provided, saves the annotated image
                to ~/clothing_debug/save_name.png
        """
        img_copy = image.copy()

        for idx, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])

            # Draw point
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

            # Coordinates text
            cv2.putText(
                img_copy, f"({x},{y})", (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1
            )

            # Index text
            cv2.putText(
                img_copy, f"{idx}", (x-10, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )

        if save_name:
            save_dir = os.path.join(os.path.expanduser("~"), "clothing_debug")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_name + ".png")
            cv2.imwrite(save_path, img_copy)
            print(f"Saved visualization → {save_path}")

        return img_copy

        
    
def main(args=None):
    rclpy.init(args=args)
    node = GetLandmarksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
