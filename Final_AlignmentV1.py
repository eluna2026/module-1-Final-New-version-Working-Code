import sys
import os
import inspect
import time
import numpy as np
import cv2

# --- Newport & Pylon Imports ---
import clr
from pypylon import pylon 
from System.Text import StringBuilder

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
    QHBoxLayout, QPushButton, QTabWidget, QToolButton, QDialog,
    QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QDialogButtonBox, QFrame, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

# ==========================================
# NEWPORT DLL CONFIGURATION
# ==========================================
strCurrFile = os.path.abspath(inspect.stack()[0][1])
strPathDllFolder = os.path.dirname(strCurrFile)
sys.path.append(strPathDllFolder)

try:
    clr.AddReference("UsbDllWrap")
    from Newport.USBComm import *
except Exception as e:
    print(f"Warning: Could not load Newport DLL. {e}")

# ==========================================
# 1. IMAGE & HARDWARE FUNCTIONS 
# ==========================================
def find_iris_grid(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100, 
        param1=50, param2=30, minRadius=50, maxRadius=400
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        first_circle = circles[0][0]
        iris_x, iris_y, radius = int(first_circle[0]), int(first_circle[1]), int(first_circle[2])
        
        cv2.circle(image, (iris_x, iris_y), radius, (0, 255, 0), 2)
        cv2.circle(image, (iris_x, iris_y), 5, (0, 0, 255), -1)
        return iris_x, iris_y, radius
    return None, None, None

def find_laser_center(image, iris_x=None, iris_y=None, radius=None):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    if iris_x is not None and iris_y is not None and radius is not None:
        mask = np.zeros_like(gray)
        cv2.circle(mask, (iris_x, iris_y), int(radius - 15), 255, -1)
        gray = cv2.bitwise_and(gray, mask)
        
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    
    if max_val < 200: return None, None

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    
    if M["m00"] != 0: 
        laser_x, laser_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        laser_x, laser_y = int(max_loc[0]), int(max_loc[1])
    
    cv2.drawMarker(image, (laser_x, laser_y), (255, 0, 0), cv2.MARKER_CROSS, 40, 2)
    return laser_x, laser_y

def move_motor_no_wait(oUSB, strDeviceKey, channel, steps):
    if steps == 0: return
    strBldr = StringBuilder(128)
    oUSB.Query(strDeviceKey, f"{channel}PR{steps}", strBldr)

def stop_motors(oUSB, strDeviceKey, ch1, ch2):
    strBldr = StringBuilder(128)
    try:
        oUSB.Query(strDeviceKey, f"{ch1}ST", strBldr)
        oUSB.Query(strDeviceKey, f"{ch2}ST", strBldr)
    except: pass

def adjust_hardware_alignment(oUSB, strDeviceKey, delta_x, delta_y, actuator_num):
    error_distance = max(abs(delta_x), abs(delta_y))

    # --- ASYMMETRIC DEADBANDS ---
    # Stops the infinite hunting loop by respecting the physical optical lever arm.
    if actuator_num == 1 and error_distance <= 10:
        return 0.0
    elif actuator_num == 2 and error_distance <= 12:
        return 0.0

    if error_distance < 15:
        GAIN = 0.5
        MIN_STEPS = 1
        MAX_STEPS = 3    
    elif error_distance < 30:
        GAIN = 0.75  
        MIN_STEPS = 2  
        MAX_STEPS = 20   
    else:
        GAIN = 1.0  
        MIN_STEPS = 15  
        MAX_STEPS = 600  

    if actuator_num == 1:
        CH_X, CH_Y = 1, 2
        STEPS_PER_PIXEL_X, STEPS_PER_PIXEL_Y = 20.0, -20.0
    else:
        CH_X, CH_Y = 3, 4
        STEPS_PER_PIXEL_X, STEPS_PER_PIXEL_Y = -2.0, -2.0
        MAX_STEPS = min(MAX_STEPS, 4)

    steps_x = int(delta_x * STEPS_PER_PIXEL_X * GAIN)
    steps_y = int(delta_y * STEPS_PER_PIXEL_Y * GAIN)

    if 0 < steps_x < MIN_STEPS: steps_x = MIN_STEPS
    elif 0 > steps_x > -MIN_STEPS: steps_x = -MIN_STEPS
    elif steps_x > MAX_STEPS: steps_x = MAX_STEPS
    elif steps_x < -MAX_STEPS: steps_x = -MAX_STEPS

    if 0 < steps_y < MIN_STEPS: steps_y = MIN_STEPS
    elif 0 > steps_y > -MIN_STEPS: steps_y = -MIN_STEPS
    elif steps_y > MAX_STEPS: steps_y = MAX_STEPS
    elif steps_y < -MAX_STEPS: steps_y = -MAX_STEPS

    if steps_x != 0: move_motor_no_wait(oUSB, strDeviceKey, CH_X, steps_x)
    if steps_y != 0: move_motor_no_wait(oUSB, strDeviceKey, CH_Y, steps_y)

    max_steps_taken = max(abs(steps_x), abs(steps_y))
    return (max_steps_taken / 2000.0) + 0.05

# ==========================================
# 2. THE HARDWARE THREAD (The Bridge)
# ==========================================
class HardwareThread(QThread):
    frame_ready = pyqtSignal(int, np.ndarray)  
    log_msg = pyqtSignal(str)                  
    status_msg = pyqtSignal(str)               
    laser_pos_update = pyqtSignal(int, int, int) 

    def __init__(self):
        super().__init__()
        self.is_running = True
        
        self.is_aligning = False  
        self.was_aligning = False
        self.alignment_cooldown = 0.0 
        
        self.manual_target_active = False
        self.was_manual_aligning = False
        self.manual_cam_idx = 0
        self.manual_x = 0
        self.manual_y = 0
        
        self.cam1_locked = False
        self.cam2_locked = False
        self.sentry_timer = 0.0  
        
        self.cam1_stable_count = 0
        self.cam2_stable_count = 0
        
        self.cam1_drift_count = 0
        self.cam2_drift_cont = 0
        self.system_locked_stop_sent = False
        
        self.save_images = False
        self.save_interval = 3
        self.last_save_time = time.time()

    def update_camera_settings(self, exposure, gain):
        if hasattr(self, 'cameras') and self.cameras.IsOpen():
            for i, cam in enumerate(self.cameras):
                try: cam.ExposureTime.SetValue(float(exposure))
                except: 
                    try: cam.ExposureTimeAbs.SetValue(float(exposure))
                    except: pass
                try: cam.Gain.SetValue(float(gain))
                except: 
                    try: cam.GainRaw.SetValue(int(gain))
                    except: pass
            self.log_msg.emit(f"Hardware updated: Exposure={exposure}, Gain={gain}")

    def execute_manual_move(self, cam_idx, x, y):
        self.is_aligning = False 
        self.manual_target_active = True
        self.manual_cam_idx = cam_idx
        self.manual_x = x
        self.manual_y = y

    def stop_all_movement(self):
        self.is_aligning = False
        self.manual_target_active = False
        self.status_msg.emit("All movement stopped. Holding position.")

    def run(self):
        # --- SPLIT TOLERANCES ---
        CAM1_TOLERANCE_PX = 10  
        CAM2_TOLERANCE_PX = 12  
        DRIFT_TOLERANCE_PX = 22 # Bumped slightly so normal noise doesn't instantly wake it up
        
        STABLE_FRAMES_REQUIRED = 5
        DRIFT_FRAMES_REQUIRED = 5

        try:
            self.oUSB = USB(True)
            if not self.oUSB.OpenDevices(0, True):
                self.log_msg.emit("ERROR: Could not open Newport USB devices.")
                return
            oDeviceTable = self.oUSB.GetDeviceTable()
            oEnumerator = oDeviceTable.GetEnumerator()
            oEnumerator.MoveNext()
            self.strDeviceKey = str(oEnumerator.Key)
            self.log_msg.emit(f"Connected Newport Controller: {self.strDeviceKey}")
        except Exception as e:
            self.log_msg.emit(f"Hardware init skipped or failed: {e}")
            return

        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            CAM1_SN = "25191527" 
            CAM2_SN = "25191524" 
            
            self.cameras = pylon.InstantCameraArray(2)
            cam1_found, cam2_found = False, False
            for dev in devices:
                sn = dev.GetSerialNumber()
                if sn == CAM1_SN:
                    self.cameras[0].Attach(tlFactory.CreateDevice(dev))
                    self.cameras[0].SetCameraContext(0)
                    cam1_found = True
                elif sn == CAM2_SN:
                    self.cameras[1].Attach(tlFactory.CreateDevice(dev))
                    self.cameras[1].SetCameraContext(1)
                    cam2_found = True
                    
            if not (cam1_found and cam2_found):
                self.log_msg.emit("ERROR: Could not find both cameras!")
                return

            self.cameras.Open()
            for i, cam in enumerate(self.cameras):
                try: cam.ExposureAuto.SetValue("Off")
                except: pass
                try: cam.ExposureTime.SetValue(7000.0)
                except: 
                    try: cam.ExposureTimeAbs.SetValue(7000.0)
                    except: pass
                try: cam.GainAuto.SetValue("Off")
                except: pass
                try: cam.Gain.SetValue(0.0)
                except: 
                    try: cam.GainRaw.SetValue(0)
                    except: pass

                try: cam.AcquisitionFrameRateEnable.SetValue(True)
                except: pass
                try: cam.AcquisitionFrameRate.SetValue(30.0)
                except: 
                    try: cam.AcquisitionFrameRateAbs.SetValue(30.0)
                    except: pass

            self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByUser)
            self.log_msg.emit("Cameras started at 30 FPS. Waiting for frames...")
            latest_frames = {0: None, 1: None}

            while self.is_running and self.cameras.IsGrabbing():
                grab_result = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                if grab_result.GrabSucceeded():
                    cam_idx = grab_result.GetCameraContext()
                    raw_image = grab_result.Array
                    image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR) if len(raw_image.shape) == 2 else raw_image.copy()
                    latest_frames[cam_idx] = image
                grab_result.Release()

                if latest_frames[0] is not None and latest_frames[1] is not None:
                    img1, img2 = latest_frames[0], latest_frames[1]

                    i1_x, i1_y, r1 = find_iris_grid(img1)
                    l1_x, l1_y = find_laser_center(img1, i1_x, i1_y, r1)
                    err1_x, err1_y = 0, 0
                    if i1_x is not None and l1_x is not None:
                        self.laser_pos_update.emit(0, l1_x, l1_y)
                        err1_x, err1_y = l1_x - i1_x, l1_y - i1_y
                        cv2.line(img1, (i1_x, i1_y), (l1_x, l1_y), (0, 255, 255), 2)

                    i2_x, i2_y, r2 = find_iris_grid(img2)
                    l2_x, l2_y = find_laser_center(img2, i2_x, i2_y, r2)
                    err2_x, err2_y = 0, 0
                    if i2_x is not None and l2_x is not None:
                        self.laser_pos_update.emit(1, l2_x, l2_y)
                        err2_x, err2_y = l2_x - i2_x, l2_y - i2_y
                        cv2.line(img2, (i2_x, i2_y), (l2_x, l2_y), (0, 255, 255), 2)
                        
                    err_dist1 = max(abs(err1_x), abs(err1_y)) if l1_x is not None else 999
                    err_dist2 = max(abs(err2_x), abs(err2_y)) if l2_x is not None else 999

                    err_pct1 = 0.0
                    err_pct2 = 0.0

                    if r1 is not None and r1 > 0:
                        err_mag1 = (err1_x ** 2 + err1_y ** 2) ** 0.5
                        err_pct1 = (err_mag1 / r1) * 100

                    if r2 is not None and r2 > 0:
                        err_mag2 = (err2_x ** 2 + err2_y ** 2) ** 0.5
                        err_pct2 = (err_mag2 / r2) * 100

                    if self.is_aligning:
                        if self.cam1_locked and self.cam2_locked:
                            if not self.system_locked_stop_sent:
                                stop_motors(self.oUSB, self.strDeviceKey, 1, 2)
                                stop_motors(self.oUSB, self.strDeviceKey, 3, 4)
                                self.system_locked_stop_sent = True
                                self.log_msg.emit("System aligned. Motors stopped; monitoring for drift.")

                            if err_dist1 > DRIFT_TORLERANCE_PX:
                                self.cam1_drift_count += 1
                            else:
                                self.cam1_drift_count = 0

                            if self.cam1_drift_count >= DRIFT_FRAMES_REQUIRED:
                                self.cam1_locked = False
                                self.cam1_stable_count = 0
                                self.cam1_drift_count = 0
                                self.system_locked_stop_sent = False
                                self.log_msg.emit("Camera 1 drift detected. Re-aligning actuator 1.")

                            if self.cam2_drift_count >= DRIFT_FRAMES_REQUIRED:
                                self.cam2_locked = False
                                self.cam2_stable_count = 0
                                self.cam2_drift_count = 0
                                self.system_locked_stop_sent = False
                                self.log_msg.emit("Camera 2 drift detected. Re-aligning actuator 2.")
 
                                drift_detected = False
                                
                                if err_dist1 > DRIFT_TOLERANCE_PX:
                                    self.cam1_locked = False
                                    self.cam1_stable_count = 0
                                    drift_detected = True
                                if err_dist2 > DRIFT_TOLERANCE_PX:
                                    self.cam2_locked = False
                                    self.cam2_stable_count = 0
                                    drift_detected = True
                                    
                                if drift_detected:
                                    self.log_msg.emit("Drift detected! Waking up motors...")
                                    
                                self.sentry_timer = time.time() + 3.0
                        else:
                            if self.cam1_locked:
                                if err_dist1 > DRIFT_TOLERANCE_PX:
                                    self.cam1_locked = False
                                    self.cam1_stable_count = 0
                            else:
                                if err_dist1 <= CAM1_TOLERANCE_PX:
                                    self.cam1_stable_count += 1
                                    if self.cam1_stable_count >= STABLE_FRAMES_REQUIRED:
                                        self.cam1_locked = True
                                        stop_motors(self.oUSB, self.strDeviceKey, 1, 2)
                                        self.alignment_cooldown = time.time() + 0.5 
                                else:
                                    self.cam1_stable_count = 0
                                    
                            if self.cam2_locked:
                                if err_dist2 > DRIFT_TOLERANCE_PX:
                                    self.cam2_locked = False
                                    self.cam2_stable_count = 0
                            else:
                                if err_dist2 <= CAM2_TOLERANCE_PX:
                                    self.cam2_stable_count += 1
                                    if self.cam2_stable_count >= STABLE_FRAMES_REQUIRED:
                                        self.cam2_locked = True
                                        stop_motors(self.oUSB, self.strDeviceKey, 3, 4)
                                        self.alignment_cooldown = time.time() + 0.5 
                                else:
                                    self.cam2_stable_count = 0

                            if self.cam1_locked and self.cam2_locked:
                                self.sentry_timer = time.time() + 3.0

                        new_status = "" 
                        if l1_x is None:
                            cv2.putText(img1, "BEAM LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            new_status = "Beam lost! Waiting for manual intervention..."
                        elif not self.cam1_locked:
                            cv2.putText(img1, "ALIGNING M1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                            new_status = "Aligning Camera 1..."
                            if time.time() > self.alignment_cooldown:
                                self.log_msg.emit(
                                    f"Aligning Camera 1 | Beam: X= {l1_x}, Y={l1_y} | "
                                    f"dX={err1_x}px, dY={err1_y}px | Error={err_pct1:.2f}%"
                                )

                                cooldown = adjust_hardware_alignment(self.oUSB, self.strDeviceKey, err1_x, err1_y, 1)
                                self.alignment_cooldown = time.time() + cooldown
                        
                        elif self.cam1_locked and not self.cam2_locked:
                            cv2.putText(img1, "M1 LOCKED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            cv2.putText(img2, "ALIGNING M2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                            new_status = "Camera 1 Locked. Aligning Camera 2..."
                            if time.time() > self.alignment_cooldown:
                                
                                self.log_msg.emit(
                                    f"Aligning Camera 2 | Beam: X={l2_x}, Y={l2_y} | "
                                    f"dX={err2_x}px, dY={err2_y}px | Error={err_pct2:.2f}%"
                                )
                                
                                cooldown = adjust_hardware_alignment(self.oUSB, self.strDeviceKey, err2_x, err2_y, 2)
                                self.alignment_cooldown = time.time() + cooldown
                        elif self.cam1_locked and self.cam2_locked:
                            cv2.putText(img1, "SYSTEM LOCKED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            cv2.putText(img2, "SYSTEM LOCKED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            time_left = max(0, int(self.sentry_timer - time.time()))
                            new_status = f"System Aligned. Double-checking drift in {time_left}s..."

                        if not hasattr(self, 'current_status') or new_status != self.current_status:
                            self.status_msg.emit(new_status)
                            self.current_status = new_status
                            
                    elif self.manual_target_active:
                        new_status = ""
                        
                        if self.manual_cam_idx == 0:
                            if l1_x is None:
                                cv2.putText(img1, "BEAM LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                new_status = "Beam lost! Cannot complete manual move."
                            else:
                                err_x = l1_x - self.manual_x
                                err_y = l1_y - self.manual_y
                                dist = max(abs(err_x), abs(err_y))
                                
                                cv2.drawMarker(img1, (self.manual_x, self.manual_y), (0, 255, 0), cv2.MARKER_STAR, 30, 2)
                                cv2.line(img1, (self.manual_x, self.manual_y), (l1_x, l1_y), (0, 255, 255), 2)
                                
                                if dist <= CAM1_TOLERANCE_PX:
                                    self.manual_target_active = False
                                    stop_motors(self.oUSB, self.strDeviceKey, 1, 2)
                                    new_status = "Manual target reached!"
                                    self.log_msg.emit("Manual mode complete. Brakes engaged.")
                                else:
                                    new_status = "Moving to manual target on Camera 1..."
                                    if time.time() > self.alignment_cooldown:
                                        self.log_msg.emit(f"Manual Cam 1 Pos: X={l1_x}, Y={l1_y} | Target: X={self.manual_x}, Y={self.manual_y} -> Adjusting...")
                                        cooldown = adjust_hardware_alignment(self.oUSB, self.strDeviceKey, err_x, err_y, 1)
                                        self.alignment_cooldown = time.time() + cooldown
                                        
                        elif self.manual_cam_idx == 1:
                            if l2_x is None:
                                cv2.putText(img2, "BEAM LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                                new_status = "Beam lost! Cannot complete manual move."
                            else:
                                err_x = l2_x - self.manual_x
                                err_y = l2_y - self.manual_y
                                dist = max(abs(err_x), abs(err_y))
                                
                                cv2.drawMarker(img2, (self.manual_x, self.manual_y), (0, 255, 0), cv2.MARKER_STAR, 30, 2)
                                cv2.line(img2, (self.manual_x, self.manual_y), (l2_x, l2_y), (0, 255, 255), 2)
                                
                                if dist <= CAM2_TOLERANCE_PX:
                                    self.manual_target_active = False
                                    stop_motors(self.oUSB, self.strDeviceKey, 3, 4)
                                    new_status = "Manual target reached!"
                                    self.log_msg.emit("Manual mode complete. Brakes engaged.")
                                else:
                                    new_status = "Moving to manual target on Camera 2..."
                                    if time.time() > self.alignment_cooldown:
                                        self.log_msg.emit(f"Manual Cam 2 Pos: X={l2_x}, Y={l2_y} | Target: X={self.manual_x}, Y={self.manual_y} -> Adjusting...")
                                        cooldown = adjust_hardware_alignment(self.oUSB, self.strDeviceKey, err_x, err_y, 2)
                                        self.alignment_cooldown = time.time() + cooldown

                        if not hasattr(self, 'current_status') or new_status != self.current_status:
                            self.status_msg.emit(new_status)
                            self.current_status = new_status

                    else:
                        if self.was_aligning or self.was_manual_aligning:
                            stop_motors(self.oUSB, self.strDeviceKey, 1, 2)
                            stop_motors(self.oUSB, self.strDeviceKey, 3, 4)
                    
                    self.was_aligning = self.is_aligning
                    self.was_manual_aligning = self.manual_target_active

                    if self.save_images and (time.time() - self.last_save_time >= self.save_interval):
                        cv2.imwrite(f"cam1_{int(time.time())}.png", img1)
                        cv2.imwrite(f"cam2_{int(time.time())}.png", img2)
                        self.log_msg.emit("Images saved to disk.")
                        self.last_save_time = time.time()

                    self.frame_ready.emit(0, img1)
                    self.frame_ready.emit(1, img2)
                    latest_frames = {0: None, 1: None} 

        except Exception as e:
            self.log_msg.emit(f"Camera Loop Error: {e}")

    def stop(self):
        self.is_running = False
        self.wait() 
        try:
            self.cameras.StopGrabbing()
            self.cameras.Close()
            self.oUSB.CloseDevices()
        except: pass

# ==========================================
# 3. GUI CLASSES
# ==========================================
class ClickableCameraView(QLabel):
    clicked = pyqtSignal(str, int, int)

    def __init__(self, camera_name):
        super().__init__()
        self.camera_name = camera_name
        self.setText(f"{camera_name}\nLoading camera feed...")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(350, 300)
        self.setStyleSheet("background-color: black; color: white; border: 2px solid gray; font-size: 16px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = int(event.position().x()), int(event.position().y())
            self.clicked.emit(self.camera_name, x, y)

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)
        layout = QFormLayout()

        self.save_images_checkbox = QCheckBox("Save images during alignment")
        self.save_interval_input = QSpinBox()
        self.save_interval_input.setRange(1, 60)
        self.save_interval_input.setSuffix(" sec")

        self.gain_input = QDoubleSpinBox()
        self.gain_input.setRange(0.0, 100.0)
        self.gain_input.setSingleStep(0.1)

        self.exposure_input = QSpinBox()
        self.exposure_input.setRange(1, 10000)
        self.exposure_input.setSuffix(" ms")

        layout.addRow("Save images:", self.save_images_checkbox)
        layout.addRow("Save interval:", self.save_interval_input)
        layout.addRow("Gain:", self.gain_input)
        layout.addRow("Exposure time:", self.exposure_input)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

class AlignerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Laser Alignment System")
        self.setGeometry(100, 100, 1100, 650)
        
        self.save_images = False
        self.save_interval = 3
        self.gain = 0.0
        self.exposure_time = 7000

        self.current_x = 0.0
        self.current_y = 0.0
        self.target_camera = None
        self.target_x = None
        self.target_y = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.tabs.addTab(self.create_camera_tab(), "Cameras")
        self.tabs.addTab(self.create_logs_tab(), "Logs")

        alignment_panel = self.create_alignment_panel()
        main_layout.addWidget(alignment_panel)

        self.hw_thread = HardwareThread()
        self.hw_thread.frame_ready.connect(self.display_camera_frame)
        self.hw_thread.laser_pos_update.connect(self.update_current_position)
        self.hw_thread.log_msg.connect(self.log)
        self.hw_thread.status_msg.connect(self.set_status_message)
        self.hw_thread.start() 

        self.log("App started.")
        self.set_status_message("Cameras warming up...")

    def create_camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        self.camera_status_label = QLabel()
        self.camera_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_status_label.setStyleSheet("background-color: #2b2b2b; color: white; padding: 10px; border-radius: 6px; font-size: 16px; font-weight: bold;")
        layout.addWidget(self.camera_status_label)

        camera_row = QHBoxLayout()
        
        frame1 = QFrame()
        frame1.setFrameShape(QFrame.Shape.Box)
        layout1 = QVBoxLayout(frame1)
        layout1.addWidget(QLabel("<b>Camera 1</b>", alignment=Qt.AlignmentFlag.AlignCenter))
        self.cam1_view = ClickableCameraView("Camera 1")
        self.cam1_view.clicked.connect(self.set_target_position)
        layout1.addWidget(self.cam1_view)
        
        frame2 = QFrame()
        frame2.setFrameShape(QFrame.Shape.Box)
        layout2 = QVBoxLayout(frame2)
        layout2.addWidget(QLabel("<b>Camera 2</b>", alignment=Qt.AlignmentFlag.AlignCenter))
        self.cam2_view = ClickableCameraView("Camera 2")
        self.cam2_view.clicked.connect(self.set_target_position)
        layout2.addWidget(self.cam2_view)

        camera_row.addWidget(frame1)
        camera_row.addWidget(frame2)
        layout.addLayout(camera_row)
        return tab

    def create_logs_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)
        return tab

    def create_alignment_panel(self):
        panel = QFrame()
        panel.setFixedWidth(280)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        layout.addWidget(QLabel("<b>Alignment Controls</b>", alignment=Qt.AlignmentFlag.AlignCenter))
        
        self.current_position_label = QLabel()
        self.target_position_label = QLabel()
        layout.addWidget(self.current_position_label)
        layout.addWidget(self.target_position_label)
        
        self.update_position_display()
        self.update_target_display()

        go_button = QPushButton("Go to Target")
        go_button.clicked.connect(self.go_to_target)
        layout.addWidget(go_button)

        start_btn = QPushButton("Start Alignment")
        stop_btn = QPushButton("Stop")
        start_btn.clicked.connect(self.start_alignment)
        stop_btn.clicked.connect(self.stop_alignment)
        layout.addWidget(start_btn)
        layout.addWidget(stop_btn)

        self.save_images_label = QLabel()
        layout.addWidget(self.save_images_label)
        self.update_settings_display()

        layout.addStretch()
        return panel

    def open_settings(self):
        dialog = SettingsDialog()
        dialog.save_images_checkbox.setChecked(self.save_images)
        dialog.save_interval_input.setValue(self.save_interval)
        dialog.gain_input.setValue(self.gain)
        dialog.exposure_input.setValue(self.exposure_time)

        if dialog.exec():
            self.save_images = dialog.save_images_checkbox.isChecked()
            self.save_interval = dialog.save_interval_input.value()
            self.gain = dialog.gain_input.value()
            self.exposure_time = dialog.exposure_input.value()
            
            self.update_settings_display()
            
            self.hw_thread.save_images = self.save_images
            self.hw_thread.save_interval = self.save_interval
            self.hw_thread.update_camera_settings(self.exposure_time, self.gain)

    def display_camera_frame(self, cam_index, img_data):
        if cam_index == 0: self.cam1_view.update_image(img_data)
        elif cam_index == 1: self.cam2_view.update_image(img_data)

    def update_current_position(self, cam_idx, x, y):
        self.current_x = x
        self.current_y = y
        self.update_position_display(f"Camera {cam_idx + 1}")

    def set_target_position(self, camera_name, x, y):
        self.target_camera = camera_name
        self.target_x = x
        self.target_y = y
        self.update_target_display()
        self.set_status_message(f"Target selected on {camera_name}: X={x}, Y={y}")

    def go_to_target(self):
        if self.target_x is None:
            self.set_status_message("Select a target first!")
            return
        
        cam_idx = 0 if self.target_camera == "Camera 1" else 1
        
        self.log(f"Manual override: Target set for {self.target_camera} at X={self.target_x}, Y={self.target_y}")
        self.set_status_message(f"Executing manual target move on {self.target_camera}...")
        
        self.hw_thread.execute_manual_move(cam_idx, self.target_x, self.target_y)

    def update_position_display(self, active_cam="None"):
        self.current_position_label.setText(f"Laser Tracking ({active_cam}):\nX = {self.current_x}\nY = {self.current_y}")

    def update_target_display(self):
        if self.target_x is None:
            self.target_position_label.setText("Target position:\nNone selected")
        else:
            self.target_position_label.setText(f"Target position ({self.target_camera}):\nX = {self.target_x}\nY = {self.target_y}")

    def update_settings_display(self):
        if self.save_images:
            self.save_images_label.setText(f"Save images: ON ({self.save_interval}s)")
        else:
            self.save_images_label.setText("Save images: OFF")

    def start_alignment(self):
        self.hw_thread.manual_target_active = False
        self.hw_thread.is_aligning = True

        self.hw_thread.cam1_locked = False
        self.hw_thread.cam2_locked = False
        self.hw_thread.cam1_stable_count = 0
        self.hw_thread.cam2_stable_count = 0
        self.hw_thread.cam1_drift_count = 0
        self.hw_thread.cam2_drift_count = 0
        self.hw_thread.system_locked_stop_sent = False
        self.hw_thread.alignment_cooldown = 0.0

        self.set_status_message("Auto-alignment Running...")
        self.log("Started alignment algorithm.")

    def stop_alignment(self):
        self.hw_thread.stop_all_movement()
        self.log("Stopped all alignment and halted motors.")

    def set_status_message(self, msg):
        self.camera_status_label.setText(msg)

    def log(self, msg):
        if hasattr(self, "log_box"):
            self.log_box.append(msg)
        print(msg)

    def closeEvent(self, event):
        self.log("Shutting down hardware cleanly...")
        self.hw_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AlignerApp()
    window.show()
    sys.exit(app.exec())