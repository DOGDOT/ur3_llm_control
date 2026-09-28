# UR3 LLM Manipulation & Task Planning

## 1. Giới thiệu dự án

Hệ thống được thiết kế theo kiến trúc 4 tầng phân cấp:
- **LLM Task Planner:** Chuyển đổi khẩu lệnh tự nhiên thành chuỗi kỹ năng thao tác (Robot Skills: `pick`, `place`, `home`).
- **State-Aware Intelligence:** Tự động ước lượng trạng thái thế giới (vị trí vật thể, vùng chiếm dụng) để lập kế hoạch vòng kín (Closed-Loop), giải tỏa vật cản trước khi xếp đặt theo cấu hình cá nhân hóa MSSV.
- **Task Validator & Robot Skills:** Kiểm tra tính hợp lệ hình học/ngữ nghĩa và điều phối thực thi.
- **MoveIt 2 & TF Tracking:** Sinh quỹ đạo chuyển động mượt mà, tối ưu hóa ràng buộc hướng tránh tự va chạm (`self-collision`).

---

## 2. Yêu cầu môi trường

- **Hệ điều hành:** Ubuntu 22.04 LTS
- **Framework:** ROS 2 Humble Hawksbill
- **Các gói ROS 2 phụ thuộc:**
  - `moveit2`, `moveit_msgs`
  - `shape_msgs`, `visualization_msgs`, `tf2_ros`
  - `ur_simulation_gazebo`, `ur_moveit_config`
- **Thư viện Python:**
  - `openai`, `pyyaml`
- **LLM Gateway:** 9Router (hoặc OpenAI-compatible proxy tương đương)

---

## 3. Hướng dẫn cài đặt & Khởi chạy

### Bước 1: Build Workspace
```bash
cd ~/ur_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3_llm_control --symlink-install
source install/setup.bash
```

### Bước 2: Bật cổng dịch vụ LLM
Mở terminal và kích hoạt proxy LLM (mặc định cổng 20128):

```bash
9router
```

### Bước 3: Khởi chạy môi trường mô phỏng MoveIt 2 & RViz

```bash
source /opt/ros/humble/setup.bash
source ~/ur_ws/install/setup.bash
ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py ur_type:=ur3
```

### Bước 4: Chạy Node điều khiển tác vụ

```bash
source /opt/ros/humble/setup.bash
source ~/ur_ws/install/setup.bash
ros2 run ur3_llm_control skill_executor
```

## 4. PUT YOUR REQUEST 
### warning !!!: don't make all zone full, that project haven't swap cube