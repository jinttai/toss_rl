# ==================== 디버깅 코드 시작 ====================
import sys
import os
print("--- 파이썬 환경 진단 시작 ---")
print(f"실행된 파이썬 경로: {sys.executable}")
print("\n--- 파이썬이 모듈을 찾는 경로 목록 (sys.path) ---")
for path in sys.path:
    print(path)
print("\n--- PYTHONPATH 환경 변수 ---")
print(os.environ.get('PYTHONPATH', 'PYTHONPATH 변수가 설정되지 않았습니다.'))
print("--- 파이썬 환경 진단 끝 ---\n\n")
# ===================== 디버깅 코드 끝 =====================
import os
import carb
# 최신 Isaac Sim API를 사용합니다.
from isaacsim.simulation_app import SimulationApp
from isaacsim.core.simulation_context import SimulationContext
from isaacsim.core.world import World
from isaacsim.core.robots import Robot
from isaacsim.core.utils.prims import delete_prim_if_exists

# --------------------------------------------------------------------------
# 1. 시뮬레이션 앱 초기 설정
# 모든 설정을 하나의 딕셔너리 안에 포함시켜 전달합니다.
# --------------------------------------------------------------------------
kit = SimulationApp(
    {
        "headless": True,
        "physics_dt": 1.0 / 60.0,
        "rendering_dt": 1.0 / 60.0,
    }
)

# --------------------------------------------------------------------------
# 2. 시뮬레이션 및 월드 환경 설정
# --------------------------------------------------------------------------
# SimulationContext와 World 객체를 직접 생성합니다.
simulation_context = SimulationContext()
world = World(stage_units_in_meters=1.0)

# 시뮬레이션에 사용할 물리 엔진과 장치를 설정합니다.
simulation_context.set_setting("/physics/solverType", "TGS") # TGS 솔버 사용
simulation_context.set_setting("/physics/gpu", 0) # 사용할 GPU ID

# --------------------------------------------------------------------------
# 3. 에셋 및 로봇 설정
# --------------------------------------------------------------------------
# !!! 중요 !!!
# 시뮬레이션하려는 MuJoCo XML 파일의 '절대 경로'를 여기에 입력하세요.
# 예: "/home/user/my_project/models/my_robot.xml"
MUJOCO_FILE_PATH = "/home/chengu/toss_ws/mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml" 

# MuJoCo 에셋 로드 함수
def load_mujoco_asset(asset_path: str, prim_path: str = "/World/MujocoRobot"):
    """지정된 경로의 MuJoCo 파일을 Isaac Sim으로 로드합니다."""
    # MJCF Importer 확장 기능이 활성화되었는지 확인합니다.
    # 헤드리스 모드에서는 종종 자동으로 로드되지만 명시적으로 호출하는 것이 안전합니다.
    carb.log_info("Enabling MJCF importer extension...")
    kit.app.get_extension_manager().set_extension_enabled_immediate("omni.importer.mjcf", True)
    
    # 기존에 동일한 경로의 프리미티브가 있다면 삭제하여 충돌을 방지합니다.
    delete_prim_if_exists(prim_path)
    
    # MJCF(MuJoCo) 파일을 임포트합니다.
    # 성공 여부(result)와 생성된 프리미티브의 경로(prim_path_out)를 반환합니다.
    result, prim_path_out = kit.app.get_app_interface().get_extension_manager().get_extension_interface("omni.importer.mjcf").import_mujoco_file(asset_path, prim_path)

    if result:
        carb.log_info(f"Successfully imported MuJoCo file: {asset_path}")
        # 임포트된 프리미티브를 사용하여 로봇 객체를 생성합니다.
        return Robot(prim_path=prim_path_out, name="mujoco_robot")
    else:
        carb.log_error(f"Failed to import MuJoCo file: {asset_path}")
        return None

# --------------------------------------------------------------------------
# 4. 시뮬레이션 실행
# --------------------------------------------------------------------------
# 월드에 기본 바닥 평면을 추가합니다.
world.scene.add_default_ground_plane()

# MuJoCo 로봇을 로드하고 월드에 추가합니다.
print(f"Attempting to load MuJoCo asset from: {MUJOCO_FILE_PATH}")
mujoco_robot = load_mujoco_asset(MUJOCO_FILE_PATH)

if mujoco_robot is None:
    print("Exiting due to failed MuJoCo import.")
    kit.app.stop()
    exit()

# 월드를 초기화합니다 (장면에 추가된 모든 객체들을 준비시킵니다).
world.reset()
# 로봇 객체의 내부 상태(관절 정보 등)를 초기화합니다.
mujoco_robot.initialize()

# 시뮬레이션을 시작합니다.
simulation_context.play()
print("Simulation started in headless mode. Running for 500 steps...")

# 시뮬레이션 루프
for i in range(500):
    # 물리 스텝을 진행합니다 (렌더링은 건너뜁니다).
    simulation_context.step(render=False)

    # 100 스텝마다 로봇의 관절 상태를 출력합니다.
    if i % 100 == 0:
        current_joint_positions = mujoco_robot.get_joint_positions()
        print(f"Step {i}: Joint Positions (first 3) = {current_joint_positions[:3]}")
    
    # 이 부분에 강화학습 정책으로부터 받은 행동(action)을 적용하는 로직을 추가할 수 있습니다.
    # 예:
    # observation = get_observation()
    # action = policy.get_action(observation)
    # mujoco_robot.get_articulation_controller().apply_action(action)

print("Simulation finished.")

# --------------------------------------------------------------------------
# 5. 종료
# --------------------------------------------------------------------------
# 시뮬레이션 중지 및 Isaac Sim 앱 종료
simulation_context.stop()
kit.app.stop()
print("Isaac Sim application stopped.")