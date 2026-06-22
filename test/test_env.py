import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

try:
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    from calvin_env.envs.play_table_env import PlayTableSimEnv

    print("✅ 基础依赖库及环境包引入成功。")

except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def main():
    config_dir = "/root/autodl-tmp/calvin_env/conf"

    print("🚀 正在通过 Hydra 读取配置文件...")

    try:
        # 避免重复初始化 Hydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(
            config_dir=config_dir,
            version_base=None,
            job_name="test_calvin_env",
        ):
            cfg = compose(
                config_name="config_data_collection",
                overrides=[
                    "env.show_gui=False",
                    "cameras=static_and_gripper",
                ],
            )

        print("📋 配置文件成功加载。")
        print(f"📷 相机配置名称: {list(cfg.cameras.keys())}")

        for name in cfg.cameras:
            print(
                f"  - {name}: "
                f"{cfg.cameras[name].width}×{cfg.cameras[name].height}"
            )

    except Exception:
        print("❌ Hydra 配置解析失败：")
        traceback.print_exc()
        return

    env = None

    try:
        print("🎬 正在创建 PlayTableSimEnv...")

        env = PlayTableSimEnv(
            robot_cfg=cfg.robot,
            seed=cfg.seed,
            use_vr=False,
            bullet_time_step=cfg.env.bullet_time_step,
            cameras=cfg.cameras,
            show_gui=False,
            scene_cfg=cfg.scene,
            use_scene_info=cfg.env.use_scene_info,
            use_egl=False,
        )

        print("🎉 环境实例化成功！")

        print("🔄 正在执行 env.reset()...")
        obs = env.reset()

        print("✅ 环境重置成功！")
        print(f"📦 观测键: {list(obs.keys())}")

        if "rgb_obs" not in obs:
            print("⚠️ 未发现 rgb_obs。")

            if isinstance(obs, dict):
                for key, value in obs.items():
                    print(f"{key}: {type(value)}")

            return

        rgb_dict = obs["rgb_obs"]

        print(f"📸 可用相机: {list(rgb_dict.keys())}")

        # 优先使用 gripper 相机
        if "rgb_gripper" in rgb_dict:
            camera_key = "rgb_gripper"
        else:
            camera_key = list(rgb_dict.keys())[0]

        print(f"📷 使用相机: {camera_key}")

        image = rgb_dict[camera_key]

        # Tensor -> NumPy
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()

        image = np.asarray(image)

        print(f"🖼️ 图像形状: {image.shape}")
        print(f"🖼️ 图像类型: {image.dtype}")

        # (C, H, W) -> (H, W, C)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.transpose(1, 2, 0)

        # (H, W, 1) -> (H, W)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[:, :, 0]

        # 浮点数裁剪
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"CALVIN View ({camera_key})")
        plt.axis("off")

        output_path = "figure/calvin_env_smoke_test.png"

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print("🎉 烟雾测试通过！")
        print(f"💾 图像已保存至: {os.path.abspath(output_path)}")

    except Exception:
        print("❌ 环境运行失败：")
        traceback.print_exc()

    finally:
        env = None


if __name__ == "__main__":
    main()
