import pyvista as pv

import numpy as np
import os
import vtk
import sys

# ================= 配置区域 =================
# 请确保路径正确 (指向你生成的 .vtu 文件)
vtu_file_path = "./results_comparison/BEST_HybridFEM_idx54.vtu"
save_output_dir = "./vis_output_full_1"
# ===========================================


def read_vtu_as_polydata(filepath):
    """强制读取并修复点云拓扑"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    print(f"�� 正在使用 vtkXMLPolyDataReader 强制读取: {filepath}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()

    mesh = pv.wrap(reader.GetOutput())

    # 手动添加 Vertex 单元，使其成为合法的 Point Cloud
    if mesh.n_points > 0 and mesh.n_cells == 0:
        print("   ⚠️ 检测到无单元拓扑，正在生成顶点单元 (Vertex Cells)...")
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(mesh)
        vertex_filter.Update()
        mesh = pv.wrap(vertex_filter.GetOutput())

    return mesh


def plot_all_results(vtu_path, output_dir):
    sys.stdout.flush()

    try:
        mesh = read_vtu_as_polydata(vtu_path)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # ============ 统一调整汽车姿态：让车头不再“朝下” ============
    # 这里以绕 X 轴旋转 90 度为例，你可以根据实际效果改成
    # rotate_x(180)、rotate_y(90)、rotate_z(180) 等
    mesh = mesh.copy()
    mesh.rotate_x(90, inplace=True)
    # ======================================================

    print(f"✅ 读取成功! 点数: {mesh.n_points}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    basename = os.path.basename(vtu_path).replace(".vtu", "")

    # 提取车身 (Is_Surface == 1)
    if "Is_Surface" in mesh.array_names:
        mask = mesh["Is_Surface"] == 1
        if np.any(mask):
            car = mesh.extract_points(mask)
        else:
            print("⚠️ 警告: Is_Surface 全为 0，无法提取车身")
            return
    else:
        print("⚠️ 警告: 没有 Is_Surface 字段，默认使用整个网格")
        car = mesh

    # =========================================================
    # 1. 绘制压力对比 (GT vs Pred vs Error)
    # =========================================================
    if "Pressure_GT" in car.array_names and "Pressure_Pred" in car.array_names:
        print("�� [1/3] 正在绘制压力场三视图 (GT, Pred, Error)...")

        # --- A. 计算统一的颜色范围 (让 GT 和 Pred 可比) ---
        p_min = min(np.min(car["Pressure_GT"]), np.min(car["Pressure_Pred"]))
        p_max = max(np.max(car["Pressure_GT"]), np.max(car["Pressure_Pred"]))
        print(f"   -> 压力范围: [{p_min:.2f}, {p_max:.2f}] Pa")

        # --- B. 绘制 Ground Truth ---
        p = pv.Plotter(off_screen=True)
        p.add_mesh(
            car,
            scalars="Pressure_GT",
            cmap="coolwarm",
            clim=[p_min, p_max],
            render_points_as_spheres=True,
            point_size=8,
            scalar_bar_args={"title": "Pressure GT (Pa)"},  # stitle -> scalar_bar_args
        )
        p.view_isometric()
        save_path = os.path.join(output_dir, f"{basename}_press_GT.png")
        p.screenshot(save_path)
        print(f"   ✅ 保存: {save_path}")

        # --- C. 绘制 Prediction ---
        p = pv.Plotter(off_screen=True)
        p.add_mesh(
            car,
            scalars="Pressure_Pred",
            cmap="coolwarm",
            clim=[p_min, p_max],
            render_points_as_spheres=True,
            point_size=8,
            scalar_bar_args={"title": "Pressure Pred (Pa)"},
        )
        p.view_isometric()
        save_path = os.path.join(output_dir, f"{basename}_press_Pred.png")
        p.screenshot(save_path)
        print(f"   ✅ 保存: {save_path}")

        # --- D. 绘制 Error (Diff) ---
        error = car["Pressure_Pred"] - car["Pressure_GT"]
        car["Pressure_Error_Calc"] = error
        max_err = np.max(np.abs(error))
        if max_err == 0:
            max_err = 1.0

        p = pv.Plotter(off_screen=True)
        p.add_mesh(
            car,
            scalars="Pressure_Error_Calc",
            cmap="bwr",
            clim=[-max_err, max_err],
            render_points_as_spheres=True,
            point_size=8,
            scalar_bar_args={"title": "Error (Pred - GT)"},
        )
        p.view_isometric()
        save_path = os.path.join(output_dir, f"{basename}_press_Error.png")
        p.screenshot(save_path)
        print(f"   ✅ 保存: {save_path}")

    # =========================================================
    # 2. 绘制稀疏传感器输入 (Input)
    # =========================================================
    if "Is_Sensor" in mesh.array_names:
        print("�� [2/3] 正在绘制稀疏传感器输入...")
        sensor_mask = mesh["Is_Sensor"] == 1
        if np.any(sensor_mask):
            sensors = mesh.extract_points(sensor_mask)

            p = pv.Plotter(off_screen=True)
            # 画淡灰色的车身轮廓作为背景
            p.add_mesh(
                car,
                color="gray",
                opacity=0.1,
                render_points_as_spheres=True,
                point_size=2,
            )
            # 画高亮的传感器点
            p.add_mesh(
                sensors,
                color="red",
                render_points_as_spheres=True,
                point_size=15,
                label="Sensors",
            )
            p.add_legend()
            p.view_isometric()
            save_path = os.path.join(output_dir, f"{basename}_input_sensors.png")
            p.screenshot(save_path)
            print(f"   ✅ 保存: {save_path}")

    # =========================================================
    # 3. 绘制流线 (Streamlines)
    # =========================================================
    if "Velocity_GT" in mesh.array_names:
        print("�� [3/3] 正在绘制流线图...")
        mesh.set_active_vectors("Velocity_GT")
        bounds = mesh.bounds

        # 根据车身坐标调整 seed 位置
        line_source = pv.Line(
            pointa=(bounds[0] + 0.1, bounds[2] + 0.2, bounds[4] + 0.2),
            pointb=(bounds[0] + 0.1, bounds[3] - 0.2, bounds[5] - 0.2),
            resolution=40,
        )

        try:
            streamlines = mesh.streamlines_from_source(
                line_source,
                vectors="Velocity_GT",
                max_time=20.0,
                surface_streamlines=False,
            )
            if streamlines.n_points > 0:
                p = pv.Plotter(off_screen=True)
                p.add_mesh(mesh.outline(), color="black")
                p.add_mesh(
                    car,
                    color="gray",
                    opacity=0.5,
                    render_points_as_spheres=True,
                    point_size=4,
                )
                p.add_mesh(
                    streamlines,
                    scalars="Velocity_GT",  # 使用速度向量的范数上色也可以: scalars=streamlines["Velocity_GT"][:, 0] 等
                    cmap="jet",
                    scalar_bar_args={"title": "Velocity"},
                    line_width=2,
                )
                p.view_isometric()
                save_path = os.path.join(output_dir, f"{basename}_streamlines.png")
                p.screenshot(save_path)
                print(f"   ✅ 保存: {save_path}")
            else:
                print("   ⚠️ 警告: 生成流线点数为 0，可能 Seed 位置不对")
        except Exception as e:
            print(f"   ⚠️ 流线绘制跳过: {e}")


if __name__ == "__main__":
    plot_all_results(vtu_file_path, save_output_dir)
