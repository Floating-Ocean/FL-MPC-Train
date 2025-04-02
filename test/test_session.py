import unittest
import uuid
import time
import os
from multiprocessing import Manager, Process

from train.session import open_session, check_classify_acc, get_available_models


class SessionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manager = Manager()
        cls.task_id = uuid.uuid4()
        cls.test_output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_step_0_get_models(self):
        print(get_available_models())

    def test_step_1_successful_training(self):
        training_status = self.manager.dict()
        p = Process(target=open_session,
                    args=(self.task_id, 10, 'mnist', self.test_output_dir, training_status))
        p.start()

        try:
            status_history = []

            # 模拟前端轮询
            while True:
                current_status = training_status.get(self.task_id)

                if not current_status:
                    time.sleep(0.1)
                    continue

                # 记录状态变化
                if len(status_history) == 0 or status_history[-1] != current_status:
                    status_history.append(current_status)
                    print(f"状态更新 [{current_status['status']}]: {current_status.get('data', 'None')}")

                # 终止条件
                if current_status["status"] in ("FINISHED", "FAILED"):
                    break

                time.sleep(0.5)  # 模拟500ms轮询间隔

            # 验证最终状态
            self.assertEqual("FINISHED", current_status["status"],
                             f"训练失败，最终状态: {current_status}")

            # 验证输出文件
            expected_files = [
                f"{self.task_id}.pth",
                f"{self.task_id}_metrics.png"
            ]
            for f in expected_files:
                self.assertTrue(
                    os.path.exists(os.path.join(self.test_output_dir, f)),
                    f"文件 {f} 未生成"
                )

        finally:
            if p.is_alive():
                p.terminate()
            p.join()

    def test_step_2_failed_training(self):
        training_status = self.manager.dict()
        # 使用无效数据集测试失败场景
        p = Process(target=open_session,
                    args=(self.task_id, 2, 'invalid_dataset', self.test_output_dir, training_status))
        p.start()

        try:
            start_time = time.time()
            timeout = 60

            while time.time() - start_time < timeout:
                current_status = training_status.get(self.task_id)
                if current_status and current_status["status"] == "FAILED":
                    break
                time.sleep(0.1)
            else:
                self.fail("失败场景未正确处理")

            # 验证错误信息
            error_data = current_status["data"]
            self.assertIn("Unknown dataset", error_data)
            print(f"预期错误捕获成功: {error_data[:100]}...")

        finally:
            if p.is_alive():
                p.terminate()
            p.join()

    def test_step_3_img_acc(self):
        acc_dict = self.manager.dict()
        model_path = os.path.join(self.test_output_dir, f"{self.task_id}.pth")
        img_path = os.path.join(os.path.dirname(__file__), "test_mnist.png")
        p = Process(target=check_classify_acc, args=(model_path, img_path, acc_dict))
        p.start()
        p.join()
        self.assertIn("result", acc_dict)
        print("测试结果: ", acc_dict['result'])
        print("最有可能是: ", max(acc_dict['result'], key=acc_dict['result'].get))


if __name__ == '__main__':
    unittest.main()