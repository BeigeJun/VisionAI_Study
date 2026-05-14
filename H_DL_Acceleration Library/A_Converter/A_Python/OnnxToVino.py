from openvino import convert_model, save_model
import os

model_path = "D:/0. Model_Save_Folder/Model_Save_Folder_HA"

onnx_path = os.path.join(model_path, "model.onnx")
ir_xml    = os.path.join(model_path, "model.xml")
ov_model = convert_model(onnx_path)
save_model(ov_model, ir_xml)
print("Finish")

