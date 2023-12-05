import tensorflow.lite as tflite

def LoadModelTFlite(modelpath):
    model = tflite.Interpreter(model_path=modelpath)
    return model

def TFlitePredict(model, input_tensor):
    model.resize_tensor_input(0, input_tensor.shape, strict=True)
    model.allocate_tensors()

    input_tensor_index = model.get_input_details()[0]["index"]

    model.set_tensor(input_tensor_index, input_tensor)
    model.invoke()

    out_details = model.get_output_details()

    if len(out_details) == 1:
        return model.get_tensor(out_details[0]["index"])

    out_indexs = [None] * len(out_details)
    for _layer in out_details:
        index = int(_layer["name"].split(":")[-1])
        out_indexs[index] = _layer["index"]

    return [model.get_tensor(index) for index in out_indexs]


def cropBox(image, detection, margin):
    x1, y1, w, h = detection['box']
    x1 -= margin
    y1 -= margin
    w += 2*margin
    h += 2*margin
    if x1 < 0:
        w += x1
        x1 = 0
    if y1 < 0:
        h += y1
        y1 = 0
    return image[y1:y1+h, x1:x1+w]